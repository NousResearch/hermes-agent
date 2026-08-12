"""Idle-triggered compaction: interaction with the compression guards.

The idle trigger (``agent/turn_context.py``, opt-in via
``compression.idle_compact_after_seconds``) does not compress anything
itself — it routes through ``agent._compress_context`` (the forwarder to
``agent.conversation_compression.compress_context``), which owns ALL of the
automatic-compaction guards:

- the per-session compression lock (added after the idle-compaction PR was
  written — an idle-triggered compress racing a turn-triggered one on the
  same session_id must be serialized, not forked),
- the persisted summary-failure cooldown,
- the anti-thrash / fallback-streak breaker.

These tests pin that composition end-to-end with a real ``AIAgent`` wired to
a real ``SessionDB``: when a guard says no, the idle path must be a strict
no-op for the turn (no compressor call, no session rotation, no flush
re-baseline, no user-message re-anchor).
"""

from __future__ import annotations

import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB

from agent.turn_context import build_turn_context

from tests.agent.test_compression_concurrent_fork import _build_agent_with_db


def _prep_idle_agent(db: SessionDB, session_id: str, *, idle_after: int = 60,
                     idle_gap: float = 3600.0):
    """Real AIAgent (mock compressor) primed so the idle trigger is eligible."""
    agent = _build_agent_with_db(db, session_id)
    agent.compression_enabled = True
    agent.compression_idle_compact_after_seconds = idle_after
    agent._last_activity_ts = time.time() - idle_gap
    # The idle block reads these from the compressor; give the MagicMock real
    # numbers so the floor computation and the preflight gate behave.
    agent.context_compressor.threshold_tokens = 100_000
    agent.context_compressor.summary_target_ratio = 0.20
    agent.context_compressor.protect_first_n = 2
    agent.context_compressor.protect_last_n = 2
    # No active failure cooldown unless a test installs one.
    agent.context_compressor.get_active_compression_failure_cooldown = (
        lambda *a, **k: None
    )
    agent._cached_system_prompt = "SYSTEM"
    return agent


def _run_prologue(agent, history, user_message="hello again"):
    """Invoke ``build_turn_context`` the way ``conversation_loop`` does.

    The token-threshold preflight gate is pinned False so these tests
    exercise the IDLE trigger in isolation (the preflight path has its own
    coverage in ``test_turn_context.py``).
    """
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
         patch("agent.turn_context._should_run_preflight_estimate",
               return_value=False), \
         patch("agent.turn_context.estimate_request_tokens_rough",
               return_value=999_999):
        return build_turn_context(
            agent=agent,
            user_message=user_message,
            system_message=None,
            conversation_history=history,
            task_id=None,
            stream_callback=None,
            persist_user_message=None,
            restore_or_build_system_prompt=lambda *a, **k: None,
            install_safe_stdio=lambda: None,
            sanitize_surrogates=lambda s: s,
            summarize_user_message_for_log=lambda s: str(s),
            set_session_context=lambda _sid: None,
            set_current_write_origin=lambda _o: None,
            ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
        )


def _history(n: int = 20) -> list:
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_idle_compaction_runs_through_guarded_path_and_releases_lock(
    tmp_path: Path,
) -> None:
    """Happy path: the idle trigger reaches the real ``compress_context``.

    It must acquire + release the per-session lock, invoke the compressor
    exactly once, and (rotation mode) rotate the session — proving the trigger
    is wired through the guarded entrypoint rather than calling the compressor
    directly.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "IDLE_HAPPY"
    db.create_session(sid, source="cli")
    agent = _prep_idle_agent(db, sid)

    ctx = _run_prologue(agent, _history())

    agent.context_compressor.compress.assert_called_once()
    # Rotation mode (in_place=False in the shared fixture) creates a child.
    assert agent.session_id != sid
    # The lock keyed on the OLD session id must not leak.
    assert db.get_compression_lock_holder(sid) is None
    # The turn continues on the compacted transcript, with the user-message
    # anchor pointing at a live user row in the rebuilt list.
    assert 0 <= ctx.current_turn_user_idx < len(ctx.messages)
    assert ctx.messages[ctx.current_turn_user_idx].get("role") == "user"


def test_idle_compaction_defers_to_held_compression_lock(tmp_path: Path) -> None:
    """An idle-triggered compress racing another path must sit the round out.

    The per-session lock landed after the idle-compaction PR: when another
    path (turn-triggered preflight, background-review fork) already holds the
    lock, ``compress_context`` returns the input list unchanged. The idle
    block must treat that skip as a strict no-op: no compressor call, no
    rotation, no flush re-baseline, anchor untouched.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "IDLE_LOCKED"
    db.create_session(sid, source="cli")
    assert db.try_acquire_compression_lock(sid, "external_holder") is True

    agent = _prep_idle_agent(db, sid)
    history = _history()

    ctx = _run_prologue(agent, history)

    # Skipped: the compressor never ran and the session did not rotate.
    agent.context_compressor.compress.assert_not_called()
    assert agent.session_id == sid
    # The external holder still owns the lock (we must not have stolen or
    # released someone else's lease).
    assert db.get_compression_lock_holder(sid) == "external_holder"
    # Turn state untouched: full history + this turn's user message, anchor on
    # the just-appended message, flush baseline not re-baselined to None-then-
    # doubled semantics.
    assert len(ctx.messages) == len(history) + 1
    assert ctx.current_turn_user_idx == len(ctx.messages) - 1
    assert ctx.messages[ctx.current_turn_user_idx]["content"] == "hello again"


def test_idle_compaction_respects_anti_thrash_breaker(tmp_path: Path) -> None:
    """A tripped ineffective-compression breaker must block the idle trigger.

    The breaker lives in ``ContextCompressor._automatic_compression_blocked``
    and is consulted by ``compress_context`` for every non-forced entrypoint.
    The idle path is an automatic entrypoint, so two prior ineffective
    compactions must silence it too.
    """
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "IDLE_THRASH"
    db.create_session(sid, source="cli")
    agent = _prep_idle_agent(db, sid)

    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, sid)
    compressor._ineffective_compression_count = 2  # breaker tripped
    compressor.compress = MagicMock()
    agent.context_compressor = compressor

    ctx = _run_prologue(agent, _history())

    compressor.compress.assert_not_called()
    assert agent.session_id == sid
    assert len(ctx.messages) == len(_history()) + 1


def test_idle_compaction_respects_persisted_failure_cooldown(
    tmp_path: Path,
) -> None:
    """An active summary-failure cooldown must gate the idle trigger up front.

    The idle predicate itself consults
    ``get_active_compression_failure_cooldown`` — with a persisted cooldown in
    state.db the trigger must not even reach ``_compress_context``.
    """
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "IDLE_COOLDOWN"
    db.create_session(sid, source="cli")
    db.record_compression_failure_cooldown(sid, 4_000_000_000.0, "timeout")

    agent = _prep_idle_agent(db, sid)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100_000
    ):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, sid)
    compressor.compress = MagicMock()
    agent.context_compressor = compressor
    agent._compress_context = MagicMock()

    ctx = _run_prologue(agent, _history())

    agent._compress_context.assert_not_called()
    compressor.compress.assert_not_called()
    assert agent.session_id == sid
    assert len(ctx.messages) == len(_history()) + 1


def test_idle_compaction_disabled_by_default(tmp_path: Path) -> None:
    """With the default config (0) a huge idle gap must never trigger."""
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "IDLE_OFF"
    db.create_session(sid, source="cli")
    agent = _prep_idle_agent(db, sid, idle_after=0, idle_gap=10_000_000.0)
    agent._compress_context = MagicMock()

    ctx = _run_prologue(agent, _history())

    agent._compress_context.assert_not_called()
    agent.context_compressor.compress.assert_not_called()
    assert agent.session_id == sid
    assert len(ctx.messages) == len(_history()) + 1
