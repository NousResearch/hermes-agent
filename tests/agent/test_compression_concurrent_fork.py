"""Regression: prevent transcript fork when two paths compress the same session_id.

Damien's incident (Discord, 2026-05-28): a long Hermes session in a Discord
gateway hit the compression threshold at the end of a turn.  The parent agent
finished delivering the response and ``conversation_loop.py`` fired
``_spawn_background_review(...)`` — which builds a forked ``AIAgent`` that
inherits ``agent.session_id`` (see ``agent/background_review.py``::
``review_agent.session_id = agent.session_id``).  Roughly two seconds later
a synthetic ``Background process proc_… completed`` event arrived and
started a fresh turn on the same parent ``session_id`` (still cached in the
gateway's ``SessionEntry``).  Both paths hit preflight compression on the
same parent transcript and called ``_compress_context`` concurrently.  Each
ended the parent and created its own CHILD session in ``state.db``, both
parented to the same old id.  The gateway's ``SessionEntry`` only caught one
rotation; the other child became an orphan that silently accumulated writes.

Repro shape on Damien's machine:

  parent 20260527_234659_e65f0e  ended_at=set  end_reason='compression'
  child  20260528_113619_fc80e1  parent=20260527_234659_e65f0e  (in SessionEntry)
  child  <orphan>                parent=20260527_234659_e65f0e  (silent writes)

This regression simulates the two concurrent ``compress_context`` calls
against a shared ``state.db`` and asserts that the per-session compression
lock added in this PR prevents the orphan child.  Without the lock the
fixture deterministically produces 2 children; with the lock, exactly 1.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent_with_db(
    db: SessionDB,
    session_id: str,
    *,
    stub_compressor: bool = True,
):
    """Build an AIAgent that's wired to ``db`` and pinned to ``session_id``."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    # Stub the compressor so it returns deterministic output and DOESN'T make
    # an LLM call.  Sleep inside compress() so the two threads' rotations
    # actually overlap — without that the OS could happen to serialize them
    # and hide the bug.
    if not stub_compressor:
        return agent

    compressor = MagicMock()

    def _compress_with_overlap(*_a, **_kw):
        time.sleep(0.15)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    compressor.compress.side_effect = _compress_with_overlap
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    # The compressor is a stub — the one-time compression-model feasibility
    # probe would resolve a REAL auxiliary provider (credential pools, live
    # token exchanges over the network on some dev machines). That makes the
    # first _compress_context call in a test nondeterministically slow (>2s)
    # and flakes event-based timing assertions. Mark it done: these tests
    # exercise locking/fencing/rotation, never aux-model feasibility.
    agent._compression_feasibility_checked = True
    # These tests cover the ROTATION fallback path (forking, child sessions,
    # lock contention) — pin in_place=False so they keep exercising it
    # regardless of the global default (which flipped to True in #38763).
    agent.compression_in_place = False
    return agent


def _count_children(db: SessionDB, parent_sid: str) -> int:
    """Count rows in state.db whose parent_session_id == parent_sid."""
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?",
        (parent_sid,),
    ).fetchall()
    return len(rows)


def _live_child_id(db: SessionDB, parent_sid: str) -> str | None:
    """The single child id of ``parent_sid``, or None when there is none.

    Fails loudly on more than one child: callers use this to prove the agents
    converged on the winner's session, so a multi-child state is a fork and
    must not be silently reduced to 'the first row'.
    """
    rows = db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?",
        (parent_sid,),
    ).fetchall()
    assert len(rows) <= 1, f"expected at most one child of {parent_sid}, got {rows!r}"
    return rows[0][0] if rows else None


def _wait_for_touch(touch_calls: list[str], value: str, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value in touch_calls:
            return
        time.sleep(0.01)
    pytest.fail(f"Timed out waiting for touch activity {value!r}; calls={touch_calls!r}")


def test_compression_activity_heartbeat_touches_agent_during_long_compress(tmp_path: Path) -> None:
    """Long compression must refresh agent activity so gateway watchdogs do not fire."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)

    def _slow_compress(*_a, **_kw):
        _wait_for_touch(touch_calls, "context compression in progress")
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert touch_calls[0] == "context compression started"
    assert "context compression in progress" in touch_calls
    assert touch_calls[-1] == "context compression completed"
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_emits_client_status_events(tmp_path: Path) -> None:
    """The heartbeat must re-emit the compacting status, not just DB touches.

    Remote transports (e.g. the Android relay app) run idle-progress turn
    watchdogs that ``session.interrupt`` a turn after ~180s with no gateway
    events. Compression is silent on the event stream, so without periodic
    status heartbeats a long compression is killed mid-flight and retriggers
    forever on sessions near the context ceiling.
    """
    from agent.conversation_compression import (
        COMPACTION_HEARTBEAT_STATUS,
        is_compaction_progress_status,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_STATUS_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)
    status_events: list[tuple[str, str]] = []
    setattr(
        agent,
        "status_callback",
        lambda event, message: status_events.append((event, message)),
    )

    def _slow_compress(*_a, **_kw):
        _wait_for_touch(touch_calls, "context compression in progress")
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    agent._compress_context(messages, "sys", approx_tokens=120_000)

    heartbeats = [e for e in status_events if e[1] == COMPACTION_HEARTBEAT_STATUS]
    assert heartbeats, "no heartbeat status reached the client"
    # Same "lifecycle" key as the other compaction statuses so the TUI gateway
    # re-tags it to kind="compacting" and Telegram edits one bubble in place.
    assert {event for event, _ in heartbeats} == {"lifecycle"}
    assert is_compaction_progress_status(COMPACTION_HEARTBEAT_STATUS)
    # Exactly one routine start line precedes the first heartbeat; the
    # heartbeat no longer re-emits a start of its own (adapters without
    # send_or_update_status would otherwise post two messages).
    assert status_events[0][1] != COMPACTION_HEARTBEAT_STATUS
    # Every heartbeat is a periodic tick: none may precede the first
    # "in progress" DB touch, which is what start() would have produced.
    first_tick_touch = touch_calls.index("context compression in progress")
    assert first_tick_touch >= 1  # "started" touch came first
    assert len(heartbeats) <= touch_calls.count("context compression in progress")


def test_compression_heartbeat_is_silent_for_quiet_context_engines(tmp_path: Path) -> None:
    """A context engine that suppresses the routine start status opens no
    visible compaction phase; the heartbeat must not open one either (there
    would be no terminal edge to close it)."""
    from agent.conversation_compression import COMPACTION_HEARTBEAT_STATUS

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_QUIET_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)
    status_events: list[tuple[str, str]] = []
    setattr(
        agent,
        "status_callback",
        lambda event, message: status_events.append((event, message)),
    )

    def _slow_compress(*_a, **_kw):
        _wait_for_touch(touch_calls, "context compression in progress")
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with patch(
        "agent.conversation_compression.automatic_compaction_status_message",
        return_value="",
    ):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert all(m != COMPACTION_HEARTBEAT_STATUS for _, m in status_events)
    assert "context compression in progress" in touch_calls  # DB touches still ran


def test_lock_contender_preserves_terminal_compaction_lifecycle(tmp_path: Path) -> None:
    """A lock loser still closes the structured compaction lifecycle.

    The gateway independently filters this routine notice for chat surfaces
    unless ``compression.progress_notices`` is enabled.  The low-level event
    must remain available so the desktop can retire its compaction phase.
    """
    from agent.conversation_compression import COMPACTION_DONE_STATUS

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "LOCK_CONTENDER_STATUS_TEST"
    db.create_session(session_id, source="discord")
    assert db.try_acquire_compression_lock(session_id, "winner", ttl_seconds=60)

    agent = _build_agent_with_db(db, session_id)
    status_events: list[tuple[str, str]] = []
    setattr(
        agent,
        "status_callback",
        lambda event, message: status_events.append((event, message)),
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    returned, _system_prompt = agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
    )

    assert returned is messages
    assert getattr(agent, "_compression_skipped_due_to_lock", None) == "winner"
    assert status_events.count(("compacted", COMPACTION_DONE_STATUS)) == 1


def test_failed_session_split_does_not_announce_compaction_complete(tmp_path: Path) -> None:
    """A failed durable split must not emit a successful completion edge."""
    from agent.conversation_compression import COMPACTION_DONE_STATUS

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "FAILED_SPLIT_STATUS_TEST"
    db.create_session(session_id, source="discord")
    agent = _build_agent_with_db(db, session_id)
    setattr(agent, "compression_in_place", False)
    db.publish_compression_child = MagicMock(side_effect=RuntimeError("split boom"))
    status_events: list[tuple[str, str]] = []
    setattr(
        agent,
        "status_callback",
        lambda event, message: status_events.append((event, message)),
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    db.publish_compression_child.assert_called_once()
    assert ("compacted", COMPACTION_DONE_STATUS) not in status_events
    assert db.get_compression_lock_holder(session_id) is None


def test_failed_in_place_split_does_not_announce_compaction_complete(tmp_path: Path) -> None:
    """An in-place persistence failure must not emit a completion edge."""
    from agent.conversation_compression import COMPACTION_DONE_STATUS

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "FAILED_IN_PLACE_STATUS_TEST"
    db.create_session(session_id, source="discord")
    agent = _build_agent_with_db(db, session_id)
    setattr(agent, "compression_in_place", True)
    db.archive_and_compact = MagicMock(side_effect=RuntimeError("archive boom"))
    status_events: list[tuple[str, str]] = []
    setattr(
        agent,
        "status_callback",
        lambda event, message: status_events.append((event, message)),
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    db.archive_and_compact.assert_called_once()
    assert ("compacted", COMPACTION_DONE_STATUS) not in status_events
    assert getattr(agent, "session_id", None) == session_id
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_stops_on_compress_exception(tmp_path: Path) -> None:
    """Exception paths must stop the heartbeat and release the compression lock."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_FAIL_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)

    def _failing_compress(*_a, **_kw):
        _wait_for_touch(touch_calls, "context compression in progress")
        raise RuntimeError("compress boom")

    agent.context_compressor.compress.side_effect = _failing_compress
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="compress boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert touch_calls[0] == "context compression started"
    assert "context compression in progress" in touch_calls
    assert touch_calls[-1] == "context compression failed"
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_ignores_touch_errors(tmp_path: Path) -> None:
    """Activity touch failures must not affect compression success semantics."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_TOUCH_ERROR_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = 0.1
    agent._touch_activity = lambda _desc, **_kw: (_ for _ in ()).throw(RuntimeError("touch boom"))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed[0]["content"] == "[CONTEXT COMPACTION] summary"
    assert db.get_compression_lock_holder(session_id) is None


def test_compression_activity_heartbeat_strict_signature_fallback_releases_lock(tmp_path: Path) -> None:
    """Strict compressor signatures still compress while heartbeat cleanup runs.

    Main inspects the engine signature up front (_supported_compression_kwargs)
    instead of catching TypeError, so a strict-signature engine is invoked
    exactly once with only the kwargs it accepts. The heartbeat (with a
    non-numeric configured interval falling back to the default) must still
    wrap the call and stop cleanly.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_TYPEERROR_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    agent._compression_activity_heartbeat_interval = "not-a-number"
    touch_calls: list[str] = []
    agent._touch_activity = lambda desc, **_kw: touch_calls.append(desc)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    strict_calls: list[int | None] = []

    def _strict_compress(messages, current_tokens=None):
        strict_calls.append(current_tokens)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] strict summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress = _strict_compress

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert compressed[0]["content"] == "[CONTEXT COMPACTION] strict summary"
    assert touch_calls[0] == "context compression started"
    assert touch_calls[-1] == "context compression completed"
    assert db.get_compression_lock_holder(session_id) is None
    assert strict_calls == [120_000]


def test_compression_activity_heartbeat_nonfinite_interval_falls_back(tmp_path: Path) -> None:
    """Non-finite heartbeat intervals must not reach Event.wait()."""
    from agent.conversation_compression import _CompressionActivityHeartbeat

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_NONFINITE_INTERVAL_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    touch_calls: list[str] = []
    touch_provenances: list = []

    def _capture(desc, *, provenance=None, force_persist=False):
        touch_calls.append(desc)
        touch_provenances.append(provenance)

    agent._touch_activity = _capture

    heartbeat = _CompressionActivityHeartbeat(agent, interval_seconds=float("inf"))

    assert heartbeat._interval_seconds == 60.0
    heartbeat.start()
    heartbeat.stop()
    assert touch_calls == ["context compression started", "context compression completed"]
    from agent.session_activity import ActivityProvenance

    assert touch_provenances == [
        ActivityProvenance.AGENT_COMPRESSION,
        ActivityProvenance.AGENT_COMPRESSION,
    ]


def test_compression_heartbeat_stop_persists_completed_over_in_progress(
    tmp_path: Path,
) -> None:
    """/compress is outside run_conversation, so turn-end clear never runs.

    Heartbeat progress stamps persist to SessionDB; completion is often
    rate-limited out. stop() must force-persist the terminal label so idle
    sessions show "context compression completed", not a stale "in progress".
    """
    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HEARTBEAT_PERSIST_COMPLETED_TEST"
    db.create_session(session_id, source="test")

    agent = _build_agent_with_db(db, session_id)
    # Long interval: only start/stop touch; we inject the progress stamp.
    hb = _CompressionActivityHeartbeat(agent, interval_seconds=3600.0)
    hb.start()

    agent._session_activity_last_persist_mono = 0.0
    agent._touch_activity(
        "context compression in progress",
        provenance=ActivityProvenance.AGENT_COMPRESSION,
    )
    row = db.get_session(session_id)
    assert row["last_activity_description"] == "context compression in progress"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION.value

    # Mimic the common case: completion falls inside the 60s persist window.
    agent._session_activity_last_persist_mono = time.monotonic()
    hb.stop("context compression completed")

    row = db.get_session(session_id)
    assert row["last_activity_description"] == "context compression completed"
    assert row["last_activity_provenance"] == ActivityProvenance.AGENT_COMPRESSION.value
    assert agent._last_activity_desc == "context compression completed"
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION


def test_compression_heartbeat_does_not_clobber_timeout_provenance() -> None:
    """Detached heartbeat/stop must not overwrite a host timeout stamp."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        _last_activity_desc="context compression timed out",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    hb._touch("context compression in progress")
    hb.stop("context compression completed")

    assert agent.touches == []
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION_TIMEOUT
    assert agent._last_activity_desc == "context compression timed out"


def test_compression_heartbeat_does_not_clobber_cooldown_provenance() -> None:
    """Cooldown/abort stamps must also survive a late heartbeat stop."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_COOLDOWN,
        _last_activity_desc="compression blocked (cooldown: 30s remaining)",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    hb._touch("context compression in progress")
    hb.stop("context compression failed")

    assert agent.touches == []
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION_COOLDOWN


def test_compression_heartbeat_start_republishes_after_terminal_provenance() -> None:
    """A new compression episode may overwrite a prior timeout/cooldown stamp."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        _last_activity_desc="context compression timed out",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    hb.start()
    hb.stop()

    assert agent.touches[0] == (
        "context compression started",
        ActivityProvenance.AGENT_COMPRESSION,
    )
    assert agent.touches[-1] == (
        "context compression completed",
        ActivityProvenance.AGENT_COMPRESSION,
    )
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION


def test_compression_heartbeat_does_not_rearm_after_unknown_provenance() -> None:
    """After a terminal stamp, UNKNOWN must not re-arm a detached heartbeat."""
    from types import SimpleNamespace

    from agent.conversation_compression import _CompressionActivityHeartbeat
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION_TIMEOUT,
        _last_activity_desc="context compression timed out",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    hb = _CompressionActivityHeartbeat(agent, interval_seconds=60.0)
    # First tick observes TIMEOUT and latches silent.
    hb._touch("context compression in progress")
    assert hb._suppressed is True
    # Turn continues / ends and clears labels to UNKNOWN — must stay silent.
    agent._last_activity_provenance = ActivityProvenance.UNKNOWN
    agent._last_activity_desc = "calling model"
    hb._touch("context compression in progress")
    hb.stop("context compression completed")

    assert agent.touches == []
    assert agent._last_activity_provenance is ActivityProvenance.UNKNOWN
    assert agent._last_activity_desc == "calling model"


def test_compression_heartbeat_stops_when_commit_fence_cancelled() -> None:
    """Host fence cancel must silence detached heartbeat refresh and late stop."""
    from types import SimpleNamespace

    from agent.conversation_compression import (
        CompressionCommitFence,
        _CompressionActivityHeartbeat,
    )
    from agent.session_activity import ActivityProvenance

    agent = SimpleNamespace(
        _last_activity_provenance=ActivityProvenance.AGENT_COMPRESSION,
        _last_activity_desc="context compression started",
        touches=[],
    )

    def _touch(desc, *, provenance=None, force_persist=False):
        agent.touches.append((desc, provenance))
        agent._last_activity_provenance = provenance
        agent._last_activity_desc = desc

    agent._touch_activity = _touch

    fence = CompressionCommitFence()
    assert fence.cancel_before_commit() is True

    hb = _CompressionActivityHeartbeat(
        agent, interval_seconds=60.0, commit_fence=fence
    )
    hb._touch("context compression in progress")
    hb.stop("context compression completed")

    assert agent.touches == []
    assert hb._suppressed is True
    assert agent._last_activity_provenance is ActivityProvenance.AGENT_COMPRESSION
    assert agent._last_activity_desc == "context compression started"


def test_concurrent_compression_does_not_fork_session(tmp_path: Path) -> None:
    """Two AIAgents that share a session_id MUST NOT both rotate it.

    Without the per-session compression lock this fixture deterministically
    produces 2 child sessions (transcript fork). With the lock at most one
    path rotates: normally exactly 1 canonical child, or — under heavy DB
    write contention that makes the winner's child create_session exhaust its
    retries — 0, because _compress_context safely rolls back to the parent
    instead of orphaning a child. The forbidden outcome is 2+ (the fork).
    """
    db = SessionDB(db_path=tmp_path / "state.db")

    parent_sid = "PARENT_TEST_SESSION"
    db.create_session(parent_sid, source="discord")

    # Two agents on the same session_id, both wired to the same db —
    # mirrors the parent-turn agent + the background-review fork right
    # after a turn ends.
    agent_a = _build_agent_with_db(db, parent_sid)
    agent_b = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def run(agent):
        try:
            agent._compress_context(messages, "sys", approx_tokens=120_000)
        except Exception:
            # Surface to the test if either raises — should not happen.
            raise

    t_a = threading.Thread(target=run, args=(agent_a,), name="main_turn")
    t_b = threading.Thread(target=run, args=(agent_b,), name="review_fork")
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    # The invariant Damien's incident is about: the parent must NEVER end up
    # with two (or more) children — that is the transcript fork. The lock
    # guarantees only one path rotates.
    #
    # Zero children is also a valid, non-forking outcome: under heavy DB write
    # contention the winner's child ``create_session`` can exhaust its retry
    # budget, and ``_compress_context`` deliberately rolls the live id back to
    # the (still-indexed) parent rather than stranding an orphan child — see
    # the create-failure rollback in agent/conversation_compression.py. That
    # safe rollback leaves 0 children and is correct. So the contract is
    # ``children <= 1``; only ``>= 2`` is the bug. Asserting an exact ``== 1``
    # made this test flaky under the concurrent CI load that triggers the
    # contention rollback (#54465 churn surfaced it).
    n_children = _count_children(db, parent_sid)
    assert n_children <= 1, (
        f"Compression lock failed: parent session has {n_children} children in "
        "state.db (transcript fork). This is Damien's incident shape — see the "
        "test docstring. Two or more children means the lock did not serialize "
        "the concurrent rotations."
    )

    # Every agent that moved off the parent must have landed on the SAME id.
    # Counting movers is the wrong contract: the loser can legitimately end up
    # on the child too, without rotating anything itself — it takes the lock
    # after the winner released it, sees the parent was already rotated, and
    # _adopt_live_compression_child() points it at the winner's single child
    # (the "compression recovery: stale session=... adopted live child=..."
    # path). That convergence is the fix working, not a fork; the fork is two
    # DIFFERENT live ids. Asserting ``movers <= 1`` failed on that healthy
    # outcome under concurrent load.
    moved = {a.session_id for a in (agent_a, agent_b) if a.session_id != parent_sid}
    assert len(moved) <= 1, (
        f"Expected at most one post-compression session id, got {sorted(moved)}. "
        "Two distinct ids means the lock didn't serialize them (transcript fork)."
    )
    assert len(moved) == n_children, (
        f"Inconsistent state: agents live on {sorted(moved)} but {n_children} "
        "child session(s) exist — rotation and child creation diverged."
    )
    if moved:
        child = _live_child_id(db, parent_sid)
        assert moved == {child}, (
            f"Agents live on {sorted(moved)} but the parent's only child is "
            f"{child} — an agent is writing to a session outside the lineage."
        )

    # The lock must be released after both paths finished, regardless of
    # whether the winner committed a child or rolled back.
    assert db.get_compression_lock_holder(parent_sid) is None, (
        "Compression lock leaked: still held after both paths completed."
    )


def test_durable_message_committed_before_lease_is_adopted(
    tmp_path: Path,
) -> None:
    """A durable row absent from the caller snapshot must still be compressed.

    Previously this path aborted and returned the stale snapshot unchanged,
    which permanently wedged busy sessions: every compress attempt saw the
    DB ahead of the in-memory list, logged "changed before lease
    acquisition", and never called the compressor. Adopting the durable
    transcript keeps the late-committed turn and lets compression proceed.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PRE_LEASE_DURABLE_RACE"
    db.create_session(parent_sid, source="webui")
    db.append_message(parent_sid, "user", "old durable")

    # Frontend takes its snapshot, then another producer commits before this
    # compressor acquires the lease.
    stale_snapshot = [{"role": "user", "content": "old durable"}]
    db.append_message(parent_sid, "assistant", "late committed before lease")
    agent = _build_agent_with_db(db, parent_sid)

    returned, _system_prompt = agent._compress_context(
        stale_snapshot, "sys", approx_tokens=120_000
    )

    agent.context_compressor.compress.assert_called_once()
    compressed_arg = agent.context_compressor.compress.call_args.args[0]
    assert [m["content"] for m in compressed_arg] == [
        "old durable",
        "late committed before lease",
    ]
    # Must not echo the stale snapshot — compression proceeded on the
    # adopted durable transcript (rotation publishes a child session).
    assert returned is not stale_snapshot
    assert returned[0]["content"] == "[CONTEXT COMPACTION] summary"
    assert agent.session_id != parent_sid
    child_id = _live_child_id(db, parent_sid)
    assert child_id is not None
    assert child_id == agent.session_id





def test_fence_cancelled_compression_leaves_lock_reacquirable(tmp_path: Path) -> None:
    """A fence-cancelled attempt must not poison the per-session lock.

    Lock-release verification for the hygiene-timeout path: after the gateway
    times out and cancels a hygiene compression at the commit fence, the very
    next attempt on the same session (e.g. the user running ``/compress``)
    must acquire the compression lock and commit normally. A leaked lock here
    would silently block every future compaction for the session until TTL
    expiry.
    """
    from agent.conversation_compression import CompressionCommitFence

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HYGIENE_LOCK_REACQUIRE"
    db.create_session(session_id, source="telegram")

    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    summary_started = threading.Event()
    release_summary = threading.Event()

    def _slow_summary(*_args, **_kwargs):
        summary_started.set()
        assert release_summary.wait(timeout=5)
        agent.context_compressor._proactive_prune_rearm_tokens = 0
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _slow_summary
    agent.context_compressor._proactive_prune_rearm_tokens = 120_000
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    fence = CompressionCommitFence()
    result = {}

    def _run_compression() -> None:
        result["value"] = agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
            commit_fence=fence,
        )

    worker = threading.Thread(target=_run_compression, name="fenced-hygiene")
    worker.start()
    assert summary_started.wait(timeout=2)
    assert fence.cancel_before_commit() is True
    release_summary.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    # Cancelled attempt: no mutation, and — the invariant under test — the
    # per-session compression lock is fully released.
    assert result["value"][0] is messages
    assert agent.context_compressor._proactive_prune_rearm_tokens == 120_000
    assert db.get_compression_lock_holder(session_id) is None

    # The NEXT attempt (no fence — a manual /compress retry) must be able to
    # acquire the lock and commit an in-place compaction normally.
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] retry summary"},
        {"role": "user", "content": "tail"},
    ]
    retried, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert retried is not messages
    assert len(retried) < len(messages)
    assert agent.session_id == session_id  # in-place: same session id
    assert agent._last_compaction_in_place is True
    assert db.get_compression_lock_holder(session_id) is None


def test_commit_fence_waits_for_an_active_commit() -> None:
    """A timeout that loses the fence race cannot overlap the live turn."""
    from agent.conversation_compression import CompressionCommitFence

    fence = CompressionCommitFence()
    assert fence.begin_commit() is True
    assert fence.try_cancel_before_commit() is None
    cancel_started = threading.Event()
    cancel_finished = threading.Event()
    result = {}

    def _cancel() -> None:
        cancel_started.set()
        result["cancelled"] = fence.cancel_before_commit()
        cancel_finished.set()

    waiter = threading.Thread(target=_cancel, name="hygiene-timeout-fence")
    waiter.start()
    try:
        assert cancel_started.wait(timeout=2)
        assert not cancel_finished.is_set()
    finally:
        fence.finish_commit()
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert result["cancelled"] is False


def test_total_deadline_cancellation_retains_lock_until_worker_cleanup() -> None:
    """A total-ceiling timeout must exclude retries while its worker is alive."""
    from agent.conversation_compression import CompressionCommitFence

    released = threading.Event()
    fence = CompressionCommitFence(total_ceiling_seconds=1.0)
    fence.register_cancelled_lock_release(released.set)
    fence.retain_compression_lock_until_worker_done()

    now = time.monotonic()
    with patch(
        "agent.conversation_compression.time.monotonic",
        return_value=now + 2.0,
    ):
        assert fence.is_cancelled
        assert fence.try_cancel_before_commit() is True
    fence.release_cancelled_compression_lock()

    assert not released.is_set()


def test_delayed_contender_adopts_unique_rotated_child(tmp_path: Path) -> None:
    """A stale agent must continue on the winner's compacted child transcript."""
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "STALE_PARENT"
    child_sid = "CANONICAL_CHILD"
    db.create_session(parent_sid, source="webui")
    db.end_session(parent_sid, "compression")
    db.create_session(child_sid, source="webui", parent_session_id=parent_sid)
    compacted = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "compacted tail"},
    ]
    db.replace_messages(child_sid, compacted)

    agent = _build_agent_with_db(db, parent_sid)
    stale_messages = [
        {"role": "user", "content": "stale"},
        {"role": "assistant", "content": "x" * 1000},
    ]
    recovered, _system_prompt = agent._compress_context(
        stale_messages, "sys", approx_tokens=120_000
    )

    assert agent.session_id == child_sid
    assert [(m["role"], m["content"]) for m in recovered] == [
        ("user", "[CONTEXT COMPACTION] summary"),
        ("assistant", "compacted tail"),
    ]
    assert agent._session_db_created is True
    assert agent._flushed_db_message_session_id == child_sid
    assert agent._last_flushed_db_idx == len(recovered)
    agent.context_compressor.compress.assert_not_called()
    lifecycle_args, lifecycle_kwargs = agent.context_compressor.on_session_start.call_args
    assert lifecycle_args == (child_sid,)
    assert lifecycle_kwargs["boundary_reason"] == "compression"
    assert lifecycle_kwargs["old_session_id"] == parent_sid
    assert lifecycle_kwargs["session_db"] is db








def _no_consecutive_user_roles(messages: list) -> bool:
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    return all(
        not (roles[i] == roles[i + 1] == "user") for i in range(len(roles) - 1)
    )


def test_restored_anchor_never_creates_consecutive_user_roles() -> None:
    """Anchor restoration must preserve strict role alternation (#55677).

    The original insertion helper could land the human anchor directly next
    to user-role scaffolding (index-0 insert before a leading synthetic user
    turn, or a bare scaffolding-only transcript), producing user/user
    adjacency that strict chat templates reject.
    """
    from agent.conversation_compression import _insert_real_user_anchor

    anchor = {"role": "user", "content": "REAL HUMAN ASK"}

    # Leading synthetic user turn before the assistant summary.
    compressed = [
        {
            "role": "user",
            "content": "[System: Your previous response was truncated ...]",
            "_empty_recovery_synthetic": True,
        },
        {"role": "assistant", "content": "summary"},
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]",
            "_todo_snapshot_synthetic": True,
        },
    ]
    _insert_real_user_anchor(compressed, dict(anchor))
    assert _no_consecutive_user_roles(compressed)
    assert any(m.get("content", "").startswith("REAL HUMAN ASK") for m in compressed)

    # Scaffolding-only transcript: the anchor is merged, not inserted
    # adjacent, and the merged turn leads with the human ask.
    compressed = [
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]",
            "_todo_snapshot_synthetic": True,
        },
    ]
    _insert_real_user_anchor(compressed, dict(anchor))
    assert _no_consecutive_user_roles(compressed)
    assert len(compressed) == 1
    assert compressed[0]["content"].startswith("REAL HUMAN ASK")
    assert not compressed[0].get("_todo_snapshot_synthetic")




def test_compression_persists_child_handoff_immediately(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "HEADLESS_PREFLIGHT_PARENT"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)
    child_sid = agent.session_id

    assert child_sid != parent_sid
    assert db.get_session(parent_sid)["end_reason"] == "compression"
    assert len(db.get_messages(child_sid)) == len(compressed)

    agent._flush_messages_to_session_db(compressed, None)
    assert len(db.get_messages(child_sid)) == len(compressed)




def test_rotation_publish_failure_restores_proactive_prune_runway(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "PRUNE_RUNWAY_ROLLBACK_PARENT"
    db.create_session(
        parent_sid,
        source="cli",
        model_config={"keep": "value", "_proactive_prune_rearm_tokens": 120_000},
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    db.append_messages_batch(parent_sid, messages)
    for message in messages:
        message["_db_persisted"] = True
    agent = _build_agent_with_db(db, parent_sid)
    agent.context_compressor._proactive_prune_rearm_tokens = 120_000

    def _compress(*_args, **_kwargs):
        agent.context_compressor._proactive_prune_rearm_tokens = 0
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _compress
    durable_before = db.get_messages_as_conversation(parent_sid)
    with patch.object(
        db,
        "publish_compression_child",
        side_effect=RuntimeError("publish failed"),
    ):
        returned, _sp = agent._compress_context(
            messages, "sys", approx_tokens=120_000,
        )

    assert returned is messages
    assert agent.session_id == parent_sid
    assert agent.context_compressor._proactive_prune_rearm_tokens == 120_000
    assert db.get_messages_as_conversation(parent_sid) == durable_before
    assert json.loads(db.get_session(parent_sid)["model_config"]) == {
        "keep": "value",
        "_proactive_prune_rearm_tokens": 120_000,
    }


def test_full_in_place_compression_atomically_clears_durable_prune_runway(
    tmp_path: Path,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "IN_PLACE_CLEARS_PRUNE_RUNWAY"
    db.create_session(
        session_id,
        source="cli",
        model_config={"keep": "value", "_proactive_prune_rearm_tokens": 120_000},
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    db.append_messages_batch(session_id, messages)
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent.context_compressor._proactive_prune_rearm_tokens = 120_000

    compressed, _sp = agent._compress_context(
        messages, "sys", approx_tokens=120_000,
    )

    assert agent.session_id == session_id
    assert [message["content"] for message in db.get_messages_as_conversation(session_id)] == [
        message["content"] for message in compressed
    ]
    assert json.loads(db.get_session(session_id)["model_config"]) == {"keep": "value"}


def test_rotation_child_starts_without_durable_prune_runway(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ROTATION_CLEARS_PRUNE_RUNWAY"
    db.create_session(
        parent_sid,
        source="cli",
        model_config={"keep": "parent", "_proactive_prune_rearm_tokens": 120_000},
    )
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    db.append_messages_batch(parent_sid, messages)
    agent = _build_agent_with_db(db, parent_sid)

    agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert agent.session_id != parent_sid
    child_config = json.loads(db.get_session(agent.session_id)["model_config"])
    assert "_proactive_prune_rearm_tokens" not in child_config
    assert json.loads(db.get_session(parent_sid)["model_config"])[
        "_proactive_prune_rearm_tokens"
    ] == 120_000


@pytest.mark.parametrize("in_place", [False, True])
def test_equal_copy_compression_result_does_not_rewrite_session(
    tmp_path: Path,
    in_place: bool,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = f"EQUAL_COPY_NOOP_{in_place}"
    db.create_session(parent_sid, source="cli")

    agent = _build_agent_with_db(db, parent_sid)
    setattr(agent, "compression_in_place", in_place)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    compressor = getattr(agent, "context_compressor")
    compressor.compress.side_effect = lambda incoming, **_kw: list(incoming)

    with patch.object(
        db,
        "archive_and_compact",
        wraps=db.archive_and_compact,
    ) as archive_and_compact:
        returned, _sp = agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
        )

    assert returned is messages
    assert getattr(agent, "session_id") == parent_sid
    assert _count_children(db, parent_sid) == 0
    parent = db.get_session(parent_sid)
    assert parent is not None
    assert parent["end_reason"] is None
    assert db.get_compression_lock_holder(parent_sid) is None
    archive_and_compact.assert_not_called()




def test_post_compress_exception_stops_lock_refresher(tmp_path: Path, monkeypatch) -> None:
    """A warning-path exception after compress() returns must still release the lock."""
    real_try_acquire = SessionDB.try_acquire_compression_lock

    def _short_ttl(self, session_id: str, holder: str, ttl_seconds: float = 300.0) -> bool:
        return real_try_acquire(self, session_id, holder, ttl_seconds=0.15)

    monkeypatch.setattr(SessionDB, "try_acquire_compression_lock", _short_ttl)

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "REFRESH_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_ttl_seconds = 0.15
    agent._compression_lock_refresh_interval = 0.05
    agent.context_compressor._last_summary_error = "summary failed"
    agent._emit_warning = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("warn boom"))

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="warn boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    time.sleep(0.25)
    assert db.try_acquire_compression_lock(parent_sid, "probe", ttl_seconds=1.0) is True








def test_signature_introspection_exception_releases_lock_and_refresher(
    tmp_path: Path, monkeypatch
) -> None:
    """Capability inspection failures must not leak the acquired lock lease."""
    from agent.conversation_compression import (
        _CompressionLockLeaseRefresher as RealLeaseRefresher,
    )

    refreshers = []

    class RecordingLeaseRefresher(RealLeaseRefresher):
        def start(self):
            refreshers.append(self)
            return super().start()

    monkeypatch.setattr(
        "agent.conversation_compression._CompressionLockLeaseRefresher",
        RecordingLeaseRefresher,
    )

    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "SIGNATURE_EXCEPTION_TEST"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    agent._compression_lock_refresh_interval = 0.1

    class SignatureBomb:
        calls = 0

        @property
        def __signature__(self):
            raise RuntimeError("signature boom")

        def __call__(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("engine must not run after signature failure")

    bomb = SignatureBomb()
    agent.context_compressor.compress = bomb
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    with pytest.raises(RuntimeError, match="signature boom"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert bomb.calls == 0
    assert db.get_compression_lock_holder(parent_sid) is None
    assert len(refreshers) == 1
    assert not refreshers[0]._thread.is_alive()








def _make_legacy_session_db_class() -> type:
    """Model the class retained in ``sys.modules`` before the lock API existed.

    During the real version-skew incident, a re-imported compression module
    imports the same still-loaded ``hermes_state`` module, whose ``SessionDB``
    class is old. The test replaces that module attribute with this lockless
    class and forwards all persistence operations to a current real database.
    """
    source_path = inspect.getfile(SessionDB)
    namespace = {"__name__": "hermes_state"}
    source = '''
class SessionDB:
    def __init__(self, real_db):
        self._real = real_db

    def __getattribute__(self, name):
        if name in {"_real", "__class__"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)
'''
    exec(compile(source, source_path, "exec"), namespace)
    return namespace["SessionDB"]


class _NominalSessionDBImpostor:
    """A proxy that spoofs names but lacks the real SessionDB source contract."""

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    def create_session(self, *args, **kwargs):
        return self._real.create_session(*args, **kwargs)

    def __getattr__(self, name):
        if name == "try_acquire_compression_lock":
            raise AttributeError(name)
        return getattr(self._real, name)


_NominalSessionDBImpostor.__module__ = "hermes_state"
_NominalSessionDBImpostor.__name__ = "SessionDB"


class _BrokenLockLookupDB:
    """A present lock API whose instance lookup fails unexpectedly."""

    def __init__(self, real_db: SessionDB, error: Exception) -> None:
        self._real = real_db
        self._error = error

    def try_acquire_compression_lock(self, *_args, **_kwargs):
        raise AssertionError("the broken lookup must not resolve to a callable")

    def __getattribute__(self, name):
        if name == "try_acquire_compression_lock":
            raise object.__getattribute__(self, "_error")
        if name in {"_real", "_error", "__class__"}:
            return object.__getattribute__(self, name)
        return getattr(object.__getattribute__(self, "_real"), name)


class _NonCallableLockAPI:
    """A present lock API descriptor that resolves to a non-callable value."""

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    try_acquire_compression_lock = None

    def __getattr__(self, name):
        return getattr(self._real, name)










@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("simulated lock-table corruption"),
        AttributeError("simulated internal lock attribute error"),
        TypeError("simulated internal lock type error"),
    ],
)
def test_real_lock_api_internal_errors_fail_closed_skips_compression(
    tmp_path: Path, monkeypatch, error: Exception
) -> None:
    """Errors after a real lock API resolves must preserve session lineage.

    ``AttributeError`` only means version skew while resolving the method. This
    test injects failures beneath the real ``SessionDB.try_acquire...`` body,
    proving that an internal AttributeError or TypeError cannot take the
    structural-absence compatibility path.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "ERRORING_LOCK_TEST"
    db.create_session(parent_sid, source="discord")

    def _fail_lock_write(_fn):
        raise error

    monkeypatch.setattr(db, "_execute_write", _fail_lock_write)
    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Skipped: messages returned verbatim, no rotation, compressor never ran.
    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    assert _count_children(db, parent_sid) == 0
    agent.context_compressor.compress.assert_not_called()




def test_review_fork_compacts_oversized_snapshot_in_memory(tmp_path: Path) -> None:
    """An oversized review snapshot replays warm on the first request, then
    compacts in memory before further requests — without mutating the parent.

    Regression for #93057: the fork historically pinned ``compression_enabled =
    False`` because it shares the parent's session_id (issue #38727). That
    left the review's private snapshot unbounded — every follow-up request in
    the review tool loop replayed the whole snapshot (350k-384k input tokens
    per request in production). The fix detaches the fork's compressor from
    the parent's SessionDB/session_id and enables in-memory-only compaction.

    This test drives the REAL ``_run_review_in_thread`` + ``run_conversation``
    with a threshold-crossing snapshot across two provider requests and
    asserts:
      • the FIRST request replays the full snapshot untouched (warm
        prompt-cache parity) — no compaction summary, middle turns present;
      • compression actually fired before the SECOND request (a real
        threshold crossing, not just setup-time binding state), and that
        request carries the compaction summary and none of the middle
        snapshot turns;
      • the fork keeps the parent's session_id (prompt-cache parity) but its
        agent-level AND compressor-level session bindings are detached;
      • the parent's durable transcript, session row, and child-session graph
        are byte-for-byte unchanged after the compaction ran.
    """
    import agent.background_review as br

    parent_sid = "REVIEW_FORK_IN_MEMORY_COMPACTION_93057"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(parent_sid, source="discord")
    db.append_message(parent_sid, role="user", content="durable parent turn")
    durable_before = db.get_messages(parent_sid)
    session_row_before = tuple(
        db._conn.execute(
            "SELECT id, parent_session_id, ended_at, end_reason FROM sessions WHERE id = ?",
            (parent_sid,),
        ).fetchone()
    )
    parent = _build_agent_with_db(db, parent_sid)
    parent._cached_system_prompt = "stable parent prompt"

    snapshot = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"review turn {i} " + "x" * 200,
        }
        for i in range(24)
    ]

    captured = {}

    def _tool_response(prompt_tokens: int) -> SimpleNamespace:
        message = SimpleNamespace(
            content=None,
            reasoning_content=None,
            reasoning=None,
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    type="function",
                    function=SimpleNamespace(
                        name="web_search", arguments='{"query": "x"}'
                    ),
                )
            ],
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
            model="test/model",
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
                total_tokens=prompt_tokens + 1,
            ),
        )

    def _final_response():
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    index=0,
                    message=SimpleNamespace(
                        role="assistant",
                        content="review complete",
                        tool_calls=None,
                        reasoning_content=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=10, total_tokens=110
            ),
            model="test/model",
        )

    # The worker does a local ``from run_agent import AIAgent``; patching the
    # class method covers that import path.
    from run_agent import AIAgent

    real_run_conversation = AIAgent.run_conversation

    def _run_threshold_crossing_review(self, *args, **kwargs):
        captured["compression_enabled"] = self.compression_enabled
        captured["compression_in_place"] = self.compression_in_place
        captured["session_id"] = self.session_id
        captured["session_db"] = self._session_db
        captured["input_budget"] = getattr(
            self, "_review_input_token_budget", "missing"
        )
        captured["defer_first_request"] = getattr(
            self, "_review_defer_compaction_before_first_response", "missing"
        )
        captured["compressor_session_db"] = getattr(
            self.context_compressor, "_session_db", "missing"
        )
        captured["compressor_session_id"] = getattr(
            self.context_compressor, "_session_id", "missing"
        )
        # Stub the fork's compressor so compaction output is deterministic
        # and no aux-LLM call happens; the trigger/commit paths stay real.
        self.context_compressor.threshold_tokens = 1
        self.context_compressor.protect_first_n = 1
        self.context_compressor.protect_last_n = 1
        self.context_compressor.compress = MagicMock(
            return_value=[
                {"role": "user", "content": "[CONTEXT COMPACTION] review summary"},
                {"role": "assistant", "content": "summary acknowledged"},
            ]
        )
        # Compress on the first pressure check after the first response, then
        # stand down so the compacted request proceeds instead of looping.
        _should_compress_calls = {"count": 0}

        def _should_compress(_tokens):
            _should_compress_calls["count"] += 1
            return _should_compress_calls["count"] == 1

        self.context_compressor.should_compress = MagicMock(
            side_effect=_should_compress
        )
        self.context_compressor.should_compress_info = MagicMock(
            return_value=(True, "over threshold")
        )
        self.context_compressor.should_compress_preflight = MagicMock(
            return_value=True
        )
        self.context_compressor.should_defer_preflight_to_real_usage = MagicMock(
            return_value=False
        )
        self.context_compressor.get_active_compression_failure_cooldown = MagicMock(
            return_value=None
        )
        self.context_compressor.select_context = MagicMock(return_value=None)
        self._compression_feasibility_checked = True
        self.client = MagicMock()
        self.client.chat.completions.create.side_effect = [
            _tool_response(100),
            _final_response(),
        ]
        self._disable_streaming = True
        self._use_prompt_caching = False

        def _fake_execute_tool_calls(assistant_message, messages, *_args):
            tool_call = assistant_message.tool_calls[0]
            messages.append(
                {
                    "role": "tool",
                    "name": tool_call.function.name,
                    "tool_call_id": tool_call.id,
                    "content": "ok",
                }
            )

        self._execute_tool_calls = _fake_execute_tool_calls

        result = real_run_conversation(self, *args, **kwargs)
        captured["compression_calls"] = self.context_compressor.compress.call_count
        create = self.client.chat.completions.create
        captured["create_calls"] = create.call_count
        captured["outbound"] = [
            call.kwargs.get("messages") for call in create.call_args_list
        ]
        return result

    try:
        with patch.object(AIAgent, "run_conversation", _run_threshold_crossing_review):
            br._run_review_in_thread(parent, snapshot, "review this conversation")

        assert captured["compression_calls"] >= 1, (
            "FIX REGRESSION: the review fork did not compact its oversized "
            "snapshot. compression_enabled was historically set False on the "
            "fork, which removed the only bound on the replayed snapshot "
            "(issue #93057)."
        )
        assert captured["create_calls"] == 2, (
            f"expected a 2-request review (tool call + final), "
            f"got {captured['create_calls']}"
        )
        first_outbound, second_outbound = captured["outbound"]
        first_contents = [str(m.get("content", "")) for m in first_outbound]
        second_contents = [str(m.get("content", "")) for m in second_outbound]
        # Warm-cache parity: the first request replays the full snapshot
        # untouched — middle turns present, no compaction summary yet.
        assert any("review turn 12" in text for text in first_contents), (
            "the review fork's FIRST request must replay the full snapshot "
            "(warm prompt-cache read) — compaction must not rewrite it before "
            "the first provider call"
        )
        assert not any(
            "[CONTEXT COMPACTION]" in text for text in first_contents
        ), f"first request was compacted prematurely: {first_contents!r}"
        # The SECOND request carries the compaction summary and none of the
        # middle snapshot turns.
        assert any(
            "[CONTEXT COMPACTION] review summary" in text
            for text in second_contents
        ), f"outbound request did not contain the compaction summary: {second_contents!r}"
        assert not any("review turn 12" in text for text in second_contents), (
            "outbound request still replays the middle of the snapshot — "
            "the review replayed an unbounded transcript despite compaction"
        )
        assert captured["session_id"] == parent_sid, (
            "Review fork should inherit the parent's session_id for "
            "prompt-cache parity."
        )
        assert captured["session_db"] is None, (
            "Review fork must keep its agent-level SessionDB detached "
            "(_session_db=None) so compaction can never persist."
        )
        assert captured["compressor_session_db"] is None, (
            "Review fork's compressor must be detached from the parent "
            "SessionDB, otherwise durable cooldown/streak writes land on "
            "the parent's row (issue #93057)."
        )
        assert captured["compressor_session_id"] == ""
        assert captured["compression_enabled"] is True
        assert captured["compression_in_place"] is True
        assert captured["defer_first_request"] is True
        assert isinstance(captured["input_budget"], int) and captured["input_budget"] > 0

        # Parent session must be byte-for-byte unchanged after the review
        # compacted its private snapshot.
        assert parent.session_id == parent_sid
        assert db.get_messages(parent_sid) == durable_before
        session_row_after = tuple(
            db._conn.execute(
                "SELECT id, parent_session_id, ended_at, end_reason FROM sessions WHERE id = ?",
                (parent_sid,),
            ).fetchone()
        )
        assert session_row_after == session_row_before
        assert _count_children(db, parent_sid) == 0
    finally:
        db.close()


def test_review_fork_fails_closed_when_compressor_rebind_raises(
    tmp_path: Path, caplog
) -> None:
    """A failed compressor detachment must keep the fork's compression OFF.

    Regression for the #93057 adversarial review: if ``bind_session_state``
    cannot sever the engine's binding to the parent's SessionDB/session_id,
    enabling compression would run it against the parent's live session
    binding — durable cooldown/streak/ineffective-count writes on the
    parent's row and the sibling-session race behind #38727 re-opened. The
    fork must fail CLOSED: keep the historical ``compression_enabled = False``
    behavior and warn. The review still runs (the iteration cap and the
    aggregate input budget still bound it).
    """
    import agent.background_review as br
    from agent.context_compressor import ContextCompressor

    parent_sid = "REVIEW_FORK_REBIND_FAIL_CLOSED_93057"

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(parent_sid, source="discord")
    parent = _build_agent_with_db(db, parent_sid)
    parent._cached_system_prompt = "stable parent prompt"

    snapshot = [
        {
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"review turn {i}",
        }
        for i in range(8)
    ]

    captured = {}

    def _capture_fork_flags(self, *args, **kwargs):
        captured["compression_enabled"] = self.compression_enabled
        captured["compression_in_place"] = self.compression_in_place
        captured["input_budget"] = getattr(
            self, "_review_input_token_budget", "missing"
        )
        return {
            "completed": True,
            "final_response": "review complete",
            "api_call_count": 0,
        }

    # The worker does a local ``from run_agent import AIAgent``; patching the
    # class method covers that import path.
    from run_agent import AIAgent

    _real_bind = ContextCompressor.bind_session_state

    def _failing_bind(self, session_db=None, session_id=""):
        # Only the detachment rebind may fail; any other binding passes
        # through so the fork's construction path stays intact.
        if session_db is None:
            raise RuntimeError("detachment boom")
        return _real_bind(self, session_db, session_id)

    try:
        with (
            patch.object(AIAgent, "run_conversation", _capture_fork_flags),
            patch.object(
                ContextCompressor, "bind_session_state", _failing_bind
            ),
        ):
            with caplog.at_level(logging.WARNING, logger="agent.background_review"):
                br._run_review_in_thread(parent, snapshot, "review this conversation")

        assert captured["compression_enabled"] is False, (
            "FIX REGRESSION: a failed compressor rebind must leave "
            "compression_enabled False on the review fork (fail-closed). "
            "Enabling compression with the engine still bound to the "
            "parent's session re-opens the #38727 race (issue #93057)."
        )
        assert any(
            "detachment failed" in record.message for record in caplog.records
        ), (
            "the failed rebind must log a warning so operators can see the "
            "fork fell back to the pre-fix behavior"
        )
        assert (
            isinstance(captured["input_budget"], int) and captured["input_budget"] > 0
        ), (
            "the aggregate input budget must still be armed on the fail-closed "
            "path — it bounds the review even when compaction stays off"
        )
    finally:
        db.close()


# ── Lease-refresher bounded-failure tolerance (salvage follow-up, #54465) ────
# A single falsy refresh (transient DB blip) must NOT permanently kill the
# lease — only a *persistent* failure (genuine lost-ownership) should stop the
# refresher after a bounded number of consecutive failures. Without this, one
# escaped lock-contention error silently reintroduces the TTL-expiry wedge the
# PR set out to fix.


class _FlakyRefreshDB:
    """A db whose refresh_compression_lock returns a scripted sequence."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def refresh_compression_lock(self, session_id, holder, ttl_seconds=300.0):
        self.calls += 1
        if self._results:
            return self._results.pop(0)
        return True  # steady-state success after the scripted prefix


def _no_sleep(refresher) -> None:
    """Make the refresher loop iterate without real wall-clock sleeps.

    ``_stop.wait(interval)`` returns False (keep looping) instantly instead of
    blocking for the (clamped) interval, so count-based tests stay fast and
    deterministic — the loop's termination is driven by the failure cap / the
    scripted db, not by timing.
    """
    refresher._stop.wait = lambda _interval: False  # type: ignore[assignment]








def test_lease_refresher_failure_window_is_bounded_by_ttl() -> None:
    """Persistent failure stops within one lease's worth of time, not forever.

    The contract (not a magic count): the give-up window
    ``cap * refresh_interval`` must be <= the TTL, so a stuck refresher can
    never hold the lock past its TTL. We assert that relationship directly
    rather than freezing a literal cap (behavior contract over snapshot).
    """
    from agent.conversation_compression import _CompressionLockLeaseRefresher

    ttl, interval = 10.0, 2.0  # cap should be int(10/2) = 5
    db = _FlakyRefreshDB([False] * 50)  # never recovers (lost ownership)
    refresher = _CompressionLockLeaseRefresher(
        db, "sess", "holder", ttl_seconds=ttl, refresh_interval_seconds=interval
    )
    _no_sleep(refresher)
    refresher._run()

    cap = refresher._max_consecutive_failures
    assert cap == int(ttl / interval), "cap must derive from ttl/interval"
    # Stops at the cap — not on the first failure, not forever.
    assert db.calls == cap
    # The invariant that makes the cap honest: total tolerance <= one TTL.
    assert cap * interval <= ttl, (
        f"give-up window {cap * interval}s must not exceed the lease TTL {ttl}s"
    )


def test_hard_interrupt_aborts_compression_and_unblocks_session_writes(tmp_path: Path) -> None:
    """Ctrl+C must abort an interrupt-protected summary without leaving the
    session write-blocked behind its compression lease."""
    from agent import auxiliary_client as aux

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_INTERRUPT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")

    agent = _build_agent_with_db(db, session_id)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    original_messages = copy.deepcopy(messages)

    def _cancelled_compress(current, *_args, **_kwargs):
        agent._hard_interrupt_requested.set()
        assert aux._aux_interrupt_cancel_requested() is True
        # F3 isolation: the engine only ever sees the pooled snapshot, so an
        # in-place mutation lands on ``current`` (the worker's copy), never on
        # the caller's live list. The rollback contract is that the RETURNED
        # transcript equals the pre-compression one.
        current[0]["content"] = "must be rolled back"
        raise aux.AuxiliaryExplicitCancellation()

    agent.context_compressor.compress.side_effect = _cancelled_compress

    compressed, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed == original_messages
    assert messages == original_messages
    assert db.get_compression_lock_holder(session_id) is None
    db.append_message(session_id, "assistant", "writes recovered")


def test_late_hard_interrupt_restores_full_compressor_attempt_state_and_retry(
    tmp_path: Path,
) -> None:
    """A stop after provider success but before compress() returns is a true no-op."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "LATE_HARD_INTERRUPT_STATE_TEST"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    provider_returned = threading.Event()
    allow_compress_return = threading.Event()
    shared_telemetry = {"shared": [1, 2, 3]}
    state_fields = {
        "_previous_summary": "old-summary",
        "_summary_has_user_turn": False,
        "compression_count": 4,
        "_last_compression_savings_pct": 37.5,
        "_ineffective_compression_count": 1,
        "_anti_thrash_recovery_deadline": 123.0,
        "_fallback_compression_streak": 1,
        "_verify_compaction_cleared_threshold": False,
        "_last_compression_made_progress": False,
        "_summary_failure_cooldown_until": 456.0,
        "_cooldown_persist_failed": True,
        "_last_summary_error": "old-error",
        "_consecutive_timeout_failures": 2,
        "_last_summary_dropped_count": 3,
        "_last_summary_fallback_used": True,
        "_last_compress_aborted": False,
        "_last_summary_auth_failure": True,
        "_last_summary_network_failure": True,
        "_last_aux_model_failure_error": "old-aux-error",
        "_last_aux_model_failure_model": "old-aux-model",
        "_summary_model_fallen_back": True,
        "summary_model": "old-summary-model",
        "_last_compression_telemetry": shared_telemetry,
        "_active_compression_telemetry": shared_telemetry,
        "_compression_telemetry_seed": {"seed": [3]},
    }
    for name, value in state_fields.items():
        setattr(agent.context_compressor, name, copy.deepcopy(value))
    restored_shared_telemetry = copy.deepcopy(shared_telemetry)
    agent.context_compressor._last_compression_telemetry = restored_shared_telemetry
    agent.context_compressor._active_compression_telemetry = restored_shared_telemetry

    def _provider_succeeded_then_waits(*_args, **_kwargs):
        for name in state_fields:
            setattr(agent.context_compressor, name, f"mutated-{name}")
        provider_returned.set()
        assert allow_compress_return.wait(timeout=5)
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] cancelled summary"},
            {"role": "user", "content": "tail"},
        ]

    agent.context_compressor.compress.side_effect = _provider_succeeded_then_waits
    result: dict[str, tuple] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    worker.start()
    assert provider_returned.wait(timeout=2)
    agent.hard_interrupt("cancel after provider return")
    allow_compress_return.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["value"][0] is messages
    assert {
        name: copy.deepcopy(getattr(agent.context_compressor, name))
        for name in state_fields
    } == state_fields
    assert (
        agent.context_compressor._active_compression_telemetry
        is agent.context_compressor._last_compression_telemetry
    )
    assert db.get_compression_lock_holder(session_id) is None

    agent.clear_interrupt()
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] retry summary"},
        {"role": "user", "content": "tail"},
    ]
    retried, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )
    assert retried is not messages
    assert retried[0]["content"] == "[CONTEXT COMPACTION] retry summary"


def test_force_cancel_restores_newer_durable_cooldown_captured_under_lease(
    tmp_path: Path,
) -> None:
    """A stale forced attempt rolls back to the lease-protected durable row."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "FORCE_CANCEL_DURABLE_COOLDOWN"
    db.create_session(session_id, source="cli")

    # B binds first and therefore has no local cooldown. A then persists a
    # newer cooldown for the same durable session before B acquires its lease.
    stale_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    writer_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    stale = stale_agent.context_compressor
    writer = writer_agent.context_compressor
    assert isinstance(stale, ContextCompressor)
    assert isinstance(writer, ContextCompressor)
    assert stale._summary_failure_cooldown_until == 0.0

    writer._record_compression_failure_cooldown(120.0, "newer durable failure")
    durable_before = tuple(
        db._conn.execute(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    )
    assert durable_before[0] is not None

    stale_seed = {"seed": ["truly-pre-attempt"]}
    stale._compression_telemetry_seed = copy.deepcopy(stale_seed)
    stale._previous_summary = "pre-attempt-summary"
    stale_agent._compression_feasibility_checked = True
    stale_agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _clear_then_hard_cancel() -> None:
        real_clear(stale)
        stale_agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    # Exercise the built-in force=True mutation point deterministically: force
    # clears the durable cooldown, then the frozen host cancellation unwinds it.
    stale._clear_compression_failure_cooldown = _clear_then_hard_cancel

    compressed, _prompt = stale_agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    assert compressed is messages
    durable_after = tuple(
        db._conn.execute(
            "SELECT compression_failure_cooldown_until, compression_failure_error "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
    )
    assert durable_after == durable_before
    assert stale._summary_failure_cooldown_until > time.monotonic()
    assert stale._last_summary_error == "newer durable failure"
    assert stale._cooldown_persist_failed is False
    assert stale._compression_telemetry_seed == stale_seed
    assert stale._previous_summary == "pre-attempt-summary"
    assert db.get_compression_lock_holder(session_id) is None

    # A future compressor refresh must still observe the exact row rather than
    # the cancelled force attempt having permanently cleared it.
    future_agent = _build_agent_with_db(
        db, session_id, stub_compressor=False
    )
    future = future_agent.context_compressor.get_active_compression_failure_cooldown(
        refresh=True
    )
    assert future is not None
    assert future["error"] == "newer durable failure"


def test_unrelated_interrupted_error_propagates_and_releases_compression_lease(
    tmp_path: Path,
) -> None:
    """A plugin/OS InterruptedError is a failure, not an explicit transaction abort."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "UNRELATED_INTERRUPT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")

    agent = _build_agent_with_db(db, session_id)
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _provider_interrupted(*_args, **_kwargs):
        messages[0]["content"] = "must be rolled back"
        raise InterruptedError("provider syscall interrupted")

    agent.context_compressor.compress.side_effect = _provider_interrupted

    with pytest.raises(InterruptedError, match="provider syscall interrupted"):
        agent._compress_context(messages, "sys", approx_tokens=120_000)

    assert db.get_compression_lock_holder(session_id) is None
    db.append_message(session_id, "assistant", "writes recovered")


def test_redirect_interrupt_remains_protected_during_compression(tmp_path: Path) -> None:
    """Redirects use interrupt_requested=True/message=None; only the atomic
    hard-cancel event may override summary protection."""
    from agent import auxiliary_client as aux

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "REDIRECT_COMPRESSION_TEST"
    db.create_session(session_id, source="cli")
    agent = _build_agent_with_db(db, session_id)
    agent._interrupt_requested = True
    agent._interrupt_message = None
    agent._pending_redirect = "new correction"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    def _protected_noop(current, **_kwargs):
        assert aux._aux_interrupt_cancel_requested() is False
        return copy.deepcopy(current)

    agent.context_compressor.compress.side_effect = _protected_noop

    compressed, _prompt = agent._compress_context(
        messages, "sys", approx_tokens=120_000
    )

    assert compressed == messages
    assert db.get_compression_lock_holder(session_id) is None


def test_hard_cancel_between_compress_return_and_commit_begin_wins_atomically(
    tmp_path: Path,
) -> None:
    """The hard-stop admission and commit admission share one fence lock."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_COMMIT_RACE"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    before_commit = threading.Event()
    allow_commit_check = threading.Event()

    class _CommitBarrierList(list):
        def __eq__(self, other):
            before_commit.set()
            assert allow_commit_check.wait(timeout=5)
            return super().__eq__(other)

    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: _CommitBarrierList(
        [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
    )
    archive_spy = MagicMock(wraps=db.archive_and_compact)
    db.archive_and_compact = archive_spy
    result: dict[str, tuple] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    worker.start()
    assert before_commit.wait(timeout=2)

    agent.hard_interrupt("cancel before commit admission")
    allow_commit_check.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert result["value"][0] is messages
    archive_spy.assert_not_called()
    assert db.get_compression_lock_holder(session_id) is None


def test_hard_stop_waits_for_commit_already_admitted(tmp_path: Path) -> None:
    """A surfaced stop never races an untracked post-return transcript commit."""
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "HARD_CANCEL_AFTER_COMMIT_ADMISSION"
    db.create_session(session_id, source="tui")
    agent = _build_agent_with_db(db, session_id)
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    commit_started = threading.Event()
    allow_commit = threading.Event()
    stop_returned = threading.Event()
    original_archive = db.archive_and_compact

    def _blocked_archive(*args, **kwargs):
        commit_started.set()
        assert allow_commit.wait(timeout=5)
        return original_archive(*args, **kwargs)

    db.archive_and_compact = _blocked_archive
    agent.context_compressor.compress.side_effect = lambda *_a, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compression_result: dict[str, tuple] = {}
    compression = threading.Thread(
        target=lambda: compression_result.setdefault(
            "value", agent._compress_context(messages, "sys", approx_tokens=120_000)
        ),
        daemon=True,
    )
    compression.start()
    assert commit_started.wait(timeout=2)

    stop = threading.Thread(
        target=lambda: (
            agent.hard_interrupt("stop after commit admission"),
            stop_returned.set(),
        ),
        daemon=True,
    )
    stop.start()
    assert not stop_returned.wait(timeout=0.1)
    allow_commit.set()
    compression.join(timeout=5)
    stop.join(timeout=5)

    assert not compression.is_alive()
    assert not stop.is_alive()
    assert stop_returned.is_set()
    assert compression_result["value"][0][0]["content"] == (
        "[CONTEXT COMPACTION] summary"
    )
    assert agent._hard_interrupt_requested.is_set()
    assert db.get_compression_lock_holder(session_id) is None


@pytest.mark.parametrize("deadline_offset", [-10.0, 0.05, None])
def test_force_cancel_restores_exact_expired_or_expiring_cooldown_row(
    tmp_path: Path,
    deadline_offset: float | None,
) -> None:
    """Cancellation preserves raw cooldown columns even after their deadline."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = f"RAW_COOLDOWN_{deadline_offset}"
    db.create_session(session_id, source="cli")
    deadline = time.time() + deadline_offset if deadline_offset is not None else None
    db.restore_compression_failure_cooldown_row(
        session_id,
        {
            "session_exists": True,
            "cooldown_until": deadline,
            "error": "expired-but-exact",
        },
    )
    before = db.get_compression_failure_cooldown_row(session_id)

    agent = _build_agent_with_db(db, session_id, stub_compressor=False)
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor)
    # A stale local persistence-failure marker must not suppress restoration
    # once the raw durable row was captured authoritatively under the lease.
    compressor._cooldown_persist_failed = True
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _mutate_then_cancel() -> None:
        real_clear(compressor)
        if deadline_offset is not None and deadline_offset > 0:
            assert deadline is not None
            while time.time() <= deadline:
                time.sleep(0.005)
        agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    compressor._clear_compression_failure_cooldown = _mutate_then_cancel

    compressed, _prompt = agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    assert compressed is messages
    assert db.get_compression_failure_cooldown_row(session_id) == before
    assert db.get_compression_lock_holder(session_id) is None


def test_cooldown_rollback_failure_surfaces_and_releases_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed compensating write cannot masquerade as a mutation-free cancel."""
    from agent.auxiliary_client import AuxiliaryExplicitCancellation
    from agent.context_compressor import ContextCompressor

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "COOLDOWN_ROLLBACK_WRITE_FAILURE"
    db.create_session(session_id, source="cli")
    db.record_compression_failure_cooldown(
        session_id,
        time.time() + 120.0,
        "must-restore",
    )
    agent = _build_agent_with_db(db, session_id, stub_compressor=False)
    compressor = agent.context_compressor
    assert isinstance(compressor, ContextCompressor)
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    real_clear = ContextCompressor._clear_compression_failure_cooldown

    def _mutate_then_cancel() -> None:
        real_clear(compressor)
        agent._hard_interrupt_requested.set()
        raise AuxiliaryExplicitCancellation()

    compressor._clear_compression_failure_cooldown = _mutate_then_cancel

    def _rollback_write_fails(_self, _session_id, _snapshot) -> None:
        raise sqlite3.OperationalError("forced rollback write failure")

    monkeypatch.setattr(
        SessionDB,
        "restore_compression_failure_cooldown_row",
        _rollback_write_fails,
    )

    with pytest.raises(sqlite3.OperationalError, match="forced rollback write failure"):
        agent._compress_context(
            messages,
            "sys",
            approx_tokens=120_000,
            force=True,
        )

    assert db.get_compression_lock_holder(session_id) is None


def test_exact_cooldown_restore_api_propagates_sqlite_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "RAW_COOLDOWN_WRITE_FAILURE"
    db.create_session(session_id, source="test")

    def _write_fails(_callback) -> None:
        raise sqlite3.OperationalError("forced low-level write failure")

    monkeypatch.setattr(db, "_execute_write", _write_fails)

    with pytest.raises(sqlite3.OperationalError, match="forced low-level write failure"):
        db.restore_compression_failure_cooldown_row(
            session_id,
            {
                "session_exists": True,
                "cooldown_until": time.time() + 10.0,
                "error": "must propagate",
            },
        )


def test_failed_split_arms_failure_cooldown(tmp_path: Path) -> None:
    """Regression #97948 symptom B: a failed split/archive must arm the
    compression failure cooldown so the next automatic turn cannot
    immediately re-run the identical doomed compression."""
    from agent.conversation_compression import _SPLIT_FAILURE_COOLDOWN_SECONDS

    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "SPLIT_FAIL_COOLDOWN_TEST"
    db.create_session(session_id, source="test")
    agent = _build_agent_with_db(db, session_id)
    setattr(agent, "compression_in_place", True)
    db.archive_and_compact = MagicMock(side_effect=RuntimeError("archive boom"))
    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    agent._compress_context(
        messages,
        "sys",
        approx_tokens=120_000,
        force=True,
    )

    cooldown_calls = (
        agent.context_compressor._record_compression_failure_cooldown.call_args_list
    )
    assert len(cooldown_calls) == 1, (
        "split failure must arm the failure cooldown (#97948 symptom B)"
    )
    seconds, error = cooldown_calls[0].args
    assert seconds == _SPLIT_FAILURE_COOLDOWN_SECONDS
    assert "session_split_failed" in str(error)
