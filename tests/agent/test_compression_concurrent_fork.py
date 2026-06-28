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

import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


def _build_agent_with_db(db: SessionDB, session_id: str):
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
    compressor = MagicMock()

    def _compress_with_overlap(*_a, **_kw):
        time.sleep(0.25)
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


def test_concurrent_compression_does_not_fork_session(tmp_path: Path) -> None:
    """Two AIAgents that share a session_id MUST NOT both rotate it.

    Without the per-session compression lock this fixture deterministically
    produces 2 child sessions (transcript fork).  With the lock the second
    path aborts cleanly, leaving exactly 1 canonical child.
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

    # Exactly one canonical child — not two orphans.
    assert _count_children(db, parent_sid) == 1, (
        "Compression lock failed: parent session has multiple children in state.db "
        "(transcript fork). This is Damien's incident shape — see the test docstring."
    )

    # And exactly one of the two agents actually rotated its session_id; the
    # other should still hold the parent_sid (its compression was skipped).
    rotated = sum(
        1 for a in (agent_a, agent_b) if a.session_id != parent_sid
    )
    assert rotated == 1, (
        f"Expected exactly one agent to rotate session_id, got {rotated}. "
        "Both agents rotating means the lock didn't serialize them."
    )

    # The lock must be released after the winner finished.
    assert db.get_compression_lock_holder(parent_sid) is None, (
        "Compression lock leaked: still held after both rotations completed."
    )


def test_skipped_compression_returns_messages_unchanged(tmp_path: Path) -> None:
    """The loser of the lock race must return its input messages verbatim.

    Callers (preflight compression in ``conversation_loop.py``) detect the
    no-op via ``len(returned) == len(input)`` and stop the auto-compress
    retry loop.  If the skipped path returned the compressed view, that
    detection would break and the caller would mutate the conversation
    without going through state.db rotation.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "LOSER_TEST"
    db.create_session(parent_sid, source="discord")

    # Pre-acquire the lock so the agent's compress_context sees it held.
    held = db.try_acquire_compression_lock(parent_sid, "external_holder")
    assert held is True

    agent = _build_agent_with_db(db, parent_sid)
    messages = [{"role": "user", "content": "m1"}, {"role": "user", "content": "m2"}]

    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Skipped: messages returned verbatim, no rotation
    assert compressed is messages or compressed == messages
    assert agent.session_id == parent_sid
    # Compressor was never called (the skip happens before .compress())
    agent.context_compressor.compress.assert_not_called()


class _NoLockSubsystemDB:
    """Wraps a real SessionDB but simulates a pre-#34351 version skew.

    A long-lived process can hold ``hermes_state.SessionDB`` bound to the
    OLD class in memory (no compression-lock methods) while a lazily
    re-imported ``conversation_compression.py`` calls the NEW lock code.
    ``try_acquire_compression_lock`` then raises ``AttributeError`` — which
    is NOT a ``sqlite3.Error``, so the method's own fail-open guard never
    runs.  Before the fix the exception propagated to the outer agent loop,
    which printed the error and retried; compression never succeeded, the
    token count never dropped, and the loop re-triggered compaction forever.
    """

    def __init__(self, real_db: SessionDB) -> None:
        self._real = real_db

    def try_acquire_compression_lock(self, *_a, **_k):  # noqa: D401
        raise AttributeError(
            "'SessionDB' object has no attribute 'try_acquire_compression_lock'"
        )

    def get_compression_lock_holder(self, *_a, **_k):
        raise AttributeError("'SessionDB' object has no attribute 'get_compression_lock_holder'")

    def release_compression_lock(self, *_a, **_k):
        raise AttributeError("'SessionDB' object has no attribute 'release_compression_lock'")

    def __getattr__(self, name):
        # Everything else (create_session, append, rotation helpers) goes to
        # the real db so the post-lock compression + rotation path runs.
        return getattr(self._real, name)


def test_missing_lock_subsystem_fails_open_not_infinite_loop(tmp_path: Path) -> None:
    """Version skew (no lock methods) must fail OPEN, not raise into the loop.

    Reproduces the "API call #47/#48/#49 ... has no attribute
    try_acquire_compression_lock" infinite-compaction spin: when the lock
    subsystem is absent, ``_compress_context`` must skip locking and proceed
    with compression (so the loop makes progress and terminates) instead of
    letting the ``AttributeError`` escape to the retry loop.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    parent_sid = "SKEW_TEST_SESSION"
    db.create_session(parent_sid, source="discord")

    agent = _build_agent_with_db(db, parent_sid)
    # Swap in the lock-less wrapper AFTER construction (the agent already
    # holds a normal db reference; we only break the lock methods).
    agent._session_db = _NoLockSubsystemDB(db)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]

    # MUST NOT raise AttributeError. Before the fix this raised and the
    # outer loop would retry forever.
    compressed, _sp = agent._compress_context(messages, "sys", approx_tokens=120_000)

    # Compression actually ran (proceeded past the broken lock) and made
    # progress, so the auto-compress loop would terminate.
    agent.context_compressor.compress.assert_called_once()
    assert len(compressed) < len(messages), (
        "Compression made no progress despite failing open — loop would still spin."
    )
    # Session rotated (compression succeeded end-to-end).
    assert agent.session_id != parent_sid


def test_review_fork_disables_compression_to_prevent_stale_parent_fork() -> None:
    """The background-review fork must set ``compression_enabled = False``
    so it can never compress the parent it shares a session_id with
    (issue #38727).

    The per-session compression lock only serialises a SAME-WINDOW concurrent
    race. It does NOT stop a stale parent from being compressed again in a
    LATER turn: if ``review_agent`` had won the race, its new child session is
    never adopted by the gateway (the fork is single-lifecycle and dies right
    after one ``run_conversation``), so the foreground path would start the
    next turn from the stale parent and compress it AGAIN — leaving the same
    parent with two sibling children.

    The fix makes the review fork never trigger compression at all. Both
    compression trigger sites in ``agent/conversation_loop.py`` gate on
    ``agent.compression_enabled`` BEFORE calling ``_compress_context``:
      • preflight (``if agent.compression_enabled and len(messages) > ...``)
      • mid-loop  (``if agent.compression_enabled and _compressor.should_compress(...)``)
    so a fork with the flag cleared never reaches the rotation path.

    This test pins the contract at the source: ``_run_review_in_thread``
    must set ``review_agent.compression_enabled = False`` on the fork it
    builds. It calls the real worker synchronously with
    ``AIAgent.run_conversation`` patched (so no LLM call happens) and
    captures the constructed review agent to assert the flag.
    """
    import tempfile

    import agent.background_review as br

    captured = {}

    def _fake_run_conversation(self, *_a, **_k):
        captured["compression_enabled"] = self.compression_enabled
        captured["session_id"] = self.session_id
        return {"final_response": "", "messages": []}

    parent_sid = "REVIEW_FORK_FLAG_TEST"

    with tempfile.TemporaryDirectory() as td:
        db = SessionDB(db_path=Path(td) / "state.db")
        parent = None
        try:
            db.create_session(parent_sid, source="discord")
            parent = _build_agent_with_db(db, parent_sid)

            # The worker does a local ``from run_agent import AIAgent``; patching
            # the class method covers that import path.
            from run_agent import AIAgent

            with patch.object(AIAgent, "run_conversation", _fake_run_conversation):
                br._run_review_in_thread(
                    parent,
                    [{"role": "user", "content": "hi"}],
                    "review this conversation",
                )
        finally:
            # Windows keeps sqlite files locked until every connection owner is
            # closed.  Close explicitly before TemporaryDirectory attempts to
            # delete state.db.
            if parent is not None:
                parent.close()
            db.close()

    assert captured, (
        "_run_review_in_thread never reached run_conversation — the spawn path "
        "changed; update this test to capture the review AIAgent."
    )
    assert captured["session_id"] == parent_sid, (
        "Review fork should inherit the parent's session_id (shared id is the "
        "whole reason compression must be disabled)."
    )
    assert captured["compression_enabled"] is False, (
        "FIX REGRESSION: background-review fork did NOT disable compression. "
        "It shares the parent's session_id, so an enabled fork can rotate the "
        "parent into an orphan child (issue #38727). The trigger gates in "
        "conversation_loop.py only short-circuit when compression_enabled is "
        "False — this flag MUST be cleared on the review fork."
    )
