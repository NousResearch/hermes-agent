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
