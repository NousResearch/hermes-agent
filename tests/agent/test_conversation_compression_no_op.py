"""Tests for #57771: infinite compression loop when compression makes no progress.

When `ContextCompressor.compress()` returns the input messages unchanged
(no compressible window found), `compress_context()` does not detect this
no-op and proceeds with full session rotation. This creates an infinite
loop where each pass saves only a handful of messages but rotates to a fresh
child session. The anti-thrashing guard (`_ineffective_compression_count >= 2`)
stops the loop within a single turn, but the next turn's preflight check
re-triggers compression because the compressed messages still exceed the threshold.

Fix: detect when `len(compressed) == len(messages)` and skip session rotation.
The compressor has its own anti-thrashing logic but it's defeated because
`compress_context()` unconditionally rotates the session.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

from hermes_state import SessionDB
from agent.conversation_compression import compress_context


def _stub_compressor_no_progress():
    """Compressor that does NOT reduce message count.

    Mirrors the real-world bug behavior: the compressor found no
    compressible window and returns the input messages unchanged.
    """
    return SimpleNamespace(
        compress=lambda messages, **kwargs: list(messages),  # same length
        last_prompt_tokens=1000,
        context_length=200000,
        compression_count=0,
        threshold_tokens=None,
        _last_compress_aborted=False,
        _last_summary_error=None,
    )


def _stub_memory_manager():
    """Stub memory manager with no-op hooks for the methods compress_context calls."""
    return SimpleNamespace(
        on_pre_compress=lambda msgs: None,
        on_post_compress=lambda *a, **k: None,
        on_session_switch=lambda *a, **k: None,
    )


def _build_agent(db, sid, *, compression_in_place=False):
    """Construct a minimal AIAgent wired to db and pinned to sid.

    compression_in_place defaults to False so the rotation path is exercised.
    The bug #57771 only manifests in the rotation branch — when
    in_place=True the session id is preserved by design and rotation never
    runs. Tests that need the in-place behavior should pass
    compression_in_place=True explicitly.
    """
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=sid,
            skip_context_files=True,
            skip_memory=True,
        )
    # Override the agent's compression_in_place — the real bug is in the
    # rotation branch, not the in-place branch.
    agent.compression_in_place = compression_in_place
    agent._emit_status = lambda *_a, **_k: None
    agent._emit_warning = lambda *_a, **_k: None
    agent._memory_manager = _stub_memory_manager()
    return agent


class TestNoProgressCompressionDoesNotRotateSession:
    """The core bug fix: when compression makes no progress, session must NOT rotate."""

    def test_session_id_unchanged_when_no_progress(self, tmp_path):
        """len(compressed) == len(messages) → agent.session_id must NOT change."""
        db = SessionDB(tmp_path / "state.db")
        original = "session-no-rotate-1"
        db.create_session(session_id=original, source="cli", model="m", model_config={})
        agent = _build_agent(db, original)
        agent.context_compressor = _stub_compressor_no_progress()

        messages = [{"role": "user", "content": f"m{i}"} for i in range(50)]
        compress_context(agent, messages, "sp")

        assert agent.session_id == original, (
            f"session_id changed from {original!r} to {agent.session_id!r} "
            f"despite no compression progress (len unchanged)"
        )

    def test_messages_returned_unchanged_when_no_progress(self, tmp_path):
        """When no progress, returned messages must equal input (no info loss)."""
        db = SessionDB(tmp_path / "state.db")
        sid = "session-msgs-unchanged"
        db.create_session(session_id=sid, source="cli", model="m", model_config={})
        agent = _build_agent(db, sid)
        agent.context_compressor = _stub_compressor_no_progress()

        messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        new_messages, _ = compress_context(agent, messages, "sp")

        assert len(new_messages) == len(messages)
        assert new_messages == messages

    def test_repeated_no_progress_does_not_drift_session(self, tmp_path):
        """5 consecutive no-progress compressions → agent.session_id stays put.

        Per the bug report: 17 rotations in ~4 hours. After the fix,
        repeated no-progress compressions must NOT create new child sessions.
        """
        db = SessionDB(tmp_path / "state.db")
        sid = "session-no-drift"
        db.create_session(session_id=sid, source="cli", model="m", model_config={})
        agent = _build_agent(db, sid)
        agent.context_compressor = _stub_compressor_no_progress()

        messages = [{"role": "user", "content": f"m{i}"} for i in range(50)]

        for i in range(5):
            compress_context(agent, messages, "sp")

        assert agent.session_id == sid, (
            f"After 5 no-progress compressions, session_id drifted to {agent.session_id!r}"
        )

    def test_no_progress_does_not_create_orphan_sessions(self, tmp_path):
        """The DB must not have orphan child sessions when no progress was made.

        The bug caused ~474 messages to remain in the DB but with rotating
        parent_session_id chains, leaving orphan rows. After the fix, the
        session row count should NOT grow on no-progress compressions.
        """
        db = SessionDB(tmp_path / "state.db")
        sid = "session-no-orphans"
        db.create_session(session_id=sid, source="cli", model="m", model_config={})
        agent = _build_agent(db, sid)
        agent.context_compressor = _stub_compressor_no_progress()

        # Snapshot session count before
        cur = db._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sessions")
        count_before = cur.fetchone()[0]

        messages = [{"role": "user", "content": f"m{i}"} for i in range(50)]
        for _ in range(3):
            compress_context(agent, messages, "sp")

        cur.execute("SELECT COUNT(*) FROM sessions")
        count_after = cur.fetchone()[0]

        # No new sessions should have been created
        assert count_after == count_before, (
            f"DB session count grew from {count_before} to {count_after} after "
            f"3 no-progress compressions — should be unchanged"
        )


class TestAbortedCompressionStillReturnsUnchanged:
    """Regression check: the existing aborted-compression guard must keep working."""

    def test_aborted_compression_does_not_rotate(self, tmp_path):
        """If _last_compress_aborted=True, return unchanged and don't rotate."""
        db = SessionDB(tmp_path / "state.db")
        original = "session-aborted-1"
        db.create_session(session_id=original, source="cli", model="m", model_config={})
        agent = _build_agent(db, original)

        # Compressor that reports it aborted
        compressor = SimpleNamespace(
            compress=lambda messages, **kwargs: list(messages),
            last_prompt_tokens=1000,
            context_length=200000,
            compression_count=0,
            threshold_tokens=None,
            _last_compress_aborted=True,
            _last_summary_error="aux LLM timed out",
        )
        agent.context_compressor = compressor

        messages = [{"role": "user", "content": f"m{i}"} for i in range(10)]
        compress_context(agent, messages, "sp")

        # Aborted compression should not rotate
        assert agent.session_id == original
