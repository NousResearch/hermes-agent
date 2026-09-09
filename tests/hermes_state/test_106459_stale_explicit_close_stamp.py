"""Regression tests for #106459 — an over-limit session must not become permanently
uncompressible because its row carries a stale explicit-close stamp.

``publish_compression_child`` used to fail closed on ANY non-automatic ``end_reason``
(``tui_close``, ``cli_close``, ``webhook_complete``, ``new_session``, gateway-recovery and
cron reasons), and ``end_session()`` is first-stamp-wins, so nothing ever cleared it. Every
turn then ran the compression, discarded its result at publication, kept the oversized
history and hit ``Context length exceeded … Cannot compress further`` again; manual
``/compress`` reported "No changes". The agent's pre-flush guard re-implemented the same
taxonomy and aborted even earlier.

Contract under test (single verdict owner ``_compression_parent_obstacle``): a writer
holding the compression lease is provably still the conversation's writer, so an explicit
close with NO continuation child is stale and is cleared inside the publish transaction,
exactly like an automatic stamp (#88197). Lineage owned elsewhere still fails closed: a
``'compression'`` stamp, a reset boundary, or an explicit close whose continuation child was
already published.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from hermes_state_common import _RESET_END_REASONS

STALE_EXPLICIT_CLOSES = ["tui_close", "cli_close", "webhook_complete", "new_session"]


@pytest.fixture
def db(tmp_path: Path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _publish(db: SessionDB, parent: str, child: str) -> None:
    db.publish_compression_child(
        parent_session_id=parent,
        child_session_id=child,
        source="tui",
        messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        require_compression_lease=False,
    )


def _stamp(db: SessionDB, session_id: str, reason: str) -> None:
    db.end_session(session_id, reason)
    row = db.get_session(session_id)
    assert row["ended_at"] is not None and row["end_reason"] == reason


class TestPublishClearsStaleExplicitClose:
    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_explicit_close_without_continuation_publishes(self, db: SessionDB, reason: str, caplog) -> None:
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        db.append_message(parent, "user", content="hello")
        _stamp(db, parent, reason)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            _publish(db, parent, f"C_{reason}")

        parent_row = db.get_session(parent)
        assert parent_row["end_reason"] == "compression"  # the true boundary, not the stale stamp
        assert parent_row["ended_at"] is not None
        child_row = db.get_session(f"C_{reason}")
        assert child_row is not None and child_row["parent_session_id"] == parent
        assert any(reason in rec.getMessage() and "stale" in rec.getMessage() for rec in caplog.records), (
            "clearing an explicit-close stamp must be logged with the stale reason")

    def test_repeated_rotation_is_not_wedged(self, db: SessionDB) -> None:
        """The field shape: the stamp lands once, and every later rotation must still publish."""
        parent = "P_repeat"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "cli_close")
        _publish(db, parent, "C_1")
        _publish(db, "C_1", "C_2")
        assert db.get_session("C_1")["end_reason"] == "compression"
        assert db.get_session("C_2")["parent_session_id"] == "C_1"


class TestLineageOwnedElsewhereStillFailsClosed:
    def test_explicit_close_with_published_continuation_fails_closed(self, db: SessionDB) -> None:
        parent = "P_superseded"
        db.create_session(parent, source="tui")
        _publish(db, parent, "C_first")  # a continuation now exists
        db.reopen_session(parent)
        _stamp(db, parent, "tui_close")

        with pytest.raises(RuntimeError, match="already ended.*published continuation"):
            _publish(db, parent, "C_second")
        assert db.get_session("C_second") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"  # untouched

    @pytest.mark.parametrize("reason", ["session_reset", "session_switch", "idle", "daily"])
    def test_reset_boundary_fails_closed(self, db: SessionDB, reason: str) -> None:
        assert reason in _RESET_END_REASONS
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason)
        with pytest.raises(RuntimeError, match=f"already ended.*{reason} boundary"):
            _publish(db, parent, f"C_{reason}")
        assert db.get_session(f"C_{reason}") is None

    def test_compression_stamp_fails_closed(self, db: SessionDB) -> None:
        parent = "P_compression"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "compression")
        with pytest.raises(RuntimeError, match="already ended.*closed by compression"):
            _publish(db, parent, "C_x")


class TestVerdictIsSharedWithTheAgentGuard:
    """The pre-flush guard must agree with publish, or a durable flush is skipped (or written for nothing)."""

    def test_read_only_verdict_matches_publish(self, db: SessionDB) -> None:
        for sid, reason in [("live", None), ("auto", "ws_disconnect"), ("stale", "tui_close")]:
            db.create_session(sid, source="tui")
            if reason:
                _stamp(db, sid, reason)
            assert db.compression_parent_deliberately_ended(sid) is False, (sid, reason)
        for sid, reason in [("reset", "session_reset"), ("comp", "compression")]:
            db.create_session(sid, source="tui")
            _stamp(db, sid, reason)
            assert db.compression_parent_deliberately_ended(sid) is True, (sid, reason)
        db.create_session("superseded", source="tui")
        _publish(db, "superseded", "superseded_child")
        db.reopen_session("superseded")
        _stamp(db, "superseded", "cli_close")
        assert db.compression_parent_deliberately_ended("superseded") is True
        assert db.compression_parent_deliberately_ended("missing") is False
        assert db.compression_parent_deliberately_ended("") is False

    def test_agent_guard_delegates_to_the_store(self, db: SessionDB) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        db.create_session("stale", source="tui")
        _stamp(db, "stale", "tui_close")
        assert _parent_deliberately_ended(db, "stale") is False
        db.create_session("reset", source="tui")
        _stamp(db, "reset", "session_reset")
        assert _parent_deliberately_ended(db, "reset") is True

    def test_agent_guard_fails_open_and_keeps_the_taxonomy_fallback(self) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        broken = MagicMock()
        broken.compression_parent_deliberately_ended.side_effect = RuntimeError("db unavailable")
        assert _parent_deliberately_ended(broken, "x") is False

        class TaxonomyOnlyStore:  # a stand-in without the verdict: old behaviour is retained
            def get_session(self, session_id):
                return {"ended_at": 1.0, "end_reason": "tui_close"}

        assert _parent_deliberately_ended(TaxonomyOnlyStore(), "x") is True


class TestRotationEndToEnd:
    def _build_agent(self, db: SessionDB, session_id: str):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                platform="tui",
                quiet_mode=True,
                session_db=db,
                session_id=session_id,
                skip_context_files=True,
                skip_memory=True,
            )
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        compressor._last_summary_auth_failure = False
        compressor._last_aux_model_failure_model = None
        compressor._last_aux_model_failure_error = None
        agent.context_compressor = compressor
        agent.compression_in_place = False  # rotation path
        return agent

    def test_stale_explicit_close_does_not_wedge_rotation(self, db: SessionDB) -> None:
        """#106459 end-to-end: the live session's row carries an explicit-close stamp, and
        auto-compaction still rotates instead of computing and discarding forever."""
        parent = "PARENT_106459_E2E"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close")

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id != parent, "rotation aborted on the stale explicit-close stamp"
        assert db.get_session(parent)["end_reason"] == "compression"
        child_row = db.get_session(agent.session_id)
        assert child_row is not None and child_row["parent_session_id"] == parent

    def test_reset_boundary_still_aborts_rotation(self, db: SessionDB) -> None:
        parent = "PARENT_106459_RESET"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "session_reset")

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == "session_reset"
