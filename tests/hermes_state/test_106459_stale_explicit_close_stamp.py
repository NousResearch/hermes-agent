"""Regression tests for #106459 — an over-limit session must not become permanently
uncompressible because its row carries a stale explicit-close stamp, and a close the user
makes mid-compression must never be resurrected.

``publish_compression_child`` used to fail closed on ANY non-automatic ``end_reason``
(``tui_close``, ``cli_close``, ``webhook_complete``, ``new_session``, gateway-recovery and
cron reasons), and ``end_session()`` is first-stamp-wins, so nothing ever cleared it. Every
turn then ran the compression, discarded its result at publication, kept the oversized
history and hit ``Context length exceeded … Cannot compress further`` again; manual
``/compress`` reported "No changes". The agent's pre-flush guard re-implemented the same
taxonomy and aborted even earlier.

Contract under test (single verdict owner ``_compression_parent_obstacle``): an explicit
close is ordered against this attempt's compression lease. A stamp written BEFORE the lease
was acquired is stale — the writer took the lease and has driven the row since — and is
cleared inside the publish transaction like an automatic stamp (#88197). A close written
AFTER the lease was acquired is the user closing the session while compression runs
(``session.close`` waits five seconds for the turn thread, then stamps ``tui_close``) and
is preserved: publish fails closed and no child is created. Without a lease of our own the
stamp cannot be proven stale and also fails closed. Lineage owned elsewhere always fails
closed: a ``'compression'`` stamp, a reset boundary, or an explicit close whose
continuation child was already published.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from hermes_state_common import _RESET_END_REASONS

STALE_EXPLICIT_CLOSES = ["tui_close", "cli_close", "webhook_complete", "new_session"]
HOLDER = "compression-writer"


@pytest.fixture
def db(tmp_path: Path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _publish(db: SessionDB, parent: str, child: str, *, holder: str | None = None) -> None:
    db.publish_compression_child(
        parent_session_id=parent,
        child_session_id=child,
        source="tui",
        messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        require_compression_lease=holder is not None,
        compression_lock_holder=holder,
    )


def _stamp(db: SessionDB, session_id: str, reason: str, *, age: float = 0.0) -> None:
    """End the row with *reason*; ``age`` backdates the stamp so its order against a lease
    acquired afterwards is unambiguous."""
    db.end_session(session_id, reason)
    if age:
        db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time() - age, session_id))
    row = db.get_session(session_id)
    assert row["ended_at"] is not None and row["end_reason"] == reason


def _lease(db: SessionDB, session_id: str, holder: str = HOLDER) -> str:
    assert db.try_acquire_compression_lock(session_id, holder, ttl_seconds=300.0)
    return holder


class TestStaleStampBeforeTheLeaseIsCleared:
    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_explicit_close_before_the_lease_publishes(self, db: SessionDB, reason: str, caplog) -> None:
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        db.append_message(parent, "user", content="hello")
        _stamp(db, parent, reason, age=60.0)  # the #106459 shape: the mark predates this attempt
        holder = _lease(db, parent)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            _publish(db, parent, f"C_{reason}", holder=holder)

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
        _stamp(db, parent, "cli_close", age=60.0)
        _publish(db, parent, "C_1", holder=_lease(db, parent))
        _publish(db, "C_1", "C_2", holder=_lease(db, "C_1"))
        assert db.get_session("C_1")["end_reason"] == "compression"
        assert db.get_session("C_2")["parent_session_id"] == "C_1"


class TestDeliberateCloseAfterTheLeaseIsPreserved:
    def test_close_during_compression_fails_closed(self, db: SessionDB) -> None:
        """The reviewer's probe: acquire the lease, then the user closes the session
        (``session.close`` stamps ``tui_close`` after its five-second wait), then publish."""
        parent = "P_closed_mid_compression"
        db.create_session(parent, source="tui")
        holder = _lease(db, parent)
        time.sleep(0.01)
        _stamp(db, parent, "tui_close")

        with pytest.raises(RuntimeError, match="already ended.*after this compression attempt acquired its lease"):
            _publish(db, parent, "C_never", holder=holder)

        assert db.get_session("C_never") is None
        row = db.get_session(parent)
        assert row["end_reason"] == "tui_close" and row["ended_at"] is not None  # the user's close survives

    def test_close_at_the_same_instant_fails_closed(self, db: SessionDB) -> None:
        parent = "P_tie"
        db.create_session(parent, source="tui")
        holder = _lease(db, parent)
        acquired_at = db._read_one("SELECT acquired_at FROM compression_locks WHERE session_id = ?", (parent,))[0]
        db.end_session(parent, "tui_close")
        db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (acquired_at, parent))

        with pytest.raises(RuntimeError, match="after this compression attempt acquired its lease"):
            _publish(db, parent, "C_tie", holder=holder)
        assert db.get_session(parent)["end_reason"] == "tui_close"

    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_without_a_lease_an_explicit_close_cannot_be_proven_stale(self, db: SessionDB, reason: str) -> None:
        parent = f"P_nolease_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason, age=60.0)
        with pytest.raises(RuntimeError, match="already ended.*no compression-lease ordering"):
            _publish(db, parent, f"C_{reason}")  # require_compression_lease=False: no ordering signal
        assert db.get_session(f"C_{reason}") is None
        assert db.get_session(parent)["end_reason"] == reason


class TestLineageOwnedElsewhereStillFailsClosed:
    def test_explicit_close_with_published_continuation_fails_closed(self, db: SessionDB) -> None:
        parent = "P_superseded"
        db.create_session(parent, source="tui")
        _publish(db, parent, "C_first", holder=_lease(db, parent))  # a continuation now exists
        db.release_compression_lock(parent, HOLDER)
        db.reopen_session(parent)
        _stamp(db, parent, "tui_close", age=60.0)
        holder = _lease(db, parent, holder="second-writer")

        with pytest.raises(RuntimeError, match="already ended.*published continuation"):
            _publish(db, parent, "C_second", holder=holder)
        assert db.get_session("C_second") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"  # untouched

    @pytest.mark.parametrize("reason", ["session_reset", "session_switch", "idle", "daily"])
    def test_reset_boundary_fails_closed(self, db: SessionDB, reason: str) -> None:
        assert reason in _RESET_END_REASONS
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason, age=60.0)
        holder = _lease(db, parent)
        with pytest.raises(RuntimeError, match=f"already ended.*{reason} boundary"):
            _publish(db, parent, f"C_{reason}", holder=holder)
        assert db.get_session(f"C_{reason}") is None

    def test_compression_stamp_fails_closed(self, db: SessionDB) -> None:
        parent = "P_compression"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "compression", age=60.0)
        holder = _lease(db, parent)
        with pytest.raises(RuntimeError, match="already ended.*closed by compression"):
            _publish(db, parent, "C_x", holder=holder)


class TestVerdictIsSharedWithTheAgentGuard:
    """The pre-flush guard must agree with publish, or a durable flush is skipped (or written for nothing)."""

    def test_read_only_verdict_matches_publish(self, db: SessionDB) -> None:
        for sid, reason in [("live", None), ("auto", "ws_disconnect")]:
            db.create_session(sid, source="tui")
            if reason:
                _stamp(db, sid, reason, age=60.0)
            assert db.compression_parent_deliberately_ended(sid) is False, (sid, reason)

        db.create_session("stale", source="tui")
        _stamp(db, "stale", "tui_close", age=60.0)
        assert db.compression_parent_deliberately_ended("stale") is True  # no lease: unprovable
        _lease(db, "stale")
        assert db.compression_parent_deliberately_ended("stale", holder=HOLDER) is False
        assert db.compression_parent_deliberately_ended("stale", holder="someone-else") is True

        db.create_session("closed_mid", source="tui")
        _lease(db, "closed_mid")
        time.sleep(0.01)
        _stamp(db, "closed_mid", "tui_close")
        assert db.compression_parent_deliberately_ended("closed_mid", holder=HOLDER) is True

        for sid, reason in [("reset", "session_reset"), ("comp", "compression")]:
            db.create_session(sid, source="tui")
            _stamp(db, sid, reason, age=60.0)
            _lease(db, sid)
            assert db.compression_parent_deliberately_ended(sid, holder=HOLDER) is True, (sid, reason)

        db.create_session("superseded", source="tui")
        _publish(db, "superseded", "superseded_child", holder=_lease(db, "superseded"))
        db.release_compression_lock("superseded", HOLDER)
        db.reopen_session("superseded")
        _stamp(db, "superseded", "cli_close", age=60.0)
        _lease(db, "superseded")
        assert db.compression_parent_deliberately_ended("superseded", holder=HOLDER) is True
        assert db.compression_parent_deliberately_ended("missing") is False
        assert db.compression_parent_deliberately_ended("") is False

    def test_agent_guard_delegates_to_the_store(self, db: SessionDB) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        db.create_session("stale", source="tui")
        _stamp(db, "stale", "tui_close", age=60.0)
        _lease(db, "stale")
        assert _parent_deliberately_ended(db, "stale", holder=HOLDER) is False
        assert _parent_deliberately_ended(db, "stale") is True  # no lease ordering to lean on
        db.create_session("reset", source="tui")
        _stamp(db, "reset", "session_reset", age=60.0)
        _lease(db, "reset")
        assert _parent_deliberately_ended(db, "reset", holder=HOLDER) is True

    def test_agent_guard_fails_open_and_keeps_the_taxonomy_fallback(self) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        broken = MagicMock()
        broken.compression_parent_deliberately_ended.side_effect = RuntimeError("db unavailable")
        assert _parent_deliberately_ended(broken, "x", holder=HOLDER) is False

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
        """#106459 end-to-end: the live session's row carries an explicit-close stamp that predates
        the attempt, and auto-compaction still rotates instead of computing and discarding forever."""
        parent = "PARENT_106459_E2E"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close", age=60.0)

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id != parent, "rotation aborted on the stale explicit-close stamp"
        assert db.get_session(parent)["end_reason"] == "compression"
        child_row = db.get_session(agent.session_id)
        assert child_row is not None and child_row["parent_session_id"] == parent

    def test_close_during_compression_still_aborts_rotation(self, db: SessionDB) -> None:
        """The user closes the session while the summary is being produced: the close wins."""
        parent = "PARENT_106459_CLOSE_MID"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        compressor = agent.context_compressor
        summary = compressor.compress.return_value

        def close_while_summarizing(*args, **kwargs):
            time.sleep(0.01)
            db.end_session(parent, "tui_close")  # session.close lands after the lease was acquired
            return summary

        compressor.compress.side_effect = close_while_summarizing
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id == parent, "a close made mid-compression must not be resurrected"
        assert db.get_session(parent)["end_reason"] == "tui_close"

    def test_reset_boundary_still_aborts_rotation(self, db: SessionDB) -> None:
        parent = "PARENT_106459_RESET"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "session_reset", age=60.0)

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == "session_reset"
