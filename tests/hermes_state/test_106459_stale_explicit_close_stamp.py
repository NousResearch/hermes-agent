"""Regression tests for #106459 — an over-limit session must not become permanently
uncompressible because its row carries a stale explicit-close stamp, and a close the user
makes during a turn must never be resurrected.

``publish_compression_child`` used to fail closed on ANY non-automatic ``end_reason``
(``tui_close``, ``cli_close``, ``webhook_complete``, gateway-recovery and cron reasons), and
``end_session()`` is first-stamp-wins, so nothing ever cleared it. Every turn then ran the
compression, discarded its result at publication, kept the oversized history and hit
``Context length exceeded … Cannot compress further`` again; manual ``/compress`` reported
"No changes". The agent's pre-flush guard re-implemented the same taxonomy and aborted even
earlier.

Contract under test (single verdict owner ``_compression_parent_obstacle``): an explicit
close is cleared only on POSITIVE evidence that the session was driven after it — this
writer's session turn lease (``session_turn_leases.acquired_at``, taken once per turn by the
façade before the transcript is loaded) was admitted AFTER the stamp. Every sanctioned resume
clears the stamp first (``reopen_session``) and a host never admits a turn on a session it is
closing, so that order can only mean the stamp missed a still-routed session. A stamp at or
after admission is a close made during this turn (``session.close`` stamps ``tui_close``
after a five-second grace even while the turn thread runs on, and that thread later takes the
compression lease) and is preserved: publish fails closed and no child is created. The
compression lease is no evidence — it is taken inside the turn, after any such close. Without
an admitted turn of our own the stamp cannot be proven stale and also fails closed. Lineage
owned elsewhere always fails closed: a ``'compression'`` stamp, a boundary reason (the reset
reasons and CLI ``new_session``), or an explicit close whose continuation child was already
published.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from hermes_state_common import _BOUNDARY_END_REASONS, _RESET_END_REASONS

STALE_EXPLICIT_CLOSES = ["tui_close", "cli_close", "webhook_complete"]
BOUNDARIES = sorted(_BOUNDARY_END_REASONS)
HOLDER = "compression-writer"
TURN = "pid=1:turn=t1:platform=tui"


@pytest.fixture
def db(tmp_path: Path):
    handle = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield handle
    finally:
        handle.close()


def _publish(db: SessionDB, parent: str, child: str, *, holder: str | None = None,
             turn: str | None = None) -> None:
    db.publish_compression_child(
        parent_session_id=parent,
        child_session_id=child,
        source="tui",
        messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
        require_compression_lease=holder is not None,
        compression_lock_holder=holder,
        turn_lease_holder=turn,
    )


def _stamp(db: SessionDB, session_id: str, reason: str, *, age: float = 0.0) -> None:
    """End the row with *reason*; ``age`` backdates the stamp so its order against a turn
    admitted afterwards is unambiguous."""
    db.end_session(session_id, reason)
    if age:
        db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time() - age, session_id))
    row = db.get_session(session_id)
    assert row["ended_at"] is not None and row["end_reason"] == reason


def _lease(db: SessionDB, session_id: str, holder: str = HOLDER) -> str:
    """The compression lease: publication ownership, never evidence about the stamp."""
    assert db.try_acquire_compression_lock(session_id, holder, ttl_seconds=300.0)
    return holder


def _turn(db: SessionDB, session_id: str, holder: str = TURN) -> str:
    """Admit a turn on the conversation, as ``admit_durable_turn_lease`` does before a turn runs."""
    assert db.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=300.0)
    return holder


def _turn_admitted_at(db: SessionDB, session_id: str) -> float:
    key = db._session_turn_lease_key(session_id)
    return db._read_one("SELECT acquired_at FROM session_turn_leases WHERE conversation_id = ?", (key,))[0]


class TestStaleStampBeforeTheTurnIsCleared:
    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_explicit_close_before_the_turn_publishes(self, db: SessionDB, reason: str, caplog) -> None:
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        db.append_message(parent, "user", content="hello")
        _stamp(db, parent, reason, age=60.0)  # the #106459 shape: the mark predates this turn
        turn = _turn(db, parent)
        holder = _lease(db, parent)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            _publish(db, parent, f"C_{reason}", holder=holder, turn=turn)

        parent_row = db.get_session(parent)
        assert parent_row["end_reason"] == "compression"  # the true boundary, not the stale stamp
        assert parent_row["ended_at"] is not None
        child_row = db.get_session(f"C_{reason}")
        assert child_row is not None and child_row["parent_session_id"] == parent
        assert any(reason in rec.getMessage() and "stale" in rec.getMessage() for rec in caplog.records), (
            "clearing an explicit-close stamp must be logged with the stale reason")

    def test_repeated_rotation_is_not_wedged(self, db: SessionDB) -> None:
        """The field shape: the stamp lands once, and every later turn must still rotate."""
        parent = "P_repeat"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "cli_close", age=60.0)
        turn = _turn(db, parent)
        _publish(db, parent, "C_1", holder=_lease(db, parent), turn=turn)
        db.release_session_turn_lease(parent, turn)
        turn = _turn(db, "C_1", holder="pid=1:turn=t2:platform=tui")
        _publish(db, "C_1", "C_2", holder=_lease(db, "C_1"), turn=turn)
        assert db.get_session("C_1")["end_reason"] == "compression"
        assert db.get_session("C_2")["parent_session_id"] == "C_1"

    def test_stale_close_on_a_compression_child_is_ordered_by_the_conversation_lease(self, db: SessionDB) -> None:
        """The turn lease is keyed by the lineage root; a stamp on a later segment is ordered against it."""
        root = "P_root"
        db.create_session(root, source="tui")
        turn = _turn(db, root)
        _publish(db, root, "C_1", holder=_lease(db, root), turn=turn)
        db.release_session_turn_lease(root, turn)
        _stamp(db, "C_1", "tui_close", age=60.0)
        turn = _turn(db, "C_1", holder="pid=1:turn=t2:platform=tui")
        assert db._session_turn_lease_key("C_1") == root

        _publish(db, "C_1", "C_2", holder=_lease(db, "C_1"), turn=turn)
        assert db.get_session("C_1")["end_reason"] == "compression"
        assert db.get_session("C_2")["parent_session_id"] == "C_1"

    def test_a_stamp_that_lands_mid_turn_costs_that_turn_only(self, db: SessionDB) -> None:
        """No evidence yet during the turn the stamp landed in: that rotation fails closed. The next
        admitted turn is the evidence, and it rotates."""
        parent = "P_mid_turn"
        db.create_session(parent, source="tui")
        turn = _turn(db, parent)
        time.sleep(0.01)
        _stamp(db, parent, "webhook_complete")  # a foreign stale writer, during our turn
        with pytest.raises(RuntimeError, match="already ended.*after this turn was admitted"):
            _publish(db, parent, "C_never", holder=_lease(db, parent), turn=turn)
        assert db.get_session("C_never") is None
        db.release_compression_lock(parent, HOLDER)
        db.release_session_turn_lease(parent, turn)

        time.sleep(0.01)
        turn = _turn(db, parent, holder="pid=1:turn=t2:platform=tui")
        _publish(db, parent, "C_next", holder=_lease(db, parent), turn=turn)
        assert db.get_session(parent)["end_reason"] == "compression"
        assert db.get_session("C_next")["parent_session_id"] == parent


class TestDeliberateCloseDuringTheTurnIsPreserved:
    def test_close_after_the_turn_was_admitted_fails_closed(self, db: SessionDB) -> None:
        """The reviewer's probe: the turn is running, ``session.close`` stamps ``tui_close`` after its
        grace period without interrupting the turn thread, and that thread then acquires the
        compression lease and publishes."""
        parent = "P_closed_during_turn"
        db.create_session(parent, source="tui")
        turn = _turn(db, parent)
        time.sleep(0.01)
        _stamp(db, parent, "tui_close")
        holder = _lease(db, parent)  # the lease is younger than the close: no evidence either way

        with pytest.raises(RuntimeError, match="already ended.*after this turn was admitted"):
            _publish(db, parent, "C_never", holder=holder, turn=turn)

        assert db.get_session("C_never") is None
        row = db.get_session(parent)
        assert row["end_reason"] == "tui_close" and row["ended_at"] is not None  # the user's close survives

    def test_close_during_compression_fails_closed(self, db: SessionDB) -> None:
        """The earlier probe: lease first, then the close, then publish."""
        parent = "P_closed_mid_compression"
        db.create_session(parent, source="tui")
        turn = _turn(db, parent)
        holder = _lease(db, parent)
        time.sleep(0.01)
        _stamp(db, parent, "tui_close")

        with pytest.raises(RuntimeError, match="already ended.*after this turn was admitted"):
            _publish(db, parent, "C_never", holder=holder, turn=turn)
        assert db.get_session("C_never") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"

    def test_close_at_the_same_instant_fails_closed(self, db: SessionDB) -> None:
        parent = "P_tie"
        db.create_session(parent, source="tui")
        turn = _turn(db, parent)
        db.end_session(parent, "tui_close")
        db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (_turn_admitted_at(db, parent), parent))

        with pytest.raises(RuntimeError, match="after this turn was admitted"):
            _publish(db, parent, "C_tie", holder=_lease(db, parent), turn=turn)
        assert db.get_session(parent)["end_reason"] == "tui_close"

    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_without_an_admitted_turn_an_explicit_close_cannot_be_proven_stale(
        self, db: SessionDB, reason: str,
    ) -> None:
        parent = f"P_noturn_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason, age=60.0)
        holder = _lease(db, parent)  # a compression lease younger than the stamp proves nothing
        with pytest.raises(RuntimeError, match="already ended.*no admitted turn of ours"):
            _publish(db, parent, f"C_{reason}", holder=holder)
        with pytest.raises(RuntimeError, match="already ended.*no admitted turn of ours"):
            _publish(db, parent, f"C_{reason}")  # lease-less publication: same contract
        assert db.get_session(f"C_{reason}") is None
        assert db.get_session(parent)["end_reason"] == reason

    def test_a_foreign_turn_is_not_our_evidence(self, db: SessionDB) -> None:
        parent = "P_foreign_turn"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "tui_close", age=60.0)
        _turn(db, parent, holder="pid=2:turn=t9:platform=cli")
        with pytest.raises(RuntimeError, match="already ended.*no admitted turn of ours"):
            _publish(db, parent, "C_never", holder=_lease(db, parent), turn=TURN)
        assert db.get_session("C_never") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"


class TestLineageOwnedElsewhereStillFailsClosed:
    def test_explicit_close_with_published_continuation_fails_closed(self, db: SessionDB) -> None:
        parent = "P_superseded"
        db.create_session(parent, source="tui")
        turn = _turn(db, parent)
        _publish(db, parent, "C_first", holder=_lease(db, parent), turn=turn)  # a continuation now exists
        db.release_compression_lock(parent, HOLDER)
        db.release_session_turn_lease(parent, turn)
        db.reopen_session(parent)
        _stamp(db, parent, "tui_close", age=60.0)
        turn = _turn(db, parent, holder="pid=1:turn=t2:platform=tui")
        holder = _lease(db, parent, holder="second-writer")

        with pytest.raises(RuntimeError, match="already ended.*published continuation"):
            _publish(db, parent, "C_second", holder=holder, turn=turn)
        assert db.get_session("C_second") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"  # untouched

    @pytest.mark.parametrize("reason", BOUNDARIES)
    def test_boundary_fails_closed_even_when_older_than_the_turn(self, db: SessionDB, reason: str) -> None:
        assert reason in _RESET_END_REASONS or reason == "new_session"  # CLI /new is a boundary too
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason, age=60.0)
        turn = _turn(db, parent)
        with pytest.raises(RuntimeError, match=f"already ended.*{reason} boundary"):
            _publish(db, parent, f"C_{reason}", holder=_lease(db, parent), turn=turn)
        assert db.get_session(f"C_{reason}") is None
        assert db.get_session(parent)["end_reason"] == reason

    def test_compression_stamp_fails_closed(self, db: SessionDB) -> None:
        parent = "P_compression"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "compression", age=60.0)
        turn = _turn(db, parent)
        with pytest.raises(RuntimeError, match="already ended.*closed by compression"):
            _publish(db, parent, "C_x", holder=_lease(db, parent), turn=turn)


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
        assert db.compression_parent_deliberately_ended("stale") is True  # no turn: unprovable
        _lease(db, "stale")
        assert db.compression_parent_deliberately_ended("stale", turn_lease_holder=TURN) is True  # no such turn
        _turn(db, "stale")
        assert db.compression_parent_deliberately_ended("stale", turn_lease_holder=TURN) is False
        assert db.compression_parent_deliberately_ended("stale", turn_lease_holder="someone-else") is True

        db.create_session("closed_mid", source="tui")
        _turn(db, "closed_mid")
        time.sleep(0.01)
        _stamp(db, "closed_mid", "tui_close")
        assert db.compression_parent_deliberately_ended("closed_mid", turn_lease_holder=TURN) is True

        for sid, reason in [("reset", "session_reset"), ("new", "new_session"), ("comp", "compression")]:
            db.create_session(sid, source="tui")
            _stamp(db, sid, reason, age=60.0)
            _turn(db, sid)
            assert db.compression_parent_deliberately_ended(sid, turn_lease_holder=TURN) is True, (sid, reason)

        db.create_session("superseded", source="tui")
        turn = _turn(db, "superseded")
        _publish(db, "superseded", "superseded_child", holder=_lease(db, "superseded"), turn=turn)
        db.release_compression_lock("superseded", HOLDER)
        db.release_session_turn_lease("superseded", turn)
        db.reopen_session("superseded")
        _stamp(db, "superseded", "cli_close", age=60.0)
        _turn(db, "superseded", holder="pid=1:turn=t2:platform=tui")
        assert db.compression_parent_deliberately_ended(
            "superseded", turn_lease_holder="pid=1:turn=t2:platform=tui") is True
        assert db.compression_parent_deliberately_ended("missing") is False
        assert db.compression_parent_deliberately_ended("") is False

    def test_agent_guard_delegates_to_the_store(self, db: SessionDB) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        db.create_session("stale", source="tui")
        _stamp(db, "stale", "tui_close", age=60.0)
        _turn(db, "stale")
        assert _parent_deliberately_ended(db, "stale", turn_lease_holder=TURN) is False
        assert _parent_deliberately_ended(db, "stale") is True  # no admitted turn to lean on
        db.create_session("reset", source="tui")
        _stamp(db, "reset", "session_reset", age=60.0)
        _turn(db, "reset")
        assert _parent_deliberately_ended(db, "reset", turn_lease_holder=TURN) is True

    def test_agent_guard_fails_open_and_keeps_the_taxonomy_fallback(self) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        broken = MagicMock()
        broken.compression_parent_deliberately_ended.side_effect = RuntimeError("db unavailable")
        assert _parent_deliberately_ended(broken, "x", turn_lease_holder=TURN) is False

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

    @staticmethod
    def _admit_turn(db: SessionDB, agent, session_id: str) -> None:
        """What ``admit_durable_turn_lease`` does at the start of ``run_conversation``."""
        _turn(db, session_id)
        agent._active_session_turn_lease_holder = TURN
        agent._active_session_turn_lease_ttl_seconds = 300.0

    def test_stale_explicit_close_does_not_wedge_rotation(self, db: SessionDB) -> None:
        """#106459 end-to-end: the live session's row carries an explicit-close stamp that predates
        the turn, and auto-compaction still rotates instead of computing and discarding forever."""
        parent = "PARENT_106459_E2E"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close", age=60.0)
        self._admit_turn(db, agent, parent)

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id != parent, "rotation aborted on the stale explicit-close stamp"
        assert db.get_session(parent)["end_reason"] == "compression"
        child_row = db.get_session(agent.session_id)
        assert child_row is not None and child_row["parent_session_id"] == parent

    def test_close_after_the_turn_was_admitted_aborts_rotation(self, db: SessionDB) -> None:
        """The reviewer's probe end-to-end: close first, the still-running turn then compresses,
        acquires the lease second, and publish creates no child."""
        parent = "PARENT_106459_CLOSE_FIRST"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        self._admit_turn(db, agent, parent)
        time.sleep(0.01)
        db.end_session(parent, "tui_close")  # session.close, after its grace, with the turn thread alive

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id == parent, "a close made during the turn must not be resurrected"
        assert db.get_session(parent)["end_reason"] == "tui_close"
        assert db._read_one("SELECT COUNT(*) FROM sessions WHERE parent_session_id = ?", (parent,))[0] == 0

    def test_close_during_compression_still_aborts_rotation(self, db: SessionDB) -> None:
        """The user closes the session while the summary is being produced: the close wins."""
        parent = "PARENT_106459_CLOSE_MID"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        self._admit_turn(db, agent, parent)
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

    def test_without_an_admitted_turn_a_stale_close_is_not_healed(self, db: SessionDB) -> None:
        """No turn lease (a caller outside the façade): nothing proves the stamp stale, so the
        pre-#106459 contract stands and the close is preserved."""
        parent = "PARENT_106459_NO_TURN"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close", age=60.0)

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == "tui_close"

    @pytest.mark.parametrize("reason", ["session_reset", "new_session"])
    def test_boundary_still_aborts_rotation(self, db: SessionDB, reason: str) -> None:
        parent = f"PARENT_106459_{reason}"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, reason, age=60.0)
        self._admit_turn(db, agent, parent)

        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == reason
