"""Regression tests for #106459 — an over-limit session must not become permanently
uncompressible because its row carries a stale explicit-close stamp, and a close the user
makes must never be resurrected.

``publish_compression_child`` used to fail closed on ANY non-automatic ``end_reason``
(``tui_close``, ``cli_close``, ``webhook_complete``, gateway-recovery and cron reasons), and
``end_session()`` is first-stamp-wins, so nothing ever cleared it. Every turn then ran the
compression, discarded its result at publication, kept the oversized history and hit
``Context length exceeded … Cannot compress further`` again; manual ``/compress`` reported
"No changes". The agent's pre-flush guard re-implemented the same taxonomy and aborted even
earlier.

Contract under test. The store never decides on its own that an explicit close is stale.
Publication (single verdict owner ``_compression_parent_obstacle``, shared with the agent's
pre-flush guard) clears only automatic-cleanup stamps (#88197) and fails closed on every other
stamp. Turn-lease admission clears nothing: a TUI turn worker can take its lease after the host
has already closed the session. A stale explicit close is cleared only through
``reopen_if_explicitly_closed()``, called by the host that still routes the session -- the TUI,
under ``_sessions_lock`` as it starts a turn for a session it still has registered (covered in
``tests/tui_gateway/test_106459_routing_provenance_reopen.py``). That method clears only explicit
closes with no published continuation and leaves automatic, ``'compression'``, boundary (reset
reasons and CLI ``new_session``) and superseded stamps alone. Here the host's call is stood in
for by ``_host_reopen``.
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
# Holders carry this process's pid: a dead pid makes a lease reclaimable, which is not what these probe.
TURN = f"pid={os.getpid()}:turn=t1:platform=tui"
TURN_2 = f"pid={os.getpid()}:turn=t2:platform=tui"
NOT_HEALED = "healed only by the host that still routes the session"
PROVENANCE = "the test host still routes the session"


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
    """End the row with *reason*; ``age`` backdates the stamp so it is unambiguously older than
    anything that follows."""
    db.end_session(session_id, reason)
    if age:
        db._write_sql("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time() - age, session_id))
    row = db.get_session(session_id)
    assert row["ended_at"] is not None and row["end_reason"] == reason


def _lease(db: SessionDB, session_id: str, holder: str = HOLDER) -> str:
    """The compression lease: publication ownership only."""
    assert db.try_acquire_compression_lock(session_id, holder, ttl_seconds=300.0)
    return holder


def _admit(db: SessionDB, session_id: str, holder: str = TURN) -> str:
    """Admit a turn on the conversation, as ``admit_durable_turn_lease`` does inside the turn worker."""
    assert db.try_acquire_session_turn_lease(session_id, holder, ttl_seconds=300.0)
    return holder


def _host_reopen(db: SessionDB, session_id: str) -> str | None:
    """What the TUI does under ``_sessions_lock`` for a session it still has registered."""
    return db.reopen_if_explicitly_closed(session_id, provenance=PROVENANCE)


def _live(db: SessionDB, session_id: str) -> bool:
    row = db.get_session(session_id)
    return row["ended_at"] is None and row["end_reason"] is None


class TestReopenIfExplicitlyClosed:
    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_clears_each_explicit_close_and_logs_the_provenance(self, db: SessionDB, reason: str, caplog) -> None:
        sid = f"S_{reason}"
        db.create_session(sid, source="tui")
        _stamp(db, sid, reason, age=60.0)

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            assert _host_reopen(db, sid) == reason

        assert _live(db, sid)
        assert any(reason in r.getMessage() and PROVENANCE in r.getMessage() and "stale" in r.getMessage()
                   for r in caplog.records), "clearing must be logged with the reason and the host's provenance"

    @pytest.mark.parametrize("reason", ["ws_disconnect", "compression", *BOUNDARIES])
    def test_leaves_every_other_stamp_alone(self, db: SessionDB, reason: str) -> None:
        sid = f"S_{reason}"
        db.create_session(sid, source="tui")
        _stamp(db, sid, reason, age=60.0)
        assert _host_reopen(db, sid) is None
        row = db.get_session(sid)
        assert row["end_reason"] == reason and row["ended_at"] is not None

    def test_leaves_a_superseded_close_alone(self, db: SessionDB) -> None:
        parent = "P_superseded_reopen"
        db.create_session(parent, source="tui")
        _publish(db, parent, "C_first", holder=_lease(db, parent))  # a continuation now exists
        db.release_compression_lock(parent, HOLDER)
        db.reopen_session(parent)
        _stamp(db, parent, "tui_close", age=60.0)
        assert _host_reopen(db, parent) is None
        assert db.get_session(parent)["end_reason"] == "tui_close"

    def test_live_missing_and_empty_ids_are_no_ops(self, db: SessionDB) -> None:
        db.create_session("live", source="tui")
        assert _host_reopen(db, "live") is None
        assert _live(db, "live")
        assert _host_reopen(db, "missing") is None
        assert _host_reopen(db, "") is None


class TestTurnAdmissionHealsNothing:
    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_admitting_a_turn_leaves_an_explicit_close(self, db: SessionDB, reason: str) -> None:
        """A lease is not routing provenance: the store must not treat its acquisition as proof."""
        sid = f"S_admit_{reason}"
        db.create_session(sid, source="tui")
        _stamp(db, sid, reason, age=60.0)
        _admit(db, sid)
        assert db.get_session(sid)["end_reason"] == reason

    def test_a_late_turn_lease_after_the_host_closed_leaves_the_close(self, db: SessionDB) -> None:
        """Fourth review probe, store level: the TUI accepted a prompt and started a worker, the user closed
        the session (pop, grace, ``tui_close``), and only then did the worker reach ``run_conversation()``
        and take its turn lease. Nothing re-applies the close afterwards, so the lease must not clear it."""
        sid = "S_late_worker"
        db.create_session(sid, source="tui")
        _stamp(db, sid, "tui_close")  # teardown finished before the worker took its lease
        _admit(db, sid)
        assert db.get_session(sid)["end_reason"] == "tui_close"
        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, sid, "C_never", holder=_lease(db, sid))
        assert db.get_session("C_never") is None
        assert db.get_session(sid)["end_reason"] == "tui_close"


class TestHostReopenUnwedgesRotation:
    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_a_stale_close_cleared_by_the_host_lets_the_rotation_publish(self, db: SessionDB, reason: str) -> None:
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        db.append_message(parent, "user", content="hello")
        _stamp(db, parent, reason, age=60.0)  # the #106459 shape: the mark predates this turn
        assert _host_reopen(db, parent) == reason
        _admit(db, parent)
        _publish(db, parent, f"C_{reason}", holder=_lease(db, parent))

        parent_row = db.get_session(parent)
        assert parent_row["end_reason"] == "compression" and parent_row["ended_at"] is not None
        child_row = db.get_session(f"C_{reason}")
        assert child_row is not None and child_row["parent_session_id"] == parent

    def test_repeated_rotation_is_not_wedged(self, db: SessionDB) -> None:
        """The field shape: the stamp lands once, and every later turn must still rotate."""
        parent = "P_repeat"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "cli_close", age=60.0)
        _host_reopen(db, parent)
        _admit(db, parent)
        _publish(db, parent, "C_1", holder=_lease(db, parent))
        db.release_session_turn_lease(parent, TURN)
        _host_reopen(db, "C_1")  # nothing to clear on the live continuation
        _admit(db, "C_1", holder=TURN_2)
        _publish(db, "C_1", "C_2", holder=_lease(db, "C_1"))
        assert db.get_session("C_1")["end_reason"] == "compression"
        assert db.get_session("C_2")["parent_session_id"] == "C_1"

    def test_a_stale_close_on_a_compression_child_is_cleared_on_that_row(self, db: SessionDB) -> None:
        """The host reopens the row the turn writes to -- the compression tip -- not the lineage root."""
        root = "P_root"
        db.create_session(root, source="tui")
        _admit(db, root)
        _publish(db, root, "C_1", holder=_lease(db, root))
        db.release_session_turn_lease(root, TURN)
        _stamp(db, "C_1", "tui_close", age=60.0)

        assert _host_reopen(db, "C_1") == "tui_close"
        assert db.get_session(root)["end_reason"] == "compression"  # the root's stamp is lineage, untouched
        _admit(db, "C_1", holder=TURN_2)
        _publish(db, "C_1", "C_2", holder=_lease(db, "C_1"))
        assert db.get_session("C_2")["parent_session_id"] == "C_1"

    def test_a_stamp_that_lands_mid_turn_costs_that_turn_only(self, db: SessionDB) -> None:
        """A foreign writer's mis-stamp during the turn cannot be told from a deliberate close, so that
        rotation fails closed; the host's reopen on the next prompt clears it and the rotation publishes."""
        parent = "P_mid_turn"
        db.create_session(parent, source="tui")
        _host_reopen(db, parent)
        _admit(db, parent)
        _stamp(db, parent, "webhook_complete")
        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, "C_never", holder=_lease(db, parent))
        assert db.get_session("C_never") is None
        db.release_compression_lock(parent, HOLDER)
        db.release_session_turn_lease(parent, TURN)

        assert _host_reopen(db, parent) == "webhook_complete"
        _admit(db, parent, holder=TURN_2)
        _publish(db, parent, "C_next", holder=_lease(db, parent))
        assert db.get_session(parent)["end_reason"] == "compression"
        assert db.get_session("C_next")["parent_session_id"] == parent


class TestADeliberateCloseIsNeverHealedAtPublication:
    def test_close_during_compression_fails_closed(self, db: SessionDB) -> None:
        """First review probe: lease, then the user closes, then publish."""
        parent = "P_closed_mid_compression"
        db.create_session(parent, source="tui")
        _admit(db, parent)
        holder = _lease(db, parent)
        _stamp(db, parent, "tui_close")

        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, "C_never", holder=holder)
        assert db.get_session("C_never") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"  # the user's close survives

    def test_close_after_the_turn_was_admitted_fails_closed(self, db: SessionDB) -> None:
        """Second review probe: the turn is running, ``session.close`` stamps after its grace without
        interrupting the turn thread, and that thread then takes the compression lease."""
        parent = "P_closed_during_turn"
        db.create_session(parent, source="tui")
        _admit(db, parent)
        _stamp(db, parent, "tui_close")
        holder = _lease(db, parent)

        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, "C_never", holder=holder)
        assert db.get_session("C_never") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"

    def test_close_during_a_turn_that_began_on_a_stale_stamp_is_not_resurrected(self, db: SessionDB) -> None:
        """Third review probe, store only: the turn began on the stale stamp this fix targets, the user's
        close during it is a no-op write under first-stamp-wins, and publication must still refuse."""
        parent = "P_stale_then_closed"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "tui_close", age=60.0)
        _admit(db, parent)
        db.end_session(parent, "tui_close")  # no-op: the stale stamp already occupies the row
        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, "C_never", holder=_lease(db, parent))
        assert db.get_session("C_never") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"

    def test_after_the_host_reopen_a_close_during_the_turn_is_recorded_and_wins(self, db: SessionDB) -> None:
        """Third review probe on the host path: the TUI cleared the stale stamp as it started the turn, so
        the user's close during it is an actual write with a fresh timestamp, and publication preserves it."""
        parent = "P_host_then_closed"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "tui_close", age=60.0)
        assert _host_reopen(db, parent) == "tui_close"
        _admit(db, parent)
        before = time.time()
        db.end_session(parent, "tui_close")
        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, "C_never", holder=_lease(db, parent))
        row = db.get_session(parent)
        assert row["end_reason"] == "tui_close" and row["ended_at"] >= before, "the fresh close, not the stale one"

    @pytest.mark.parametrize("reason", STALE_EXPLICIT_CLOSES)
    def test_without_the_host_a_stale_close_is_not_healed(self, db: SessionDB, reason: str) -> None:
        parent = f"P_nohost_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason, age=60.0)
        _admit(db, parent)
        holder = _lease(db, parent)  # neither lease is evidence
        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, f"C_{reason}", holder=holder)
        with pytest.raises(RuntimeError, match=f"already ended.*{NOT_HEALED}"):
            _publish(db, parent, f"C_{reason}")  # lease-less publication: same contract
        assert db.get_session(f"C_{reason}") is None
        assert db.get_session(parent)["end_reason"] == reason


class TestLineageOwnedElsewhereStillFailsClosed:
    def test_explicit_close_with_published_continuation_fails_closed(self, db: SessionDB) -> None:
        parent = "P_superseded"
        db.create_session(parent, source="tui")
        _admit(db, parent)
        _publish(db, parent, "C_first", holder=_lease(db, parent))  # a continuation now exists
        db.release_compression_lock(parent, HOLDER)
        db.release_session_turn_lease(parent, TURN)
        db.reopen_session(parent)
        _stamp(db, parent, "tui_close", age=60.0)
        assert _host_reopen(db, parent) is None
        _admit(db, parent, holder=TURN_2)
        holder = _lease(db, parent, holder="second-writer")

        with pytest.raises(RuntimeError, match="already ended.*published continuation"):
            _publish(db, parent, "C_second", holder=holder)
        assert db.get_session("C_second") is None
        assert db.get_session(parent)["end_reason"] == "tui_close"  # untouched

    @pytest.mark.parametrize("reason", BOUNDARIES)
    def test_boundary_fails_closed_even_after_a_host_reopen(self, db: SessionDB, reason: str) -> None:
        assert reason in _RESET_END_REASONS or reason == "new_session"  # CLI /new is a boundary too
        parent = f"P_{reason}"
        db.create_session(parent, source="tui")
        _stamp(db, parent, reason, age=60.0)
        assert _host_reopen(db, parent) is None
        _admit(db, parent)
        with pytest.raises(RuntimeError, match=f"already ended.*{reason} boundary"):
            _publish(db, parent, f"C_{reason}", holder=_lease(db, parent))
        assert db.get_session(f"C_{reason}") is None
        assert db.get_session(parent)["end_reason"] == reason

    def test_compression_stamp_fails_closed(self, db: SessionDB) -> None:
        parent = "P_compression"
        db.create_session(parent, source="tui")
        _stamp(db, parent, "compression", age=60.0)
        assert _host_reopen(db, parent) is None
        _admit(db, parent)
        with pytest.raises(RuntimeError, match="already ended.*closed by compression"):
            _publish(db, parent, "C_x", holder=_lease(db, parent))


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
        _admit(db, "stale")
        assert db.compression_parent_deliberately_ended("stale") is True  # admission is not provenance
        _host_reopen(db, "stale")
        assert db.compression_parent_deliberately_ended("stale") is False

        db.create_session("closed_mid", source="tui")
        _admit(db, "closed_mid")
        _stamp(db, "closed_mid", "tui_close")
        assert db.compression_parent_deliberately_ended("closed_mid") is True

        for sid, reason in [("reset", "session_reset"), ("new", "new_session"), ("comp", "compression")]:
            db.create_session(sid, source="tui")
            _stamp(db, sid, reason, age=60.0)
            _host_reopen(db, sid)
            assert db.compression_parent_deliberately_ended(sid) is True, (sid, reason)

        db.create_session("superseded", source="tui")
        _publish(db, "superseded", "superseded_child", holder=_lease(db, "superseded"))
        db.release_compression_lock("superseded", HOLDER)
        db.reopen_session("superseded")
        _stamp(db, "superseded", "cli_close", age=60.0)
        _host_reopen(db, "superseded")
        assert db.compression_parent_deliberately_ended("superseded") is True
        assert db.compression_parent_deliberately_ended("missing") is False
        assert db.compression_parent_deliberately_ended("") is False

    def test_agent_guard_delegates_to_the_store(self, db: SessionDB) -> None:
        from agent.conversation_compression import _parent_deliberately_ended

        db.create_session("stale", source="tui")
        _stamp(db, "stale", "tui_close", age=60.0)
        _admit(db, "stale")
        assert _parent_deliberately_ended(db, "stale") is True
        _host_reopen(db, "stale")
        assert _parent_deliberately_ended(db, "stale") is False
        db.create_session("reset", source="tui")
        _stamp(db, "reset", "session_reset", age=60.0)
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

    @staticmethod
    def _admit_turn(db: SessionDB, agent, session_id: str) -> None:
        """What ``admit_durable_turn_lease`` does at the start of ``run_conversation``."""
        _admit(db, session_id)
        agent._active_session_turn_lease_holder = TURN
        agent._active_session_turn_lease_ttl_seconds = 300.0

    @staticmethod
    def _compress(agent) -> None:
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
        agent._compress_context(list(msgs), "sys", approx_tokens=120_000)

    def test_a_stale_close_cleared_by_the_host_does_not_wedge_rotation(self, db: SessionDB) -> None:
        """#106459 end-to-end: the live session's row carries an explicit-close stamp that predates the
        turn; the routing host clears it as the turn starts, and auto-compaction rotates."""
        parent = "PARENT_106459_E2E"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close", age=60.0)
        _host_reopen(db, parent)
        self._admit_turn(db, agent, parent)

        self._compress(agent)

        assert agent.session_id != parent, "rotation aborted on the stale explicit-close stamp"
        assert db.get_session(parent)["end_reason"] == "compression"
        child_row = db.get_session(agent.session_id)
        assert child_row is not None and child_row["parent_session_id"] == parent

    def test_without_the_host_an_admitted_turn_does_not_rotate_over_a_close(self, db: SessionDB) -> None:
        """The inverse of the previous revision's contract: a turn lease alone never clears the stamp."""
        parent = "PARENT_106459_ADMIT_ONLY"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close", age=60.0)
        self._admit_turn(db, agent, parent)

        self._compress(agent)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == "tui_close"
        assert db._read_one("SELECT COUNT(*) FROM sessions WHERE parent_session_id = ?", (parent,))[0] == 0

    def test_close_during_a_turn_the_host_reopened_aborts_rotation(self, db: SessionDB) -> None:
        """Third probe end-to-end on the host path: stale stamp, host reopen, admit, close, compress."""
        parent = "PARENT_106459_HOST_THEN_CLOSED"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, "tui_close", age=60.0)
        _host_reopen(db, parent)
        self._admit_turn(db, agent, parent)
        before = time.time()
        db.end_session(parent, "tui_close")  # session.close during the turn

        self._compress(agent)

        assert agent.session_id == parent, "a close made during the turn must not be resurrected"
        row = db.get_session(parent)
        assert row["end_reason"] == "tui_close" and row["ended_at"] >= before
        assert db._read_one("SELECT COUNT(*) FROM sessions WHERE parent_session_id = ?", (parent,))[0] == 0

    def test_close_after_the_turn_was_admitted_aborts_rotation(self, db: SessionDB) -> None:
        """Second probe end-to-end: close first, the still-running turn then compresses and takes the
        lease second; publish creates no child."""
        parent = "PARENT_106459_CLOSE_FIRST"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        self._admit_turn(db, agent, parent)
        db.end_session(parent, "tui_close")

        self._compress(agent)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == "tui_close"
        assert db._read_one("SELECT COUNT(*) FROM sessions WHERE parent_session_id = ?", (parent,))[0] == 0

    def test_close_during_compression_still_aborts_rotation(self, db: SessionDB) -> None:
        """First probe end-to-end: the user closes while the summary is being produced."""
        parent = "PARENT_106459_CLOSE_MID"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        self._admit_turn(db, agent, parent)
        compressor = agent.context_compressor
        summary = compressor.compress.return_value

        def close_while_summarizing(*args, **kwargs):
            db.end_session(parent, "tui_close")  # session.close lands after the lease was acquired
            return summary

        compressor.compress.side_effect = close_while_summarizing
        self._compress(agent)

        assert agent.session_id == parent, "a close made mid-compression must not be resurrected"
        assert db.get_session(parent)["end_reason"] == "tui_close"

    @pytest.mark.parametrize("reason", ["session_reset", "new_session"])
    def test_boundary_still_aborts_rotation(self, db: SessionDB, reason: str) -> None:
        parent = f"PARENT_106459_{reason}"
        db.create_session(parent, source="tui")
        agent = self._build_agent(db, parent)
        _stamp(db, parent, reason, age=60.0)
        _host_reopen(db, parent)
        self._admit_turn(db, agent, parent)

        self._compress(agent)

        assert agent.session_id == parent
        assert db.get_session(parent)["end_reason"] == reason
