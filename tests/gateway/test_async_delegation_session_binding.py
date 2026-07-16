"""Gateway-side session binding for async delegations (#57498, #55578).

Three invariants on the messaging-gateway surface, mirroring the TUI rules:

1. Completions are pinned to the spawning session (contributor commit).
2. A dead/ended spawning session is never resurrected: the injection is
   dropped, fail-closed (never rerouted to the peer's current session).
3. /new interrupts the old conversation's in-flight async delegations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tools.async_delegation as ad


@pytest.fixture(autouse=True)
def _reset_async_delegation():
    ad._reset_for_tests()
    yield
    ad._reset_for_tests()


def _seed_record(delegation_id, session_key="", parent_session_id="", status="running"):
    fn = MagicMock()
    with ad._records_lock:
        ad._records[delegation_id] = {
            "delegation_id": delegation_id,
            "status": status,
            "session_key": session_key,
            "parent_session_id": parent_session_id,
            "interrupt_fn": fn,
        }
    return fn


class TestInterruptForSessionByParentId:
    def test_parent_session_id_selector(self):
        mine = _seed_record("d1", session_key="agent:main:telegram:dm:1", parent_session_id="sess_old")
        other = _seed_record("d2", session_key="agent:main:telegram:dm:2", parent_session_id="sess_other")
        n = ad.interrupt_for_session(parent_session_id="sess_old")
        assert n == 1
        mine.assert_called_once()
        other.assert_not_called()

    def test_reset_interrupts_by_key_and_parent(self):
        """A /new reset passes both selectors — either match claims the record."""
        by_key = _seed_record("d1", session_key="agent:main:telegram:dm:1", parent_session_id="")
        by_parent = _seed_record("d2", session_key="", parent_session_id="sess_old")
        unrelated = _seed_record("d3", session_key="other", parent_session_id="other")
        n = ad.interrupt_for_session(
            session_key="agent:main:telegram:dm:1",
            parent_session_id="sess_old",
            reason="session_reset",
        )
        assert n == 2
        by_key.assert_called_once()
        by_parent.assert_called_once()
        unrelated.assert_not_called()


class TestGatewayPinningFailsClosed:
    """The gateway injection path must never resurrect an ended session."""

    def _make_runner(self, rows, tip=None):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        db = MagicMock()
        db.get_session = AsyncMock(side_effect=lambda sid: rows.get(sid))
        db.get_compression_tip = AsyncMock(return_value=tip)
        runner._session_db = db
        return runner

    def _resolve(self, runner, pinned_session_id):
        return asyncio.run(runner._resolve_pinned_delegation_session(pinned_session_id))

    def test_live_spawning_session_pins(self):
        runner = self._make_runner({"sess_old": {"id": "sess_old", "ended_at": None}})
        assert self._resolve(runner, "sess_old") == "sess_old"

    def test_ended_spawning_session_drops(self):
        runner = self._make_runner(
            {"sess_old": {"id": "sess_old", "ended_at": "2026-07-08T00:00:00"}}
        )
        assert self._resolve(runner, "sess_old") is None

    def test_unknown_spawning_session_drops(self):
        runner = self._make_runner({})
        assert self._resolve(runner, "sess_gone") is None

    def test_handler_routes_through_resolver(self):
        """The inner handler must gate pinning on the resolver, not raw ended_at."""
        import inspect
        from gateway.run import GatewayRunner

        src = inspect.getsource(GatewayRunner._handle_message_with_agent)
        assert "_resolve_pinned_delegation_session" in src


class TestCompressionContinuationRerouting:
    """#65779: a compression-ended parent is a continuation boundary, not a
    user-ended conversation — reroute the completion to the live tip."""

    _make_runner = TestGatewayPinningFailsClosed._make_runner
    _resolve = TestGatewayPinningFailsClosed._resolve

    def test_compression_parent_reroutes_to_live_tip(self):
        runner = self._make_runner(
            {
                "sess_parent": {
                    "id": "sess_parent",
                    "ended_at": "2026-07-16T11:09:33",
                    "end_reason": "compression",
                },
                "sess_tip": {"id": "sess_tip", "ended_at": None},
            },
            tip="sess_tip",
        )
        assert self._resolve(runner, "sess_parent") == "sess_tip"

    def test_compression_parent_with_ended_tip_drops(self):
        runner = self._make_runner(
            {
                "sess_parent": {
                    "id": "sess_parent",
                    "ended_at": "2026-07-16T11:09:33",
                    "end_reason": "compression",
                },
                "sess_tip": {
                    "id": "sess_tip",
                    "ended_at": "2026-07-16T12:00:00",
                    "end_reason": "session_reset",
                },
            },
            tip="sess_tip",
        )
        assert self._resolve(runner, "sess_parent") is None

    def test_compression_parent_without_continuation_drops(self):
        # get_compression_tip returns the input id when no continuation exists.
        runner = self._make_runner(
            {
                "sess_parent": {
                    "id": "sess_parent",
                    "ended_at": "2026-07-16T11:09:33",
                    "end_reason": "compression",
                },
            },
            tip="sess_parent",
        )
        assert self._resolve(runner, "sess_parent") is None

    def test_explicit_reset_still_drops(self):
        """Security assertion: /new stays fail-closed and never walks the chain."""
        runner = self._make_runner(
            {
                "sess_old": {
                    "id": "sess_old",
                    "ended_at": "2026-07-16T11:09:33",
                    "end_reason": "session_reset",
                },
            },
            tip="sess_child",
        )
        assert self._resolve(runner, "sess_old") is None
        runner._session_db.get_compression_tip.assert_not_called()


class TestResetHandlerInterruptsDelegations:
    def test_reset_command_calls_interrupt_for_session(self):
        """The /new handler must sever the old conversation's delegations."""
        import inspect
        from gateway import slash_commands

        src = inspect.getsource(slash_commands.GatewaySlashCommandsMixin._handle_reset_command)
        assert "interrupt_for_session" in src
        assert "session_reset" in src
