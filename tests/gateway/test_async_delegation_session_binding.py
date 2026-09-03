"""Gateway-side session binding for async delegations (#57498, #55578).

Three invariants on the messaging-gateway surface, mirroring the TUI rules:

1. Completions are pinned to the spawning session (contributor commit).
2. A dead/ended spawning session is never resurrected: the injection is
   dropped, fail-closed (never rerouted to the peer's current session).
3. /new interrupts the old conversation's in-flight async delegations.
"""

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


class TestGatewayPinningFailsClosed:
    """The gateway must follow only verified compression continuations."""

    @staticmethod
    def _entry(session_id):
        from datetime import datetime

        from gateway.config import Platform
        from gateway.session import SessionEntry

        return SessionEntry(
            session_key="agent:main:telegram:group:-100:4",
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="group",
        )

    def _make_runner(
        self,
        rows,
        *,
        compression_tip=None,
        compression_error=None,
        switched_entry=None,
    ):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore

        runner = object.__new__(GatewayRunner)
        db = MagicMock()
        db.get_session = AsyncMock(side_effect=lambda session_id: rows.get(session_id))
        db.get_compression_tip = AsyncMock(
            return_value=compression_tip,
            side_effect=compression_error,
        )
        runner._session_db = db
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock(return_value=switched_entry)
        runner.session_store.advance_compression_session = MagicMock(
            return_value=switched_entry
        )
        runner._async_session_store = AsyncSessionStore(runner.session_store)
        return runner

    @staticmethod
    def _assert_no_route_change(runner):
        getattr(runner.session_store, "switch_session").assert_not_called()
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_not_called()


    @pytest.mark.asyncio
    async def test_live_spawning_session_rebinds_from_different_route(self):
        current = self._entry("sess_current")
        pinned = self._entry("sess_live")
        runner = self._make_runner(
            {"sess_live": {"id": "sess_live", "ended_at": None}},
            switched_entry=pinned,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_live"
        )

        assert resolved is pinned
        getattr(runner.session_store, "switch_session").assert_called_once_with(
            current.session_key, "sess_live"
        )

    @pytest.mark.asyncio
    async def test_live_delegate_child_resolves_to_parent_without_rebinding_route(self):
        current = self._entry("sess_parent")
        runner = self._make_runner(
            {
                "sess_parent": {"id": "sess_parent", "ended_at": None},
                "sess_child": {
                    "id": "sess_child",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_parent"},
                },
            },
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_child")

        assert resolved is current
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_nested_delegate_child_resolves_to_gateway_parent(self):
        current = self._entry("sess_parent")
        runner = self._make_runner(
            {
                "sess_parent": {"id": "sess_parent", "ended_at": None},
                "sess_child": {
                    "id": "sess_child",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_parent"},
                },
                "sess_grandchild": {
                    "id": "sess_grandchild",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_child"},
                },
            },
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_grandchild")

        assert resolved is current
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_delegate_provenance_cycle_fails_closed_without_route_change(self):
        current = self._entry("sess_current")
        runner = self._make_runner(
            {
                "sess_a": {
                    "id": "sess_a",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_b"},
                },
                "sess_b": {
                    "id": "sess_b",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_a"},
                },
            },
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_a")

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_self_referential_delegate_provenance_fails_closed(self):
        current = self._entry("sess_current")
        runner = self._make_runner(
            {
                "sess_self": {
                    "id": "sess_self",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_self"},
                },
            },
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_self")

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_overlong_delegate_chain_fails_closed_without_route_change(self):
        from gateway.run import _MAX_DELEGATE_PROVENANCE_HOPS

        # One acyclic hop past the cap: the cycle guard cannot catch this, so
        # only the hop bound stops the walk (#92620 review).
        depth = _MAX_DELEGATE_PROVENANCE_HOPS + 2
        rows = {
            f"sess_{i}": {
                "id": f"sess_{i}",
                "ended_at": None,
                "model_config": {"_delegate_from": f"sess_{i + 1}"},
            }
            for i in range(depth)
        }
        rows[f"sess_{depth}"] = {"id": f"sess_{depth}", "ended_at": None}
        current = self._entry("sess_current")
        runner = self._make_runner(rows)

        resolved = await runner._resolve_async_delegation_session(current, "sess_0")

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_delegate_chain_at_the_hop_limit_still_resolves(self):
        from gateway.run import _MAX_DELEGATE_PROVENANCE_HOPS

        # Exactly at the bound: the cap must not reject a resolvable chain.
        depth = _MAX_DELEGATE_PROVENANCE_HOPS
        rows = {
            f"sess_{i}": {
                "id": f"sess_{i}",
                "ended_at": None,
                "model_config": {"_delegate_from": f"sess_{i + 1}"},
            }
            for i in range(depth)
        }
        rows[f"sess_{depth}"] = {"id": f"sess_{depth}", "ended_at": None}
        current = self._entry(f"sess_{depth}")
        runner = self._make_runner(rows)

        resolved = await runner._resolve_async_delegation_session(current, "sess_0")

        assert resolved is current
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_delegate_with_missing_parent_fails_closed_without_route_change(self):
        current = self._entry("sess_current")
        runner = self._make_runner(
            {
                "sess_child": {
                    "id": "sess_child",
                    "ended_at": None,
                    "model_config": {"_delegate_from": "sess_missing"},
                },
            },
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_child")

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_malformed_delegate_config_fails_closed_without_route_change(self):
        current = self._entry("sess_current")
        runner = self._make_runner(
            {
                "sess_child": {
                    "id": "sess_child",
                    "ended_at": None,
                    "model_config": "{not-json}",
                },
            },
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_child")

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_non_compression_ended_parent_drops(self):
        current = self._entry("sess_old")
        runner = self._make_runner(
            {
                "sess_old": {
                    "id": "sess_old",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "session_reset",
                }
            }
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_old"
        )

        assert resolved is None
        self._assert_no_route_change(runner)


    @pytest.mark.asyncio
    async def test_intermediate_compression_route_advances_to_same_live_tip(self):
        current = self._entry("sess_middle")
        tip = self._entry("sess_tip")
        runner = self._make_runner(
            {
                "sess_parent": {
                    "id": "sess_parent",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "compression",
                },
                "sess_middle": {
                    "id": "sess_middle",
                    "ended_at": "2026-07-08T00:01:00",
                    "end_reason": "compression",
                    "parent_session_id": "sess_parent",
                },
                "sess_tip": {
                    "id": "sess_tip",
                    "ended_at": None,
                    "parent_session_id": "sess_middle",
                },
            },
            compression_tip="sess_tip",
            switched_entry=tip,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_parent"
        )

        assert resolved is tip
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_called_once_with(current.session_key, "sess_middle", "sess_tip")

    @pytest.mark.asyncio
    async def test_compression_parent_follows_real_sessiondb_lineage(self, tmp_path):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore
        from hermes_state import AsyncSessionDB, SessionDB

        session_db = SessionDB(db_path=tmp_path / "state.db")
        session_db.create_session("sess_parent", source="telegram")
        session_db.end_session("sess_parent", end_reason="compression")
        session_db.create_session(
            "sess_tip",
            source="telegram",
            parent_session_id="sess_parent",
        )

        current = self._entry("sess_parent")
        tip = self._entry("sess_tip")
        runner = object.__new__(GatewayRunner)
        runner._session_db = AsyncSessionDB(session_db)
        runner.session_store = MagicMock()
        runner.session_store.switch_session = MagicMock(return_value=tip)
        runner.session_store.advance_compression_session = MagicMock(return_value=tip)
        runner._async_session_store = AsyncSessionStore(runner.session_store)

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_parent"
        )

        assert resolved is tip
        getattr(
            runner.session_store, "advance_compression_session"
        ).assert_called_once_with(current.session_key, "sess_parent", "sess_tip")


class TestResetHandlerInterruptsDelegations:
    def test_reset_command_calls_interrupt_for_session(self):
        """The /new handler must sever the old conversation's delegations."""
        import inspect
        from gateway import slash_commands

        src = inspect.getsource(slash_commands.GatewaySlashCommandsMixin._handle_reset_command)
        assert "interrupt_for_session" in src
        assert "session_reset" in src
