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


def test_async_delegations_schema_has_cross_adapter_columns():
    """A (#64934): the durable table carries the cross-adapter routing fields
    so a completion can be matched back to its spawning adapter."""
    import tools.async_delegation as ad

    conn = ad._connect()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(async_delegations)")}
    finally:
        conn.close()
    for col in (
        "source_adapter_id",
        "source_profile",
        "source_session_key",
        "delegation_type",
    ):
        assert col in cols, f"async_delegations missing column {col}"


class TestCrossAdapterCompletionIsolation:
    """#64934: a delegation spawned on adapter A must not be sewn onto the
    parent session when its completion routes through adapter B (multi-app
    fan-out). Drop fail-closed; the result remains in async_delegations."""

    @staticmethod
    def _feishu_entry(session_id, adapter_id):
        from datetime import datetime

        from gateway.config import Platform
        from gateway.session import SessionEntry

        aid = adapter_id.replace("%", "%25").replace(":", "%3A")
        return SessionEntry(
            session_key=f"agent:main:feishu:adapter={aid}:group:oc_chat:omt_thread",
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.FEISHU,
            chat_type="group",
        )

    @staticmethod
    def _source_key(adapter_id):
        aid = adapter_id.replace("%", "%25").replace(":", "%3A")
        return f"agent:main:feishu:adapter={aid}:group:oc_chat:omt_thread"

    def _make_runner(self, rows, *, switched_entry=None, compression_tip=None):
        from gateway.run import GatewayRunner
        from gateway.session import AsyncSessionStore

        runner = object.__new__(GatewayRunner)
        db = MagicMock()
        db.get_session = AsyncMock(side_effect=lambda sid: rows.get(sid))
        db.get_compression_tip = AsyncMock(return_value=compression_tip)
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
        runner.session_store.switch_session.assert_not_called()
        runner.session_store.advance_compression_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_cross_adapter_completion_drops_instead_of_sewing(self):
        # Completion routes through Tony's adapter (cli_aad581a8) but the
        # delegation was spawned by Pete (cli_aad7c4d).
        current = self._feishu_entry("sess_tony", "feishu:cli_aad581a8")
        runner = self._make_runner(
            {"sess_pete": {"id": "sess_pete", "ended_at": None}},
            switched_entry=self._feishu_entry("sess_pete", "feishu:cli_aad7c4d"),
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_pete",
            source_session_key=self._source_key("feishu:cli_aad7c4d"),
        )

        assert resolved is None
        self._assert_no_route_change(runner)

    @pytest.mark.asyncio
    async def test_same_adapter_completion_still_pins(self):
        current = self._feishu_entry("sess_tony", "feishu:cli_aad581a8")
        target = self._feishu_entry("sess_pete", "feishu:cli_aad581a8")
        runner = self._make_runner(
            {"sess_pete": {"id": "sess_pete", "ended_at": None}},
            switched_entry=target,
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_pete",
            source_session_key=self._source_key("feishu:cli_aad581a8"),
        )

        assert resolved is target
        runner.session_store.switch_session.assert_called_once_with(
            current.session_key, "sess_pete", expected_adapter_id="feishu:cli_aad581a8"
        )

    @pytest.mark.asyncio
    async def test_missing_source_key_falls_back_to_pin(self):
        # Legacy record (pre-#64934): empty source_session_key → original pin.
        current = self._feishu_entry("sess_tony", "feishu:cli_aad581a8")
        target = self._feishu_entry("sess_pete", "feishu:cli_aad7c4d")
        runner = self._make_runner(
            {"sess_pete": {"id": "sess_pete", "ended_at": None}},
            switched_entry=target,
        )

        resolved = await runner._resolve_async_delegation_session(current, "sess_pete")

        assert resolved is target
        runner.session_store.switch_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_cross_adapter_with_compression_still_drops(self):
        # pinned parent ended via compression; current route owns the lineage
        # (session_id == pinned) but its KEY belongs to a different adapter.
        current = self._feishu_entry("sess_pete", "feishu:cli_aad581a8")
        runner = self._make_runner(
            {
                "sess_pete": {
                    "id": "sess_pete",
                    "ended_at": "2026-07-08T00:00:00",
                    "end_reason": "compression",
                },
                "sess_child": {"id": "sess_child", "ended_at": None},
            },
            switched_entry=self._feishu_entry("sess_child", "feishu:cli_aad7c4d"),
            compression_tip="sess_child",
        )

        resolved = await runner._resolve_async_delegation_session(
            current, "sess_pete",
            source_session_key=self._source_key("feishu:cli_aad7c4d"),
        )

        assert resolved is None
        self._assert_no_route_change(runner)


class TestParentAdapterReroute:
    """#64934 (B): a cross-adapter completion is rerouted to the spawning
    adapter's source so the result lands in the parent's own session."""

    @staticmethod
    def _feishu_source(adapter_id):
        from gateway.config import Platform
        from gateway.session import SessionSource

        return SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_chat",
            chat_type="group",
            thread_id="omt_thread",
            adapter_id=adapter_id,
        )

    @staticmethod
    def _key(adapter_id):
        aid = adapter_id.replace(":", "%3A")
        return f"agent:main:feishu:adapter={aid}:group:oc_chat:omt_thread"

    def _make_runner(self, parent_source):
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner._build_process_event_source = lambda evt: parent_source
        return runner

    def test_cross_adapter_reroutes_to_parent_source(self):
        parent = self._feishu_source("feishu:cli_aad7c4d")  # Pete
        runner = self._make_runner(parent)
        current = self._feishu_source("feishu:cli_aad581a8")  # Tony

        result = runner._maybe_reroute_to_parent_adapter(
            {
                "session_key": self._key("feishu:cli_aad581a8"),
                "source_session_key": self._key("feishu:cli_aad7c4d"),
            },
            current,
        )

        assert result is parent

    def test_same_adapter_no_reroute(self):
        parent = self._feishu_source("feishu:cli_aad581a8")
        runner = self._make_runner(parent)
        current = self._feishu_source("feishu:cli_aad581a8")

        result = runner._maybe_reroute_to_parent_adapter(
            {
                "session_key": self._key("feishu:cli_aad581a8"),
                "source_session_key": self._key("feishu:cli_aad581a8"),
            },
            current,
        )

        assert result is current

    def test_missing_source_key_no_reroute(self):
        runner = self._make_runner(None)
        current = self._feishu_source("feishu:cli_aad581a8")

        result = runner._maybe_reroute_to_parent_adapter(
            {"session_key": self._key("feishu:cli_aad581a8")},
            current,
        )

        assert result is current

    def test_parent_source_unresolvable_no_reroute(self):
        # Cross-adapter but the parent source cannot be rebuilt → stay put;
        # the resolver guard then drops the sew fail-closed.
        runner = self._make_runner(None)
        current = self._feishu_source("feishu:cli_aad581a8")

        result = runner._maybe_reroute_to_parent_adapter(
            {
                "session_key": self._key("feishu:cli_aad581a8"),
                "source_session_key": self._key("feishu:cli_aad7c4d"),
            },
            current,
        )

        assert result is current

