"""Gateway-side session binding for async delegations (#57498, #55578).

Three invariants on the messaging-gateway surface, mirroring the TUI rules:

1. Completions are pinned to the spawning session (contributor commit).
2. A dead/ended spawning session is never resurrected: the injection is
   dropped, fail-closed (never rerouted to the peer's current session).
3. /new interrupts the old conversation's in-flight async delegations.
"""

from unittest.mock import AsyncMock, MagicMock

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
        runner.session_store.switch_session.assert_not_called()
        runner.session_store.advance_compression_session.assert_not_called()


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
        runner.session_store.switch_session.assert_called_once_with(
            current.session_key, "sess_live", expected_session_id=current.session_id,
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
        runner.session_store.advance_compression_session.assert_called_once_with(current.session_key, "sess_middle", "sess_tip")

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
        runner.session_store.advance_compression_session.assert_called_once_with(current.session_key, "sess_parent", "sess_tip")




@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["none", "revoke", "replace"])
async def test_pending_pin_respects_concurrent_boundary(tmp_path, boundary):
    """A non-compression re-pin that resolved its row across an await must not move the route
    after the run was invalidated (/stop) or the route was replaced (/new, /resume) meanwhile;
    an undisturbed pin still lands. Real store + real resolver; the DB lookup is event-gated.
    Scenario by the #113690 reporter."""
    import asyncio
    from types import SimpleNamespace

    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore, SessionSource, SessionStore

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="test-chat", chat_type="dm", user_id="test-user")
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    generation = runner._begin_session_run_generation(entry.session_key)
    entered, release = asyncio.Event(), asyncio.Event()

    async def get_session(session_id):
        entered.set()
        await release.wait()
        return {"id": session_id, "ended_at": None}

    runner._session_db = SimpleNamespace(get_session=AsyncMock(side_effect=get_session))
    task = asyncio.create_task(runner._resolve_async_delegation_session(entry, "test-pinned"))
    await asyncio.wait_for(entered.wait(), 3)
    expected = "test-pinned"
    if boundary != "none":
        runner._invalidate_session_run_generation(entry.session_key, reason="test boundary")
        assert not runner._is_session_run_current(entry.session_key, generation)
        expected = entry.session_id
    if boundary == "replace":
        store.switch_session(entry.session_key, "test-replacement")
        expected = "test-replacement"
    release.set()
    result = await asyncio.wait_for(task, 3)

    assert store.lookup_by_session_key(entry.session_key).session_id == expected
    if boundary == "none":
        assert result is not None and result.session_id == expected
    else:
        assert result is None

@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["readiness", "claimed"])
@pytest.mark.parametrize("fault", [
    "parent_tip_error", "parent_tip_none", "parent_tip_self", "tip_row_error",
    "tip_row_missing", "tip_row_ended", "route_row_error", "route_row_missing",
    "route_tip_error", "route_tip_none",
])
async def test_compression_uncertainty_keeps_durable_completion_retryable(
    tmp_path, monkeypatch, phase, fault, request, private_db_probe_cleanup,
):
    """Uncertain lineage never consumes the result or mutates its route; replay recovers."""
    from types import SimpleNamespace
    from typing import Any, cast

    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session import SessionSource

    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        request.addfinalizer(runner.close_all_session_db_handles)
        request.addfinalizer(store.close_all_db_handles)
        entry = store.get_or_create_session(SessionSource(
            platform=Platform.TELEGRAM, chat_id="compression-owner", chat_type="dm",
        ))
        db = store._db
        parent = entry.session_id
        db.end_session(parent, end_reason="compression")
        db.create_session("middle", source="telegram", parent_session_id=parent)
        entry = store.switch_session(entry.session_key, "middle")
        assert entry is not None
        db.end_session("middle", end_reason="compression")
        db.create_session("tip", source="telegram", parent_session_id="middle")
        before = {sid: db.get_session(sid) for sid in (parent, "middle", "tip")}
        async_db = runner._session_db
        get_session = async_db.get_session
        get_tip = async_db.get_compression_tip
        parent_reads = 0
        enabled = True
        resolver_probe = True

        def failing():
            return enabled and (resolver_probe or phase == "readiness" or parent_reads >= 2)

        async def lookup_session(sid):
            nonlocal parent_reads
            if sid == parent:
                parent_reads += 1
            if failing() and sid == {"tip_row_error": "tip", "tip_row_missing": "tip",
                                    "tip_row_ended": "tip", "route_row_error": "middle",
                                    "route_row_missing": "middle"}.get(fault):
                if fault.endswith("error"):
                    raise RuntimeError("temporary session lookup failure")
                if fault.endswith("missing"):
                    return None
                row = await get_session(sid)
                return dict(row, ended_at=1, end_reason="compression")
            return await get_session(sid)

        async def lookup_tip(sid):
            if failing() and ((sid == parent and fault.startswith("parent_tip"))
                              or (sid == "middle" and fault.startswith("route_tip"))):
                if fault.endswith("error"):
                    raise RuntimeError("temporary compression lookup failure")
                return sid if fault.endswith("self") else None
            return await get_tip(sid)

        monkeypatch.setattr(async_db, "get_session", lookup_session)
        monkeypatch.setattr(async_db, "get_compression_tip", lookup_tip)
        assert await runner._resolve_async_delegation_session(entry, parent) is None
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None and current.session_id == "middle"
        for sid, row in before.items():
            assert db.get_session(sid) == row
        resolver_probe = False
        parent_reads = 0
        resolved = []

        async def accept(event):
            current = store.lookup_by_session_key(entry.session_key)
            assert current is not None
            result = await runner._resolve_async_delegation_session(current, event.metadata["gateway_session_id"])
            assert result is not None
            resolved.append(result.session_id)
            event._gateway_accepted = True

        runner.adapters[Platform.TELEGRAM] = cast(BasePlatformAdapter, SimpleNamespace(handle_message=accept))
        event: dict[str, Any] = {"type": "async_delegation", "delegation_id": "uncertain-compression",
                                "session_key": entry.session_key, "parent_session_id": parent,
                                "dispatched_at": 1.0, "summary": "completed result", "status": "completed"}
        ad._persist_dispatch(event)
        ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})
        assert await runner._deliver_completion_notification("completed result", event) is False
        row = ad.get_durable_delegation(event["delegation_id"])
        assert row is not None
        assert (row["delivery_state"], row["delivery_attempts"]) == (
            "pending", 0 if phase == "readiness" else 1,
        )
        import sqlite3
        with sqlite3.connect(ad._db_path()) as conn:
            assert conn.execute("SELECT delivery_claim FROM async_delegations WHERE delegation_id=?",
                                (event["delegation_id"],)).fetchone() == (None,)
        assert not resolved
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None and current.session_id == "middle"
        for sid, original in before.items():
            assert db.get_session(sid) == original
        enabled = False
        assert await runner._deliver_completion_notification("completed result", event) is True
        assert resolved == ["tip"]
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None and current.session_id == "tip"
        row = ad.get_durable_delegation(event["delegation_id"])
        assert row is not None and row["delivery_state"] == "delivered"
        assert await runner._deliver_completion_notification("completed result", event) is None
        assert resolved == ["tip"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["single", "group"])
@pytest.mark.parametrize("case", [
    "single", "nested", "relabeled", "limit", "over_limit", "missing", "cycle",
    "malformed", "nonobject", "invalid_marker", "empty_marker", "null_marker",
    "markerless", "null_config", "absent_config", "generic_parent", "foreground",
    "current", "current_malformed", "child_to_current", "current_boundary",
    "parent_boundary", "child_ended", "child_boundary", "intermediate_boundary",
    "compression", "compression_foreign", "idle",
])
async def test_delegate_completion_preserves_real_route_ownership(
    tmp_path, case, delivery, request, private_db_probe_cleanup,
):
    """Real durable admission and route resolution agree before any destructive switch."""
    import json
    import sqlite3
    from types import SimpleNamespace
    from typing import Any, cast

    from gateway.platforms.base import BasePlatformAdapter
    from gateway.config import GatewayConfig, Platform
    from gateway.run import GatewayRunner, _profile_runtime_scope
    from gateway.session import SessionSource
    from tools import async_delegation as ad

    with _profile_runtime_scope(tmp_path):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
        store = runner.session_store
        request.addfinalizer(runner.close_all_session_db_handles)
        request.addfinalizer(store.close_all_db_handles)
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="owner-chat", chat_type="dm")
        entry = store.get_or_create_session(source)
        db = store._db
        parent = entry.session_id
        previous = parent
        children = []
        for index in range({"nested": 2, "intermediate_boundary": 2, "limit": 16, "over_limit": 17}.get(case, 1)):
            child = f"worker-{index}"
            db.create_session(child, source="telegram" if case == "relabeled" else "subagent",
                              parent_session_id=previous, model_config={"_delegate_from": previous})
            children.append(child)
            previous = child
        pinned = children[-1]
        configs = {
            "missing": {"_delegate_from": "missing-parent"}, "cycle": {"_delegate_from": pinned},
            "malformed": "{broken", "nonobject": [], "invalid_marker": {"_delegate_from": 7},
            "empty_marker": {"_delegate_from": ""}, "null_marker": {"_delegate_from": None},
            "markerless": {}, "null_config": "null", "absent_config": None,
            "generic_parent": {}, "current_malformed": "{broken",
        }
        if case in configs:
            config = configs[case]
            raw = config if config is None or isinstance(config, str) else json.dumps(config)
            with sqlite3.connect(db.db_path) as conn:
                conn.execute("UPDATE sessions SET model_config=? WHERE id=?", (raw, pinned))
        if case == "generic_parent":
            with sqlite3.connect(db.db_path) as conn:
                conn.execute("UPDATE sessions SET source='telegram' WHERE id=?", (pinned,))
        if case == "foreground":
            pinned = parent
        if case in {"current", "current_malformed", "current_boundary", "child_to_current"}:
            entry = store.switch_session(entry.session_key, pinned)
            assert entry is not None
            if case == "child_to_current":
                db.create_session("grandchild", source="subagent", model_config={"_delegate_from": pinned})
                children.append("grandchild")
                pinned = "grandchild"
        if case == "current_boundary":
            db.end_session(pinned, end_reason="session_reset")
        if case == "parent_boundary":
            db.end_session(parent, end_reason="session_reset")
        if case == "child_ended":
            db.end_session(pinned, end_reason="agent_close")
        if case == "child_boundary":
            db.end_session(pinned, end_reason="session_reset")
        if case == "intermediate_boundary":
            db.end_session(children[0], end_reason="session_reset")
        if case in {"child_boundary", "intermediate_boundary"}:
            # The live ancestor must not override a newer user-selected route.
            entry = store.get_or_create_session(source, force_new=True)
            children.append(entry.session_id)
        if case == "idle":
            db.end_session(parent, end_reason="idle")
        if case in {"compression", "compression_foreign"}:
            db.create_session("continuation", source="telegram", parent_session_id=parent)
            db.end_session(parent, end_reason="compression")
            children.append("continuation")
            if case == "compression_foreign":
                entry = store.get_or_create_session(source, force_new=True)
                children.append(entry.session_id)
        before = {sid: db.get_session(sid) for sid in [parent, *children]}
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None
        route_before = current.session_id
        rejected = case in {
            "over_limit", "missing", "cycle", "malformed", "nonobject", "invalid_marker",
            "empty_marker", "null_marker", "markerless", "null_config", "absent_config",
            "current_boundary", "parent_boundary", "child_boundary", "intermediate_boundary", "compression_foreign",
        }
        resolved = []

        async def accept(event):
            current = store.lookup_by_session_key(event.metadata["gateway_session_key"])
            assert current is not None
            result = await runner._resolve_async_delegation_session(current, event.metadata["gateway_session_id"])
            resolved.append(result.session_id if result else None)
            event._gateway_accepted = True

        runner.adapters[Platform.TELEGRAM] = cast(
            BasePlatformAdapter, SimpleNamespace(handle_message=accept),
        )
        events = []
        for index in range(2 if delivery == "group" else 1):
            event: dict[str, Any] = {"type": "async_delegation", "delegation_id": f"deleg-{index}",
                     "session_key": entry.session_key, "parent_session_id": pinned,
                     "dispatched_at": 1.0, "summary": "completed result", "status": "completed"}
            ad._persist_dispatch(event)
            ad._persist_completion(event, {"status": "completed", "summary": event["summary"]})
            events.append(event)
        if delivery == "group":
            await runner._deliver_async_delegation_group(events)
        else:
            await runner._deliver_completion_notification("completed result", events[0])
        # This detects the deterministic false ack: live child accepted, resolver
        # rejects its provenance, yet the durable row previously became delivered.
        for event in events:
            row = ad.get_durable_delegation(event["delegation_id"])
            assert row is not None
            assert row["delivery_state"] == ("dropped" if rejected else "delivered")
        expected = "continuation" if case == "compression" else pinned if case == "generic_parent" else route_before
        assert resolved == ([] if rejected else [expected])
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None
        result = await runner._resolve_async_delegation_session(current, pinned)
        assert (result.session_id if result else None) == (None if rejected else expected)
        current = store.lookup_by_session_key(entry.session_key)
        assert current is not None
        assert current.session_id == (route_before if rejected else expected)
        if case not in {"generic_parent", "compression"}:
            for sid, row in before.items():
                after = db.get_session(sid)
                for field in ("ended_at", "end_reason", "source", "session_key", "model_config"):
                    assert after[field] == row[field], (case, sid, field)
