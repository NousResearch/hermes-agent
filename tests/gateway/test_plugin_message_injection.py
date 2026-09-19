"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
import concurrent.futures
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    PlatformConfig,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.internal_events import create_gateway_system_event, gateway_system_event_message, new_gateway_event_receipt
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _entry(*, origin=True) -> SessionEntry:
    source = None
    if origin:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
            user_name="tester",
        )
    now = datetime.now()
    return SessionEntry(
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        created_at=now,
        updated_at=now,
        origin=source,
        platform=Platform.TELEGRAM,
    )


def _runner(entry: SessionEntry | None, adapter=None) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry),
        typed_event_recovery_owner_state=AsyncMock(return_value="none"),
        mark_typed_event_recovery_owner=AsyncMock(return_value=True),
    )
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter else {}
    runner._profile_adapters = {}
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._is_user_authorized = MagicMock(return_value=True)
    return runner


class _RoutingAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise AssertionError("network send is not expected")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


@pytest.mark.asyncio
async def test_plugin_context_routes_through_live_gateway_to_existing_session(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({
            "plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}
        })
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    source = _entry().origin
    entry = store.get_or_create_session(source)
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    pending_user_event = MessageEvent(
        text="human follow-up",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["human.jpg"],
        media_types=["image/jpeg"],
    )
    adapter._pending_messages[entry.session_key] = pending_user_event

    runner = object.__new__(GatewayRunner)
    runner.session_store = store
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._profile_adapters = {}
    runner._gateway_loop = asyncio.get_running_loop()
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    runner._queued_events = {}
    runner._is_user_authorized = MagicMock(return_value=True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert (
            context.inject_message(
                "/approve always",
                session_key=entry.session_key,
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        assert adapter._pending_messages[entry.session_key] is pending_user_event
        queued = runner._queued_events[entry.session_key][0]
        assert pending_user_event.text == "human follow-up"
        assert pending_user_event.media_urls == ["human.jpg"]
        assert pending_user_event.allow_gateway_control is True
        assert queued.text == "/approve always"
        assert queued.allow_gateway_control is False
        assert queued.metadata["gateway_session_id"] == entry.session_id
        adapter._message_handler.assert_not_awaited()

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False


@pytest.mark.asyncio
async def test_dispatch_uses_stored_origin_and_adapter_message_path():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="check the deployment",
        plugin_id="notify-plugin",
    )

    assert accepted is True
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "check the deployment"
    assert event.internal is True
    assert event.allow_gateway_control is False
    assert event.get_command() is None
    assert event.source == entry.origin
    assert event.source is not entry.origin
    runner._is_user_authorized.assert_called_once_with(
        event.source,
        allow_adapter_delegation=False,
    )
    assert event.metadata == {
        "hermes_plugin_id": "notify-plugin",
        "hermes_plugin_injection": True,
        "gateway_session_key": entry.session_key,
        "gateway_session_id": entry.session_id,
        "gateway_session_strict": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry", "with_adapter"),
    [
        (None, True),
        (_entry(origin=False), True),
        (_entry(), False),
    ],
)
async def test_dispatch_rejects_unroutable_session(entry, with_adapter):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter if with_adapter else None)

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_dispatch_rechecks_current_authorization(raises):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    if raises:
        runner._is_user_authorized.side_effect = RuntimeError("config unavailable")
    else:
        runner._is_user_authorized.return_value = False

    accepted = await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_rejects_stored_role_only_authorization(monkeypatch):
    """A stored adapter role grant must be revalidated against current core auth."""
    for key in (
        "DISCORD_ALLOWED_USERS",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)

    adapter = MagicMock(spec=BasePlatformAdapter)
    adapter.handle_message = AsyncMock()
    entry = _entry()
    entry.session_key = "agent:main:discord:dm:42"
    entry.platform = Platform.DISCORD
    source = entry.origin
    assert source is not None
    source.platform = Platform.DISCORD
    source.role_authorized = True

    runner = _runner(entry)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = GatewayConfig()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    del runner._is_user_authorized

    accepted = await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key,
        content="wake up",
        plugin_id="notify-plugin",
    )

    assert accepted is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_stops_when_gateway_drains_during_lookup():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)
    lookup_started = asyncio.Event()
    release_lookup = asyncio.Event()

    async def _lookup(_session_key):
        lookup_started.set()
        await release_lookup.wait()
        return _entry()

    runner._async_session_store.lookup_by_session_key = _lookup
    dispatch = asyncio.create_task(
        runner._dispatch_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
    )

    await lookup_started.wait()
    runner._draining = True
    release_lookup.set()

    assert await dispatch is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_base_adapter_queues_non_control_plugin_text_for_exact_session():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    source = _entry().origin
    session_key = build_session_key(source)
    adapter._active_sessions[session_key] = asyncio.Event()
    event = MessageEvent(
        text="/approve always",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": session_key},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._pending_messages[session_key] is event
    assert adapter._active_sessions[session_key].is_set() is False


@pytest.mark.asyncio
async def test_base_adapter_rejects_derived_session_mismatch():
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    event = MessageEvent(
        text="ordinary input",
        source=_entry().origin,
        internal=True,
        allow_gateway_control=False,
        metadata={"gateway_session_key": "agent:main:telegram:dm:other"},
    )

    await adapter.handle_message(event)

    adapter._message_handler.assert_not_awaited()
    assert adapter._active_sessions == {}


@pytest.mark.asyncio
async def test_scheduler_submits_dispatch_on_live_gateway_loop():
    runner = _runner(_entry())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)

    assert (
        runner._schedule_plugin_message_injection(
            session_key="agent:main:telegram:dm:42",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is True
    )

    await asyncio.sleep(0)
    runner._dispatch_plugin_message_injection.assert_awaited_once_with(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
    )


@pytest.mark.asyncio
async def test_scheduler_ignores_same_loop_task_cancellation():
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))

    blocker = asyncio.Event()

    async def _wait_for_cancellation(**_kwargs):
        await blocker.wait()

    runner._dispatch_plugin_message_injection = _wait_for_cancellation

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

        task = next(iter(runner._background_tasks))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []


@pytest.mark.asyncio
async def test_scheduler_logs_async_failure_without_callback_error(caplog):
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    callback_errors = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: callback_errors.append(context))
    runner._dispatch_plugin_message_injection = AsyncMock(
        side_effect=RuntimeError("adapter failed")
    )

    try:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )
        task = next(iter(runner._background_tasks))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert callback_errors == []
    assert "plugin=notify-plugin session=key" in caplog.text


def test_scheduler_uses_threadsafe_bridge_outside_gateway_loop():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, target_loop, **_kwargs):
        assert target_loop is loop
        coro.close()
        future = concurrent.futures.Future()
        future.set_result(True)
        return future

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit) as submit:
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

    submit.assert_called_once()


def test_scheduler_ignores_threadsafe_future_cancellation():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _submit(coro, _target_loop, **_kwargs):
        coro.close()
        future = concurrent.futures.Future()
        future.cancel()
        return future

    with (
        patch("gateway.run.safe_schedule_threadsafe", side_effect=_submit),
        patch("gateway.run.logger.warning") as warning,
    ):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is True
        )

    warning.assert_not_called()


def test_scheduler_rejects_stopped_or_closed_gateway():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop
    runner._running = False

    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()

    runner._running = True
    runner._gateway_loop = None
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )

    runner._gateway_loop = loop
    loop.is_closed.return_value = True
    assert (
        runner._schedule_plugin_message_injection(
            session_key="key",
            content="wake up",
            plugin_id="notify-plugin",
        )
        is False
    )
    loop.call_soon_threadsafe.assert_not_called()


def test_scheduler_rejects_submission_failure():
    runner = _runner(_entry())
    loop = MagicMock()
    loop.is_closed.return_value = False
    runner._gateway_loop = loop

    def _reject(coro, _target_loop, **_kwargs):
        coro.close()
        return None

    with patch("gateway.run.safe_schedule_threadsafe", side_effect=_reject):
        assert (
            runner._schedule_plugin_message_injection(
                session_key="key",
                content="wake up",
                plugin_id="notify-plugin",
            )
            is False
        )


def test_install_and_clear_gateway_injector_preserves_newer_owner():
    runner = _runner(_entry())
    manager = PluginManager()

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert manager.has_gateway_message_injector is True

        runner._clear_plugin_message_injector()
        assert manager.has_gateway_message_injector is False

        runner._install_plugin_message_injector()

        newer_owner = MagicMock()
        newer_injector = MagicMock(return_value=True)
        manager.set_gateway_message_injector(newer_owner, newer_injector)
        runner._clear_plugin_message_injector()

    assert manager.has_gateway_message_injector is True
    assert manager.inject_gateway_message(value="kept") is True
    newer_injector.assert_called_once_with(value="kept")


def _typed_event(
    entry: SessionEntry,
    *,
    event_id: str = "event-42",
    route_overrides: dict | None = None,
    eligibility_check=None,
):
    source = entry.origin
    assert source is not None
    expected_route = {
        "profile_name": source.profile or "default",
        "platform": source.platform.value,
        "user_id": str(source.user_id or ""),
        "chat_id": str(source.chat_id),
        "topic_id": str(source.thread_id or ""),
    }
    expected_route.update(route_overrides or {})
    content, marker = create_gateway_system_event(
        content="[T3 continuation] work completed",
        session_key=entry.session_key,
        expected_session_id=entry.session_id,
        event_id=event_id,
        event_kind="external_tool_completed",
        plugin_id="notify-plugin",
        expected_route=expected_route,
        eligibility_check=eligibility_check or (lambda: True),
    )
    return content, marker, new_gateway_event_receipt()



@pytest.mark.asyncio
async def test_system_event_dispatch_is_strict_and_uses_native_adapter_path():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(entry)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )

    adapter.handle_message.assert_awaited_once()
    routed = adapter.handle_message.await_args.args[0]
    assert routed.text == content
    assert routed.message_id is None
    assert routed.gateway_system_event is marker
    assert routed.gateway_event_receipt is receipt
    assert routed.metadata["gateway_session_strict"] is True
    assert not receipt.done()
    runner._is_user_authorized.assert_called_once_with(
        routed.source, allow_adapter_delegation=False
    )



@pytest.mark.asyncio
async def test_system_event_rechecks_eligibility_at_actual_turn_admission():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    eligibility_check = MagicMock(return_value=True)
    content, marker, receipt = _typed_event(
        entry, eligibility_check=eligibility_check
    )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )
        routed = adapter.handle_message.await_args.args[0]
        admitted_source = await runner._admit_gateway_system_event_turn(
            routed, entry
        )

    assert admitted_source == entry.origin
    assert eligibility_check.call_count == 2
    assert receipt.running()



@pytest.mark.asyncio
async def test_typed_event_requires_lease_and_durable_owner_before_history():
    entry = _entry()
    runner = _runner(entry, SimpleNamespace(handle_message=AsyncMock()))
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(
        text=content,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        assert not await runner._mark_durable_active_turn(
            event,
            entry.session_key,
            entry.session_id,
            turn_lease_acquired=False,
        )

    assert receipt.result(timeout=2)["status"] == "agent_error"
    runner._async_session_store.mark_typed_event_recovery_owner.assert_not_awaited()



@pytest.mark.asyncio
async def test_sequential_distinct_typed_events_replace_owner_without_user_turn():
    entry = _entry()
    runner = _runner(entry, SimpleNamespace(handle_message=AsyncMock()))
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True

    events = []
    for event_id in ("event-first", "event-second"):
        content, marker, receipt = _typed_event(entry, event_id=event_id)
        events.append(
            MessageEvent(
                text=content,
                source=entry.origin,
                internal=True,
                allow_gateway_control=False,
                gateway_system_event=marker,
                gateway_event_receipt=receipt,
            )
        )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        for event in events:
            assert await runner._mark_durable_active_turn(
                event,
                entry.session_key,
                entry.session_id,
                turn_lease_acquired=True,
            )

    calls = runner._async_session_store.mark_typed_event_recovery_owner.await_args_list
    assert [call.args[2] for call in calls] == [
        events[0].gateway_system_event.event_id,
        events[1].gateway_system_event.event_id,
    ]



@pytest.mark.asyncio
async def test_typed_owner_persistence_failure_aborts_before_admission():
    entry = _entry()
    runner = _runner(entry, SimpleNamespace(handle_message=AsyncMock()))
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    runner._async_session_store.mark_typed_event_recovery_owner.return_value = False
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(
        text=content,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        assert not await runner._mark_durable_active_turn(
            event,
            entry.session_key,
            entry.session_id,
            turn_lease_acquired=True,
        )

    assert receipt.result(timeout=2)["status"] == "busy"



@pytest.mark.asyncio
async def test_system_event_rechecks_full_route_at_actual_turn_admission():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(entry)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )
        routed = adapter.handle_message.await_args.args[0]
        assert entry.origin is not None
        entry.origin.user_id = "different-user"
        admitted_source = await runner._admit_gateway_system_event_turn(
            routed, entry
        )

    assert admitted_source is None
    assert receipt.result(timeout=2)["status"] == "route_mismatch"



@pytest.mark.asyncio
async def test_system_event_accepts_native_default_profile_topic_route(tmp_path):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="1000000001",
        chat_type="dm",
        user_id="1000000001",
        thread_id="42",
    )
    entry = SessionEntry(
        session_key=build_session_key(source),
        session_id="session-fixture-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        origin=source,
        platform=Platform.TELEGRAM,
    )
    assert entry.session_key == "agent:main:telegram:dm:1000000001:42"
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    native_store = SessionStore(
        sessions_dir=tmp_path / "sessions", config=runner.config
    )
    runner.session_store = native_store
    runner._async_session_store._store = native_store
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(entry)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )

    adapter.handle_message.assert_awaited_once()
    assert marker.expected_route.profile_name == "default"
    assert marker.expected_route.topic_id == "42"
    assert not receipt.done()



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("profile_name", "other-profile"),
        ("user_id", "different-user"),
        ("topic_id", "different-topic"),
    ],
)
async def test_system_event_rejects_wrong_route_tuple_with_valid_key_and_session(
    field, wrong_value
):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(
        entry, route_overrides={field: wrong_value}
    )

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )

    assert receipt.result(timeout=2)["status"] == "route_mismatch"
    adapter.handle_message.assert_not_awaited()



def test_system_event_message_metadata_excludes_route_and_authority():
    entry = _entry()
    content, marker, _receipt = _typed_event(entry)

    message = gateway_system_event_message(marker, content)

    assert message["display_metadata"] == {
        "schema_version": 1,
        "event_kind": "external_tool_completed",
        "event_id": "event-42",
        "plugin_id": "notify-plugin",
    }
    assert "eligibility_check" not in repr(marker)



@pytest.mark.asyncio
async def test_system_event_fails_closed_when_physical_session_changes():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(entry)
    changed = _entry()
    changed.session_id = "replacement-session"
    runner._async_session_store.lookup_by_session_key.return_value = changed

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )

    assert receipt.result(timeout=2)["status"] == "session_mismatch"
    adapter.handle_message.assert_not_awaited()



@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry_field", "expected_status"),
    [("suspended", "unauthorized"), ("resume_pending", "busy")],
)
async def test_system_event_rejects_session_recovery_conflict(
    entry_field, expected_status
):
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    setattr(entry, entry_field, True)
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(entry)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )

    assert receipt.result(timeout=2)["status"] == expected_status
    adapter.handle_message.assert_not_awaited()



@pytest.mark.asyncio
async def test_system_event_rejects_malformed_recovery_owner():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._async_session_store.typed_event_recovery_owner_state.return_value = (
        "conflict"
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True
    content, marker, receipt = _typed_event(entry)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        await runner._dispatch_plugin_system_event(
            content=content, system_event=marker, receipt=receipt
        )

    assert receipt.result(timeout=2)["status"] == "busy"
    adapter.handle_message.assert_not_awaited()



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["exception", "awaitable"])
async def test_turn_admission_fails_closed_for_invalid_eligibility_result(mode):
    entry = _entry()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    runner._run_agent = AsyncMock()

    def raises():
        raise RuntimeError("binding unavailable")

    async def returns_awaitable():
        return True

    check = raises if mode == "exception" else returns_awaitable
    content, marker, receipt = _typed_event(entry, eligibility_check=check)
    event = MessageEvent(
        text=content,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        admitted_source = await runner._admit_gateway_system_event_turn(
            event, entry
        )

    assert admitted_source is None
    assert receipt.result(timeout=2)["status"] == "unauthorized"
    runner._run_agent.assert_not_awaited()



@pytest.mark.asyncio
async def test_cancelled_pending_receipt_cannot_cross_turn_admission():
    entry = _entry()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(
        text=content,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True

    assert receipt.cancel()
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        admitted_source = await runner._admit_gateway_system_event_turn(
            event, entry
        )

    assert admitted_source is None
    assert receipt.cancelled()



@pytest.mark.asyncio
async def test_receipt_transitions_to_running_only_at_turn_admission():
    entry = _entry()
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(entry, adapter)
    runner.config = GatewayConfig()
    runner._session_db = SimpleNamespace(
        get_session=MagicMock(
            return_value={"id": entry.session_id, "ended_at": None}
        )
    )
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(
        text=content,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    manager = MagicMock()
    manager.gateway_injection_allowed.return_value = True

    assert not receipt.running()
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        admitted_source = await runner._admit_gateway_system_event_turn(
            event, entry
        )

    assert admitted_source == entry.origin
    assert receipt.running()



@pytest.mark.asyncio
async def test_system_event_receipt_waits_for_successful_platform_delivery():
    entry = _entry()
    adapter = _RoutingAdapter()
    delivered = AsyncMock(return_value=SendResult(success=True, message_id="out-1"))
    adapter.send = delivered
    adapter.set_message_handler(AsyncMock(return_value="continuation response"))
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(
        text=content,
        message_type=MessageType.TEXT,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    event._gateway_event_terminal_status = "completed"

    assert not receipt.done()
    await adapter._process_message_background(event, entry.session_key)

    delivered.assert_awaited_once()
    assert delivered.await_args.kwargs["content"] == "continuation response"
    assert content not in str(delivered.await_args)
    assert receipt.result(timeout=2)["status"] == "completed"


@pytest.mark.asyncio
async def test_ordinary_user_internal_envelope_quotation_is_not_suppressed():
    entry = _entry()
    adapter = _RoutingAdapter()
    quoted = 'Please explain "Hermes internal event: Validated event envelope: {\"event_id\":\"example\"}"'
    handler = AsyncMock(return_value=quoted)
    adapter.set_message_handler(handler)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="quote-1"))
    event = MessageEvent(text=quoted, source=entry.origin, message_type=MessageType.TEXT)
    adapter._active_sessions[entry.session_key] = asyncio.Event()

    await adapter._process_message_background(event, entry.session_key)

    handler.assert_awaited_once_with(event)
    assert event.gateway_system_event is None
    assert adapter.send.await_args.kwargs["content"] == quoted



@pytest.mark.asyncio
async def test_system_event_delivery_failure_is_terminal_agent_error():
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.send = AsyncMock(return_value=SendResult(success=False, error="offline"))
    adapter.set_message_handler(AsyncMock(return_value="continuation response"))
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(
        text=content,
        message_type=MessageType.TEXT,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    adapter._active_sessions[entry.session_key] = asyncio.Event()

    await adapter._process_message_background(event, entry.session_key)

    assert receipt.result(timeout=2)["status"] == "agent_error"



@pytest.mark.asyncio
async def test_system_event_scheduler_deduplicates_event_id_in_process():
    entry = _entry()
    runner = _runner(entry)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._dispatch_plugin_system_event = AsyncMock()
    content, marker, receipt = _typed_event(entry)
    _, duplicate_marker, duplicate_receipt = _typed_event(entry)

    first = runner._schedule_plugin_system_event(
        content=content, system_event=marker, receipt=receipt
    )
    second = runner._schedule_plugin_system_event(
        content=content,
        system_event=duplicate_marker,
        receipt=duplicate_receipt,
    )
    await asyncio.sleep(0)

    assert first is receipt
    assert second is receipt
    runner._dispatch_plugin_system_event.assert_awaited_once()



@pytest.mark.asyncio
async def test_system_event_is_never_written_to_legacy_shutdown_spool():
    entry = _entry()
    adapter = _RoutingAdapter()
    content, marker, receipt = _typed_event(entry)
    internal = MessageEvent(
        text=content,
        message_type=MessageType.TEXT,
        source=entry.origin,
        internal=True,
        allow_gateway_control=False,
        gateway_system_event=marker,
        gateway_event_receipt=receipt,
    )
    ordinary = MessageEvent(
        text="ordinary queued user text",
        message_type=MessageType.TEXT,
        source=entry.origin,
    )
    adapter._pending_messages = {
        entry.session_key: internal,
        "agent:main:telegram:dm:other": ordinary,
    }

    with patch("gateway.shutdown_flush.flush_pending_to_file") as flush:
        await adapter.cancel_background_tasks()

    assert receipt.result(timeout=2)["status"] == "stopping"
    flushed = flush.call_args.args[0]
    assert list(flushed.values()) == [ordinary]



@pytest.mark.asyncio
async def test_typed_busy_receipt_can_retry_without_entering_human_queue():
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(text=content, source=entry.origin, internal=True,
                         allow_gateway_control=False, gateway_system_event=marker,
                         gateway_event_receipt=receipt)
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    adapter._heal_stale_session_lock = lambda key: None
    human = MessageEvent(text="human stays intact", source=entry.origin)
    adapter._pending_messages[entry.session_key] = human
    await adapter.handle_message(event)
    assert receipt.result(timeout=2)["status"] == "busy"
    assert adapter._pending_messages[entry.session_key] is human
    adapter._message_handler.assert_not_awaited()

    runner = _runner(entry)
    runner._gateway_loop = asyncio.get_running_loop()
    async def refuse_then_accept(**kwargs):
        from gateway.internal_events import resolve_gateway_event_receipt
        resolve_gateway_event_receipt(kwargs["receipt"], "busy" if refuse_then_accept.calls == 0 else "completed",
                                      event=kwargs["system_event"])
        refuse_then_accept.calls += 1
    refuse_then_accept.calls = 0
    runner._dispatch_plugin_system_event = refuse_then_accept
    first = runner._schedule_plugin_system_event(content=content, system_event=marker, receipt=new_gateway_event_receipt())
    await asyncio.sleep(0)
    assert first.result(timeout=2)["status"] == "busy"
    second = runner._schedule_plugin_system_event(content=content, system_event=marker, receipt=new_gateway_event_receipt())
    await asyncio.sleep(0)
    assert second.result(timeout=2)["status"] == "completed"
    assert refuse_then_accept.calls == 2


@pytest.mark.asyncio
async def test_typed_physical_lease_is_nonblocking_and_does_not_steal_human_owner():
    from gateway.turn_lease import SessionTurnLeaseRegistry
    leases = SessionTurnLeaseRegistry()
    human = await leases.acquire("physical", owner_key="human", generation=1)
    assert await leases.try_acquire("physical", owner_key="event", generation=2) is None
    assert not human.released
    leases.release(human)
    event = await leases.try_acquire("physical", owner_key="event", generation=2)
    assert event is not None
    leases.release(event)


@pytest.mark.asyncio
async def test_typed_events_refuse_external_drain_at_schedule_and_admission():
    entry = _entry()
    runner = _runner(entry)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._external_drain_active = True
    content, marker, receipt = _typed_event(entry)
    runner._schedule_plugin_system_event(content=content, system_event=marker, receipt=receipt)
    assert receipt.result(timeout=2)["status"] == "stopping"
    assert await runner._gateway_system_event_target(marker) == ("stopping", None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [({"completed": True}, "completed"), ({}, "agent_error"), ({"completed": False}, "agent_error"), ({"completed": True, "failed": True}, "agent_error"), ({"interrupted": True}, "cancelled"), ({"completed": False, "failed": True, "compression_exhausted": True}, "agent_error")])
async def test_typed_terminal_status_requires_real_model_success(result, expected):
    entry = _entry()
    runner = _runner(entry)
    content, marker, receipt = _typed_event(entry)
    event = MessageEvent(text=content, source=entry.origin, internal=True,
                         gateway_system_event=marker, gateway_event_receipt=receipt)
    prepared = runner._PreparedTurn([], "system", content, None, None, "internal_notification")
    runner._hmwa_resolve_session = AsyncMock(return_value=(entry.origin, entry, entry.session_key))
    runner._hmwa_prepare_turn = AsyncMock(return_value=(prepared, None))
    runner._admit_gateway_system_event_turn = AsyncMock(return_value=entry.origin)
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._run_agent = AsyncMock(return_value=result)
    runner._reply_anchor_for_event = lambda event: None
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._is_session_run_current = lambda *args: True
    runner._hmwa_shape_agent_response = AsyncMock(return_value=("report", False, []))
    runner._hmwa_prepend_reasoning = lambda result, response, *args: response
    runner._hmwa_runtime_footer_line = lambda *args: None
    runner._hmwa_post_turn_hooks = AsyncMock()
    runner._hmwa_classify_turn_failure = lambda *args: (False, False, False)
    runner._hmwa_compression_exhaustion_reset = AsyncMock(return_value=("report", entry))
    if result.get("compression_exhausted"):
        runner._hmwa_compression_exhaustion_reset = GatewayRunner._hmwa_compression_exhaustion_reset.__get__(runner)
        runner._async_session_store.reset_session = AsyncMock(return_value=entry)
        runner._evict_cached_agent = MagicMock()
        runner._clear_conversation_scope = MagicMock()
        runner._sync_telegram_topic_binding = MagicMock()
    runner._hmwa_persist_turn_transcript = AsyncMock()
    runner._hmwa_deliver_turn_response = AsyncMock(return_value="report")
    runner._hmwa_agent_error_reply = AsyncMock(return_value="error")
    runner._clear_session_env = lambda tokens: None
    assert await runner._handle_message_with_agent(event, entry.origin, entry.session_key, 1) == "report"
    assert event._gateway_event_terminal_status == expected
    assert runner._run_agent.await_args.kwargs["gateway_system_event"] is marker
    assert not receipt.done()  # native adapter owns final delivery disposition
    if result.get("compression_exhausted"):
        runner._async_session_store.reset_session.assert_not_awaited()
        runner._evict_cached_agent.assert_not_called()
        runner._sync_telegram_topic_binding.assert_not_called()
        adapter = _RoutingAdapter()
        adapter.set_message_handler(AsyncMock(return_value="bounded failure"))
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        await adapter._process_message_background(event, entry.session_key)
        assert receipt.result(timeout=2)["status"] == "agent_error"
    if expected == "completed":
        runner._hmwa_post_turn_hooks.side_effect = RuntimeError("post-turn failure")
        assert await runner._handle_message_with_agent(event, entry.origin, entry.session_key, 1) == "error"
        assert event._gateway_event_terminal_status == "agent_error"

@pytest.mark.asyncio
async def test_native_gateway_readiness_uses_host_turn_context_and_live_target(tmp_path, monkeypatch):
    from gateway.session_context import plugin_gateway_turn
    from tools.thread_context import propagate_context_to_thread
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_ID", "foreign-env-session")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}}))
    entry = _entry()
    runner = _runner(entry, _RoutingAdapter())
    runner._gateway_loop = asyncio.get_running_loop()
    runner._session_key_for_source = lambda source: entry.session_key
    runner._session_db = SimpleNamespace(get_session=lambda sid: {"id": sid, "ended_at": None})
    manager = PluginManager(scope_key=str(tmp_path))
    ctx = PluginContext(PluginManifest(name="notify-plugin", key="notify-plugin", source="user"), manager)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    manager.set_gateway_message_injector(runner, runner._schedule_plugin_system_event)
    ctx._gateway_observer_ready = True
    manager._gateway_tasks = {("notify-plugin", "observer"): SimpleNamespace(done=lambda: False)}
    turn = SimpleNamespace(source=entry.origin, session_key=entry.session_key, session_id=entry.session_id)
    agent = SimpleNamespace(session_id=entry.session_id, api_mode="codex_responses", provider="openai-codex")
    assert ctx.current_gateway_destination() is None
    def probe():
        target = ctx.current_gateway_destination()
        return target, ctx.gateway_continuation_readiness(target)
    with plugin_gateway_turn(runner, turn, agent):
        import contextvars
        copied = contextvars.copy_context()
        target, ticket = await asyncio.to_thread(propagate_context_to_thread(probe))
        assert target["session_id"] == entry.session_id
        assert ticket["ready"] is True
        entry.session_id = "replaced-physical-session"
        _, ticket = await asyncio.to_thread(propagate_context_to_thread(probe))
        assert ticket["ready"] is False
        entry.session_id = turn.session_id
        runner._is_user_authorized.return_value = False
        _, ticket = await asyncio.to_thread(propagate_context_to_thread(probe))
        assert ticket["ready"] is False
    assert ctx.current_gateway_destination() is None
    assert copied.run(ctx.current_gateway_destination) is None
    turn.gateway_system_event = object()
    with plugin_gateway_turn(runner, turn, agent):
        assert ctx.current_gateway_destination() is None
    with plugin_gateway_turn(object(), turn, agent):
        assert ctx.current_gateway_destination() is None
