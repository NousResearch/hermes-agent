"""Tests for plugin-triggered turns in existing gateway sessions."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import hermes_yaml as yaml

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    PlatformConfig,
    SendResult,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
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
        _store=runner.session_store, lookup_by_session_key=AsyncMock(return_value=entry)
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
async def test_scheduler_logs_async_failure_without_callback_error():
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


@pytest.mark.asyncio
async def test_keyed_injection_is_durable_and_replay_does_not_start_second_turn(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(return_value=None)

    request = dict(session_key=entry.session_key, content="Where is the patch?",
                   plugin_id="notify-plugin", idempotency_key="evt_123", idle_only=True,
                   enabled_toolsets=["memory"])
    assert runner._schedule_plugin_message_injection(**request) is True
    task = next(iter(runner._background_tasks))
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    adapter.send.assert_awaited_once()
    assert "Delegated request from notify-plugin" in adapter.send.await_args.args[1]
    from gateway.plugin_injection_ledger import state
    receipt = state("notify-plugin", "evt_123")
    assert receipt["state"] == "dispatched"
    assert runner._schedule_plugin_message_injection(**request) is True
    await asyncio.sleep(0)
    adapter.send.assert_awaited_once()
    # A new runner against the same state.db sees the already claimed key and
    # cannot start a second turn after an accepted-but-uncertain process exit.
    replacement = _runner(entry, adapter)
    replacement._gateway_loop = asyncio.get_running_loop()
    replacement._thread_metadata_for_source = MagicMock(return_value=None)
    assert replacement._schedule_plugin_message_injection(**request) is True
    assert replacement._background_tasks == set()
    adapter.send.assert_awaited_once()
    with patch("gateway.run.safe_schedule_threadsafe"):
        assert runner._schedule_plugin_message_injection(
            **{**request, "content": "different request"}) is False


@pytest.mark.asyncio
async def test_keyed_injection_refuses_busy_session_without_queuing(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    adapter._active_sessions[entry.session_key] = asyncio.Event()
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="hello", plugin_id="notify-plugin",
        idempotency_key="evt_busy", idle_only=True) is True
    task = next(iter(runner._background_tasks))
    await asyncio.gather(task, return_exceptions=True)
    from gateway.plugin_injection_ledger import state
    receipt = state("notify-plugin", "evt_busy")
    assert receipt["state"] == "deferred"
    assert receipt["last_error"] == "session busy"
    adapter._message_handler.assert_not_awaited()

    adapter._active_sessions.clear()
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="hello", plugin_id="notify-plugin",
        idempotency_key="evt_busy", idle_only=True) is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    adapter.send.assert_awaited_once()
    assert state("notify-plugin", "evt_busy")["state"] == "dispatched"


@pytest.mark.asyncio
async def test_retry_after_busy_race_does_not_repeat_telegram_notice(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    sends = 0

    async def send(chat_id, content, reply_to=None, metadata=None):
        nonlocal sends
        sends += 1
        if sends == 1:
            adapter._active_sessions[entry.session_key] = asyncio.Event()
        return SendResult(success=True)

    adapter.send = send
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    request = dict(session_key=entry.session_key, content="Question",
                   plugin_id="notify-plugin", idempotency_key="evt_race", idle_only=True)

    assert runner._schedule_plugin_message_injection(**request) is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    from gateway.plugin_injection_ledger import state
    assert state("notify-plugin", "evt_race")["state"] == "notice_deferred"
    assert sends == 1

    adapter._active_sessions.clear()
    assert runner._schedule_plugin_message_injection(**request) is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    assert state("notify-plugin", "evt_race")["state"] == "dispatched"
    assert sends == 1
    assert runner._schedule_plugin_message_injection(**request) is True
    assert sends == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("after_notice", [False, True])
async def test_deferred_key_is_bound_to_original_session_generation(
    tmp_path, monkeypatch, after_notice,
):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock())
    if not after_notice:
        adapter._active_sessions[entry.session_key] = asyncio.Event()
    sends = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sends.append(content)
        if after_notice:
            adapter._active_sessions[entry.session_key] = asyncio.Event()
        return SendResult(success=True)

    adapter.send = send
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    request = dict(session_key=entry.session_key, content="Question",
                   plugin_id="notify-plugin", idempotency_key="evt_generation", idle_only=True)

    assert runner._schedule_plugin_message_injection(**request) is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    from gateway.plugin_injection_ledger import state
    assert state("notify-plugin", "evt_generation")["session_id"] == "session-42"
    assert state("notify-plugin", "evt_generation")["state"] == (
        "notice_deferred" if after_notice else "deferred")

    adapter._active_sessions.clear()
    entry.session_id = "session-after-new"
    assert runner._schedule_plugin_message_injection(**request) is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    assert state("notify-plugin", "evt_generation")["state"] == "refused"
    assert state("notify-plugin", "evt_generation")["last_error"] == "session generation changed"
    assert len(sends) == (1 if after_notice else 0)
    adapter._message_handler.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("pre_turn_state,expected_notice_count", [
    ("scheduled", 1), ("notice_sent", 0),
])
async def test_dead_owner_pre_turn_state_resumes_once(
    tmp_path, monkeypatch, pre_turn_state, expected_notice_count,
):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    request = dict(session_key=entry.session_key, content="Question",
                   plugin_id="notify-plugin", idempotency_key="evt_crash")
    from gateway import plugin_injection_ledger as ledger
    from hermes_cli.sqlite_util import transaction
    assert ledger.claim("notify-plugin", "evt_crash", entry.session_key, "Question") == "new"
    assert ledger.bind_session("notify-plugin", "evt_crash", entry.session_id) is True
    with transaction(ledger._connect()) as conn:
        conn.execute("""
            UPDATE plugin_injections SET state=?, owner_pid=99999999, owner_started_at=1
            WHERE plugin_id='notify-plugin' AND idempotency_key='evt_crash'
        """, (pre_turn_state,))

    assert runner._schedule_plugin_message_injection(**request) is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    assert ledger.state("notify-plugin", "evt_crash")["state"] == "dispatched"
    assert adapter.send.await_count == expected_notice_count
    assert runner._schedule_plugin_message_injection(**request) is True
    assert adapter.send.await_count == expected_notice_count


def test_injection_status_closes_its_database_connection(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from gateway import plugin_injection_ledger as ledger
    ledger.claim("notify-plugin", "evt_close", "session-key", "Question")
    original_connect = ledger._connect
    closed = []

    class TrackedConnection:
        def __init__(self):
            self.connection = original_connect()

        def execute(self, *args):
            return self.connection.execute(*args)

        def close(self):
            closed.append(True)
            self.connection.close()

    monkeypatch.setattr(ledger, "_connect", TrackedConnection)
    assert ledger.state("notify-plugin", "evt_close")["state"] == "scheduled"
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("final_succeeds", [True, False])
async def test_keyed_injection_reports_actual_final_delivery(tmp_path, monkeypatch, final_succeeds):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()

    async def answer(event):
        event._heartbeat_execution_started = True
        return "Daphne's answer"

    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        if len(sent) == 1:
            return SendResult(success=True)
        return SendResult(success=final_succeeds, error=None if final_succeeds else "send_path_degraded")

    adapter.set_message_handler(answer)
    adapter.send = send
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(return_value=None)

    assert runner._schedule_plugin_message_injection(
        session_key=entry.session_key, content="Question", plugin_id="notify-plugin",
        idempotency_key="evt_answer") is True
    while runner._background_tasks:
        await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
    while adapter._background_tasks:
        await asyncio.gather(*list(adapter._background_tasks), return_exceptions=True)
    from gateway.plugin_injection_ledger import state
    receipt = state("notify-plugin", "evt_answer")
    assert sent[:2] == ["Delegated request from notify-plugin:\n\nQuestion", "Daphne's answer"]
    assert receipt["state"] == "turn_complete"
    assert receipt["response"] == "Daphne's answer"
    assert receipt["delivery_state"] == ("delivered" if final_succeeds else "failed")


@pytest.mark.asyncio
async def test_missing_injected_toolset_fails_before_agent_creation():
    runner = _runner(_entry())
    runner._get_proxy_url = MagicMock(return_value=None)
    runner._run_agent_display_settings = MagicMock(
        return_value=SimpleNamespace(enabled_toolsets=["messaging", "mempalace"]))
    runner._run_agent_build_turn_context = MagicMock()

    with pytest.raises(RuntimeError, match="not enabled"):
        await runner._run_agent_inner(
            message="delegated question", context_prompt="", history=[],
            source=_entry().origin, session_id="session-42",
            injected_toolsets=["mempalace-coordination"])
    runner._run_agent_build_turn_context.assert_not_called()


@pytest.mark.asyncio
async def test_plugin_context_exposes_keyed_gateway_receipt(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({
        "plugins": {"entries": {"notify-plugin": {"allow_gateway_injection": True}}}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    entry = _entry()
    adapter = _RoutingAdapter()
    adapter.set_message_handler(AsyncMock(return_value=None))
    adapter.send = AsyncMock(return_value=SendResult(success=True))
    runner = _runner(entry, adapter)
    runner._gateway_loop = asyncio.get_running_loop()
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"), manager)

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._install_plugin_message_injector()
        assert context.inject_message(
            "Question", session_key=entry.session_key, idempotency_key="evt_api",
            idle_only=True, enabled_toolsets=["memory"]) is True
        while runner._background_tasks:
            await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
        assert context.injection_status("evt_api")["state"] == "dispatched"
        assert context.injection_status("missing") is None
        runner._clear_plugin_message_injector()
