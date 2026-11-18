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
    # The scheduler's synchronous profile check reads the store; dispatch reads the async facade.
    runner.session_store = SimpleNamespace(lookup_by_session_key=lambda _key: None)
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
    runner._is_user_authorized = MagicMock(return_value=True)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="notify-plugin", key="notify-plugin", source="user"),
        manager,
    )

    # The runner publishes process-wide; expose this standalone manager through the legacy slot so
    # the real publisher stamps it while this integration test still exercises the scheduler.
    with patch("hermes_cli.plugins._plugin_manager", manager):
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
        queued = runner._session_state(entry.session_key).conversation.queued_events[0]
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
async def test_scheduler_forwards_expected_session_id_to_the_keyed_dispatch():
    runner = _runner(_entry())
    loop = asyncio.get_running_loop()
    runner._gateway_loop = loop
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)

    assert runner._schedule_plugin_message_injection(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
        expected_session_id="session-42",
    ) is True
    await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)

    runner._dispatch_plugin_message_injection.assert_awaited_once_with(
        session_key="agent:main:telegram:dm:42",
        content="wake up",
        plugin_id="notify-plugin",
        plugin_home=None,
        expected_session_id="session-42",
    )


@pytest.mark.parametrize("kwargs", [
    {"expected_session_id": "session-42"},
    {"origin": {"platform": "telegram", "chat_id": "42"}, "expected_session_id": "session-42"},
    {"keyless": True, "expected_session_id": "session-42"},
    {"session_key": "agent:main:telegram:dm:42", "expected_session_id": ""},
    {"session_key": "agent:main:telegram:dm:42", "expected_session_id": 42},
])
def test_scheduler_rejects_expected_session_id_without_a_valid_keyed_pin(kwargs):
    runner = _runner(_entry())
    runner._gateway_loop = MagicMock(is_closed=MagicMock(return_value=False))
    runner._dispatch_plugin_message_injection = AsyncMock(return_value=True)

    assert runner._schedule_plugin_message_injection(
        content="wake up", plugin_id="notify-plugin", **kwargs) is False
    runner._dispatch_plugin_message_injection.assert_not_awaited()


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

    # The runner publishes process-wide; expose this standalone manager through the legacy slot.
    with patch("hermes_cli.plugins._plugin_manager", manager):
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


async def _real_owner(tmp_path, monkeypatch, human_message_id="human-1"):
    """A real SessionAuthority/Discord session with one committed human turn, ready for dispatch."""
    from tests.gateway.test_automation_notification_category import _native_owner

    runner, authority, adapter, source, metadata = await _native_owner(
        tmp_path, monkeypatch, human_message_id)
    runner._running = True
    runner._draining = False
    runner._background_tasks = set()
    return runner, authority, adapter, source, metadata


async def _admit_human_after_reset(runner, authority, source):
    from tests.gateway.test_automation_compression_media import _admit_human
    await _admit_human(runner, authority, MessageEvent(
        text="after reset", source=source, message_id="after-reset"))


@pytest.mark.asyncio
async def test_expected_session_id_holds_the_current_session_through_dispatch(tmp_path, monkeypatch):
    """The optional pin is forwarded through the host lookup and commits to the current conversation."""
    runner, authority, _adapter, _source, metadata = await _real_owner(tmp_path, monkeypatch)
    route, session_id = metadata["gateway_session_key"], metadata["gateway_session_id"]
    try:
        assert await runner._dispatch_plugin_message_injection(
            session_key=route, content="expected-session success", plugin_id="probe",
            expected_session_id=session_id) is True
        rows = authority.persistence.admissions.list(
            session_id=authority.logical_owner(session_id), pending_only=False)
        assert any((r.payload.get("native_text_v1") or {}).get("automation") for r in rows)
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_plugin_context_expected_session_id_reaches_the_durable_fifo(tmp_path, monkeypatch):
    """The public plugin call carries the pin through the scheduler into the same admission."""
    from pathlib import Path

    runner, authority, _adapter, _source, metadata = await _real_owner(tmp_path, monkeypatch)
    runner._gateway_loop = asyncio.get_running_loop()
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="probe", key="probe", source="user"), manager)
    (Path(manager.home_path) / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"probe": {"allow_gateway_injection": True}}}}),
        encoding="utf-8",
    )
    try:
        with patch("hermes_cli.plugins._plugin_manager", manager):
            runner._install_plugin_message_injector()
            try:
                assert context.inject_message(
                    "through the public API",
                    session_key=metadata["gateway_session_key"],
                    expected_session_id=metadata["gateway_session_id"],
                ) is True
                await asyncio.gather(*list(runner._background_tasks), return_exceptions=True)
            finally:
                runner._clear_plugin_message_injector()
        rows = authority.persistence.admissions.list(
            session_id=authority.logical_owner(metadata["gateway_session_id"]), pending_only=False)
        assert any((r.payload.get("native_text_v1") or {}).get("automation") for r in rows)
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_expected_session_id_allows_a_compression_continuation(tmp_path, monkeypatch):
    """The pin is a conversation pin, not an exact transcript id: compression keeps the same
    conversation and must keep accepting the pre-compression session id."""
    from persistence.conversation.lineage import CompressionChild
    from persistence.conversation.messages import NewMessage
    from tests.fixtures.storage_seed import seed_messages

    runner, authority, _adapter, _source, metadata = await _real_owner(tmp_path, monkeypatch)
    route, root = metadata["gateway_session_key"], metadata["gateway_session_id"]
    tip = root + "-compressed"
    try:
        seed_messages(authority.persistence, root, {"role": "user", "content": "human"})
        authority.persistence.lineage.publish_compression_child(
            root, CompressionChild(session_id=tip, source="discord"),
            [NewMessage.from_live({"role": "user", "content": "summary"})],
            require_compression_lease=False)
        assert runner.session_store.advance_compression_session(route, root, tip) is not None
        assert await runner._dispatch_plugin_message_injection(
            session_key=route, content="compression stays", plugin_id="probe",
            expected_session_id=root) is True
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_expected_session_id_refuses_a_reset_before_the_host_lookup(tmp_path, monkeypatch):
    """The caller's pin survives until dispatch: a /new between the call and the lookup must not
    redirect the plugin turn into the fresh conversation, even after that conversation has a human turn."""
    runner, authority, _adapter, source, metadata = await _real_owner(tmp_path, monkeypatch)
    route, session_id = metadata["gateway_session_key"], metadata["gateway_session_id"]
    real_lookup = runner.async_session_store.lookup_by_session_key
    lookup_started, release_lookup = asyncio.Event(), asyncio.Event()

    async def delayed_lookup(key):
        lookup_started.set()
        await release_lookup.wait()
        return await real_lookup(key)

    monkeypatch.setattr(runner.async_session_store, "lookup_by_session_key", delayed_lookup)
    dispatch = asyncio.create_task(runner._dispatch_plugin_message_injection(
        session_key=route, content="must not jump", plugin_id="probe",
        expected_session_id=session_id))
    try:
        await lookup_started.wait()
        fresh = runner.session_store.reset_session(route)
        assert fresh is not None
        await _admit_human_after_reset(runner, authority, source)
        release_lookup.set()
        assert await dispatch is False
        assert not any((r.payload.get("native_text_v1") or {}).get("automation")
                       for r in authority.persistence.admissions.list(
                           session_id=authority.logical_owner(session_id), pending_only=False))
        # The fresh conversation is admissible on its own pin: the refusal above is the stale pin,
        # not a missing provenance/route.
        assert await runner._dispatch_plugin_message_injection(
            session_key=route, content="fresh is allowed", plugin_id="probe",
            expected_session_id=fresh.session_id) is True
    finally:
        release_lookup.set()
        await asyncio.gather(dispatch, return_exceptions=True)
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_expected_session_id_fails_closed_without_a_session_authority():
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner = _runner(_entry(), adapter)

    assert await runner._dispatch_plugin_message_injection(
        session_key="agent:main:telegram:dm:42", content="pinned", plugin_id="notify-plugin",
        expected_session_id="session-42") is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_expected_session_id_refuses_a_reset_between_lookup_and_admission(tmp_path, monkeypatch):
    """The existing admission fence is rechecked after the lookup: the pinned session may compress,
    but a /new committed before the FIFO write refuses the injection."""
    runner, authority, _adapter, _source, metadata = await _real_owner(tmp_path, monkeypatch)
    route, session_id = metadata["gateway_session_key"], metadata["gateway_session_id"]
    real_admit = authority.admit_automation

    async def reset_then_admit(*args, **kwargs):
        runner.session_store.reset_session(route)
        return await real_admit(*args, **kwargs)

    monkeypatch.setattr(authority, "admit_automation", reset_then_admit)
    try:
        assert await runner._dispatch_plugin_message_injection(
            session_key=route, content="must not commit", plugin_id="probe",
            expected_session_id=session_id) is False
    finally:
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
async def test_under_a_session_authority_injection_enters_the_durable_fifo(monkeypatch):
    """With an authority serving the scope, a plugin turn is committed as a trusted producer
    admission into the session's FIFO; it never runs on the in-process adapter lane, and a
    refused admission is reported as not routed instead of falling back to that lane."""
    import gateway.session_authorities as authorities
    from persistence.core.errors import AuthorityRefusal

    adapter = SimpleNamespace(handle_message=AsyncMock())
    entry = _entry()
    runner = _runner(entry, adapter)
    authority = SimpleNamespace(admit_automation=AsyncMock())
    monkeypatch.setattr(authorities, "active_authority", lambda _runner: authority)

    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="check the deployment", plugin_id="notify-plugin") is True
    adapter.handle_message.assert_not_awaited()
    admitted_adapter, event, identity = authority.admit_automation.await_args.args
    assert admitted_adapter is adapter and event.internal is True and event.message_id == identity
    assert event.metadata == {"gateway_session_key": entry.session_key, "gateway_session_id": entry.session_id}
    assert event.text == "check the deployment" and "notify-plugin" in identity

    authority.admit_automation.side_effect = AuthorityRefusal("not_found")
    assert await runner._dispatch_plugin_message_injection(
        session_key=entry.session_key, content="again", plugin_id="notify-plugin") is False
    adapter.handle_message.assert_not_awaited()
