"""Behavior contracts for the immutable adapter-ingress observer seam."""

import asyncio
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, _ingress_snapshot
from gateway.platforms.event import IngressEventSnapshot, MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class _ObserverAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(
    *, text: str = "hello", internal: bool = False, chat_id: str = "chat-1"
) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="dm",
            user_id="user-1",
        ),
        message_id="message-1",
        internal=internal,
    )


@pytest.mark.asyncio
async def test_idle_event_is_observed_once_before_processing_and_return_is_ignored():
    adapter = _ObserverAdapter()
    order = []
    processed = asyncio.Event()

    async def observer(snapshot, session_key, _authorization_facts):
        order.append(("observer", snapshot.message_id, session_key))
        return {"action": "skip"}

    async def handler(event):
        order.append(("handler", event.message_id, adapter._event_session_key(event)))
        processed.set()
        return None

    adapter.set_ingress_observer(observer)
    adapter.set_message_handler(handler)
    await adapter.handle_message(_event())
    await asyncio.wait_for(processed.wait(), timeout=1)

    assert [item[0] for item in order] == ["observer", "handler"]
    assert order[0][1:] == order[1][1:]


@pytest.mark.asyncio
async def test_busy_event_is_observed_once_before_busy_routing():
    adapter = _ObserverAdapter()
    event = _event(text="follow-up")
    session_key = adapter._event_session_key(event)
    adapter._active_sessions[session_key] = asyncio.Event()
    order = []

    def observer(snapshot, observed_key, _authorization_facts):
        order.append(("observer", snapshot.message_id, observed_key))
        return False

    async def busy_handler(observed, observed_key):
        order.append(("busy", observed.message_id, observed_key))
        return True

    adapter.set_ingress_observer(observer)
    adapter.set_message_handler(lambda _event: None)
    adapter.set_busy_session_handler(busy_handler)
    await adapter.handle_message(event)

    assert [item[0] for item in order] == ["observer", "busy"]
    assert all(item[1] == event.message_id and item[2] == session_key for item in order)


@pytest.mark.asyncio
async def test_internal_event_is_observed_even_without_a_message_handler():
    adapter = _ObserverAdapter()
    event = _event(internal=True)
    observed = []

    def observer(snapshot, session_key, _authorization_facts):
        observed.append((snapshot, session_key))

    adapter.set_ingress_observer(observer)
    await adapter.handle_message(event)

    assert len(observed) == 1
    assert isinstance(observed[0][0], IngressEventSnapshot)
    assert observed[0][0].internal is True
    assert observed[0][1] == adapter._event_session_key(event)


@pytest.mark.asyncio
async def test_snapshot_is_frozen_body_free_and_cannot_mutate_live_dispatch():
    adapter = _ObserverAdapter()
    event = _event(text="PRIVATE BODY")
    event.source.profile = "original-profile"
    event.source.thread_id = "thread-1"
    event.raw_message = {"private": ["sdk payload"]}
    event.metadata = {"private": {"nested": True}, "whatsapp_from_owner": True}
    event.media_urls = ["/private/attachment.jpg"]
    event.media_types = ["image/jpeg"]
    event.allow_gateway_control = False
    expected_key = adapter._event_session_key(event)
    live_source_before = dict(vars(event.source))
    processed = asyncio.Event()

    def observer(snapshot, session_key, authorization_facts):
        assert session_key == expected_key
        assert authorization_facts.user_name is None
        for omitted in ("text", "reply_to_text", "raw_message", "metadata", "media_urls"):
            assert not hasattr(snapshot, omitted)
        assert snapshot.media_count == 1
        assert snapshot.media_types == ("image/jpeg",)
        for target, name, value in (
            (snapshot, "message_id", "changed"),
            (snapshot, "internal", True),
            (snapshot, "allow_gateway_control", True),
            (snapshot, "media_types", ("changed",)),
            (snapshot.source, "user_id", "attacker"),
            (snapshot.source, "profile", "attacker"),
            (snapshot.source, "thread_id", "attacker"),
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(target, name, value)
        with pytest.raises((AttributeError, TypeError)):
            snapshot.source.new_field = "attacker"

    async def handler(observed):
        assert observed is event
        assert adapter._event_session_key(observed) == expected_key
        assert observed.text == "PRIVATE BODY"
        processed.set()
        return None

    adapter.set_ingress_observer(observer)
    adapter.set_message_handler(handler)
    await adapter.handle_message(event)
    await asyncio.wait_for(processed.wait(), timeout=1)

    assert vars(event.source) == live_source_before
    assert event.text == "PRIVATE BODY"
    assert event.raw_message == {"private": ["sdk payload"]}
    assert event.metadata == {"private": {"nested": True}, "whatsapp_from_owner": True}
    assert event.media_urls == ["/private/attachment.jpg"]
    assert event.media_types == ["image/jpeg"]
    assert event.internal is False
    assert event.allow_gateway_control is False


@pytest.mark.asyncio
async def test_observer_failure_is_fail_open_and_logs_no_private_detail(caplog):
    adapter = _ObserverAdapter()
    processed = asyncio.Event()

    def broken_observer(_snapshot, _session_key, _authorization_facts):
        raise RuntimeError("PRIVATE_EXCEPTION_DETAIL")

    async def handler(_event):
        processed.set()
        return None

    event = _event()
    private_key = adapter._event_session_key(event)
    adapter.set_ingress_observer(broken_observer)
    adapter.set_message_handler(handler)
    await adapter.handle_message(event)
    await asyncio.wait_for(processed.wait(), timeout=1)

    assert "RuntimeError" in caplog.text
    assert "PRIVATE_EXCEPTION_DETAIL" not in caplog.text
    assert private_key not in caplog.text


def _runner_with_adapter(adapter, message_handler, busy_handler, *, authorized=True):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.session_store = object()
    runner._busy_text_mode = "queue"
    runner.adapters = {adapter.platform: adapter}
    runner._profile_adapters = {}
    runner._is_user_authorized_for_source = Mock(return_value=authorized)

    async def platform_event_handler(_event, _source):
        return None

    runner._wire_adapter_handlers(
        adapter,
        message_handler=message_handler,
        fatal_error_handler=lambda _adapter: None,
        busy_session_handler=busy_handler,
        authorization_check=lambda *_args, **_kwargs: True,
        platform_event_handler=platform_event_handler,
    )
    adapter.set_topic_recovery_fn(None)
    return runner


@pytest.mark.asyncio
async def test_registered_plugin_gets_detached_idle_and_busy_snapshots_once_with_auth():
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="ingress-observer-fixture", source="user"), manager
    )
    seen = []

    def plugin_observer(snapshot, session_key, authorized, **kwargs):
        for target, name, value in (
            (snapshot, "internal", not snapshot.internal),
            (snapshot, "allow_gateway_control", False),
            (snapshot.source, "user_id", "mutated"),
            (snapshot.source, "profile", "mutated"),
            (snapshot.source, "thread_id", "mutated"),
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(target, name, value)
        for omitted in ("text", "raw_message", "metadata", "media_urls"):
            assert not hasattr(snapshot, omitted)
        seen.append((snapshot, session_key, authorized, kwargs))

    context.register_hook("gateway_ingress_observed", plugin_observer)
    manager._discovered = True
    adapter = _ObserverAdapter()
    idle_processed = asyncio.Event()

    async def message_handler(_event):
        idle_processed.set()
        return None

    async def busy_handler(_event, _session_key):
        return True

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner = _runner_with_adapter(adapter, message_handler, busy_handler)
        idle = _event()
        await adapter.handle_message(idle)
        await asyncio.wait_for(idle_processed.wait(), timeout=1)

        busy = _event(text="busy", chat_id="chat-2")
        busy.message_id = "message-2"
        busy_key = adapter._event_session_key(busy)
        adapter._active_sessions[busy_key] = asyncio.Event()
        await adapter.handle_message(busy)

    assert [(item[0].message_id, item[1]) for item in seen] == [
        ("message-1", adapter._event_session_key(idle)),
        ("message-2", busy_key),
    ]
    assert all(item[2] is True for item in seen)
    assert all(
        {"gateway", "event", "adapter", "raw_message"}.isdisjoint(item[3])
        for item in seen
    )
    assert all(not hasattr(item[0], "text") for item in seen)
    detached_sources = [call.args[0] for call in runner._is_user_authorized_for_source.call_args_list]
    assert len(detached_sources) == 2
    assert all(source is not idle.source and source is not busy.source for source in detached_sources)
    assert all(source._transport_adapter_ref() is adapter for source in detached_sources)


def test_auth_false_error_and_internal_are_tri_state_without_admission(caplog):
    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized_for_source = Mock(return_value=False)
    runner._admit_bot_message_for_source = Mock()
    runner._hm_offer_pairing_code = Mock()
    snapshot = _ingress_snapshot(_event())

    assert runner._ingress_authorization_verdict(snapshot) is False
    runner._is_user_authorized_for_source.side_effect = RuntimeError("PRIVATE AUTH")
    assert runner._ingress_authorization_verdict(snapshot) is None
    assert runner._ingress_authorization_verdict(
        dataclasses.replace(snapshot, internal=True)
    ) is None
    runner._is_user_authorized_for_source.side_effect = None
    rejected = dataclasses.replace(
        snapshot,
        source=dataclasses.replace(snapshot.source, profile_route_rejected=True),
    )
    assert runner._ingress_authorization_verdict(rejected) is False
    runner._admit_bot_message_for_source.assert_not_called()
    runner._hm_offer_pairing_code.assert_not_called()
    assert "PRIVATE AUTH" not in caplog.text


@pytest.mark.asyncio
async def test_multiplex_observation_preserves_live_source_and_session_routing(tmp_path):
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="multiplex-ingress-fixture", source="user"), manager
    )
    seen = []
    context.register_hook(
        "gateway_ingress_observed",
        lambda snapshot, session_key, authorized: seen.append(
            (snapshot, session_key, authorized)
        ),
    )
    manager._discovered = True
    adapter = _ObserverAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    runner.session_store = object()
    runner._busy_text_mode = "queue"
    runner.adapters = {adapter.platform: adapter}
    runner._profile_adapters = {}
    auth_sources = []

    def authorize(source):
        auth_sources.append(source)
        return True

    runner._is_user_authorized_for_source = authorize
    processed = asyncio.Event()

    async def handler(_event):
        processed.set()
        return None

    async def busy_handler(_event, _session_key):
        return True

    async def platform_event_handler(_event, _source):
        return None

    with patch("gateway.run.get_hermes_home", return_value=str(tmp_path)), patch(
        "hermes_cli.plugins.get_plugin_manager", return_value=manager
    ):
        runner._wire_adapter_handlers(
            adapter,
            message_handler=handler,
            fatal_error_handler=lambda _adapter: None,
            busy_session_handler=busy_handler,
            authorization_check=lambda *_args, **_kwargs: True,
            platform_event_handler=platform_event_handler,
        )
        event = _event()
        source_before = dict(vars(event.source))
        key_before = adapter._event_session_key(event)
        await adapter.handle_message(event)
        await asyncio.wait_for(processed.wait(), timeout=1)

    assert dict(vars(event.source)) == source_before
    assert adapter._event_session_key(event) == key_before
    assert seen[0][1:] == (key_before, True)
    assert auth_sources[0] is not event.source
    assert auth_sources[0]._transport_adapter_ref() is adapter
    assert auth_sources[0]._authorization_profile_home == Path(tmp_path)


@pytest.mark.asyncio
async def test_runner_wiring_has_no_auth_or_routing_effect_without_registered_plugin():
    manager = PluginManager()
    manager._discovered = True
    adapter = _ObserverAdapter()
    idle_processed = asyncio.Event()

    async def message_handler(_event):
        idle_processed.set()
        return None

    async def busy_handler(_event, _session_key):
        return True

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner = _runner_with_adapter(adapter, message_handler, busy_handler)
        await adapter.handle_message(_event())
        await asyncio.wait_for(idle_processed.wait(), timeout=1)

    runner._is_user_authorized_for_source.assert_not_called()


def test_async_gateway_ingress_plugin_callback_is_rejected_without_registration():
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="async-ingress-fixture", source="user"), manager
    )

    async def async_observer(**_kwargs):
        return None

    with pytest.raises(ValueError, match="requires a fast synchronous callback"):
        context.register_hook("gateway_ingress_observed", async_observer)

    assert manager.has_hook("gateway_ingress_observed") is False


def test_plugin_callback_exception_log_omits_exception_text_and_repr(caplog):
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="broken-ingress-fixture", source="user"), manager
    )

    def broken_observer(**_kwargs):
        raise RuntimeError("PRIVATE_PLUGIN_EXCEPTION")

    context.register_hook("gateway_ingress_observed", broken_observer)
    manager.invoke_hook(
        "gateway_ingress_observed",
        snapshot=_ingress_snapshot(_event()),
        session_key="PRIVATE_SESSION_KEY",
        authorized=True,
    )

    assert "broken_observer" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "PRIVATE_PLUGIN_EXCEPTION" not in caplog.text
    assert "PRIVATE_SESSION_KEY" not in caplog.text


def test_runner_closes_awaitable_hook_result_without_executing_or_leaking(caplog):
    runner = object.__new__(GatewayRunner)
    executed = []
    returned = []

    async def should_not_run():
        executed.append(True)

    def returns_coroutine(**_kwargs):
        result = should_not_run()
        returned.append(result)
        return result

    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="coroutine-ingress-fixture", source="user"), manager
    )
    context.register_hook("gateway_ingress_observed", returns_coroutine)
    manager._discovered = True
    snapshot = _ingress_snapshot(_event())
    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        runner._handle_gateway_ingress_observed(snapshot, "private-session-key", True)

    assert executed == []
    assert len(returned) == 1
    assert returned[0].cr_frame is None
    assert "coroutine" in caplog.text
    assert "private-session-key" not in caplog.text
