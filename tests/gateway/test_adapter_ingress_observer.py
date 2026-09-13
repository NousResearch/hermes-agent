"""Behavior contracts for the adapter-level MessageEvent observer seam."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
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

    async def observer(event, session_key):
        order.append(("observer", event.message_id, session_key))
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

    def observer(observed, observed_key):
        order.append(("observer", observed, observed_key))
        return False

    async def busy_handler(observed, observed_key):
        order.append(("busy", observed, observed_key))
        return True

    adapter.set_ingress_observer(observer)
    adapter.set_message_handler(lambda _event: None)
    adapter.set_busy_session_handler(busy_handler)
    await adapter.handle_message(event)

    assert [item[0] for item in order] == ["observer", "busy"]
    assert all(item[1] is event and item[2] == session_key for item in order)


@pytest.mark.asyncio
async def test_internal_event_is_observed_even_without_a_message_handler():
    adapter = _ObserverAdapter()
    event = _event(internal=True)
    observed = []

    def observer(candidate, session_key):
        observed.append((candidate, session_key))

    adapter.set_ingress_observer(observer)
    await adapter.handle_message(event)

    assert observed == [(event, adapter._event_session_key(event))]


@pytest.mark.asyncio
async def test_observation_precedes_authorization_and_does_not_admit_the_sender():
    adapter = _ObserverAdapter()
    event = _event()
    order = []
    processed = asyncio.Event()

    def authorization_check(user_id, chat_type, chat_id):
        order.append("authorization")
        return False

    def observer(observed, _session_key):
        assert observed is event
        assert order == []
        order.append("observer")

    async def handler(observed):
        assert adapter._is_sender_authorized(
            observed.source.user_id,
            observed.source.chat_type,
            observed.source.chat_id,
        ) is False
        order.append("handler")
        processed.set()
        return None

    adapter.set_authorization_check(authorization_check)
    adapter.set_ingress_observer(observer)
    adapter.set_message_handler(handler)
    await adapter.handle_message(event)
    await asyncio.wait_for(processed.wait(), timeout=1)

    assert order == ["observer", "authorization", "handler"]


@pytest.mark.asyncio
async def test_observer_failure_is_fail_open_and_logs_no_private_detail(caplog):
    idle_adapter = _ObserverAdapter()
    processed = asyncio.Event()

    def broken_observer(_event, _session_key):
        raise RuntimeError("PRIVATE_EXCEPTION_DETAIL")

    async def handler(_event):
        processed.set()
        return None

    idle_adapter.set_ingress_observer(broken_observer)
    idle_adapter.set_message_handler(handler)
    await idle_adapter.handle_message(_event())
    await asyncio.wait_for(processed.wait(), timeout=1)

    busy_adapter = _ObserverAdapter()
    busy_processed = asyncio.Event()
    busy_event = _event(text="busy")
    busy_key = busy_adapter._event_session_key(busy_event)
    busy_adapter._active_sessions[busy_key] = asyncio.Event()

    async def busy_handler(_event, _session_key):
        busy_processed.set()
        return True

    busy_adapter.set_ingress_observer(broken_observer)
    busy_adapter.set_message_handler(handler)
    busy_adapter.set_busy_session_handler(busy_handler)
    await busy_adapter.handle_message(busy_event)
    await asyncio.wait_for(busy_processed.wait(), timeout=1)

    assert processed.is_set()
    assert busy_processed.is_set()
    assert "RuntimeError" in caplog.text
    assert "PRIVATE_EXCEPTION_DETAIL" not in caplog.text
    assert busy_key not in caplog.text


def _runner_with_adapter(adapter, message_handler, busy_handler):
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner.session_store = object()
    runner._busy_text_mode = "queue"

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
async def test_registered_plugin_observer_receives_idle_and_busy_events_once():
    manager = PluginManager()
    context = PluginContext(
        PluginManifest(name="ingress-observer-fixture", source="user"), manager
    )
    seen = []
    context.register_hook(
        "gateway_ingress_observed",
        lambda event, gateway, session_key: seen.append(
            (event.message_id, gateway, session_key)
        ),
    )
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

    assert [(message_id, key) for message_id, _gateway, key in seen] == [
        ("message-1", adapter._event_session_key(idle)),
        ("message-2", busy_key),
    ]
    assert all(gateway is runner for _message_id, gateway, _key in seen)


@pytest.mark.asyncio
async def test_runner_wiring_has_no_routing_effect_without_a_registered_plugin():
    manager = PluginManager()
    manager._discovered = True
    adapter = _ObserverAdapter()
    idle_processed = asyncio.Event()
    busy_processed = asyncio.Event()

    async def message_handler(_event):
        idle_processed.set()
        return None

    async def busy_handler(_event, _session_key):
        busy_processed.set()
        return True

    with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
        _runner_with_adapter(adapter, message_handler, busy_handler)
        await adapter.handle_message(_event())
        await asyncio.wait_for(idle_processed.wait(), timeout=1)

        busy = _event(text="busy", chat_id="chat-2")
        busy_key = adapter._event_session_key(busy)
        adapter._active_sessions[busy_key] = asyncio.Event()
        await adapter.handle_message(busy)
        await asyncio.wait_for(busy_processed.wait(), timeout=1)

    assert idle_processed.is_set()
    assert busy_processed.is_set()


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


def test_runner_closes_awaitable_hook_result_without_executing_or_leaking(caplog):
    runner = object.__new__(GatewayRunner)
    executed = []

    async def should_not_run():
        executed.append(True)

    result = should_not_run()
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), patch(
        "hermes_cli.lifecycle.invoke_hook", return_value=[result]
    ):
        runner._handle_gateway_ingress_observed(_event(), "private-session-key")

    assert executed == []
    assert result.cr_frame is None
    assert "coroutine" in caplog.text
    assert "private-session-key" not in caplog.text
