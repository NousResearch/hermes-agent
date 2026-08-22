import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import Platform, SessionSource


@pytest.mark.asyncio
async def test_gateway_input_route_rewrites_off_event_loop_and_notices(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="gateway-session")
        ),
    )
    runner._goal_active_for_input_route = lambda _session_id: False
    runner._session_key_for_source = lambda _source: "gateway-key"
    runner._deliver_platform_notice = AsyncMock()
    caller_thread = threading.get_ident()
    callback_threads = []

    def route(**_payload):
        callback_threads.append(threading.get_ident())
        return "/grill-with-docs brief", "Routed"

    monkeypatch.setattr("hermes_cli.lifecycle.route_pre_user_input", route)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    event = MessageEvent(
        text="brief",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", user_id="user"
        ),
    )

    routed = await runner._route_pre_user_input(event)

    assert routed.text == "/grill-with-docs brief"
    assert callback_threads and callback_threads[0] != caller_thread
    runner._deliver_platform_notice.assert_awaited_once_with(event.source, "Routed")


@pytest.mark.asyncio
async def test_gateway_input_route_skips_active_goal(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="gateway-session")
        ),
    )
    runner._goal_active_for_input_route = lambda _session_id: True
    runner._session_key_for_source = lambda _source: "gateway-key"
    runner._deliver_platform_notice = AsyncMock()
    called = False

    def route(**_payload):
        nonlocal called
        called = True
        return "changed", None

    monkeypatch.setattr("hermes_cli.lifecycle.route_pre_user_input", route)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    event = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", user_id="user"
        ),
    )

    assert await runner._route_pre_user_input(event) is event
    assert called is False


@pytest.mark.asyncio
async def test_gateway_input_route_skips_unknown_goal_state(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(
            return_value=SimpleNamespace(session_id="gateway-session")
        ),
    )
    runner._goal_active_for_input_route = lambda _session_id: None
    called = False

    def route(**_payload):
        nonlocal called
        called = True
        return "changed", None

    monkeypatch.setattr("hermes_cli.lifecycle.route_pre_user_input", route)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: True)
    event = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", user_id="user"
        ),
    )

    assert await runner._route_pre_user_input(event) is event
    assert called is False


@pytest.mark.asyncio
async def test_gateway_input_route_skips_session_lookup_without_subscriber(monkeypatch):
    class FailingSessionStore:
        def __init__(self, store):
            self._store = store

        async def get_or_create_session(self, _source):
            raise AssertionError("session lookup must be skipped")

    runner = object.__new__(GatewayRunner)
    runner.session_store = object()
    runner._async_session_store = FailingSessionStore(runner.session_store)
    monkeypatch.setattr("hermes_cli.lifecycle.has_hook", lambda _name: False)
    event = MessageEvent(
        text="plain input",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="chat", user_id="user"
        ),
    )

    assert await runner._route_pre_user_input(event) is event
