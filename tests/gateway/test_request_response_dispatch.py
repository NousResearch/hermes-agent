"""Contracts for synchronous request/response platform adapters."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


class _RequestResponseAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class _NoGatewayCommandsAdapter(_RequestResponseAdapter):
    request_dispatch_allows_gateway_commands = False


def _event(text: str = "restart gateway") -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="peer-context",
            chat_type="dm",
            thread_id="caller-thread",
        ),
        message_id="task-1",
    )


def test_dispatch_request_preprocesses_and_returns_handler_response():
    adapter = _RequestResponseAdapter()
    adapter.set_topic_recovery_fn(lambda source: "recovered-thread")
    seen = []

    async def handler(event):
        seen.append((event.text, event.source.thread_id))
        return "final response"

    adapter.set_message_handler(handler)

    result = asyncio.run(adapter.dispatch_request(_event()))

    assert result == "final response"
    assert seen == [("/restart", "recovered-thread")]


def test_dispatch_request_requires_installed_handler():
    adapter = _RequestResponseAdapter()

    with pytest.raises(RuntimeError, match="message handler"):
        asyncio.run(adapter.dispatch_request(_event("hello")))


def test_dispatch_request_can_preserve_plaintext_command_phrase():
    adapter = _NoGatewayCommandsAdapter()
    seen = []

    async def handler(event):
        seen.append(event.text)
        return "ok"

    adapter.set_message_handler(handler)

    assert asyncio.run(adapter.dispatch_request(_event("restart gateway"))) == "ok"
    assert seen == ["restart gateway"]


def test_dispatch_request_rejects_slash_command_before_handler():
    adapter = _NoGatewayCommandsAdapter()
    handler = AsyncMock(return_value="should not run")
    adapter.set_message_handler(handler)

    with pytest.raises(ValueError, match="gateway commands"):
        asyncio.run(adapter.dispatch_request(_event("  /restart")))

    handler.assert_not_awaited()


def test_request_session_interrupt_delegates_all_context():
    adapter = _RequestResponseAdapter()
    callback = AsyncMock()
    adapter.set_session_interrupt_handler(callback)
    source = _event("hello").source

    handled = asyncio.run(
        adapter.request_session_interrupt(
            source,
            interrupt_reason="Remote task canceled",
            invalidation_reason="remote_cancel",
        )
    )

    assert handled is True
    callback.assert_awaited_once_with(
        source,
        interrupt_reason="Remote task canceled",
        invalidation_reason="remote_cancel",
    )


def test_request_session_interrupt_without_handler_is_safe():
    adapter = _RequestResponseAdapter()

    assert asyncio.run(
        adapter.request_session_interrupt(_event("hello").source)
    ) is False


def test_request_session_interrupt_has_no_caller_supplied_session_key():
    import inspect

    parameters = inspect.signature(
        BasePlatformAdapter.request_session_interrupt
    ).parameters

    assert "session_key" not in parameters


def test_runner_interrupt_binding_overwrites_forged_platform_and_profile():
    import dataclasses
    from types import SimpleNamespace

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._interrupt_and_clear_session = AsyncMock()
    adapter = _RequestResponseAdapter()
    callback = runner._make_adapter_session_interrupt_handler(
        adapter,
        profile_name="profile-a",
    )
    adapter.set_session_interrupt_handler(callback)
    forged = _event("hello").source
    forged = dataclasses.replace(
        forged,
        platform=Platform.DISCORD,
        profile="profile-b",
    )

    asyncio.run(callback(forged, interrupt_reason="cancel", invalidation_reason="remote"))

    args = runner._interrupt_and_clear_session.await_args
    assert args.args[0] == "agent:profile-a:telegram:dm:peer-context:caller-thread"
    assert args.args[1].platform is Platform.TELEGRAM
    assert args.args[1].profile == "profile-a"


def test_secondary_profile_interrupt_binding_reaches_only_own_namespace():
    from types import SimpleNamespace

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=True,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._interrupt_and_clear_session = AsyncMock()
    adapter = _RequestResponseAdapter()
    source = _event("hello").source
    callback = runner._make_adapter_session_interrupt_handler(
        adapter,
        profile_name="secondary",
    )
    adapter.set_session_interrupt_handler(callback)

    asyncio.run(callback(source, interrupt_reason="cancel", invalidation_reason="remote"))

    session_key = runner._interrupt_and_clear_session.await_args.args[0]
    assert session_key.startswith("agent:secondary:telegram:")
    assert "profile-a" not in session_key


def test_non_multiplex_interrupt_binding_stays_in_legacy_main_namespace():
    import dataclasses
    from types import SimpleNamespace

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(
        multiplex_profiles=False,
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._interrupt_and_clear_session = AsyncMock()
    adapter = _RequestResponseAdapter()
    callback = runner._make_adapter_session_interrupt_handler(
        adapter,
        profile_name="active-named-profile",
    )
    adapter.set_session_interrupt_handler(callback)
    forged = dataclasses.replace(_event("hello").source, profile="other")

    asyncio.run(callback(forged, interrupt_reason="cancel", invalidation_reason="remote"))

    args = runner._interrupt_and_clear_session.await_args.args
    assert args[0].startswith("agent:main:telegram:")
    assert args[1].profile == "default"


def test_disconnect_revokes_session_interrupt_authority():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._adapter_disconnect_timeout_secs = lambda: 0
    runner.config = type(
        "Config",
        (),
        {
            "multiplex_profiles": True,
            "group_sessions_per_user": True,
            "thread_sessions_per_user": False,
        },
    )()
    runner._interrupt_and_clear_session = AsyncMock()
    adapter = _RequestResponseAdapter()
    stale_callback = runner._make_adapter_session_interrupt_handler(
        adapter,
        profile_name="secondary",
    )
    adapter.set_session_interrupt_handler(stale_callback)

    asyncio.run(runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM))

    assert asyncio.run(adapter.request_session_interrupt(_event("hello").source)) is False
    asyncio.run(
        stale_callback(
            _event("hello").source,
            interrupt_reason="stale",
            invalidation_reason="stale",
        )
    )
    runner._interrupt_and_clear_session.assert_not_awaited()
