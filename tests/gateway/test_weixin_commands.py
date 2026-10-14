"""Weixin diagnostics are local commands behind the normal gateway admission and slash-policy gates."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms import weixin
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.test_unknown_command import _make_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [True, False])
async def test_echo_is_local_only_after_authorization(authorized):
    adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, extra={"account_id": "bot", "dm_policy": "pairing"}))
    adapter.send = AsyncMock()
    runner = _make_runner()
    runner.config = GatewayConfig(platforms={Platform.WEIXIN: adapter.config})
    runner.adapters = {Platform.WEIXIN: adapter}
    runner._is_user_authorized = lambda source: authorized
    runner._run_agent = AsyncMock(side_effect=AssertionError("/echo must not call an LLM"))
    event = MessageEvent(text="/echo 连接测试", message_type=MessageType.COMMAND,
                         source=adapter.build_source(chat_id="speaker", user_id="speaker"), message_id="echo")
    await runner._handle_message(event)
    messages = [call.args[1] for call in adapter.send.await_args_list]
    assert ("连接测试" in messages) is authorized
    runner._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_slash_access_denial_does_not_toggle_account_debug_mode():
    from gateway.run_inbound_commands import dispatch_platform_command

    adapter = weixin.WeixinAdapter(PlatformConfig(enabled=True, extra={"account_id": "bot"}))
    adapter.send = AsyncMock()
    runner = _make_runner()
    runner._intake_adapter_for = lambda source: adapter
    runner._check_slash_access = lambda source, command: "Access denied"
    event = MessageEvent(text="/toggle-debug", source=adapter.build_source(chat_id="speaker", user_id="speaker"))
    assert await dispatch_platform_command(runner, event, event.source, "toggle-debug") == (True, "Access denied")
    assert adapter._weixin_debug is False
    adapter.send.assert_not_awaited()
