"""A reply past MAX_SPLIT_MESSAGES goes out as one file on every path the #153476 review found (#152012)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from tests.gateway.test_reply_file_handoff import _telegram, _body, _Bot
from plugins.platforms.telegram.adapter import TelegramAdapter
from plugins.platforms.telegram import adapter as tg

@pytest.fixture(autouse=True)
def parse(monkeypatch):
    monkeypatch.setattr(tg, 'ParseMode', SimpleNamespace(MARKDOWN_V2='MarkdownV2'))

@pytest.mark.asyncio
async def test_rich_refusal_recounts_before_legacy_flood():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='fake', extra={'rich_messages': True}))
    adapter._telegram_chat_outbound_slot_secs = 0
    bot = _Bot()
    async def reject(*args, **kwargs):
        raise RuntimeError('Bad Request: unsupported rich message')
    bot.do_api_request = reject
    adapter._bot = bot
    text = '\n\n'.join('<details><summary>Section</summary>\n\n' + _body(18000) + '\n\n</details>' for _ in range(4))
    await adapter._send_with_retry('4242', text)
    sends = sum(k == 'send' for k, _ in bot.calls)
    assert bot.files == [text.encode()]
    assert sends <= 8

@pytest.mark.asyncio
async def test_discord_tool_long_reply_uses_file(monkeypatch):
    import json
    import aiohttp
    from hermes_agent.gateway.config import Platform
    from hermes_agent.tools.send_message_tool import _send_to_platform
    from plugins.platforms.discord import adapter as dc
    from tests.tools.test_send_message_tool import _StreamingAiohttpResponse, _discord_entry
    calls = []
    class Session:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def post(self, url, **kwargs):
            calls.append(kwargs)
            return _StreamingAiohttpResponse(200, json.dumps({'id':str(len(calls))}).encode())
    _discord_entry()
    monkeypatch.setattr(aiohttp, 'ClientSession', Session)
    monkeypatch.setattr(dc, '_standalone_is_forum', AsyncMock(return_value=False))
    await _send_to_platform(Platform.DISCORD, PlatformConfig(enabled=True, token='fake'), '111', _body(60000))
    assert any('data' in c for c in calls)

@pytest.mark.asyncio
async def test_file_segment_remains_known_as_delivered():
    adapter = _telegram()
    text = _body(60000)
    c = GatewayStreamConsumer(adapter, '4242')
    c.on_delta(text)
    c.on_segment_break()
    # After a tool round the agent streams the same reply again, led by a "\n\n" paragraph break.
    again = '\n\n' + text
    for i in range(0, len(again), 1500):
        c.on_delta(again[i:i + 1500])
    c.finish(text)
    await c.run()
    from hermes_agent.gateway.run_turn import GatewayTurnMixin
    confirmed = GatewayTurnMixin._run_agent_stream_confirmed_final_delivery(c, text)
    if not confirmed:
        await adapter._send_with_retry('4242', text)
    assert adapter._bot.files == [text.encode()]

@pytest.mark.asyncio
async def test_completed_commentary_reply_also_hands_off():
    adapter = _telegram()
    text = _body(60000)
    c = GatewayStreamConsumer(adapter, '4242')
    c.on_commentary(text)
    c.finish(text)
    await c.run()
    assert adapter._bot.files == [text.encode()]
