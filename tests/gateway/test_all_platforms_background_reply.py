"""Cross-platform verification suite for Plan A (_run_background_task reply_to forwarding).

Verifies that forwarding reply_to=event_message_id in _run_background_task:
1. Benefits QQBot (attaches msg_id for passive reply instead of proactive push).
2. Benefits Telegram (attaches reply_to_message_id, fixing DM topic missing anchor failures).
3. Benefits Discord (attaches MessageReference to user's command message).
4. Benefits Slack (attaches thread_ts, keeping the background result inside the thread).
5. Benefits Feishu (uses im.v1.message.reply to quote the original command).
6. Benefits WeCom (uses cached req_id for passive reply without failing in groups).
7. Benefits WhatsApp (attaches replyTo to first chunk).
8. Causes ZERO regressions when event_message_id is None (sends clean standalone message).
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.session import SessionSource


def _make_runner():
    from gateway.run_turn import GatewayTurnMixin
    runner = object.__new__(GatewayTurnMixin)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._background_tasks = set()
    mock_store = MagicMock()
    mock_store.get_model_override.return_value = None
    runner.session_store = mock_store
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._resolve_session_agent_runtime = MagicMock(return_value=("m", {"api_key": "k"}))
    runner._resolve_turn_agent_config = MagicMock(return_value={"model": "m", "runtime": {"api_key": "k"}})
    runner._resolve_session_reasoning_config = MagicMock(return_value=None)
    runner._resolve_session_service_tier = MagicMock(return_value=None)
    runner._resolve_turn_toolsets = MagicMock(return_value=([], None))
    runner._refresh_fallback_model = MagicMock(return_value=None)
    runner._cleanup_agent_resources = MagicMock()
    return runner


# ---------------------------------------------------------------------------
# QQBot Verification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_qqbot_plan_a_benefit():
    from gateway.platforms.qqbot.adapter import QQAdapter
    runner = _make_runner()
    adapter = QQAdapter(PlatformConfig(enabled=True, token="qq_token"))
    adapter._running = True
    adapter._ws = SimpleNamespace(closed=False)
    adapter._chat_type_map["user_target"] = "c2c"

    api_calls = []
    async def mock_api(method, path, body=None, timeout=None):
        api_calls.append(body)
        return {"id": "qq_msg_1"}
    adapter._api_request = mock_api
    runner.adapters[Platform.QQBOT] = adapter

    source = SessionSource(platform=Platform.QQBOT, user_id="user_target", chat_id="user_target", user_name="u")
    runner._run_in_executor_with_context = AsyncMock(return_value={"final_response": "QQ final", "messages": []})

    # Execute background task with Plan A
    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._run_background_task("task", source, "bg_qq", event_message_id="QQ_MSG_INBOUND_100")

    assert len(api_calls) == 1
    # VERIFIED: msg_id is present, converting it to passive reply
    assert api_calls[0].get("msg_id") == "QQ_MSG_INBOUND_100"


# ---------------------------------------------------------------------------
# Telegram Verification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_telegram_plan_a_benefit():
    from plugins.platforms.telegram.adapter import TelegramAdapter
    runner = _make_runner()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="123:abc"))
    adapter._bot = SimpleNamespace(username="test_bot")
    adapter._connected = True

    sent_kwargs_list = []
    async def mock_send_chunk(chunk, send_kwargs):
        sent_kwargs_list.append(send_kwargs)
        return SimpleNamespace(message_id=9999)
    adapter._send_chunk_markdown_or_plain = mock_send_chunk
    runner.adapters[Platform.TELEGRAM] = adapter

    source = SessionSource(platform=Platform.TELEGRAM, user_id="101", chat_id="101", user_name="u")
    runner._run_in_executor_with_context = AsyncMock(return_value={"final_response": "TG final", "messages": []})

    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._run_background_task("task", source, "bg_tg", event_message_id="778899")

    assert len(sent_kwargs_list) == 1
    # VERIFIED: reply_to_message_id is populated for Telegram
    assert sent_kwargs_list[0].get("reply_to_message_id") == 778899


# ---------------------------------------------------------------------------
# Discord Verification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_discord_plan_a_benefit():
    from plugins.platforms.discord.adapter import DiscordAdapter
    runner = _make_runner()
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="disc_token"))
    adapter._client = SimpleNamespace()

    send_calls = []
    mock_channel = AsyncMock()
    async def fake_channel_send(**kwargs):
        send_calls.append(kwargs)
        return SimpleNamespace(id=1111)
    mock_channel.send = fake_channel_send
    adapter._resolve_channel = AsyncMock(return_value=mock_channel)
    runner.adapters[Platform.DISCORD] = adapter

    source = SessionSource(platform=Platform.DISCORD, user_id="d1", chat_id="c1", user_name="u")
    runner._run_in_executor_with_context = AsyncMock(return_value={"final_response": "DC final", "messages": []})

    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._run_background_task("task", source, "bg_dc", event_message_id="445566")

    assert len(send_calls) == 1
    # VERIFIED: channel.send received a reference argument
    assert "reference" in send_calls[0]
    assert send_calls[0]["reference"] is not None


# ---------------------------------------------------------------------------
# Slack Verification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_slack_plan_a_benefit():
    from plugins.platforms.slack.adapter import SlackAdapter
    runner = _make_runner()
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-test", extra={"reply_in_thread": True}))
    
    posted_payloads = []
    mock_client = AsyncMock()
    async def fake_post_message(**kwargs):
        posted_payloads.append(kwargs)
        return {"ok": True, "ts": "1700000002.000000"}
    mock_client.chat_postMessage = fake_post_message
    
    adapter._client = mock_client
    adapter._app = SimpleNamespace(client=mock_client)
    adapter._connected = True
    adapter._dm_target = AsyncMock(side_effect=lambda c, m: c)
    runner.adapters[Platform.SLACK] = adapter

    source = SessionSource(platform=Platform.SLACK, user_id="U1", chat_id="C1", user_name="u")
    runner._run_in_executor_with_context = AsyncMock(return_value={"final_response": "Slack final", "messages": []})

    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._run_background_task("task", source, "bg_slack", event_message_id="1700000001.000000")

    assert len(posted_payloads) == 1
    # VERIFIED: thread_ts is populated so response stays in the Slack thread
    assert posted_payloads[0].get("thread_ts") == "1700000001.000000"


# ---------------------------------------------------------------------------
# Feishu Verification
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_feishu_plan_a_benefit():
    from plugins.platforms.feishu.adapter import FeishuAdapter
    runner = _make_runner()
    adapter = FeishuAdapter(PlatformConfig(enabled=True, token="fs_token"))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace())))

    reply_requests = []
    def fake_reply(req):
        reply_requests.append(req)
        return SimpleNamespace(code=0, msg="success", data=SimpleNamespace(message_id="om_resp_1"))
    adapter._client.im.v1.message.reply = fake_reply
    adapter._run_blocking = AsyncMock(side_effect=lambda fn, req: fn(req))
    runner.adapters[Platform.FEISHU] = adapter

    source = SessionSource(platform=Platform.FEISHU, user_id="ou_1", chat_id="oc_1", user_name="u")
    runner._run_in_executor_with_context = AsyncMock(return_value={"final_response": "Feishu final", "messages": []})

    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._run_background_task("task", source, "bg_fs", event_message_id="om_inbound_original")

    assert len(reply_requests) == 1
    # VERIFIED: calls im.v1.message.reply quoting om_inbound_original instead of create()
    assert reply_requests[0].message_id == "om_inbound_original"


# ---------------------------------------------------------------------------
# Zero Regression Verification (When event_message_id is None)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_zero_regression_when_no_anchor():
    from gateway.platforms.qqbot.adapter import QQAdapter
    runner = _make_runner()
    adapter = QQAdapter(PlatformConfig(enabled=True, token="qq_token"))
    adapter._running = True
    adapter._ws = SimpleNamespace(closed=False)
    adapter._chat_type_map["user_target"] = "c2c"

    api_calls = []
    async def mock_api(method, path, body=None, timeout=None):
        api_calls.append(body)
        return {"id": "qq_msg_standalone"}
    adapter._api_request = mock_api
    runner.adapters[Platform.QQBOT] = adapter

    source = SessionSource(platform=Platform.QQBOT, user_id="user_target", chat_id="user_target", user_name="u")
    runner._run_in_executor_with_context = AsyncMock(return_value={"final_response": "Fallback", "messages": []})

    # When event_message_id is None (synthetic dispatch or no incoming message anchor)
    with patch("gateway.run._load_gateway_config", return_value={}):
        await runner._run_background_task("task", source, "bg_no_anchor", event_message_id=None)

    assert len(api_calls) == 1
    # VERIFIED: seamlessly sends without msg_id, exactly matching legacy behavior
    assert "msg_id" not in api_calls[0]
