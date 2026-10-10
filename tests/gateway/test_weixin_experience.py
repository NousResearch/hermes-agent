"""Tencent voice shortcuts, native tool lifecycle and stale-token guards through real gateway seams."""

import queue
import asyncio
import json
import socket
import ssl
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.platforms import weixin
from gateway.platforms.event import MessageType
from gateway.platforms.weixin_experience import WeixinSessionPausedError
from gateway.platforms.weixin_protocol import classify_network_error, sanitize_bot_agent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from hermes_cli.config import atomic_config_write


def configured_adapter(tmp_path, **extra):
    atomic_config_write(tmp_path / "config.yaml", {
        "platforms": {"weixin": {"enabled": True, "extra": {
            "account_id": "bot", "dm_policy": "allowlist", "allow_from": ["speaker"], **extra,
        }}}, "stt": {"enabled": False},
    })
    adapter = weixin.WeixinAdapter(load_gateway_config().platforms[Platform.WEIXIN])
    adapter._poll_session = Mock()
    adapter._token = ""
    adapter._enqueue_text_event = Mock()
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_platform_voice_transcript_needs_neither_download_nor_stt(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path)
    download = AsyncMock(side_effect=AssertionError("The provided transcript must avoid audio download"))
    monkeypatch.setattr(weixin, "_download_and_decrypt_media", download)
    await adapter._process_message({
        "from_user_id": "speaker", "message_id": 18446744073709551615,
        "item_list": [{"type": weixin.ITEM_VOICE, "voice_item": {
            "text": "今天的天气怎么样", "media": {"encrypt_query_param": "test"},
        }}],
    })
    event = adapter._enqueue_text_event.call_args.args[0]
    assert "今天的天气怎么样" in event.text
    assert event.message_id == "18446744073709551615"
    assert event.message_type == MessageType.TEXT and event.media_urls == []
    runner = GatewayRunner.__new__(GatewayRunner)
    assert runner._classify_inbound_media(event, False)[1] == []
    download.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provided", ["", "   "])
async def test_voice_without_platform_text_reaches_audio_pipeline(tmp_path, monkeypatch, provided):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path)
    monkeypatch.setattr(weixin, "_download_and_decrypt_media", AsyncMock(return_value=b"SILK"))
    await adapter._process_message({"from_user_id": "speaker", "item_list": [{
        "type": weixin.ITEM_VOICE, "voice_item": {"text": provided, "media": {"encrypt_query_param": "test"}},
    }]})
    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.VOICE
    assert event.media_types == ["audio/silk"]


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [('{"exit_code":0}', "completed"), ('{"exit_code":1,"error":"failed"}', "failed")])
async def test_gateway_tool_callbacks_send_correlated_ilink_progress(tmp_path, monkeypatch, result, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path)
    adapter._send_session, adapter._token = Mock(), "token"
    active = {"value": True}
    calls = []

    async def send_items(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            active["value"] = False
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_items", send_items)
    ctx = SimpleNamespace(progress_queue=queue.Queue(), _run_still_current=lambda: active["value"],
                          agent_holder=[None], source=adapter.build_source(chat_id="speaker"),
                          _progress_reply_to=None, _progress_metadata=None)
    turn = TurnRunner(None, ctx)
    turn.native_tool_start_callback("call-1", "terminal", {"command": "whoami"})
    turn.native_tool_complete_callback("call-1", "terminal", {}, result)
    await turn._send_native_task_card_progress(adapter)
    start, end = (call["item_list"][0] for call in calls)
    assert start["type"] == 11 and end["type"] == 12
    assert start["tool_call_start_item"] == {"tool_name": "terminal", "tool_call_id": "call-1"}
    assert end["tool_call_result_item"]["status"] == expected
    assert end["tool_call_result_item"]["tool_call_id"] == "call-1"
    assert calls[0]["run_id"] == calls[1]["run_id"]
    assert adapter._weixin_progress == {}


@pytest.mark.parametrize("extra,display,enabled", [({}, {}, True), ({"reply_progress_messages": False}, {}, False),
                                                  ({}, {"tool_progress": "off"}, False)])
def test_weixin_native_progress_honors_adapter_and_display_config(tmp_path, monkeypatch, extra, display, enabled):
    from gateway import run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path, **extra)
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"display": display})
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._resolve_turn_toolsets = lambda *args: ([], [])
    runner._delivery_adapter_for = lambda source: adapter
    settings = runner._run_agent_display_settings(adapter.build_source(chat_id="speaker"))
    assert settings._native_slack_task_cards is enabled


@pytest.mark.asyncio
async def test_expired_bot_token_stops_all_requests_until_cooldown_ends(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path)
    clock = {"now": 0}
    monkeypatch.setattr(weixin.time, "monotonic", lambda: clock["now"])
    monkeypatch.setitem(weixin._LIVE_ADAPTERS, "token", adapter)
    request = AsyncMock(side_effect=[{"errcode": -14}, {"ret": 0}])
    monkeypatch.setattr(weixin, "_api_request", request)
    args = dict(base_url=weixin.ILINK_BASE_URL, endpoint=weixin.EP_SEND_TYPING, payload={}, token="token", timeout_ms=1000)
    with pytest.raises(WeixinSessionPausedError, match="token expired"):
        await weixin._api_post(Mock(), **args)
    with pytest.raises(WeixinSessionPausedError):
        await weixin._api_post(Mock(), **{**args, "endpoint": weixin.EP_GET_UPDATES})
    assert request.await_count == 1
    clock["now"] = 3601
    assert await weixin._api_post(Mock(), **args) == {"ret": 0}


@pytest.mark.asyncio
async def test_native_progress_keeps_all_concurrent_tool_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path)
    adapter._send_session, adapter._token = Mock(), "token"
    items, active = [], {"value": True}

    async def send_items(*args, **kwargs):
        items.extend(kwargs["item_list"])
        active["value"] = len(items) < 24
        return {"ret": 0}

    monkeypatch.setattr(weixin, "_send_items", send_items)
    ctx = SimpleNamespace(progress_queue=queue.Queue(), _run_still_current=lambda: active["value"],
                          agent_holder=[None], source=adapter.build_source(chat_id="speaker"),
                          _progress_reply_to=None, _progress_metadata=None)
    turn = TurnRunner(None, ctx)
    for index in range(12):
        turn.native_tool_start_callback(str(index), "terminal", {})
    for index in range(12):
        turn.native_tool_complete_callback(str(index), "terminal", {}, '{"exit_code":0}')
    await asyncio.wait_for(turn._send_native_task_card_progress(adapter), timeout=5)
    assert {item["tool_call_result_item"]["tool_call_id"] for item in items if item["type"] == 12} == {str(index) for index in range(12)}


@pytest.mark.asyncio
async def test_runtime_bot_agent_metadata_is_sanitized_and_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path, bot_agent="Hermes/1.0 (wechat) 不合法 Other/2.0", route_tag=42)
    monkeypatch.setitem(weixin._LIVE_ADAPTERS, "token", adapter)
    request = AsyncMock(return_value={"ret": 0})
    monkeypatch.setattr(weixin, "_api_request", request)
    await weixin._api_post(Mock(), base_url=weixin.ILINK_BASE_URL, endpoint=weixin.EP_GET_CONFIG, payload={}, token="token", timeout_ms=1000)
    assert json.loads(request.call_args.kwargs["body"])["base_info"]["bot_agent"] == "Hermes/1.0 (wechat) Other/2.0"
    assert request.call_args.kwargs["headers"]["SKRouteTag"] == "42"
    assert len(sanitize_bot_agent("Valid/1 " * 100).encode()) <= 256
    assert sanitize_bot_agent("不合法") == "Hermes"


@pytest.mark.parametrize("exception,category", [(socket.gaierror(-2, "name"), "dns"), (ConnectionRefusedError(), "tcp"),
                                               (ssl.SSLError(), "tls"), (TimeoutError(), "timeout")])
def test_network_failures_keep_their_category_through_wrapped_errors(exception, category):
    wrapper = RuntimeError("transport")
    wrapper.__cause__ = exception
    assert classify_network_error(wrapper) == category


@pytest.mark.asyncio
async def test_failed_typing_config_fetches_back_off_and_recover(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = configured_adapter(tmp_path)
    clock = {"now": 0}
    monkeypatch.setattr(weixin.time, "monotonic", lambda: clock["now"])
    get_config = AsyncMock(side_effect=[OSError("offline"), {"ret": 0, "typing_ticket": "ticket"}])
    monkeypatch.setattr(weixin, "_get_config", get_config)
    assert await adapter._fetch_typing_ticket(Mock(), "speaker", None, "test") is None
    assert await adapter._fetch_typing_ticket(Mock(), "speaker", None, "test") is None
    assert get_config.await_count == 1
    clock["now"] = 2
    assert await adapter._fetch_typing_ticket(Mock(), "speaker", None, "test") == "ticket"
    assert adapter._typing_cache.get("speaker") == "ticket"
