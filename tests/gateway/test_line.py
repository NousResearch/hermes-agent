import asyncio
import base64
import hashlib
import hmac
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageType
from gateway.platforms.line import LineAdapter


def _parse_signed_event(secret: str, payload: dict):
    body = json.dumps(payload, separators=(",", ":"))
    signature = base64.b64encode(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest()).decode()

    from linebot.v3.webhook import WebhookParser

    return WebhookParser(secret).parse(body, signature, as_payload=False)[0]


def _make_signed_line_event(*, secret: str, source: dict, text: str = "hello", message_id: str = "111"):
    payload = {
        "destination": "UDEST",
        "events": [
            {
                "type": "message",
                "mode": "active",
                "timestamp": 1710000000000,
                "source": source,
                "webhookEventId": f"evt-{message_id}",
                "deliveryContext": {"isRedelivery": False},
                "replyToken": f"reply-{message_id}",
                "message": {
                    "type": "text",
                    "id": message_id,
                    "quoteToken": f"quote-{message_id}",
                    "text": text,
                },
            }
        ],
    }
    return _parse_signed_event(secret, payload)


def _make_signed_line_audio_event(*, secret: str, source: dict, message_id: str = "333"):
    payload = {
        "destination": "UDEST",
        "events": [
            {
                "type": "message",
                "mode": "active",
                "timestamp": 1710000000000,
                "source": source,
                "webhookEventId": f"evt-{message_id}",
                "deliveryContext": {"isRedelivery": False},
                "replyToken": f"reply-{message_id}",
                "message": {
                    "type": "audio",
                    "id": message_id,
                    "duration": 1234,
                    "contentProvider": {"type": "line"},
                },
            }
        ],
    }
    return _parse_signed_event(secret, payload)


def test_apply_env_overrides_loads_line_credentials(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "line-token")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "line-secret")
    monkeypatch.setenv("LINE_HOME_CHANNEL", "Uhome123")
    monkeypatch.setenv("LINE_HOME_CHANNEL_NAME", "LINE Home")

    config = GatewayConfig()
    _apply_env_overrides(config)

    assert Platform.LINE in config.platforms
    pconfig = config.platforms[Platform.LINE]
    assert pconfig.enabled is True
    assert pconfig.token == "line-token"
    assert pconfig.extra["channel_secret"] == "line-secret"

    home = config.get_home_channel(Platform.LINE)
    assert home is not None
    assert home.chat_id == "Uhome123"
    assert home.name == "LINE Home"


@pytest.mark.asyncio
async def test_process_text_message_normalizes_dm_event():
    adapter = LineAdapter(PlatformConfig(enabled=True, token="token", extra={"channel_secret": "secret"}))
    event = _make_signed_line_event(secret="secret", source={"type": "user", "userId": "U123"}, text="hello line")

    captured = {}

    async def fake_handle_message(message_event):
        captured["event"] = message_event

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    await adapter._process_text_message(event)

    message_event = captured["event"]
    assert message_event.text == "hello line"
    assert message_event.message_id == "111"
    assert message_event.source.platform == Platform.LINE
    assert message_event.source.chat_id == "U123"
    assert message_event.source.user_id == "U123"
    assert message_event.source.chat_type == "dm"
    assert adapter._reply_tokens_by_message_id["111"] == "reply-111"


@pytest.mark.asyncio
async def test_process_text_message_normalizes_group_event():
    adapter = LineAdapter(PlatformConfig(enabled=True, token="token", extra={"channel_secret": "secret"}))
    event = _make_signed_line_event(
        secret="secret",
        source={"type": "group", "groupId": "Cgroup123", "userId": "Umember456"},
        text="group hello",
        message_id="222",
    )

    captured = {}

    async def fake_handle_message(message_event):
        captured["event"] = message_event

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    await adapter._process_text_message(event)

    message_event = captured["event"]
    assert message_event.source.chat_id == "Cgroup123"
    assert message_event.source.user_id == "Umember456"
    assert message_event.source.chat_type == "group"


@pytest.mark.asyncio
async def test_process_audio_message_caches_blob(monkeypatch, tmp_path):
    adapter = LineAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"channel_secret": "secret", "multimodal_grace_period_seconds": 0.05},
        )
    )
    event = _make_signed_line_audio_event(secret="***", source={"type": "user", "userId": "Uaudio"})

    class DummyBlobApi:
        def get_message_content(self, message_id):
            return b"audio-bytes"

    adapter._blob_api = DummyBlobApi()

    captured = {}

    async def fake_handle_message(message_event):
        captured["event"] = message_event

    monkeypatch.setattr("gateway.platforms.line.cache_audio_from_bytes", lambda data, ext=".m4a": str(tmp_path / f"voice{ext}"))
    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    await adapter._process_audio_message(event)
    await asyncio.sleep(0.2)

    message_event = captured["event"]
    assert message_event.source.chat_id == "Uaudio"
    assert message_event.message_type == MessageType.VOICE
    assert message_event.media_urls == [str(tmp_path / "voice.m4a")]
    assert message_event.media_types == ["audio/m4a"]
    assert adapter._reply_tokens_by_message_id["333"] == "reply-333"


@pytest.mark.asyncio
async def test_send_uses_reply_then_push_batches(monkeypatch):
    adapter = LineAdapter(PlatformConfig(enabled=True, token="token", extra={"channel_secret": "secret"}))

    class DummyMessagingApi:
        def reply_message(self, request, **kwargs):
            return {"request": request, "kwargs": kwargs}

        def push_message(self, request, **kwargs):
            return {"request": request, "kwargs": kwargs}

    adapter._messaging_api = DummyMessagingApi()
    adapter._remember_reply_token("msg-1", "reply-token")

    calls = []

    async def fake_call(func, *args, **kwargs):
        calls.append((func.__name__, args[0], kwargs))
        return {"ok": True}

    monkeypatch.setattr(adapter, "_call_sync_api", fake_call)
    monkeypatch.setattr(adapter, "truncate_message", lambda content, max_length: ["1", "2", "3", "4", "5", "6"])

    result = await adapter.send(chat_id="U123", content="ignored", reply_to="msg-1")

    assert result.success is True
    assert [call[0] for call in calls] == ["reply_message", "push_message"]
    assert len(calls[0][1].messages) == 5
    assert len(calls[1][1].messages) == 1
    assert calls[0][1].reply_token == "reply-token"
    assert calls[1][1].to == "U123"
