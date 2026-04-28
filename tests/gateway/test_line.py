import asyncio
import base64
import hashlib
import hmac
import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides
from gateway.platforms.base import MessageEvent, MessageType
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


@pytest.mark.asyncio
async def test_send_voice_caches_served_audio_before_building_public_url(monkeypatch, tmp_path):
    adapter = LineAdapter(
        PlatformConfig(
            enabled=True,
            token="token",
            extra={"channel_secret": "secret", "public_base_url": "https://example.test"},
        )
    )

    class DummyMessagingApi:
        def push_message(self, request, **kwargs):
            return {"request": request, "kwargs": kwargs}

    adapter._messaging_api = DummyMessagingApi()

    source_audio = tmp_path / "reply.mp3"
    source_audio.write_bytes(b"ID3" + b"\x00" * 32)
    served_audio = tmp_path / "served.m4a"
    served_audio.write_bytes(b"M4A" + b"\x00" * 32)

    calls = []

    async def fake_call(func, *args, **kwargs):
        calls.append((func.__name__, args[0], kwargs))
        return {"ok": True}

    monkeypatch.setattr(adapter, "_call_sync_api", fake_call)
    monkeypatch.setattr(adapter, "_prepare_outbound_audio", lambda path: (str(source_audio), 1234))
    monkeypatch.setattr(adapter, "_cache_served_audio", lambda path: str(served_audio))

    result = await adapter.send_voice(chat_id="U123", audio_path=str(source_audio))

    assert result.success is True
    assert [call[0] for call in calls] == ["push_message"]
    req = calls[0][1]
    assert req.messages[0].original_content_url == "https://example.test/media/line/served.m4a"


@pytest.mark.asyncio
async def test_send_schedules_quota_snapshots_in_background(monkeypatch):
    adapter = LineAdapter(PlatformConfig(enabled=True, token="token", extra={"channel_secret": "secret"}))

    class DummyMessagingApi:
        def push_message(self, request, **kwargs):
            return {"request": request, "kwargs": kwargs}

    adapter._messaging_api = DummyMessagingApi()

    calls = []
    quota_stages = []

    async def fake_call(func, *args, **kwargs):
        calls.append((func.__name__, args[0], kwargs))
        return {"ok": True}

    monkeypatch.setattr(adapter, "_call_sync_api", fake_call)
    monkeypatch.setattr(adapter, "truncate_message", lambda content, max_length: ["hello"])
    monkeypatch.setattr(adapter, "_schedule_quota_snapshot", lambda stage, **metadata: quota_stages.append((stage, metadata)))

    result = await adapter.send(chat_id="U123", content="hello")

    assert result.success is True
    assert [call[0] for call in calls] == ["push_message"]
    assert [stage for stage, _ in quota_stages] == ["before_send", "after_send"]


@pytest.mark.asyncio
async def test_group_multimodal_batch_is_scoped_by_sender():
    adapter = LineAdapter(
        PlatformConfig(
            enabled=True,
            token="token",
            extra={"channel_secret": "secret", "multimodal_grace_period_seconds": 0.01},
        )
    )

    source_a = adapter.build_source(chat_id="Cgroup123", chat_name="Cgroup123", chat_type="group", user_id="Ualpha")
    source_b = adapter.build_source(chat_id="Cgroup123", chat_name="Cgroup123", chat_type="group", user_id="Ubeta")

    event_a = MessageEvent(
        text="[Image]",
        message_type=MessageType.PHOTO,
        source=source_a,
        message_id="img-a",
        media_urls=["/tmp/a.jpg"],
        media_types=["image/jpeg"],
    )
    event_b = MessageEvent(
        text="[Voice message]",
        message_type=MessageType.VOICE,
        source=source_b,
        message_id="audio-b",
        media_urls=["/tmp/b.m4a"],
        media_types=["audio/m4a"],
    )

    captured = []

    async def fake_handle_message(message_event):
        captured.append(message_event)

    adapter.handle_message = fake_handle_message  # type: ignore[method-assign]

    await adapter._enqueue_multimodal_event(event_a)
    await adapter._enqueue_multimodal_event(event_b)
    await asyncio.sleep(0.05)

    assert len(captured) == 2
    assert {event.source.user_id for event in captured} == {"Ualpha", "Ubeta"}
    assert sorted(event.message_id for event in captured) == ["audio-b", "img-a"]


@pytest.mark.asyncio
async def test_keep_typing_accepts_stop_event():
    adapter = LineAdapter(PlatformConfig(enabled=True, token="token", extra={"channel_secret": "secret"}))
    calls = []

    async def fake_send_typing(chat_id, metadata=None):
        calls.append((chat_id, metadata))

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(adapter, "send_typing", fake_send_typing)
    stop_event = asyncio.Event()
    stop_event.set()
    try:
        await adapter._keep_typing("U123", stop_event=stop_event)
    finally:
        monkeypatch.undo()

    assert calls == []
