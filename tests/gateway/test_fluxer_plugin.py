"""Tests for the Fluxer platform-plugin adapter.

The Fluxer plugin is intentionally text-first here: REST send + Gateway
MESSAGE_CREATE normalization. Rich media can layer on once Fluxer's self-hosting
surface stabilizes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_fluxer = load_plugin_adapter("fluxer")

FluxerAdapter = _fluxer.FluxerAdapter
check_requirements = _fluxer.check_requirements
validate_config = _fluxer.validate_config
is_connected = _fluxer.is_connected
register = _fluxer.register
_env_enablement = _fluxer._env_enablement
_build_identify_payload = _fluxer._build_identify_payload


def test_platform_enum_resolves_via_plugin_scan():
    from gateway.config import Platform

    p = Platform("fluxer")
    assert p.value == "fluxer"
    assert Platform("fluxer") is p


def test_check_requirements_needs_token_only(monkeypatch):
    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert check_requirements() is False


def test_check_requirements_true_when_token_configured(monkeypatch):
    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "app.secret")

    assert check_requirements() is True


def test_validate_config_uses_env_or_extra(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert validate_config(PlatformConfig(enabled=True)) is False
    assert validate_config(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret"},
        )
    ) is True
    assert validate_config(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    ) is True


def test_is_connected_mirrors_validate(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert is_connected(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret"},
        )
    ) is True
    assert is_connected(PlatformConfig(enabled=True)) is False


def test_env_enablement_none_when_unset(monkeypatch):
    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert _env_enablement() is None


def test_env_enablement_seeds_config(monkeypatch):
    monkeypatch.setenv("FLUXER_BASE_URL", "https://fluxer.example/")
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "app.secret")
    monkeypatch.setenv("FLUXER_HOME_CHANNEL", "chan-1")
    monkeypatch.setenv("FLUXER_HOME_CHANNEL_NAME", "Fluxer Home")

    seed = _env_enablement()

    assert seed["base_url"] == "https://fluxer.example/"
    assert seed["bot_token"] == "app.secret"
    assert seed["home_channel"] == {"chat_id": "chan-1", "name": "Fluxer Home"}


def test_adapter_init_and_platform_identity(monkeypatch):
    from gateway.config import Platform, PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example/", "bot_token": "app.secret"},
        )
    )

    assert adapter.base_url == "https://fluxer.example"
    assert adapter.api_base_url == "https://fluxer.example/api"
    assert adapter.bot_token == "app.secret"
    assert adapter.platform is Platform("fluxer")
    assert adapter._running is False


def test_adapter_defaults_to_official_hosted_api(monkeypatch):
    from gateway.config import PlatformConfig

    monkeypatch.delenv("FLUXER_BASE_URL", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret"},
        )
    )

    assert adapter.base_url == "https://api.fluxer.app/v1"
    assert adapter.api_base_url == "https://api.fluxer.app/v1"
    assert adapter.bot_token == "app.secret"


def test_build_identify_payload_contains_bot_token_and_client_properties():
    payload = _build_identify_payload("app.secret")

    assert payload["op"] == 2
    assert payload["d"]["token"] == "app.secret"
    assert payload["d"]["properties"]["browser"] == "hermes"
    assert payload["d"]["properties"]["device"] == "hermes"


@pytest.mark.asyncio
async def test_send_posts_channel_message_and_reply_reference(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(return_value={"id": "msg-1"})

    result = await adapter.send("chan-1", "Hello, Fluxer!", reply_to="msg-0")

    adapter._request.assert_awaited_once_with(
        "POST",
        "/channels/chan-1/messages",
        json={"content": "Hello, Fluxer!", "message_reference": {"message_id": "msg-0"}},
    )
    assert result.success is True
    assert result.message_id == "msg-1"


@pytest.mark.asyncio
async def test_send_splits_long_channel_messages(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(side_effect=[{"id": "msg-1"}, {"id": "msg-2"}])

    result = await adapter.send("chan-1", "a" * 4500, reply_to="msg-0")

    assert result.success is True
    assert result.message_id == "msg-1"
    assert result.raw_response["message_ids"] == ["msg-1", "msg-2"]
    assert adapter._request.await_count == 2
    first = adapter._request.await_args_list[0]
    second = adapter._request.await_args_list[1]
    assert first.kwargs["json"]["message_reference"] == {"message_id": "msg-0"}
    assert len(first.kwargs["json"]["content"]) <= 4000
    assert "message_reference" not in second.kwargs["json"]
    assert len(second.kwargs["json"]["content"]) <= 4000


@pytest.mark.asyncio
async def test_send_image_file_posts_multipart_payload(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig

    image = tmp_path / "zofka.png"
    image.write_bytes(b"png-data")
    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._multipart_request = AsyncMock(return_value={"id": "msg-img"})

    result = await adapter.send_image_file("chan-1", str(image), caption="look")

    assert result.success is True
    assert result.message_id == "msg-img"
    adapter._multipart_request.assert_awaited_once()
    call = adapter._multipart_request.await_args
    assert call is not None
    _, path = call.args
    kwargs = call.kwargs
    assert path == "/channels/chan-1/messages"
    assert kwargs["payload"]["content"] == "look"
    assert kwargs["payload"]["attachments"] == [{"id": 0, "filename": "zofka.png", "title": "zofka.png"}]
    assert kwargs["files"] == [("files[0]", image.resolve(), "zofka.png")]


@pytest.mark.asyncio
async def test_send_voice_marks_fluxer_voice_message(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig

    audio = tmp_path / "reply.mp3"
    audio.write_bytes(b"mp3-data")
    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._multipart_request = AsyncMock(return_value={"id": "msg-voice"})

    result = await adapter.send_voice("chan-1", str(audio), caption="spoken", duration=3, waveform="AAAA")

    assert result.success is True
    call = adapter._multipart_request.await_args
    assert call is not None
    kwargs = call.kwargs
    assert "content" not in kwargs["payload"]
    assert kwargs["payload"]["flags"] == 1 << 13
    assert kwargs["payload"]["attachments"] == [
        {"id": 0, "filename": "reply.mp3", "title": "reply.mp3", "duration": 3, "waveform": "AAAA"}
    ]


@pytest.mark.asyncio
async def test_standalone_send_routes_audio_as_voice(tmp_path, monkeypatch):
    from gateway.config import PlatformConfig

    audio = tmp_path / "reply.mp3"
    audio.write_bytes(b"mp3-data")
    pconfig = PlatformConfig(
        enabled=True,
        extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
    )

    with patch.object(FluxerAdapter, "send_voice", new=AsyncMock(return_value=_fluxer.SendResult(success=True, message_id="msg-voice"))) as send_voice:
        result = await _fluxer._standalone_send(pconfig, "chan-1", "", media_files=[str(audio)])

    assert result == {"success": True, "platform": "fluxer", "chat_id": "chan-1", "message_id": "msg-voice"}
    send_voice.assert_awaited_once()
    call = send_voice.await_args
    assert call is not None
    assert call.args[:2] == ("chan-1", str(audio))


@pytest.mark.asyncio
async def test_send_reports_retryable_error_on_request_failure(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter._request = AsyncMock(side_effect=RuntimeError("network down"))

    result = await adapter.send("chan-1", "Hello")

    assert result.success is False
    assert "network down" in result.error
    assert result.retryable is True


@pytest.mark.asyncio
async def test_message_create_dispatches_normalized_event(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "s": 42,
            "d": {
                "id": "msg-1",
                "channel_id": "chan-1",
                "channel_type": "dm",
                "content": "morning",
                "author": {"id": "user-1", "username": "Elkim", "bot": False},
                "guild_id": None,
            },
        }
    )

    assert len(seen) == 1
    event = seen[0]
    assert event.text == "morning"
    assert event.message_id == "msg-1"
    assert event.source.chat_id == "chan-1"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "user-1"
    assert event.source.user_name == "Elkim"
    assert event.source.message_id == "msg-1"


@pytest.mark.asyncio
async def test_message_create_dispatches_image_attachment(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    with patch.object(
        _fluxer,
        "cache_image_from_url",
        new=AsyncMock(return_value="/tmp/fluxer-image.jpg"),
    ) as cache_image:
        await adapter._handle_gateway_dispatch(
            {
                "op": 0,
                "t": "MESSAGE_CREATE",
                "s": 43,
                "d": {
                    "id": "msg-img",
                    "channel_id": "chan-1",
                    "content": "",
                    "author": {"id": "user-1", "username": "Elkim", "bot": False},
                    "attachments": [
                        {
                            "id": "att-1",
                            "filename": "photo.jpg",
                            "content_type": "image/jpeg",
                            "url": "https://fluxerusercontent.com/attachments/chan-1/att-1/photo.jpg",
                        }
                    ],
                },
            }
        )

    assert len(seen) == 1
    event = seen[0]
    assert event.text == ""
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls == ["/tmp/fluxer-image.jpg"]
    assert event.media_types == ["image/jpeg"]
    cache_image.assert_awaited_once_with(
        "https://fluxerusercontent.com/attachments/chan-1/att-1/photo.jpg",
        ".jpg",
    )


@pytest.mark.asyncio
async def test_message_create_dispatches_voice_attachment_when_flagged(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    with patch.object(
        _fluxer,
        "cache_audio_from_url",
        new=AsyncMock(return_value="/tmp/fluxer-voice.mp3"),
    ):
        await adapter._handle_gateway_dispatch(
            {
                "op": 0,
                "t": "MESSAGE_CREATE",
                "s": 44,
                "d": {
                    "id": "msg-voice",
                    "channel_id": "chan-1",
                    "content": "",
                    "flags": 1 << 13,
                    "author": {"id": "user-1", "username": "Elkim", "bot": False},
                    "attachments": [
                        {
                            "id": "att-voice",
                            "filename": "voice.mp3",
                            "content_type": "audio/mpeg",
                            "url": "https://fluxerusercontent.com/attachments/chan-1/att-voice/voice.mp3",
                        }
                    ],
                },
            }
        )

    assert len(seen) == 1
    event = seen[0]
    assert event.message_type == MessageType.VOICE
    assert event.media_urls == ["/tmp/fluxer-voice.mp3"]
    assert event.media_types == ["audio/mpeg"]


@pytest.mark.asyncio
async def test_message_create_ignores_own_bot_messages(monkeypatch):
    from gateway.config import PlatformConfig

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"base_url": "https://fluxer.example", "bot_token": "app.secret"},
        )
    )
    adapter.bot_user_id = "bot-1"
    adapter.handle_message = AsyncMock()

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "MESSAGE_CREATE",
            "d": {
                "id": "msg-1",
                "channel_id": "chan-1",
                "content": "echo",
                "author": {"id": "bot-1", "username": "Žofka", "bot": True},
            },
        }
    )

    adapter.handle_message.assert_not_awaited()


def test_register_metadata():
    calls = []

    class Ctx:
        def register_platform(self, **kwargs):
            calls.append(kwargs)

    register(Ctx())

    assert len(calls) == 1
    entry = calls[0]
    assert entry["name"] == "fluxer"
    assert entry["label"] == "Fluxer"
    assert entry["required_env"] == ["FLUXER_BOT_TOKEN"]
    assert entry["allowed_users_env"] == "FLUXER_ALLOWED_USERS"
    assert entry["allow_all_env"] == "FLUXER_ALLOW_ALL_USERS"
    assert entry["cron_deliver_env_var"] == "FLUXER_HOME_CHANNEL"
    assert entry["standalone_sender_fn"] is not None
    assert entry["max_message_length"] >= 2000
