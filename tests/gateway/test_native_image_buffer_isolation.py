import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._decide_image_input_mode = lambda: "native"
    return runner


def _source(chat_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="private",
        user_name=f"user-{chat_id}",
    )


def _image_event(source: SessionSource, path: str) -> MessageEvent:
    return MessageEvent(
        text="see image",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=[path],
        media_types=["image/png"],
    )


@pytest.mark.asyncio
async def test_native_image_buffer_isolated_per_session():
    runner = _make_runner()
    source_a = _source("chat-a")
    source_b = _source("chat-b")

    await runner._prepare_inbound_message_text(
        event=_image_event(source_a, "/tmp/a.png"),
        source=source_a,
        history=[],
    )
    await runner._prepare_inbound_message_text(
        event=_image_event(source_b, "/tmp/b.png"),
        source=source_b,
        history=[],
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source_a)) == ["/tmp/a.png"]
    assert runner._consume_pending_native_image_paths(build_session_key(source_b)) == ["/tmp/b.png"]


@pytest.mark.asyncio
async def test_native_image_buffer_not_cleared_by_other_sessions_without_images():
    runner = _make_runner()
    source_a = _source("chat-a")
    source_b = _source("chat-b")

    await runner._prepare_inbound_message_text(
        event=_image_event(source_a, "/tmp/a.png"),
        source=source_a,
        history=[],
    )
    await runner._prepare_inbound_message_text(
        event=MessageEvent(text="plain text", source=source_b),
        source=source_b,
        history=[],
    )

    assert runner._consume_pending_native_image_paths(build_session_key(source_a)) == ["/tmp/a.png"]
    assert runner._consume_pending_native_image_paths(build_session_key(source_b)) == []


@pytest.mark.asyncio
async def test_kimi_coding_photo_uses_text_path_not_native_buffer():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "kimi-k2.6"
    runner._base_url = None
    runner._decide_image_input_mode = lambda: "text"
    runner._enrich_message_with_vision = AsyncMock(return_value="vision summary")

    source = _source("chat-kimi")
    result = await runner._prepare_inbound_message_text(
        event=_image_event(source, "/tmp/kimi.png"),
        source=source,
        history=[],
    )

    assert result == "vision summary"
    runner._enrich_message_with_vision.assert_awaited_once_with(
        "see image",
        ["/tmp/kimi.png"],
    )
    assert runner._consume_pending_native_image_paths(build_session_key(source)) == []


@pytest.mark.asyncio
async def test_real_image_mode_decision_uses_text_for_custom_kimi_coding_endpoint(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )

    cfg = {
        "model": {
            "provider": "custom",
            "base_url": "https://api.kimi.com/coding/v1",
        },
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "custom")
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "kimi-k2.6")
    monkeypatch.setattr("agent.image_routing._lookup_supports_vision", lambda provider, model: True)

    assert runner._decide_image_input_mode() == "text"


@pytest.mark.asyncio
async def test_real_image_mode_decision_uses_text_for_named_custom_kimi_coding_endpoint(monkeypatch):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )

    cfg = {
        "model": {
            "provider": "moonshot-proxy",
        },
        "custom_providers": [
            {
                "name": "Moonshot Proxy",
                "base_url": "https://api.kimi.com/coding",
            },
        ],
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)
    monkeypatch.setattr("agent.auxiliary_client._read_main_provider", lambda: "moonshot-proxy")
    monkeypatch.setattr("agent.auxiliary_client._read_main_model", lambda: "kimi-k2.6")
    monkeypatch.setattr("agent.image_routing._lookup_supports_vision", lambda provider, model: True)

    assert runner._decide_image_input_mode() == "text"
