"""Tests for gateway /model interactive picker integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _FakePickerAdapter:
    def __init__(self):
        self._send_model_picker = AsyncMock(side_effect=self._send)
        self.calls = []
        self.on_model_selected = None

    async def send_model_picker(self, **kwargs):
        return await self._send_model_picker(**kwargs)

    async def _send(self, **kwargs):
        self.calls.append(kwargs)
        self.on_model_selected = kwargs["on_model_selected"]
        return SimpleNamespace(success=True)


def _make_runner(platform: Platform):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: _FakePickerAdapter()}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_model_notes = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    return runner


def _make_event(platform: Platform, text: str = "/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=platform, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.DISCORD])
async def test_handle_model_command_uses_platform_native_picker(tmp_path, monkeypatch, platform):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"model": {"default": "gpt-5.4", "provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"}}
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **_kwargs: [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["anthropic/claude-sonnet-4.6"],
                "total_models": 1,
                "is_current": False,
            }
        ],
    )

    runner = _make_runner(platform)
    adapter = runner.adapters[platform]

    result = await runner._handle_model_command(_make_event(platform))

    assert result is None
    adapter._send_model_picker.assert_awaited_once()
    assert adapter.calls[0]["providers"][0]["slug"] == "openrouter"
    assert adapter.calls[0]["current_model"] == "gpt-5.4"
    assert adapter.calls[0]["current_provider"] == "openai-codex"


@pytest.mark.asyncio
async def test_model_picker_callback_uses_shared_switch_pipeline(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"model": {"default": "gpt-5.4", "provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"}}
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **_kwargs: [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["anthropic/claude-sonnet-4.6"],
                "total_models": 1,
                "is_current": False,
            }
        ],
    )

    captured = {}

    def _fake_switch_model(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            new_model=kwargs["raw_input"],
            target_provider=kwargs["explicit_provider"],
            provider_label="OpenRouter",
            api_key="picker-key",
            base_url="",
            api_mode="chat_completions",
            model_info=None,
            warning_message=None,
        )

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch_model)

    runner = _make_runner(Platform.TELEGRAM)
    adapter = runner.adapters[Platform.TELEGRAM]

    await runner._handle_model_command(_make_event(Platform.TELEGRAM))
    confirmation = await adapter.on_model_selected(
        "12345", "anthropic/claude-sonnet-4.6", "openrouter"
    )

    assert captured["raw_input"] == "anthropic/claude-sonnet-4.6"
    assert captured["explicit_provider"] == "openrouter"
    assert captured["current_model"] == "gpt-5.4"
    assert captured["current_provider"] == "openai-codex"
    assert "Model switched to `anthropic/claude-sonnet-4.6`" in confirmation
    session_key = runner._session_key_for_source(_make_event(Platform.TELEGRAM).source)
    assert runner._session_model_overrides[session_key]["model"] == "anthropic/claude-sonnet-4.6"
