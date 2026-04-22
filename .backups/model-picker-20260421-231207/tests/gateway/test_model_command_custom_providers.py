"""Regression tests for gateway /model support of config.yaml custom_providers."""

import yaml
import pytest
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    return runner


class _FakeTelegramAdapter:
    def __init__(self):
        self.calls = []

    async def send_model_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(success=True, message_id="1")


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


def _provider(slug: str, name: str, total_models: int = 1):
    return {
        "slug": slug,
        "name": name,
        "is_current": False,
        "is_user_defined": False,
        "models": [f"{slug}-model"],
        "total_models": total_models,
        "source": "built-in",
    }


@pytest.mark.asyncio
async def test_handle_model_command_lists_saved_custom_provider(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                },
                "providers": {},
                "custom_providers": [
                    {
                        "name": "Local (127.0.0.1:4141)",
                        "base_url": "http://127.0.0.1:4141/v1",
                        "model": "rotator-openrouter-coding",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    result = await _make_runner()._handle_model_command(_make_event())

    assert result is not None
    assert "Local (127.0.0.1:4141)" in result
    assert "custom:local-(127.0.0.1:4141)" in result
    assert "rotator-openrouter-coding" in result


@pytest.mark.asyncio
async def test_handle_model_command_filters_telegram_picker_providers(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "providers": {},
                "custom_providers": [],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kwargs: [
            _provider("openai-codex", "OpenAI Codex", 3),
            _provider("ollama-cloud", "Ollama Cloud", 5),
            _provider("anthropic", "Anthropic", 7),
        ],
    )

    adapter = _FakeTelegramAdapter()
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: adapter}

    result = await runner._handle_model_command(_make_event())

    assert result is None
    providers = adapter.calls[0]["providers"]
    slugs = [p["slug"] for p in providers]
    assert "openai-codex" not in slugs
    assert "ollama-cloud" not in slugs
    assert "anthropic" in slugs


@pytest.mark.asyncio
async def test_handle_model_command_filters_telegram_fallback_text_list(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                },
                "providers": {},
                "custom_providers": [],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        lambda **kwargs: [
            _provider("openai-codex", "OpenAI Codex", 3),
            _provider("ollama-cloud", "Ollama Cloud", 5),
            _provider("anthropic", "Anthropic", 7),
        ],
    )

    result = await _make_runner()._handle_model_command(_make_event())

    assert result is not None
    assert "OpenAI Codex" not in result
    assert "Ollama Cloud" not in result
    assert "Anthropic" in result
