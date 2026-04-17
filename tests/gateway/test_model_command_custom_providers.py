"""Regression tests for gateway /model support of config.yaml custom_providers."""

import yaml
import pytest

from agent.models_dev import ModelInfo
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.model_switch import ModelSwitchResult


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


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
async def test_handle_model_command_keeps_model_info_metadata_when_context_override(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                }
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.parse_model_flags",
        lambda _raw: ("qwen3.6", "", False),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: ModelSwitchResult(
            success=True,
            new_model="qwen3.6",
            target_provider="openrouter",
            provider_label="OpenRouter",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            model_info=ModelInfo(
                id="qwen/qwen3.6",
                name="Qwen 3.6",
                family="qwen3.6",
                provider_id="qwen",
                context_window=128_000,
                max_output=8_192,
                tool_call=True,
            ),
            context_length=256_000,
        ),
    )

    result = await _make_runner()._handle_model_command(_make_event("/model qwen3.6"))

    assert result is not None
    assert "Context: 256,000 tokens" in result
    assert "Max output: 8,192 tokens" in result
    assert "Capabilities: tools" in result
