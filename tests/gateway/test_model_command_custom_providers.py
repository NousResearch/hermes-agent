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
async def test_handle_model_command_reports_context_override_mismatch(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "kimi-k2.5",
                    "provider": "moonshotai",
                    "base_url": "https://api.moonshot.ai/v1",
                    "context_length": 256000,
                }
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kwargs: ModelSwitchResult(
            success=True,
            new_model="glm-5-turbo",
            target_provider="zai",
            provider_changed=True,
            api_key="new-key",
            base_url="https://api.z.ai/v1",
            api_mode="chat_completions",
            provider_label="Z.AI",
            model_info=ModelInfo(
                id="glm-5-turbo",
                name="GLM-5 Turbo",
                family="glm-5",
                provider_id="zai",
                context_window=202752,
            ),
            is_global=True,
        ),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.get_context_length_mismatch_warning",
        lambda result: "config.yaml context_length (256,000) doesn't match this model.",
    )
    monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

    runner = _make_runner()
    runner._session_key_for_source = lambda source: "sess-1"
    runner._evict_cached_agent = lambda session_key: None
    runner._pending_model_notes = {}

    result = await runner._handle_model_command(_make_event("/model glm-5-turbo --global"))

    assert result is not None
    assert "Model switched to `glm-5-turbo`" in result
    assert "Warning: config.yaml context_length (256,000) doesn't match this model." in result
    assert "Saved to config.yaml (`--global`)" in result
