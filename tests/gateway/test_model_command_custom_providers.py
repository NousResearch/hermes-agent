"""Regression tests for gateway /model support of config.yaml custom_providers."""

from types import SimpleNamespace

import yaml
import pytest

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
async def test_handle_model_command_offloads_text_provider_listing(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-5.4", "provider": "openai-codex"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    calls = []

    async def fake_to_thread(fn, /, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    def fake_list_authenticated_providers(**_kwargs):
        return [{
            "name": "OpenAI Codex",
            "slug": "openai-codex",
            "is_current": True,
            "models": ["gpt-5.4"],
            "total_models": 1,
        }]

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(model_switch, "list_authenticated_providers", fake_list_authenticated_providers)

    result = await _make_runner()._handle_model_command(_make_event())

    assert "OpenAI Codex" in result
    assert "fake_list_authenticated_providers" in calls


@pytest.mark.asyncio
async def test_handle_model_picker_callback_offloads_model_switch(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-5.4", "provider": "openai-codex"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch
    from hermes_cli.model_switch import ModelSwitchResult

    calls = []
    picker_response = {}

    async def fake_to_thread(fn, /, *args, **kwargs):
        calls.append(fn.__name__)
        return fn(*args, **kwargs)

    def fake_list_picker_providers(**_kwargs):
        return [{
            "name": "OpenRouter",
            "slug": "openrouter",
            "is_current": False,
            "models": ["openai/gpt-5.4"],
            "total_models": 1,
        }]

    def fake_switch_model(**_kwargs):
        return ModelSwitchResult(
            success=True,
            new_model="openai/gpt-5.4",
            target_provider="openrouter",
            provider_label="OpenRouter",
        )

    class _PickerAdapter:
        async def send_model_picker(self, **kwargs):
            picker_response["text"] = await kwargs["on_model_selected"](
                kwargs["chat_id"],
                "openai/gpt-5.4",
                "openrouter",
            )
            return SimpleNamespace(success=True)

    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _PickerAdapter()}

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(gateway_run.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(model_switch, "list_picker_providers", fake_list_picker_providers)
    monkeypatch.setattr(model_switch, "switch_model", fake_switch_model)

    result = await runner._handle_model_command(_make_event())

    assert result is None
    assert "fake_list_picker_providers" in calls
    assert "fake_switch_model" in calls
    assert "Model switched to `openai/gpt-5.4`" in picker_response["text"]
