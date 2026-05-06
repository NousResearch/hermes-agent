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
async def test_model_text_list_uses_gateway_executor(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "gpt-5.4", "provider": "openai"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        model_switch,
        "list_authenticated_providers",
        lambda **_kwargs: [
            {
                "name": "OpenAI",
                "slug": "openai",
                "models": ["gpt-5.4"],
                "total_models": 1,
                "is_current": True,
            }
        ],
    )

    runner = _make_runner()
    calls = []

    async def fake_executor(func, *args):
        calls.append(func)
        return func(*args)

    runner._run_in_executor_with_context = fake_executor

    result = await runner._handle_model_command(_make_event("/model"))

    assert calls
    assert result is not None
    assert "**OpenAI**" in result


@pytest.mark.asyncio
async def test_model_switch_uses_gateway_executor(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "old-model", "provider": "openai"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        model_switch,
        "switch_model",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            new_model="gpt-5.4",
            target_provider="openai",
            api_key="",
            base_url="",
            api_mode="chat_completions",
            provider_label="OpenAI",
            model_info=None,
            warning_message=None,
        ),
    )
    monkeypatch.setattr(
        model_switch,
        "resolve_display_context_length",
        lambda *args, **kwargs: None,
    )

    runner = _make_runner()
    calls = []

    async def fake_executor(func, *args):
        calls.append(func)
        return func(*args)

    runner._run_in_executor_with_context = fake_executor

    result = await runner._handle_model_command(_make_event("/model gpt-5.4 --provider openai"))

    assert calls
    assert result is not None
    assert "Model switched to `gpt-5.4`" in result


class _PickerAdapter:
    def __init__(self):
        self.callback_response = None

    async def send_model_picker(
        self,
        *,
        chat_id,
        providers,
        current_model,
        current_provider,
        session_key,
        on_model_selected,
        metadata=None,
    ):
        self.callback_response = await on_model_selected(chat_id, "gpt-5.4", "openai")
        return SimpleNamespace(success=True)


@pytest.mark.asyncio
async def test_interactive_model_picker_switch_uses_gateway_executor(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "old-model", "provider": "openai"}}),
        encoding="utf-8",
    )

    import gateway.run as gateway_run
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr(
        model_switch,
        "list_picker_providers",
        lambda **_kwargs: [
            {
                "name": "OpenAI",
                "slug": "openai",
                "models": ["gpt-5.4"],
                "total_models": 1,
                "is_current": True,
            }
        ],
    )
    monkeypatch.setattr(
        model_switch,
        "switch_model",
        lambda **_kwargs: SimpleNamespace(
            success=True,
            new_model="gpt-5.4",
            target_provider="openai",
            api_key="",
            base_url="",
            api_mode="chat_completions",
            provider_label="OpenAI",
            model_info=None,
            warning_message=None,
        ),
    )
    monkeypatch.setattr(
        model_switch,
        "resolve_display_context_length",
        lambda *args, **kwargs: None,
    )

    runner = _make_runner()
    adapter = _PickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    calls = []

    async def fake_executor(func, *args):
        calls.append(func)
        return func(*args)

    runner._run_in_executor_with_context = fake_executor

    result = await runner._handle_model_command(_make_event("/model"))

    assert result is None
    assert len(calls) == 2
    assert adapter.callback_response is not None
    assert "Model switched to `gpt-5.4`" in adapter.callback_response
