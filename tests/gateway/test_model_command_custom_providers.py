"""Regression tests for gateway /model support of config.yaml custom_providers."""

import types

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


# ---------------------------------------------------------------------------
# /model configured + model.picker_configured_only threading (#13796)
# ---------------------------------------------------------------------------


def _write_picker_config(home, *, picker_configured_only=False):
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openrouter",
                    **(
                        {"picker_configured_only": True}
                        if picker_configured_only
                        else {}
                    ),
                },
                "providers": {},
                "custom_providers": [],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_handle_model_command_text_list_forwards_configured_only_flag(
    tmp_path, monkeypatch
):
    """The text-list fallback (no platform picker) must forward the config
    flag into ``list_authenticated_providers``."""
    hermes_home = tmp_path / ".hermes"
    _write_picker_config(hermes_home, picker_configured_only=True)

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    captured: dict = {}

    def _fake_list_authenticated_providers(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        _fake_list_authenticated_providers,
    )

    await _make_runner()._handle_model_command(_make_event())

    assert captured["picker_configured_only"] is True


@pytest.mark.asyncio
async def test_handle_model_command_picker_forwards_configured_only_flag(
    tmp_path, monkeypatch
):
    """When the platform supports an interactive picker, the same flag must
    flow into the picker query."""
    hermes_home = tmp_path / ".hermes"
    _write_picker_config(hermes_home, picker_configured_only=True)

    class _PickerAdapter:
        async def send_model_picker(self, **kwargs):
            return types.SimpleNamespace(success=True)

    import gateway.run as gateway_run

    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: _PickerAdapter()}

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    captured: dict = {}

    def _fake_list_authenticated_providers(**kwargs):
        captured.update(kwargs)
        return [{"slug": "anthropic", "name": "Anthropic", "models": []}]

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        _fake_list_authenticated_providers,
    )

    result = await runner._handle_model_command(_make_event())

    assert result is None  # picker took over the response
    assert captured["picker_configured_only"] is True


@pytest.mark.asyncio
async def test_handle_model_command_configured_pseudo_arg_overrides_config(
    tmp_path, monkeypatch
):
    """Even with the config flag off, ``/model configured`` should still
    surface only configured rows for that one invocation."""
    hermes_home = tmp_path / ".hermes"
    _write_picker_config(hermes_home, picker_configured_only=False)

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    captured: dict = {}

    def _fake_list_authenticated_providers(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers",
        _fake_list_authenticated_providers,
    )

    await _make_runner()._handle_model_command(_make_event(text="/model configured"))

    assert captured["picker_configured_only"] is True
