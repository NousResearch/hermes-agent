"""Regression tests for gateway /model support of config.yaml custom_providers."""

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
async def test_direct_model_switch_offloads_to_thread(tmp_path, monkeypatch):
    """A direct `/model <name>` switch must route switch_model() through
    asyncio.to_thread so the blocking models.dev HTTP fetch can't freeze the
    gateway event loop (#20525)."""
    import asyncio

    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"model": {"default": "gpt-5.4", "provider": "openrouter"}}
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    # Fail the switch so the handler returns before _finish_switch (which needs
    # full runner state) — we only care that the offload happened.
    def _fake_switch(**kwargs):
        return ModelSwitchResult(success=False, error_message="nope")

    monkeypatch.setattr("hermes_cli.model_switch.switch_model", _fake_switch)

    offloaded = []
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(func, /, *args, **kwargs):
        offloaded.append(getattr(func, "__name__", repr(func)))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _spy_to_thread)

    result = await _make_runner()._handle_model_command(_make_event("/model gpt-5.4"))

    # switch_model was offloaded to a worker thread, not run on the event loop.
    assert "_fake_switch" in offloaded
    assert result is not None and "nope" in result
