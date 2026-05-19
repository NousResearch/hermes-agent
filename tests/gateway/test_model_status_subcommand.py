"""Regression test: /model status should show current model, not switch to 'status'."""

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


def _make_event(text="/model status"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_model_status_shows_info_instead_of_switching(tmp_path, monkeypatch):
    """'/model status' should display current model info, not switch to a model named 'status'."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gemini-3.1-pro-preview",
                    "provider": "custom",
                    "base_url": "http://127.0.0.1:8317/v1",
                },
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event("/model status"))

    # Should return status text (not None which means picker was sent)
    # and should NOT have stored a session override for "status"
    session_key = runner._session_key_for_source(
        SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    )
    override = runner._session_model_overrides.get(session_key, {})
    assert override.get("model") != "status", (
        "/model status should not switch to a model named 'status'"
    )


@pytest.mark.asyncio
async def test_model_info_shows_info_instead_of_switching(tmp_path, monkeypatch):
    """'/model info' should display current model info, not switch to a model named 'info'."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openrouter",
                },
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event("/model info"))

    session_key = runner._session_key_for_source(
        SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    )
    override = runner._session_model_overrides.get(session_key, {})
    assert override.get("model") != "info", (
        "/model info should not switch to a model named 'info'"
    )
