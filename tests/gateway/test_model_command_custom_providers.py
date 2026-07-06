"""Regression tests for gateway /model support — custom providers and two-step menu."""

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
    runner._model_menu_state = {}
    return runner


def _make_event(text="/model", chat_id="12345"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
    )


def _patch_gateway(monkeypatch, tmp_path, config_dict):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(config_dict), encoding="utf-8"
    )
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    return hermes_home


_CONFIG_WITH_CUSTOM = {
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


@pytest.mark.asyncio
async def test_handle_model_command_lists_saved_custom_provider(tmp_path, monkeypatch):
    """Step 1: /model shows numbered provider list including custom providers."""
    _patch_gateway(monkeypatch, tmp_path, _CONFIG_WITH_CUSTOM)

    runner = _make_runner()
    result = await runner._handle_model_command(_make_event())

    assert result is not None
    # Provider name must appear in the numbered list.
    assert "Local (127.0.0.1:4141)" in result
    # Model preview must appear alongside the provider name.
    assert "rotator-openrouter-coding" in result
    # The response must present a numbered menu, not a raw text dump.
    assert "Select a provider:" in result


@pytest.mark.asyncio
async def test_model_menu_state_saved_after_step1(tmp_path, monkeypatch):
    """After /model, the runner must store picker state for the session."""
    _patch_gateway(monkeypatch, tmp_path, _CONFIG_WITH_CUSTOM)

    runner = _make_runner()
    event = _make_event()
    await runner._handle_model_command(event)

    session_key = runner._session_key_for_source(event.source)
    assert session_key in runner._model_menu_state
    state = runner._model_menu_state[session_key]
    assert state["step"] == "provider"
    assert len(state["providers"]) >= 1
    provider_names = [p["name"] for p in state["providers"]]
    assert "Local (127.0.0.1:4141)" in provider_names


@pytest.mark.asyncio
async def test_model_menu_cancel(tmp_path, monkeypatch):
    """Replying 'cancel' clears the menu state and returns a cancellation message."""
    _patch_gateway(monkeypatch, tmp_path, _CONFIG_WITH_CUSTOM)

    runner = _make_runner()
    step1_event = _make_event()
    await runner._handle_model_command(step1_event)

    session_key = runner._session_key_for_source(step1_event.source)
    assert session_key in runner._model_menu_state

    cancel_reply = await runner._handle_model_menu_reply(
        step1_event, session_key, "cancel"
    )
    assert "cancel" in cancel_reply.lower()
    assert session_key not in runner._model_menu_state


@pytest.mark.asyncio
async def test_model_menu_invalid_number(tmp_path, monkeypatch):
    """Replying with an out-of-range number returns a hint without clearing state."""
    _patch_gateway(monkeypatch, tmp_path, _CONFIG_WITH_CUSTOM)

    runner = _make_runner()
    step1_event = _make_event()
    await runner._handle_model_command(step1_event)

    session_key = runner._session_key_for_source(step1_event.source)
    n = len(runner._model_menu_state[session_key]["providers"])

    reply = await runner._handle_model_menu_reply(
        step1_event, session_key, str(n + 10)
    )
    assert str(n) in reply
    # State must still be active so the user can retry.
    assert session_key in runner._model_menu_state


@pytest.mark.asyncio
async def test_model_menu_step2_shows_models(tmp_path, monkeypatch):
    """Selecting a valid provider number advances to step 2 (model list)."""
    _patch_gateway(monkeypatch, tmp_path, _CONFIG_WITH_CUSTOM)

    runner = _make_runner()
    step1_event = _make_event()
    await runner._handle_model_command(step1_event)

    session_key = runner._session_key_for_source(step1_event.source)
    providers = runner._model_menu_state[session_key]["providers"]
    # Find the index of the custom provider.
    idx = next(
        (i + 1 for i, p in enumerate(providers) if p["name"] == "Local (127.0.0.1:4141)"),
        1,
    )

    step2_reply = await runner._handle_model_menu_reply(
        step1_event, session_key, str(idx)
    )
    # Should show a model list.
    assert "rotator-openrouter-coding" in step2_reply
    assert "Reply with a number" in step2_reply
    # State must advance to step "model".
    assert runner._model_menu_state[session_key]["step"] == "model"
