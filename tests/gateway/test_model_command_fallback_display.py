"""Regression tests for /model command showing the active fallback model.

When a fallback model is actively handling a session (e.g., after a 429 on
the primary), ``/model`` should display the runtime model — not the config
default.  This behaviour was restored after being lost in the v0.15.0 refactor
(PR #1660 → issue #45970).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner(**overrides):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    for k, v in overrides.items():
        setattr(runner, k, v)
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"
        ),
    )


def _fake_agent(model="gpt-4o", provider="openrouter", fallback_activated=False):
    """Minimal agent stub with fallback state."""
    return SimpleNamespace(
        model=model,
        provider=provider,
        _fallback_activated=fallback_activated,
    )


def _setup_config(tmp_path, monkeypatch, model_default="gpt-5.5", provider="openrouter"):
    """Write a config.yaml and patch _hermes_home."""
    import yaml
    import gateway.run as gateway_run

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    cfg_path = hermes_home / "config.yaml"
    cfg_path.write_text(
        yaml.safe_dump({"model": {"default": model_default, "provider": provider}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return hermes_home


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_command_shows_fallback_model_when_active(tmp_path, monkeypatch):
    """When a fallback model is active, /model should show the runtime model."""
    _setup_config(tmp_path, monkeypatch, model_default="gpt-5.5")

    runner = _make_runner()
    session_key = "telegram:12345"
    fallback_agent = _fake_agent(
        model="claude-sonnet-4", provider="anthropic", fallback_activated=True
    )
    runner._running_agents = {session_key: fallback_agent}

    # Stub _session_key_for_source and _normalize_source_for_session_key
    monkeypatch.setattr(
        runner, "_session_key_for_source", lambda source: session_key
    )
    monkeypatch.setattr(
        runner, "_normalize_source_for_session_key", lambda source: source
    )
    monkeypatch.setattr(
        runner, "_session_key_for_source", lambda source: session_key
    )

    result = await runner._handle_model_command(_make_event())

    assert result is not None
    assert "claude-sonnet-4" in result
    assert "fallback" in result.lower()


@pytest.mark.asyncio
async def test_model_command_shows_config_model_when_no_fallback(tmp_path, monkeypatch):
    """Without fallback, /model shows the config default."""
    _setup_config(tmp_path, monkeypatch, model_default="gpt-5.5")

    runner = _make_runner()
    session_key = "telegram:12345"
    # Agent exists but fallback is NOT active
    normal_agent = _fake_agent(
        model="gpt-5.5", provider="openrouter", fallback_activated=False
    )
    runner._running_agents = {session_key: normal_agent}

    monkeypatch.setattr(
        runner, "_normalize_source_for_session_key", lambda source: source
    )
    monkeypatch.setattr(
        runner, "_session_key_for_source", lambda source: session_key
    )

    result = await runner._handle_model_command(_make_event())

    assert result is not None
    assert "gpt-5.5" in result
    assert "fallback" not in result.lower()


@pytest.mark.asyncio
async def test_model_command_ignores_pending_sentinel(tmp_path, monkeypatch):
    """When the running agent is _PENDING_SENTINEL, don't crash."""
    _setup_config(tmp_path, monkeypatch, model_default="gpt-5.5")

    runner = _make_runner()
    session_key = "telegram:12345"
    runner._running_agents = {session_key: _AGENT_PENDING_SENTINEL}

    monkeypatch.setattr(
        runner, "_normalize_source_for_session_key", lambda source: source
    )
    monkeypatch.setattr(
        runner, "_session_key_for_source", lambda source: session_key
    )

    result = await runner._handle_model_command(_make_event())

    assert result is not None
    assert "gpt-5.5" in result
    assert "fallback" not in result.lower()


@pytest.mark.asyncio
async def test_model_command_no_running_agent_shows_config(tmp_path, monkeypatch):
    """When no running agent exists, /model shows the config default."""
    _setup_config(tmp_path, monkeypatch, model_default="gpt-5.5")

    runner = _make_runner()
    # No running agents at all

    session_key = "telegram:12345"
    monkeypatch.setattr(
        runner, "_normalize_source_for_session_key", lambda source: source
    )
    monkeypatch.setattr(
        runner, "_session_key_for_source", lambda source: session_key
    )

    result = await runner._handle_model_command(_make_event())

    assert result is not None
    assert "gpt-5.5" in result
    assert "fallback" not in result.lower()


@pytest.mark.asyncio
async def test_model_command_fallback_same_model_no_indicator(tmp_path, monkeypatch):
    """When fallback model equals config model, no fallback indicator shown."""
    _setup_config(tmp_path, monkeypatch, model_default="gpt-5.5")

    runner = _make_runner()
    session_key = "telegram:12345"
    # Fallback active but model is the same as config
    agent = _fake_agent(
        model="gpt-5.5", provider="openrouter", fallback_activated=True
    )
    runner._running_agents = {session_key: agent}

    monkeypatch.setattr(
        runner, "_normalize_source_for_session_key", lambda source: source
    )
    monkeypatch.setattr(
        runner, "_session_key_for_source", lambda source: session_key
    )

    result = await runner._handle_model_command(_make_event())

    assert result is not None
    assert "gpt-5.5" in result
    assert "fallback" not in result.lower()
