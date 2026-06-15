"""Tests for the gateway /apikey slash command."""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(tmp_path=None, monkeypatch=None):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._running_agents = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = MagicMock()
    runner._evict_cached_agent = MagicMock()
    if tmp_path is not None and monkeypatch is not None:
        import gateway.run as gateway_run

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        env_path = hermes_home / ".env"
        env_path.write_text("", encoding="utf-8")
        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
        monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    return runner


def _session_key():
    return "agent:main:telegram:dm:12345"


def _make_event(text):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
    )


@pytest.mark.asyncio
async def test_apikey_no_args_shows_status(monkeypatch, tmp_path):
    runner = _make_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "claude-sonnet-4", "provider": "openrouter"}},
    )
    monkeypatch.setattr(
        "hermes_cli.apikey_switch.resolve_current_key", lambda _p: "sk-abc123"
    )

    result = await runner._handle_apikey_command(_make_event("/apikey"))

    assert "Provider: openrouter" in result
    assert "Key:" in result


@pytest.mark.asyncio
async def test_apikey_hotswap_updates_override(monkeypatch, tmp_path):
    runner = _make_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "claude-sonnet-4", "provider": "openrouter"}},
    )
    agent = MagicMock()
    key = _session_key()
    runner._running_agents[key] = agent

    result = await runner._handle_apikey_command(_make_event("/apikey sk-new-key"))

    assert result is not None
    assert "✓" in result
    agent.switch_api_key.assert_called_once_with("sk-new-key", provider="openrouter")
    assert runner._session_model_overrides[key]["api_key"] == "sk-new-key"
    runner._evict_cached_agent.assert_called_once_with(key)


@pytest.mark.asyncio
async def test_apikey_save_persists_to_env(tmp_path, monkeypatch):
    runner = _make_runner(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"model": {"default": "claude-sonnet-4", "provider": "openrouter"}},
    )

    result = await runner._handle_apikey_command(_make_event("/apikey --save sk-saved-key"))

    assert "Saved to OPENROUTER_API_KEY" in result
    assert "OPENROUTER_API_KEY=sk-saved-key" in (tmp_path / ".hermes" / ".env").read_text(encoding="utf-8")
