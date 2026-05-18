"""Tests for the gateway /footer command."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/footer", platform=Platform.TELEGRAM) -> MessageEvent:
    source = SessionSource(
        platform=platform,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _write_config(path, enabled):
    path.write_text(
        yaml.safe_dump(
            {
                "display": {
                    "runtime_footer": {
                        "enabled": enabled,
                        "fields": ["model", "context_pct", "cwd"],
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


class TestFooterCommand:
    @pytest.mark.asyncio
    async def test_status_reports_current_state_without_toggling(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        _write_config(config_path, enabled=False)

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_footer_command(_make_event("/footer status"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["runtime_footer"]["enabled"] is False
        assert "OFF" in result
        assert "model, context_pct, cwd" in result
        assert "telegram" in result.lower()

    @pytest.mark.asyncio
    async def test_off_does_not_flip_disabled_footer_back_on(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        _write_config(config_path, enabled=False)

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_footer_command(_make_event("/footer off"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["runtime_footer"]["enabled"] is False
        assert "OFF" in result

    @pytest.mark.asyncio
    async def test_on_enables_footer_persistently(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        _write_config(config_path, enabled=False)

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_footer_command(_make_event("/footer on"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["runtime_footer"]["enabled"] is True
        assert "ON" in result

    @pytest.mark.asyncio
    async def test_bare_footer_toggles_current_state(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        _write_config(config_path, enabled=True)

        monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

        runner = _make_runner()
        result = await runner._handle_footer_command(_make_event("/footer"))

        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert saved["display"]["runtime_footer"]["enabled"] is False
        assert "OFF" in result
