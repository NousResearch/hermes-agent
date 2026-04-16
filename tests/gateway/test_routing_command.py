"""Tests for gateway /routing command and per-session overrides."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._session_model_overrides = {}
    runner._session_smart_routing_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._smart_model_routing = {
        "enabled": True,
        "max_simple_chars": 160,
        "max_simple_words": 28,
        "cheap_model": {
            "provider": "nous",
            "model": "google/gemini-3-flash-preview",
        },
    }
    session_key = build_session_key(_make_source())
    session_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.reset_session.return_value = session_entry
    runner.session_store._entries = {session_key: session_entry}
    runner.session_store._generate_session_key.return_value = session_key
    return runner


@pytest.mark.asyncio
async def test_routing_off_sets_session_override():
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    result = await runner._handle_routing_command(_make_event("/routing off"))

    assert runner._session_smart_routing_overrides[session_key] is False
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_routing_default_clears_session_override():
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_smart_routing_overrides[session_key] = False

    result = await runner._handle_routing_command(_make_event("/routing default"))

    assert session_key not in runner._session_smart_routing_overrides
    assert "override cleared" in result.lower()


@pytest.mark.asyncio
async def test_routing_status_reports_effective_state(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "smart_model_routing": {
                "enabled": True,
                "max_simple_chars": 160,
                "max_simple_words": 28,
                "cheap_model": {
                    "provider": "nous",
                    "model": "google/gemini-3-flash-preview",
                },
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_smart_routing_overrides[session_key] = False

    result = await runner._handle_routing_command(_make_event("/routing status"))

    assert "**Global:** `on`" in result
    assert "**This session:** `off`" in result
    assert "**Effective:** `off`" in result


@pytest.mark.asyncio
async def test_routing_global_persists_to_config(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "smart_model_routing": {
                "enabled": True,
                "max_simple_chars": 160,
                "max_simple_words": 28,
                "cheap_model": {
                    "provider": "nous",
                    "model": "google/gemini-3-flash-preview",
                },
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    runner._session_smart_routing_overrides[session_key] = False

    result = await runner._handle_routing_command(_make_event("/routing on --global"))
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert saved["smart_model_routing"]["enabled"] is True
    assert session_key not in runner._session_smart_routing_overrides
    assert "saved to config" in result.lower()
