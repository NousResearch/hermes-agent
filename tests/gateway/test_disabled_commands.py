"""Tests for gateway.disabled_commands — per-platform slash command blocking.

Verifies that commands listed in gateway.disabled_commands /
gateway.platform_disabled_commands are blocked at the dispatch layer
and hidden from help output.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(platform), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_disabled_command_blocked_at_dispatch(tmp_path, monkeypatch):
    """A globally disabled command returns a disabled message."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "gateway:\n  disabled_commands:\n    - yolo\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_runner()
    result = await runner._handle_message(_make_event("/yolo"))
    assert result is not None
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_non_disabled_command_passes_through(tmp_path, monkeypatch):
    """A command not in the disabled list still works normally."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "gateway:\n  disabled_commands:\n    - yolo\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_runner()
    # /help is NOT disabled — should produce help output
    result = await runner._handle_message(_make_event("/help"))
    assert result is not None
    assert "disabled" not in result.lower()


@pytest.mark.asyncio
async def test_platform_specific_disabled(tmp_path, monkeypatch):
    """Platform-specific disabled_commands only block on that platform."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "gateway:\n"
        "  disabled_commands:\n"
        "    - yolo\n"
        "  platform_disabled_commands:\n"
        "    telegram:\n"
        "      - model\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_runner()

    # /model is disabled on telegram
    result = await runner._handle_message(
        _make_event("/model", platform=Platform.TELEGRAM)
    )
    assert result is not None
    assert "disabled" in result.lower()

    # /yolo is globally disabled
    result = await runner._handle_message(
        _make_event("/yolo", platform=Platform.TELEGRAM)
    )
    assert result is not None
    assert "disabled" in result.lower()


@pytest.mark.asyncio
async def test_no_config_allows_all_commands(tmp_path, monkeypatch):
    """With no gateway config, all commands are allowed."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_runner()
    # /yolo should work normally (not blocked)
    result = await runner._handle_message(_make_event("/yolo"))
    # yolo toggle returns a message about YOLO mode, not "disabled"
    if result:
        assert "disabled" not in result.lower()


@pytest.mark.asyncio
async def test_alias_also_blocked(tmp_path, monkeypatch):
    """Disabling a command also blocks its aliases (e.g. /snap -> /snapshot)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "gateway:\n  disabled_commands:\n    - snapshot\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    runner = _make_runner()
    # /snap is an alias for /snapshot
    result = await runner._handle_message(_make_event("/snap"))
    assert result is not None
    assert "disabled" in result.lower()
