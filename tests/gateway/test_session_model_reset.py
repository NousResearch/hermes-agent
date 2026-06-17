"""Tests that /new (and its /reset alias) clears session-scoped overrides.

Also tests that /model consumes was_auto_reset to prevent the next message's
auto-reset cleanup from wiping a fresh model override (issue #48031).
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    runner._session_model_overrides = {}
    runner._session_reasoning_overrides = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()

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
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = None
    runner._agent_cache_lock = None  # disables _evict_cached_agent lock path
    runner._is_user_authorized = lambda _source: True
    runner._format_session_info = lambda: ""

    return runner


@pytest.mark.asyncio
async def test_new_command_clears_session_model_override():
    """/new must remove the session-scoped model override for that session."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    # Simulate a prior /model switch stored as a session override
    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "***",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides
    assert session_key not in runner._pending_model_notes


@pytest.mark.asyncio
async def test_new_command_no_override_is_noop():
    """/new with no prior model override must not raise."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides


@pytest.mark.asyncio
async def test_new_command_only_clears_own_session():
    """/new must only clear the override for the session that triggered it."""
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    other_key = "other_session_key"

    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "",
        "api_mode": "openai",
    }
    runner._session_model_overrides[other_key] = {
        "model": "claude-sonnet-4-6",
        "provider": "anthropic",
        "api_key": "***",
        "base_url": "",
        "api_mode": "anthropic",
    }
    runner._session_reasoning_overrides[session_key] = {"enabled": True, "effort": "high"}
    runner._session_reasoning_overrides[other_key] = {"enabled": True, "effort": "low"}
    runner._pending_model_notes[session_key] = "[Note: switched to gpt-4o.]"
    runner._pending_model_notes[other_key] = "[Note: switched to claude-sonnet-4-6.]"

    await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert other_key in runner._session_model_overrides
    assert session_key not in runner._session_reasoning_overrides
    assert other_key in runner._session_reasoning_overrides
    assert session_key not in runner._pending_model_notes
    assert other_key in runner._pending_model_notes


@pytest.mark.asyncio
async def test_model_command_consumes_was_auto_reset(tmp_path):
    """/model must consume was_auto_reset so the next message's cleanup
    won't wipe the fresh override (issue #48031).

    Scenario:
    1. Session auto-resets → was_auto_reset=True on session entry
    2. /model X arrives → _handle_model_command stores override
    3. Without the fix, was_auto_reset stays True → next regular message
       in _handle_message_with_agent wipes the override at line 8378.
    4. With the fix, _handle_model_command consumes was_auto_reset →
       next message sees False → override preserved.
    """
    from hermes_cli.model_switch import ModelSwitchResult

    runner = _make_runner()
    session_key = build_session_key(_make_source())
    session_entry = runner.session_store.get_or_create_session.return_value

    # Simulate auto-reset: session entry has was_auto_reset=True
    session_entry.was_auto_reset = True
    session_entry.auto_reset_reason = "idle"

    # Set up config file so _handle_model_command can read it
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "model:\n  default: gpt-4o-mini\n  provider: openai\n",
        encoding="utf-8",
    )

    # Mock switch_model to return a successful result
    mock_result = ModelSwitchResult(
        success=True,
        new_model="gpt-4o",
        target_provider="openai",
        api_key="sk-test",
        base_url="",
        api_mode="openai",
        provider_label="OpenAI",
    )

    with (
        patch("hermes_cli.model_switch.switch_model", return_value=mock_result),
        patch("gateway.run._hermes_home", hermes_home),
        patch("gateway.run._load_gateway_config", return_value={
            "model": {"default": "gpt-4o-mini", "provider": "openai"},
        }),
        patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None),
    ):
        event = _make_event("/model gpt-4o")
        await runner._handle_model_command(event)

    # Verify: override was stored
    assert session_key in runner._session_model_overrides
    assert runner._session_model_overrides[session_key]["model"] == "gpt-4o"

    # Verify: was_auto_reset was consumed
    assert session_entry.was_auto_reset is False, (
        "was_auto_reset must be consumed by _handle_model_command so the "
        "next message's auto-reset cleanup won't wipe the fresh override"
    )
