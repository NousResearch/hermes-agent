"""Tests that /new (and its /reset alias) clears session-scoped overrides."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
async def test_auto_context_refresh_rotates_after_completed_work(monkeypatch):
    """Auto context refresh should run only after a completed agent turn."""
    from gateway.run import GatewayRunner

    runner = _make_runner()
    runner._evict_cached_agent = MagicMock()
    runner._cleanup_agent_resources = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    runner._thread_metadata_for_source = MagicMock(return_value=None)
    runner._reply_anchor_for_event = MagicMock(return_value="m1")
    runner._queued_events = {}

    session_key = build_session_key(_make_source())
    old_entry = runner.session_store._entries[session_key]
    new_entry = SessionEntry(
        session_key=session_key,
        session_id="sess-2",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.reset_session.return_value = new_entry
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "context_refresh": {
                "mode": "auto_new",
                "auto_new_policy": "phase_boundary",
                "require_no_running_processes": False,
            }
        },
    )

    response = await GatewayRunner._maybe_auto_new_after_context_refresh(
        runner,
        _make_event("hello"),
        {
            "final_response": "Phase 1 is complete. Next phase will begin with validation.",
            "completed": True,
            "pending_context_refresh": {
                "mode": "auto_new",
                "session_id": "sess-1",
                "handoff_path": "/tmp/handoff.md",
            },
        },
        "done",
        session_key,
        old_entry,
    )

    runner.session_store.reset_session.assert_called_once_with(session_key)
    assert session_key in runner._pending_context_refresh_notes
    assert "/tmp/handoff.md" in runner._pending_context_refresh_notes[session_key]
    assert "Context refreshed automatically after a completed phase" in response
    assert "sess-1" in response


@pytest.mark.asyncio
async def test_auto_context_refresh_skips_interrupted_turn(monkeypatch):
    from gateway.run import GatewayRunner

    runner = _make_runner()
    runner._evict_cached_agent = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._is_telegram_topic_lane = MagicMock(return_value=False)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {
            "context_refresh": {
                "mode": "auto_new",
                "auto_new_policy": "phase_boundary",
                "require_no_running_processes": False,
            }
        },
    )
    session_key = build_session_key(_make_source())
    old_entry = runner.session_store._entries[session_key]

    response = await GatewayRunner._maybe_auto_new_after_context_refresh(
        runner,
        _make_event("hello"),
        {
            "final_response": "interrupted",
            "completed": False,
            "interrupted": True,
            "pending_context_refresh": {
                "mode": "auto_new",
                "session_id": "sess-1",
                "handoff_path": "/tmp/handoff.md",
            },
        },
        "interrupted",
        session_key,
        old_entry,
    )

    runner.session_store.reset_session.assert_not_called()
    assert not hasattr(runner, "_pending_context_refresh_notes")
    assert response == "interrupted"


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
