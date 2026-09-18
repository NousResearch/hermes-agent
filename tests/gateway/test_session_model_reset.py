"""Tests that /new (and its /reset alias) clears session-scoped overrides."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
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


class _TypingRecordingAdapter:
    """Adapter double whose class exposes a real ``interrupt_session_activity`` accepting
    ``metadata``, so the introspective call path (``getattr(type(adapter), …)`` +
    ``_accepts_keyword``) resolves the recording method instead of skipping a MagicMock."""

    def __init__(self):
        self.interrupt_calls = []

    async def send(self, *args, **kwargs):
        pass

    async def interrupt_session_activity(self, session_key, chat_id, metadata=None):
        self.interrupt_calls.append((session_key, chat_id, metadata))


class _TwoArgInterruptAdapter:
    """Older adapter spelling: ``interrupt_session_activity(session_key, chat_id)`` only."""

    def __init__(self):
        self.interrupt_calls = []

    async def send(self, *args, **kwargs):
        pass

    async def interrupt_session_activity(self, session_key, chat_id):
        self.interrupt_calls.append((session_key, chat_id))


def _make_typing_runner(adapter):
    """``_make_runner`` with the given (class-level) adapter double installed."""
    runner = _make_runner()
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


@pytest.mark.asyncio
async def test_reset_command_interrupts_session_activity():
    """#50766: the normal /new dispatch path (no agent in _running_agents) goes straight to
    _handle_reset_command, which must stop the adapter's typing loop exactly like the /stop
    path does — otherwise an orphaned _keep_typing task keeps the "typing…" indicator alive
    after every reset."""
    adapter = _TypingRecordingAdapter()
    runner = _make_typing_runner(adapter)
    source = _make_source()
    session_key = build_session_key(source)
    expected_metadata = runner._thread_metadata_for_source(source)

    await runner._handle_reset_command(_make_event("/new"))

    assert adapter.interrupt_calls == [(session_key, "c1", expected_metadata)]


@pytest.mark.asyncio
async def test_reset_command_interrupts_legacy_two_arg_adapter():
    """The introspective contract must keep calling older adapters positionally: their
    ``interrupt_session_activity(session_key, chat_id)`` has no ``metadata`` kwarg."""
    adapter = _TwoArgInterruptAdapter()
    runner = _make_typing_runner(adapter)
    session_key = build_session_key(_make_source())

    await runner._handle_reset_command(_make_event("/new"))

    assert adapter.interrupt_calls == [(session_key, "c1")]


@pytest.mark.asyncio
async def test_reset_command_without_interrupt_hook_still_resets():
    """Adapters without ``interrupt_session_activity`` (custom platforms, older adapters) must
    not break /new: the activity interrupt is best-effort, so reset still completes and clears
    session-scoped state (baseline guard)."""
    runner = _make_runner()  # MagicMock adapter: class-level hook lookup resolves to nothing
    session_key = build_session_key(_make_source())
    runner._session_model_overrides[session_key] = {
        "model": "gpt-4o",
        "provider": "openai",
        "api_key": "sk-test",
        "base_url": "",
        "api_mode": "openai",
    }

    result = await runner._handle_reset_command(_make_event("/new"))

    assert session_key not in runner._session_model_overrides
    assert result is not None
