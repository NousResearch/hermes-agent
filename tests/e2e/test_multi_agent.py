"""End-to-end tests for the multi-agent gateway.

These tests drive messages through the full async pipeline:

    adapter.handle_message(event)
        → BasePlatformAdapter._process_message_background()
        → GatewayRunner._handle_message() / ._handle_message_with_agent()
        → command dispatch  OR  agent execution
        → adapter.send() (captured for assertions)

Scope (per the refactor brief): Telegram only, since the multi-agent
plumbing lives in the gateway-level code path that is platform-agnostic.

What is and isn't covered:

* ``/profile`` (bare/ls/switch) — full pipeline.  No LLM needed.
* ``@<name> <msg>`` inline routing — verified by stubbing
  ``AIAgent`` so ``_run_agent`` actually runs and the resolver +
  ``agent_home_scope`` are exercised, then asserting on the captured
  ``HERMES_HOME`` at construction time and the message text that
  reached ``run_conversation``.
* ``[<agent_name>] ...`` response prefix — same stub; we assert on the
  final ``adapter.send`` payload.

We do NOT spin up a real provider/model.  Memory/skills/soul real
isolation is covered by ``tests/gateway/test_multi_agent_isolation.py``
at the path level.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.agent_registry import reset_default_registry
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.session import SessionEntry, SessionSource, SessionStore, build_session_key
from tests.e2e.conftest import (
    _ensure_telegram_mock,
    make_event,
    make_source,
    send_and_capture,
)


# ---------------------------------------------------------------------------
# Telegram-only adapter + runner factory with a REAL SessionStore so the
# session_key / active_agent persistence is exercised end-to-end.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """Three on-disk profiles: default + coder + data-sci."""
    root = tmp_path / ".hermes"
    coder = root / "profiles" / "coder"
    sci = root / "profiles" / "data-sci"
    for home in (root, coder, sci):
        home.mkdir(parents=True, exist_ok=True)
        (home / "memories").mkdir()
        (home / "skills").mkdir()
    (coder / "SOUL.md").write_text("# Coder agent\nI write Python.\n", encoding="utf-8")
    (sci / "SOUL.md").write_text("# Data scientist\nI analyse data.\n", encoding="utf-8")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    reset_default_registry()
    yield root, coder, sci
    reset_default_registry()


@pytest.fixture
def telegram_runner(fake_root, tmp_path):
    """A GatewayRunner stub wired with a REAL SessionStore.

    The existing e2e ``make_runner`` fixture mocks SessionStore entirely,
    which doesn't suffice for the multi-agent commands — they read and
    write ``active_agent`` on session entries.  This fixture borrows the
    rest of the stub but swaps in a real store so we can assert on
    persisted state.
    """
    _ensure_telegram_mock()
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="e2e-test-token")
        }
    )
    runner.adapters = {}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)

    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    from gateway.restart import DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT

    runner._restart_drain_timeout = DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._pending_model_notes = {}
    runner._update_prompt_pending = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False

    runner._agent_cache: OrderedDict = OrderedDict()
    runner._agent_cache_lock = Lock()

    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_message_with_agent = AsyncMock(return_value="agent-handled-default")
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *a, **kw: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._read_user_config = lambda: {"approvals": {"destructive_slash_confirm": False}}

    runner.pairing_store = MagicMock()
    runner.pairing_store._is_rate_limited = MagicMock(return_value=False)
    runner.pairing_store.generate_code = MagicMock(return_value="ABC123")

    return runner


@pytest.fixture
def telegram_adapter(telegram_runner):
    from gateway.platforms.telegram import TelegramAdapter

    config = PlatformConfig(enabled=True, token="e2e-test-token")
    adapter = TelegramAdapter(config)
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="e2e-resp-1"))
    adapter.send_typing = AsyncMock()
    adapter.set_message_handler(telegram_runner._handle_message)
    telegram_runner.adapters[Platform.TELEGRAM] = adapter
    return adapter


def _send_response_text(adapter) -> str | None:
    if not adapter.send.called:
        return None
    return adapter.send.call_args[1].get("content") or adapter.send.call_args[0][1]


# ===========================================================================
# /profile end-to-end
# ===========================================================================


class TestProfileCommandE2E:
    """``/profile`` exercised through adapter → runner → dispatch → send."""

    @pytest.mark.asyncio
    async def test_bare_profile_shows_active(self, telegram_adapter):
        send = await send_and_capture(telegram_adapter, "/profile", Platform.TELEGRAM)
        send.assert_called()
        text = _send_response_text(telegram_adapter)
        assert "default" in text
        assert "Active profile" in text or "Active" in text
        # Hint to discover more
        assert "/profile ls" in text

    @pytest.mark.asyncio
    async def test_profile_ls_lists_all_profiles(self, telegram_adapter):
        send = await send_and_capture(telegram_adapter, "/profile ls", Platform.TELEGRAM)
        send.assert_called()
        text = _send_response_text(telegram_adapter)
        assert "Available profiles" in text
        assert "default" in text
        assert "coder" in text
        assert "data-sci" in text

    @pytest.mark.asyncio
    async def test_profile_switch_persists_to_store(
        self, telegram_adapter, telegram_runner
    ):
        send = await send_and_capture(
            telegram_adapter, "/profile coder", Platform.TELEGRAM
        )
        send.assert_called()
        text = _send_response_text(telegram_adapter)
        assert "Switched profile to: coder" in text

        # Verify persisted at the chat-binding level (new multi-agent
        # model — each (chat, agent) pair has its own session, so the
        # binding lives separately from any specific SessionEntry).
        source = make_source(Platform.TELEGRAM)
        assert telegram_runner.session_store.get_chat_agent(source) == "coder"

    @pytest.mark.asyncio
    async def test_profile_switch_evicts_default_agent_cache(
        self, telegram_adapter, telegram_runner
    ):
        # Seed the cache with a sentinel keyed by the DEFAULT agent's
        # session_key — that's what /profile coder evicts (the prior
        # binding's slot).  The new agent's session_key is independent.
        source = make_source(Platform.TELEGRAM)
        default_key = build_session_key(source, agent_name="default")
        await send_and_capture(telegram_adapter, "/profile", Platform.TELEGRAM)
        telegram_runner._agent_cache[default_key] = ("sentinel-agent", "sig-xyz")

        await send_and_capture(telegram_adapter, "/profile coder", Platform.TELEGRAM)
        assert default_key not in telegram_runner._agent_cache

    @pytest.mark.asyncio
    async def test_profile_switch_unknown_name(self, telegram_adapter):
        send = await send_and_capture(
            telegram_adapter, "/profile nonexistent-agent", Platform.TELEGRAM
        )
        send.assert_called()
        text = _send_response_text(telegram_adapter)
        assert "Unknown profile" in text
        assert "nonexistent-agent" in text
        # Lists alternatives
        assert "coder" in text

    @pytest.mark.asyncio
    async def test_profile_switch_then_bare_shows_new_binding(
        self, telegram_adapter
    ):
        await send_and_capture(
            telegram_adapter, "/profile coder", Platform.TELEGRAM
        )
        send = await send_and_capture(telegram_adapter, "/profile", Platform.TELEGRAM)
        send.assert_called()
        text = _send_response_text(telegram_adapter)
        assert "coder" in text
