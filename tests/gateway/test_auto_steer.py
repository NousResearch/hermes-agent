"""Tests for the auto_steer busy-input mode.

auto_steer behaves like "steer" when the agent is running (inject via
agent.steer()), but when the agent is NOT running or steer fails, it returns
False to let normal processing handle the message instead of queueing.

This differs from "steer" mode which falls back to queue semantics.
"""
from __future__ import annotations

import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal telegram stubs so we can import gateway code without heavy deps
# ---------------------------------------------------------------------------
import sys

_tg = types.ModuleType("telegram")
_tg.constants = types.ModuleType("telegram.constants")
_ct = MagicMock()
_ct.SUPERGROUP = "supergroup"
_ct.GROUP = "group"
_ct.PRIVATE = "private"
_tg.constants.ChatType = _ct
sys.modules.setdefault("telegram", _tg)
sys.modules.setdefault("telegram.constants", _tg.constants)
sys.modules.setdefault("telegram.ext", types.ModuleType("telegram.ext"))

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.run import GatewayRunner, _AGENT_PENDING_SENTINEL
from gateway.session import SessionEntry, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m1",
    )


def _make_adapter():
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    return adapter


def _make_runner_with_mode(busy_input_mode: str = "auto_steer") -> GatewayRunner:
    """Build a minimal GatewayRunner for testing busy-input methods."""
    runner = object.__new__(GatewayRunner)
    runner._busy_input_mode = busy_input_mode
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._draining = False
    runner._busy_text_mode = busy_input_mode
    runner.adapters = {Platform.TELEGRAM: _make_adapter()}
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    # Methods called by drain/dispatch paths
    runner._reply_anchor_for_event = lambda event: event.message_id
    runner._thread_metadata_for_source = lambda *a, **kw: None
    runner._status_action_gerund = lambda: "restarting"
    return runner


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


# ---------------------------------------------------------------------------
# _load_busy_input_mode — static method, reads env/config
# ---------------------------------------------------------------------------

class TestLoadBusyInputMode:
    def test_auto_steer_accepted(self):
        """auto_steer must be recognized as a valid busy input mode."""
        import os
        os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = "auto_steer"
        try:
            mode = GatewayRunner._load_busy_input_mode()
            assert mode == "auto_steer"
        finally:
            os.environ.pop("HERMES_GATEWAY_BUSY_INPUT_MODE", None)

    def test_auto_steer_case_insensitive(self):
        import os
        os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = "AUTO_STEER"
        try:
            mode = GatewayRunner._load_busy_input_mode()
            assert mode == "auto_steer"
        finally:
            os.environ.pop("HERMES_GATEWAY_BUSY_INPUT_MODE", None)

    def test_invalid_mode_defaults(self):
        import os
        os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = "garbage"
        try:
            mode = GatewayRunner._load_busy_input_mode()
            assert mode == "interrupt"
        finally:
            os.environ.pop("HERMES_GATEWAY_BUSY_INPUT_MODE", None)

    def test_steer_still_works(self):
        import os
        os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = "steer"
        try:
            mode = GatewayRunner._load_busy_input_mode()
            assert mode == "steer"
        finally:
            os.environ.pop("HERMES_GATEWAY_BUSY_INPUT_MODE", None)

    def test_queue_still_works(self):
        import os
        os.environ["HERMES_GATEWAY_BUSY_INPUT_MODE"] = "queue"
        try:
            mode = GatewayRunner._load_busy_input_mode()
            assert mode == "queue"
        finally:
            os.environ.pop("HERMES_GATEWAY_BUSY_INPUT_MODE", None)


# ---------------------------------------------------------------------------
# _queue_during_drain_enabled
# ---------------------------------------------------------------------------

class TestQueueDuringDrain:
    def test_auto_steer_includes_drain_queue(self):
        """auto_steer should queue messages during restart drain (like steer/queue)."""
        runner = _make_runner_with_mode()
        runner._restart_requested = True
        assert runner._queue_during_drain_enabled() is True

    def test_auto_steer_no_restart_no_drain(self):
        runner = _make_runner_with_mode()
        runner._restart_requested = False
        assert runner._queue_during_drain_enabled() is False

    def test_steer_still_includes_drain_queue(self):
        runner = _make_runner_with_mode("steer")
        runner._restart_requested = True
        assert runner._queue_during_drain_enabled() is True

    def test_interrupt_excludes_drain_queue(self):
        runner = _make_runner_with_mode("interrupt")
        runner._restart_requested = True
        assert runner._queue_during_drain_enabled() is False


# ---------------------------------------------------------------------------
# _handle_active_session_busy_message — auto_steer vs steer behavior
# ---------------------------------------------------------------------------

class TestAutoSteerBusyMessage:
    @pytest.mark.asyncio
    async def test_auto_steer_running_agent_steers(self):
        """When agent IS running, auto_steer injects via steer()."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        running_agent = MagicMock()
        running_agent.steer.return_value = True
        runner._running_agents[sk] = running_agent

        event = _make_event("check the logs")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        running_agent.steer.assert_called_once_with("check the logs")
        assert result is True  # handled (steered returns True -> ack sent)

    @pytest.mark.asyncio
    async def test_auto_steer_no_agent_returns_false(self):
        """When agent is NOT running, auto_steer returns False (normal processing)."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        event = _make_event("hello")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_steer_sentinel_returns_false(self):
        """When agent is pending sentinel, auto_steer returns False (not queue)."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())
        runner._running_agents[sk] = _AGENT_PENDING_SENTINEL

        event = _make_event("hello")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_steer_steer_failure_returns_false(self):
        """When steer() returns False, auto_steer returns False (not queue)."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        running_agent = MagicMock()
        running_agent.steer.return_value = False
        runner._running_agents[sk] = running_agent

        event = _make_event("do stuff")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_steer_no_steer_method_returns_false(self):
        """When agent lacks steer(), auto_steer returns False."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        running_agent = MagicMock(spec=[])
        runner._running_agents[sk] = running_agent

        event = _make_event("hello")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_steer_empty_text_returns_false(self):
        """Empty/whitespace text with auto_steer returns False."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        running_agent = MagicMock()
        runner._running_agents[sk] = running_agent

        event = _make_event("   ")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False
        running_agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_steer_non_text_message_returns_false(self):
        """Non-TEXT message types return False (not steerable)."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        event = _make_event("hello")
        event.message_type = MessageType.PHOTO

        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False

    @pytest.mark.asyncio
    async def test_steer_mode_still_falls_back_to_queue_on_failure(self):
        """Regular steer mode falls back to queue on steer failure (not return False)."""
        runner = _make_runner_with_mode("steer")
        sk = build_session_key(_make_source())

        running_agent = MagicMock()
        running_agent.steer.return_value = False
        runner._running_agents[sk] = running_agent

        event = _make_event("hello")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        # steer mode: on failure falls back to queue, so message is in pending
        adapter = runner.adapters[Platform.TELEGRAM]
        assert sk in adapter._pending_messages

    @pytest.mark.asyncio
    async def test_auto_steer_steer_exception_returns_false(self):
        """When steer() raises an exception, auto_steer returns False."""
        runner = _make_runner_with_mode()
        sk = build_session_key(_make_source())

        running_agent = MagicMock()
        running_agent.steer.side_effect = RuntimeError("agent busy")
        runner._running_agents[sk] = running_agent

        event = _make_event("hello")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_steer_drain_queues_and_returns_true(self):
        """When draining, auto_steer queues the message and returns True."""
        runner = _make_runner_with_mode()
        runner._draining = True
        runner._restart_requested = True
        sk = build_session_key(_make_source())

        event = _make_event("hello")
        result = await runner._handle_active_session_busy_message(event, session_key=sk)

        assert result is True
        adapter = runner.adapters[Platform.TELEGRAM]
        assert sk in adapter._pending_messages


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
