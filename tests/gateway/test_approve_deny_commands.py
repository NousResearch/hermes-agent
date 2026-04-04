"""Tests for /approve and /deny gateway commands.

Verifies that dangerous command approvals use the blocking gateway approval
mechanism — the agent thread blocks until the user responds with /approve
or /deny, mirroring the CLI's synchronous input() flow.

Supports multiple concurrent approvals (parallel subagents, execute_code)
via a per-session queue.
"""

import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


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
        source=_make_source(),
        message_id="m1",
    )


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
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    return runner


def _clear_approval_state():
    from tools import approval as mod

    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


class TestBlockingGatewayApproval:
    def setup_method(self):
        _clear_approval_state()

    def test_register_and_resolve_unblocks_entry(self):
        from tools.approval import (
            _ApprovalEntry,
            _gateway_queues,
            has_blocking_approval,
            register_gateway_notify,
            resolve_gateway_approval,
            unregister_gateway_notify,
        )

        session_key = "test-session"
        register_gateway_notify(session_key, lambda d: None)
        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues.setdefault(session_key, []).append(entry)

        assert has_blocking_approval(session_key) is True

        def resolve():
            time.sleep(0.1)
            resolve_gateway_approval(session_key, "once")

        t = threading.Thread(target=resolve)
        t.start()
        resolved = entry.event.wait(timeout=5)
        t.join()

        assert resolved is True
        assert entry.result == "once"
        unregister_gateway_notify(session_key)

    def test_resolve_returns_zero_when_no_pending(self):
        from tools.approval import resolve_gateway_approval

        assert resolve_gateway_approval("nonexistent", "once") == 0

    def test_resolve_all_unblocks_multiple_entries(self):
        from tools.approval import _ApprovalEntry, _gateway_queues, resolve_gateway_approval

        session_key = "test-all"
        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        e3 = _ApprovalEntry({"command": "cmd3"})
        _gateway_queues[session_key] = [e1, e2, e3]

        count = resolve_gateway_approval(session_key, "session", resolve_all=True)
        assert count == 3
        assert all(e.event.is_set() for e in [e1, e2, e3])
        assert all(e.result == "session" for e in [e1, e2, e3])

    def test_resolve_single_pops_oldest_fifo(self):
        from tools.approval import (
            _ApprovalEntry,
            _gateway_queues,
            pending_approval_count,
            resolve_gateway_approval,
        )

        session_key = "test-fifo"
        e1 = _ApprovalEntry({"command": "first"})
        e2 = _ApprovalEntry({"command": "second"})
        _gateway_queues[session_key] = [e1, e2]

        count = resolve_gateway_approval(session_key, "once")
        assert count == 1
        assert e1.event.is_set()
        assert e1.result == "once"
        assert not e2.event.is_set()
        assert pending_approval_count(session_key) == 1


class TestApproveCommand:
    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_approve_resolves_blocking_approval(self):
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        session_key = runner._session_key_for_source(_make_source())
        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "approved" in result.lower()
        assert "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_all_resolves_multiple(self):
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        session_key = runner._session_key_for_source(_make_source())
        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all"))
        assert "2 commands" in result
        assert e1.event.is_set()
        assert e2.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_no_pending(self):
        runner = _make_runner()
        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "No pending command" in result


class TestDenyCommand:
    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_deny_resolves_blocking_approval(self):
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        session_key = runner._session_key_for_source(_make_source())
        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_deny_command(_make_event("/deny"))
        assert "denied" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "deny"


class TestFallbackNoCallback:
    def setup_method(self):
        _clear_approval_state()

    def test_no_callback_returns_approval_required(self):
        from tools.approval import _pending, check_all_command_guards

        os.environ["HERMES_EXEC_ASK"] = "1"
        os.environ["HERMES_SESSION_KEY"] = "no-callback-test"
        try:
            result = check_all_command_guards("rm -rf /important", "local")
        finally:
            os.environ.pop("HERMES_EXEC_ASK", None)
            os.environ.pop("HERMES_SESSION_KEY", None)

        assert result["approved"] is False
        assert result.get("status") == "approval_required"
        assert _pending
        pending = next(iter(_pending.values()))
        assert pending["metadata"]["source"] == "exec_ask"
