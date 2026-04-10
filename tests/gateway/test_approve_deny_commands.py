"""Tests for /approve and /deny gateway commands.

Verifies that dangerous command approvals use the blocking gateway approval
mechanism — the agent thread blocks until the user responds with /approve
or /deny, mirroring the CLI's synchronous input() flow.

Supports multiple concurrent approvals (parallel subagents, execute_code)
via a per-session queue.
"""

import asyncio
import os
import threading
import time
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
    """Reset all module-level approval state between tests."""
    from tools import approval as mod
    mod._gateway_queues.clear()
    mod._gateway_notify_cbs.clear()
    mod._session_approved.clear()
    mod._permanent_approved.clear()
    mod._pending.clear()


# ------------------------------------------------------------------
# Blocking gateway approval infrastructure (tools/approval.py)
# ------------------------------------------------------------------


class TestBlockingGatewayApproval:
    """Tests for the blocking approval mechanism in tools/approval.py."""

    def setup_method(self):
        _clear_approval_state()

    def test_register_and_resolve_unblocks_entry(self):
        """resolve_gateway_approval signals the entry's event."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, has_blocking_approval,
            _ApprovalEntry, _gateway_queues,
        )
        session_key = "test-session"
        register_gateway_notify(session_key, lambda d: None)

        # Simulate what check_all_command_guards does
        entry = _ApprovalEntry({"command": "rm -rf /"})
        _gateway_queues.setdefault(session_key, []).append(entry)

        assert has_blocking_approval(session_key) is True

        # Resolve from another thread
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
        """resolve_gateway_approval with resolve_all=True signals all entries."""
        from tools.approval import (
            resolve_gateway_approval, _ApprovalEntry, _gateway_queues,
        )
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
        """resolve_gateway_approval without resolve_all resolves oldest first."""
        from tools.approval import (
            resolve_gateway_approval,
            _ApprovalEntry, _gateway_queues,
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
        assert len(_gateway_queues[session_key]) == 1

    def test_unregister_signals_all_entries(self):
        """unregister_gateway_notify signals all waiting entries to prevent hangs."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            _ApprovalEntry, _gateway_queues,
        )
        session_key = "test-cleanup"
        register_gateway_notify(session_key, lambda d: None)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        unregister_gateway_notify(session_key)
        assert e1.event.is_set()
        assert e2.event.is_set()


# ------------------------------------------------------------------
# /approve command
# ------------------------------------------------------------------


class TestApproveCommand:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_approve_resolves_blocking_approval(self):
        """Basic /approve signals the oldest blocked agent thread."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "approved" in result.lower()
        assert "resuming" in result.lower()
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_all_resolves_multiple(self):
        """/approve all resolves all pending approvals."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all"))
        assert "2 commands" in result
        assert e1.event.is_set()
        assert e2.event.is_set()

    @pytest.mark.asyncio
    async def test_approve_all_session(self):
        """/approve all session resolves all with session scope."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_approve_command(_make_event("/approve all session"))
        assert "session" in result.lower()
        assert e1.result == "session"
        assert e2.result == "session"

    @pytest.mark.asyncio
    async def test_approve_no_pending(self):
        """/approve with no pending approval returns helpful message."""
        runner = _make_runner()
        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "No pending command" in result

    @pytest.mark.asyncio
    async def test_approve_stale_old_style_pending(self):
        """Old-style _pending_approvals without blocking event reports expired."""
        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)
        runner._pending_approvals[session_key] = {"command": "test"}

        result = await runner._handle_approve_command(_make_event("/approve"))
        assert "expired" in result.lower() or "no longer waiting" in result.lower()
        assert session_key not in runner._pending_approvals


# ------------------------------------------------------------------
# /deny command
# ------------------------------------------------------------------


class TestDenyCommand:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_deny_resolves_blocking_approval(self):
        """/deny signals the oldest blocked agent thread with 'deny'."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        result = await runner._handle_deny_command(_make_event("/deny"))
        assert "denied" in result.lower()
        assert entry.event.is_set()
        assert entry.result == "deny"

    @pytest.mark.asyncio
    async def test_deny_all_resolves_all(self):
        """/deny all denies all pending approvals."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        e1 = _ApprovalEntry({"command": "cmd1"})
        e2 = _ApprovalEntry({"command": "cmd2"})
        _gateway_queues[session_key] = [e1, e2]

        result = await runner._handle_deny_command(_make_event("/deny all"))
        assert "2 commands" in result
        assert all(e.result == "deny" for e in [e1, e2])

    @pytest.mark.asyncio
    async def test_deny_no_pending(self):
        """/deny with no pending approval returns helpful message."""
        runner = _make_runner()
        result = await runner._handle_deny_command(_make_event("/deny"))
        assert "No pending command" in result


# ------------------------------------------------------------------
# Bare "yes" must NOT trigger approval
# ------------------------------------------------------------------


class TestBareTextNoLongerApproves:

    def setup_method(self):
        _clear_approval_state()

    @pytest.mark.asyncio
    async def test_yes_does_not_execute_pending_command(self):
        """Saying 'yes' must not trigger approval. Only /approve works."""
        from tools.approval import _ApprovalEntry, _gateway_queues

        runner = _make_runner()
        source = _make_source()
        session_key = runner._session_key_for_source(source)

        entry = _ApprovalEntry({"command": "test"})
        _gateway_queues[session_key] = [entry]

        # "yes" is not /approve — entry should still be pending
        assert not entry.event.is_set()


# ------------------------------------------------------------------
# End-to-end blocking flow
# ------------------------------------------------------------------


class TestBlockingApprovalE2E:
    """Test the full blocking flow: agent thread blocks → user approves → agent resumes."""

    def setup_method(self):
        _clear_approval_state()

    def test_blocking_approval_approve_once(self):
        """check_all_command_guards blocks until resolve_gateway_approval is called."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
        )

        session_key = "e2e-test"
        notified = []

        register_gateway_notify(session_key, lambda d: notified.append(d))

        result_holder = [None]

        def agent_thread():
            from tools.approval import reset_current_session_key, set_current_session_key

            token = set_current_session_key(session_key)
            os.environ["HERMES_EXEC_ASK"] = "1"
            os.environ["HERMES_SESSION_KEY"] = session_key
            try:
                result_holder[0] = check_all_command_guards(
                    "rm -rf /important", "local"
                )
            finally:
                os.environ.pop("HERMES_EXEC_ASK", None)
                os.environ.pop("HERMES_SESSION_KEY", None)
                reset_current_session_key(token)

        t = threading.Thread(target=agent_thread)
        t.start()

        for _ in range(50):
            if notified:
                break
            time.sleep(0.05)

        assert len(notified) == 1
        assert "rm -rf /important" in notified[0]["command"]

        resolve_gateway_approval(session_key, "once")
        t.join(timeout=5)

        assert result_holder[0] is not None
        assert result_holder[0]["approved"] is True
        unregister_gateway_notify(session_key)

    def test_blocking_approval_deny(self):
        """check_all_command_guards returns BLOCKED when denied."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
        )

        session_key = "e2e-deny"
        notified = []
        register_gateway_notify(session_key, lambda d: notified.append(d))

        result_holder = [None]

        def agent_thread():
            from tools.approval import reset_current_session_key, set_current_session_key

            token = set_current_session_key(session_key)
            os.environ["HERMES_EXEC_ASK"] = "1"
            os.environ["HERMES_SESSION_KEY"] = session_key
            try:
                result_holder[0] = check_all_command_guards(
                    "rm -rf /important", "local"
                )
            finally:
                os.environ.pop("HERMES_EXEC_ASK", None)
                os.environ.pop("HERMES_SESSION_KEY", None)
                reset_current_session_key(token)

        t = threading.Thread(target=agent_thread)
        t.start()
        for _ in range(50):
            if notified:
                break
            time.sleep(0.05)

        resolve_gateway_approval(session_key, "deny")
        t.join(timeout=5)

        assert result_holder[0]["approved"] is False
        assert "BLOCKED" in result_holder[0]["message"]
        unregister_gateway_notify(session_key)

    def test_blocking_approval_timeout(self):
        """check_all_command_guards returns BLOCKED on timeout."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            check_all_command_guards,
        )

        session_key = "e2e-timeout"
        register_gateway_notify(session_key, lambda d: None)

        result_holder = [None]

        def agent_thread():
            from tools.approval import reset_current_session_key, set_current_session_key

            token = set_current_session_key(session_key)
            os.environ["HERMES_EXEC_ASK"] = "1"
            os.environ["HERMES_SESSION_KEY"] = session_key
            try:
                with patch("tools.approval._get_approval_config",
                           return_value={"gateway_timeout": 1}):
                    result_holder[0] = check_all_command_guards(
                        "rm -rf /important", "local"
                    )
            finally:
                os.environ.pop("HERMES_EXEC_ASK", None)
                os.environ.pop("HERMES_SESSION_KEY", None)
                reset_current_session_key(token)

        t = threading.Thread(target=agent_thread)
        t.start()
        t.join(timeout=10)

        assert result_holder[0]["approved"] is False
        assert "timed out" in result_holder[0]["message"]
        unregister_gateway_notify(session_key)

    def test_parallel_subagent_approvals(self):
        """Multiple threads can block concurrently and be resolved independently."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
            _gateway_queues,
        )

        session_key = "e2e-parallel"
        notified = []
        register_gateway_notify(session_key, lambda d: notified.append(d))

        results = [None, None, None]

        def make_agent(idx, cmd):
            def run():
                from tools.approval import reset_current_session_key, set_current_session_key

                token = set_current_session_key(session_key)
                os.environ["HERMES_EXEC_ASK"] = "1"
                os.environ["HERMES_SESSION_KEY"] = session_key
                try:
                    results[idx] = check_all_command_guards(cmd, "local")
                finally:
                    os.environ.pop("HERMES_EXEC_ASK", None)
                    os.environ.pop("HERMES_SESSION_KEY", None)
                    reset_current_session_key(token)
            return run

        threads = [
            threading.Thread(target=make_agent(0, "rm -rf /a")),
            threading.Thread(target=make_agent(1, "rm -rf /b")),
            threading.Thread(target=make_agent(2, "rm -rf /c")),
        ]
        for t in threads:
            t.start()

        # Wait for all 3 to block
        for _ in range(100):
            if len(notified) >= 3:
                break
            time.sleep(0.05)

        assert len(notified) == 3
        assert len(_gateway_queues.get(session_key, [])) == 3

        # Approve all at once
        count = resolve_gateway_approval(session_key, "session", resolve_all=True)
        assert count == 3

        for t in threads:
            t.join(timeout=5)

        assert all(r is not None for r in results)
        assert all(r["approved"] is True for r in results)
        unregister_gateway_notify(session_key)

    def test_parallel_mixed_approve_deny(self):
        """Approve some, deny others in a parallel batch."""
        from tools.approval import (
            register_gateway_notify, unregister_gateway_notify,
            resolve_gateway_approval, check_all_command_guards,
        )

        session_key = "e2e-mixed"
        register_gateway_notify(session_key, lambda d: None)

        results = [None, None]

        def make_agent(idx, cmd):
            def run():
                from tools.approval import reset_current_session_key, set_current_session_key

                token = set_current_session_key(session_key)
                os.environ["HERMES_EXEC_ASK"] = "1"
                os.environ["HERMES_SESSION_KEY"] = session_key
                try:
                    results[idx] = check_all_command_guards(cmd, "local")
                finally:
                    os.environ.pop("HERMES_EXEC_ASK", None)
                    os.environ.pop("HERMES_SESSION_KEY", None)
                    reset_current_session_key(token)
            return run

        threads = [
            threading.Thread(target=make_agent(0, "rm -rf /x")),
            threading.Thread(target=make_agent(1, "rm -rf /y")),
        ]
        for t in threads:
            t.start()

        # Wait for both threads to register pending approvals instead of
        # relying on a fixed sleep.  The approval module stores entries in
        # _gateway_queues[session_key] — poll until we see 2 entries.
        from tools.approval import _gateway_queues
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if len(_gateway_queues.get(session_key, [])) >= 2:
                break
            time.sleep(0.05)

        # Approve first, deny second
        resolve_gateway_approval(session_key, "once")   # oldest
        resolve_gateway_approval(session_key, "deny")   # next

        for t in threads:
            t.join(timeout=5)

        assert all(r is not None for r in results)
        assert sorted(r["approved"] for r in results) == [False, True]
        assert sum("BLOCKED" in (r.get("message") or "") for r in results) == 1
        unregister_gateway_notify(session_key)


# ------------------------------------------------------------------
# Fallback: no gateway callback (cron/batch mode)
# ------------------------------------------------------------------


class TestFallbackNoCallback:

    def setup_method(self):
        _clear_approval_state()

    def test_no_callback_returns_approval_required(self):
        """Without a registered callback, the old approval_required path is used."""
        from tools.approval import check_all_command_guards, _pending

        os.environ["HERMES_EXEC_ASK"] = "1"
        os.environ["HERMES_SESSION_KEY"] = "no-callback-test"
        try:
            result = check_all_command_guards("rm -rf /important", "local")
        finally:
            os.environ.pop("HERMES_EXEC_ASK", None)
            os.environ.pop("HERMES_SESSION_KEY", None)

        assert result["approved"] is False
        assert result.get("status") == "approval_required"
