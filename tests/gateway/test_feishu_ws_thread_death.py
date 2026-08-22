"""Tests for Feishu WebSocket thread-death recovery (PR #73202).

When the lark-oapi SDK's WS receive loop dies silently (~every 30 min),
the adapter must detect the unexpected exit and route through the
gateway's existing per-platform reconnect watcher — no custom retry loop.
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


class TestFeishuWSThreadDeathRecovery(unittest.TestCase):
    """WS thread death → retryable fatal notification via gateway watcher."""

    # ── _handle_ws_unexpected_exit positive path ──────────────────────

    def test_handle_ws_unexpected_exit_notifies_fatal_error(self):
        """_handle_ws_unexpected_exit calls _set_fatal_error(retryable=True)
        and _notify_fatal_error(), routing through the gateway's existing
        per-platform reconnect watcher."""
        adapter = FeishuAdapter(PlatformConfig())
        adapter._running = True
        adapter._ws_client = SimpleNamespace()
        adapter._ws_future = SimpleNamespace()
        adapter._ws_thread_loop = SimpleNamespace()

        fatal_handler_called = False

        async def _fake_fatal_handler(adapter):
            nonlocal fatal_handler_called
            fatal_handler_called = True

        adapter._fatal_error_handler = _fake_fatal_handler

        asyncio.run(adapter._handle_ws_unexpected_exit())

        self.assertTrue(
            fatal_handler_called,
            "_notify_fatal_error must invoke the registered fatal handler",
        )
        self.assertEqual(
            adapter._fatal_error_code,
            "feishu_ws_unexpected_exit",
        )
        self.assertTrue(
            adapter._fatal_error_retryable,
            "error must be retryable so the supervisor reconnects",
        )
        # Cleanup: dead refs must be cleared so reconnect can re-create them.
        self.assertIsNone(adapter._ws_future)
        self.assertIsNone(adapter._ws_client)
        self.assertIsNone(adapter._ws_thread_loop)

    # ── _handle_ws_unexpected_exit gate tests ─────────────────────────

    def test_handle_ws_unexpected_exit_noops_when_not_running(self):
        """_handle_ws_unexpected_exit is a no-op when _running=False
        (adapter already stopped / clean disconnect)."""
        adapter = FeishuAdapter(PlatformConfig())
        adapter._running = False
        adapter._ws_client = SimpleNamespace()

        fatal_called = False

        def _fake_set_fatal(code, message, *, retryable):
            nonlocal fatal_called
            fatal_called = True

        # Replace _set_fatal_error so we can assert it's never called.
        adapter._set_fatal_error = _fake_set_fatal

        asyncio.run(adapter._handle_ws_unexpected_exit())

        self.assertFalse(
            fatal_called,
            "_handle_ws_unexpected_exit must no-op when _running=False",
        )

    def test_handle_ws_unexpected_exit_noops_when_no_ws_client(self):
        """_handle_ws_unexpected_exit is a no-op when _ws_client is None
        (clean disconnect path cleared it)."""
        adapter = FeishuAdapter(PlatformConfig())
        adapter._running = True
        adapter._ws_client = None

        fatal_called = False

        def _fake_set_fatal(code, message, *, retryable):
            nonlocal fatal_called
            fatal_called = True

        adapter._set_fatal_error = _fake_set_fatal

        asyncio.run(adapter._handle_ws_unexpected_exit())

        self.assertFalse(
            fatal_called,
            "_handle_ws_unexpected_exit must no-op when _ws_client is None",
        )

    # ── _on_ws_thread_exit gate tests (done callback) ─────────────────

    def test_on_ws_thread_exit_gate_clean_disconnect(self):
        """_on_ws_thread_exit early-returns on clean disconnect
        (_running=False or _ws_client=None) — no fatal escalation."""
        adapter = FeishuAdapter(PlatformConfig())

        # Gate 1: _running=False (adapter stopped)
        adapter._running = False
        adapter._ws_client = SimpleNamespace()
        try:
            adapter._on_ws_thread_exit(None)
        except Exception as exc:
            self.fail(
                f"_on_ws_thread_exit must not raise on clean disconnect "
                f"(_running=False): {exc}"
            )

        # Gate 2: _ws_client=None (clean disconnect cleared it)
        adapter._running = True
        adapter._ws_client = None
        try:
            adapter._on_ws_thread_exit(None)
        except Exception as exc:
            self.fail(
                f"_on_ws_thread_exit must not raise when ws_client is None "
                f"(clean disconnect): {exc}"
            )

    def test_on_ws_thread_exit_dispatches_to_loop(self):
        """_on_ws_thread_exit schedules _handle_ws_unexpected_exit via
        call_soon_threadsafe when _running=True and _ws_client is set."""
        adapter = FeishuAdapter(PlatformConfig())

        # Set up a real thread loop (matching the disconnect test pattern
        # in test_feishu.py:132-183).
        ws_thread_loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(ws_thread_loop)
            ready.set()
            ws_thread_loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        ready.wait()

        adapter._loop = ws_thread_loop
        adapter._running = True
        adapter._ws_client = SimpleNamespace()

        # Replace _handle_ws_unexpected_exit with a flag-setting coroutine.
        handle_called = threading.Event()

        async def _fake_handle():
            handle_called.set()

        adapter._handle_ws_unexpected_exit = _fake_handle

        try:
            adapter._on_ws_thread_exit(SimpleNamespace())
            self.assertTrue(
                handle_called.wait(timeout=2.0),
                "_on_ws_thread_exit must schedule "
                "_handle_ws_unexpected_exit on the adapter loop",
            )
        finally:
            if not ws_thread_loop.is_closed():
                ws_thread_loop.call_soon_threadsafe(ws_thread_loop.stop)
            thread.join(timeout=2.0)
            if not ws_thread_loop.is_closed():
                ws_thread_loop.close()
