"""Feishu adapter disconnect lifecycle tests (#96801 / #96809 review).

Split from test_feishu.py: that module is over its FILE_LINES cap and may only shrink.
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class TestFeishuDisconnectLifecycle(unittest.TestCase):
    """#96801 / #96809 review: the local cleanup must survive a gateway-level cancellation
    landing on any await inside disconnect(), the graceful budgets must fit the gateway's
    outer adapter timeout, and concurrent / reconnecting callers must not corrupt each
    other's teardown."""

    @staticmethod
    def _make_adapter():
        from gateway.config import PlatformConfig
        from plugins.platforms.feishu.adapter import FeishuAdapter
        return FeishuAdapter(PlatformConfig())

    @staticmethod
    def _start_ws_thread_loop():
        ws_thread_loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop() -> None:
            asyncio.set_event_loop(ws_thread_loop)
            ready.set()
            ws_thread_loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        ready.wait()
        return ws_thread_loop, thread

    @staticmethod
    def _stop_ws_thread_loop(ws_thread_loop, thread) -> None:
        if not ws_thread_loop.is_closed():
            ws_thread_loop.call_soon_threadsafe(ws_thread_loop.stop)
        thread.join(timeout=2.0)
        if not ws_thread_loop.is_closed():
            ws_thread_loop.close()

    def test_disconnect_local_teardown_survives_cancellation(self):
        """A gateway-level cancel landing on the CLOSE wait must still release every
        adapter-owned resource (#96801).

        The planned-restart path cancels disconnect() at the outer adapter deadline while
        the CLOSE-ack and thread-exit waits can still be pending. Before the fix, the
        cancellation skipped the local cleanup block entirely — SDK executor, seen-id
        persistence and the Feishu app-scoped lock stayed owned by the old process, so the
        replacement gateway could not claim the app and the restart stalled for minutes.
        """
        import concurrent.futures
        adapter = self._make_adapter()
        ws_thread_loop, thread = self._start_ws_thread_loop()

        # A CLOSE that never completes: disconnect() sits on the wait when the cancel lands.
        started = threading.Event()

        async def _hanging_close() -> None:
            started.set()
            await asyncio.Event().wait()

        adapter._ws_client = SimpleNamespace(_disconnect=_hanging_close, _auto_reconnect=True)
        adapter._ws_thread_loop = ws_thread_loop
        adapter._ws_future = None
        adapter._app_lock_identity = "test-lock-identity"
        adapter._sdk_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Dedup persistence writes to the real Hermes home; it is not under test here.
        adapter._persist_seen_message_ids = lambda: None
        mark_disconnected_calls: list[int] = []
        adapter._mark_disconnected = lambda: mark_disconnected_calls.append(1)

        async def _cancel_mid_disconnect() -> None:
            task = asyncio.create_task(adapter.disconnect())
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            for _ in range(4):  # let the wait_for settle onto the hanging CLOSE
                await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(_cancel_mid_disconnect())
        finally:
            self._stop_ws_thread_loop(ws_thread_loop, thread)

        self.assertIsNone(adapter._app_lock_identity, "cancellation must not skip the app-lock release")
        self.assertTrue(mark_disconnected_calls, "cancellation must not skip the disconnected marker")
        self.assertIsNone(adapter._sdk_executor, "cancellation must not skip the SDK executor shutdown")
        self.assertIsNone(adapter._ws_thread_loop, "cancellation must not skip the loop reference cleanup")

    def test_disconnect_inner_budgets_below_gateway_outer_timeout(self):
        """The worst-case graceful disconnect must finish inside the gateway's outer
        per-adapter budget, else the outer cancellation is what runs instead of the
        graceful path (#96801). Drift guard: raise the outer budget or shrink an inner
        one, but never let the sums cross.

        5.0 mirrors gateway/run.py's ``_ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT``; that
        module is not imported here because its import side effects touch the real Hermes
        home, which the home_io_guard test shim refuses.
        """
        from plugins.platforms.feishu import adapter as feishu_adapter
        outer_budget = 5.0
        batch = feishu_adapter._DISCONNECT_BATCH_CANCEL_TIMEOUT_SECS
        websocket_mode = 2 * batch + feishu_adapter._DISCONNECT_CLOSE_TIMEOUT_SECS \
            + feishu_adapter._DISCONNECT_THREAD_EXIT_TIMEOUT_SECS
        webhook_mode = 2 * batch + feishu_adapter._DISCONNECT_WEBHOOK_TIMEOUT_SECS
        self.assertLess(websocket_mode, outer_budget)
        self.assertLess(webhook_mode, outer_budget)

    def test_disconnect_survives_ws_loop_closed_race(self):
        """``call_soon_threadsafe`` raising "Event loop is closed" between the
        ``is_closed()`` check and the handoff must not skip the local cleanup (race
        confirmed on v0.22.2 in #96801)."""
        adapter = self._make_adapter()

        def _raise_closed() -> None:
            raise RuntimeError("Event loop is closed")

        fake_loop = SimpleNamespace(
            is_closed=lambda: False,
            call_soon_threadsafe=lambda func: _raise_closed(),
        )
        adapter._ws_client = None  # skip the CLOSE branch; the loop handoff is the target
        adapter._ws_thread_loop = fake_loop
        adapter._ws_future = None
        adapter._app_lock_identity = "race-lock"
        adapter._persist_seen_message_ids = lambda: None
        adapter._mark_disconnected = lambda: None

        asyncio.run(adapter.disconnect())  # must not raise

        self.assertIsNone(adapter._app_lock_identity, "the loop-closed race must not skip the app-lock release")
        self.assertIsNone(adapter._ws_thread_loop, "the loop-closed race must not skip reference cleanup")

    def test_concurrent_disconnects_share_one_teardown(self):
        """A second disconnect() arriving while the first is mid-CLOSE must share the
        in-flight teardown instead of racing a second CLOSE lifecycle through the same
        thread loop (#96809 review)."""
        adapter = self._make_adapter()
        ws_thread_loop, thread = self._start_ws_thread_loop()

        close_calls: list[int] = []
        release_close = asyncio.Event()

        async def _slow_close() -> None:
            close_calls.append(1)
            await release_close.wait()

        adapter._ws_client = SimpleNamespace(_disconnect=_slow_close, _auto_reconnect=True)
        adapter._ws_thread_loop = ws_thread_loop
        adapter._ws_future = None
        adapter._app_lock_identity = "shared-lock"
        adapter._persist_seen_message_ids = lambda: None
        adapter._mark_disconnected = lambda: None

        async def _scenario() -> None:
            first = asyncio.create_task(adapter.disconnect())
            for _ in range(200):  # the first disconnect is now inside the CLOSE wait
                if close_calls:
                    break
                await asyncio.sleep(0.01)
            second = asyncio.create_task(adapter.disconnect())
            await asyncio.sleep(0)  # let the second task reach its join branch
            release_close.set()
            await asyncio.gather(first, second)

        try:
            asyncio.run(_scenario())
        finally:
            self._stop_ws_thread_loop(ws_thread_loop, thread)

        self.assertEqual(len(close_calls), 1, "concurrent disconnects must share one CLOSE lifecycle")
        self.assertIsNone(adapter._app_lock_identity, "the shared teardown must still release the app lock")

    def test_failed_app_lock_release_keeps_identity_retryable(self):
        """A release that fails must keep the identity so a later disconnect()/retry can
        try again, and a successful release clears it (#96809 review)."""
        from plugins.platforms.feishu import adapter as feishu_adapter
        adapter = self._make_adapter()
        adapter._app_lock_identity = "app-1"

        with patch.object(feishu_adapter, "release_scoped_lock", side_effect=OSError("lock dir gone")):
            adapter._release_app_lock_sync()
        self.assertEqual(adapter._app_lock_identity, "app-1", "a failed release must stay retryable")

        with patch.object(feishu_adapter, "release_scoped_lock") as release_mock:
            adapter._release_app_lock_sync()
        release_mock.assert_called_once_with(feishu_adapter._FEISHU_APP_LOCK_SCOPE, "app-1")
        self.assertIsNone(adapter._app_lock_identity, "a successful release clears the identity")

    def test_disconnect_finalizer_defers_to_newer_connection(self):
        """A reconnect that re-claims the app while an old disconnect is still mid-wait
        owns the state from then on: the old disconnect's finalizer must not release the
        new lock or clear the new WebSocket handles (same-PID scoped locks self-reacquire,
        so this is reachable — #96809 review)."""
        adapter = self._make_adapter()
        ws_thread_loop, thread = self._start_ws_thread_loop()

        started = threading.Event()
        release_close = asyncio.Event()

        async def _slow_close() -> None:
            started.set()
            await release_close.wait()

        adapter._ws_client = SimpleNamespace(_disconnect=_slow_close, _auto_reconnect=True)
        adapter._ws_thread_loop = ws_thread_loop
        adapter._ws_future = None
        adapter._app_lock_identity = "app-1"
        adapter._persist_seen_message_ids = lambda: None
        adapter._mark_disconnected = lambda: None

        async def _scenario() -> None:
            task = asyncio.create_task(adapter.disconnect())
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            # The reconnect claims the app (bumps the generation) and publishes fresh
            # handles with the SAME lock identity — scoped locks self-reacquire per PID.
            adapter._lifecycle_generation += 1
            adapter._ws_thread_loop = "NEW-LOOP"
            adapter._ws_future = "NEW-FUTURE"
            release_close.set()
            await task  # completes normally; no cancellation needed

        try:
            asyncio.run(_scenario())
        finally:
            self._stop_ws_thread_loop(ws_thread_loop, thread)

        self.assertEqual(adapter._app_lock_identity, "app-1", "the old finalizer must not drop the new connection's lock")
        self.assertEqual(adapter._ws_thread_loop, "NEW-LOOP", "the old finalizer must not clear the new loop handle")
        self.assertEqual(adapter._ws_future, "NEW-FUTURE", "the old finalizer must not clear the new thread future")


if __name__ == "__main__":
    unittest.main()
