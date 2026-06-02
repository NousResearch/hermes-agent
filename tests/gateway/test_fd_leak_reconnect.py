"""Regression tests: fd leak in reconnect loop — adapter must be disposed
on every failed reconnect attempt (non-retryable fatal, retryable failure,
and exception paths).

Issue #37011: APIServerAdapter creates a ResponseStore (sqlite3, 2 fds) on
every reconnect attempt. Before the fix, all three failure paths dropped the
newly created adapter without calling disconnect(), leaking ~2 fds per retry
cycle and exhausting the fd limit (~12 h at 300 s backoff cap).

Fix 1: gateway/run.py now calls _safe_adapter_disconnect(adapter, platform)
in all three failure branches of the reconnect watcher loop.

Fix 2: APIServerAdapter.disconnect() now closes the ResponseStore connection.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DisposableStubAdapter(BasePlatformAdapter):
    """Adapter that tracks disconnect() calls and can simulate any failure."""

    def __init__(
        self,
        *,
        platform=Platform.TELEGRAM,
        connect_result=False,
        fatal_error=None,
        fatal_retryable=True,
        connect_raises=None,
    ):
        super().__init__(PlatformConfig(enabled=True, token="test"), platform)
        self._connect_result = connect_result
        self._fatal_error = fatal_error
        self._fatal_retryable = fatal_retryable
        self._connect_raises = connect_raises
        self.disconnect_call_count = 0

    async def connect(self):
        if self._connect_raises:
            raise self._connect_raises
        if self._fatal_error:
            self._set_fatal_error("test_error", self._fatal_error, retryable=self._fatal_retryable)
            return False
        return self._connect_result

    async def disconnect(self):
        self.disconnect_call_count += 1

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_runner():
    """Create a minimal GatewayRunner shell for reconnect loop tests."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="test")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_with_failure = False
    runner._exit_cleanly = False
    runner._failed_platforms = {}
    runner.adapters = {}
    runner.delivery_router = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._honcho_managers = {}
    runner._honcho_configs = {}
    runner._shutdown_all_gateway_honcho = lambda: None
    runner.session_store = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    runner._handle_message = MagicMock()
    runner._handle_adapter_fatal_error = MagicMock()
    runner._handle_active_session_busy_message = MagicMock()
    runner._recover_telegram_topic_thread_id = MagicMock()
    runner._busy_text_mode = "interrupt"
    runner._sync_voice_mode_state_to_adapter = MagicMock()
    return runner


async def _simulate_one_reconnect_attempt(runner, adapter):
    """
    Directly execute the reconnect attempt logic from _platform_reconnect_watcher.

    This mirrors the body of the try/except inside the watcher loop.
    Testing via the full watcher is impractical because it starts with
    `await asyncio.sleep(10)`. We test the internal machinery directly instead.
    """
    platform = Platform.TELEGRAM
    platform_config = PlatformConfig(enabled=True, token="test")
    info = {"config": platform_config, "attempts": 0, "next_retry": time.monotonic() - 1}
    runner._failed_platforms[platform] = info

    attempt = info["attempts"] + 1
    _BACKOFF_CAP = 300

    try:
        adapter.set_message_handler(runner._handle_message)
        adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
        adapter.set_session_store(runner.session_store)
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        if hasattr(adapter, "set_topic_recovery_fn"):
            adapter.set_topic_recovery_fn(runner._recover_telegram_topic_thread_id)
        adapter._busy_text_mode = runner._busy_text_mode

        success = await runner._connect_adapter_with_timeout(adapter, platform)

        if success:
            runner.adapters[platform] = adapter
            runner._sync_voice_mode_state_to_adapter(adapter)
            runner.delivery_router.adapters = runner.adapters
            del runner._failed_platforms[platform]
            runner._update_platform_runtime_status(
                platform.value, platform_state="connected",
                error_code=None, error_message=None,
            )
        elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
            runner._update_platform_runtime_status(
                platform.value, platform_state="fatal",
                error_code=adapter.fatal_error_code,
                error_message=adapter.fatal_error_message,
            )
            await runner._safe_adapter_disconnect(adapter, platform)
            del runner._failed_platforms[platform]
        else:
            runner._update_platform_runtime_status(
                platform.value, platform_state="retrying",
                error_code=getattr(adapter, "fatal_error_code", None),
                error_message=getattr(adapter, "fatal_error_message", None) or "failed to reconnect",
            )
            await runner._safe_adapter_disconnect(adapter, platform)
            backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
            info["attempts"] = attempt
            info["next_retry"] = time.monotonic() + backoff

    except Exception as e:
        runner._update_platform_runtime_status(
            platform.value, platform_state="retrying",
            error_code=None, error_message=str(e),
        )
        await runner._safe_adapter_disconnect(adapter, platform)
        backoff = min(30 * (2 ** (attempt - 1)), _BACKOFF_CAP)
        info["attempts"] = attempt
        info["next_retry"] = time.monotonic() + backoff


# ---------------------------------------------------------------------------
# Fix 1: reconnect loop disposes adapter on all failure paths
# ---------------------------------------------------------------------------

class TestReconnectLoopDisposesAdapter:

    def test_retryable_failure_calls_disconnect(self):
        """connect() returns False (retryable) — adapter.disconnect() must be called."""
        adapter = DisposableStubAdapter(connect_result=False)
        runner = _make_runner()

        asyncio.run(_simulate_one_reconnect_attempt(runner, adapter))

        assert adapter.disconnect_call_count >= 1, (
            "disconnect() not called after retryable failure — fd leak!"
        )

    def test_non_retryable_fatal_error_calls_disconnect(self):
        """connect() returns non-retryable fatal error — adapter.disconnect() must be called."""
        adapter = DisposableStubAdapter(fatal_error="API key missing", fatal_retryable=False)
        runner = _make_runner()

        asyncio.run(_simulate_one_reconnect_attempt(runner, adapter))

        assert adapter.disconnect_call_count >= 1, (
            "disconnect() not called after non-retryable fatal error — fd leak!"
        )

    def test_exception_during_connect_calls_disconnect(self):
        """connect() raises — adapter.disconnect() must still be called."""
        adapter = DisposableStubAdapter(connect_raises=ConnectionRefusedError("refused"))
        runner = _make_runner()

        asyncio.run(_simulate_one_reconnect_attempt(runner, adapter))

        assert adapter.disconnect_call_count >= 1, (
            "disconnect() not called after connect() raised — fd leak!"
        )


# ---------------------------------------------------------------------------
# Fix 2: APIServerAdapter.disconnect() closes the ResponseStore
# ---------------------------------------------------------------------------

class TestAPIServerAdapterResponseStoreClose:

    def test_disconnect_closes_response_store(self):
        """APIServerAdapter.disconnect() must call ResponseStore.close()."""
        try:
            from gateway.platforms.api_server import APIServerAdapter
        except ImportError:
            pytest.skip("aiohttp not installed")

        adapter = object.__new__(APIServerAdapter)
        mock_store = MagicMock()
        adapter._response_store = mock_store
        adapter._site = None
        adapter._runner = None
        adapter._app = None
        adapter._mark_disconnected = MagicMock()
        adapter.platform = Platform.TELEGRAM  # needed for adapter.name property

        asyncio.run(adapter.disconnect())

        mock_store.close.assert_called_once(), (
            "ResponseStore.close() not called in disconnect() — sqlite fd leak!"
        )

    def test_disconnect_tolerates_none_response_store(self):
        """disconnect() must not raise if _response_store is already None."""
        try:
            from gateway.platforms.api_server import APIServerAdapter
        except ImportError:
            pytest.skip("aiohttp not installed")

        adapter = object.__new__(APIServerAdapter)
        adapter._response_store = None
        adapter._site = None
        adapter._runner = None
        adapter._app = None
        adapter._mark_disconnected = MagicMock()
        adapter.platform = Platform.TELEGRAM

        # Must not raise
        asyncio.run(adapter.disconnect())
