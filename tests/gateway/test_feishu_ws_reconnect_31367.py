"""Regression tests for issue #31367.

The lark_oapi WebSocket receive loop exits with
``no close frame received or sent`` roughly every 30 minutes on Feishu's
hosted WS server.  Before the fix, ``ws_client.start()`` returning was
never observed by the adapter — the ``_ws_future`` resolved silently and
the gateway eventually escalated to a full process restart that
disconnected EVERY platform (DingTalk, Slack, etc.) and notified every
active session.  ~48 restarts/day for users running Feishu.

The fix adds ``FeishuAdapter._on_ws_thread_exit`` (a done-callback on
``_ws_future``) and ``_reconnect_websocket_after_unexpected_exit``
that recovers in-place with bounded backoff.  Only after the
adapter-level retries are exhausted do we surface
``_set_fatal_error(retryable=True)`` so the gateway's existing
per-platform watcher can take over — the OTHER platforms keep running.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.feishu import (
    FeishuAdapter,
    _FEISHU_WS_RECONNECT_BASE_DELAY_SECONDS,
    _FEISHU_WS_RECONNECT_MAX_ATTEMPTS,
)


@pytest.fixture
def adapter(monkeypatch):
    """A FeishuAdapter with the runtime knobs tests can poke directly."""
    monkeypatch.setenv("FEISHU_APP_ID", "cli_app")
    monkeypatch.setenv("FEISHU_APP_SECRET", "secret_app")
    a = FeishuAdapter(PlatformConfig())
    return a


def _make_resolved_future(exc: Exception | None = None) -> asyncio.Future:
    """A future that is already done with optional exception, no loop bound."""
    fut: asyncio.Future = asyncio.Future()
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(None)
    return fut


# ---------------------------------------------------------------------------
# _on_ws_thread_exit — gating logic
# ---------------------------------------------------------------------------


class TestOnWsThreadExitGating:
    """Callback must distinguish clean disconnect from unexpected death."""

    def test_no_op_when_running_is_false(self, adapter):
        """Disconnect set ``_running = False`` first — callback no-ops."""
        adapter._running = False
        adapter._ws_client = MagicMock()  # would-be in-flight client
        adapter._loop = MagicMock()

        adapter._on_ws_thread_exit(_make_resolved_future())

        adapter._loop.call_soon_threadsafe.assert_not_called()
        assert adapter._ws_reconnect_in_progress is False

    def test_no_op_when_ws_client_already_cleared(self, adapter):
        """``_disable_websocket_auto_reconnect`` clears ``_ws_client`` —
        the callback must treat that as a clean disconnect.
        """
        adapter._running = True
        adapter._ws_client = None
        adapter._loop = MagicMock()

        adapter._on_ws_thread_exit(_make_resolved_future())

        adapter._loop.call_soon_threadsafe.assert_not_called()

    def test_no_op_when_reconnect_already_in_progress(self, adapter):
        """Re-entrant invocations during recovery are dropped."""
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._loop = MagicMock()
        adapter._ws_reconnect_in_progress = True

        adapter._on_ws_thread_exit(_make_resolved_future())

        adapter._loop.call_soon_threadsafe.assert_not_called()

    def test_unexpected_death_schedules_reconnect_on_main_loop(self, adapter):
        """The interesting case — fires ``call_soon_threadsafe(_schedule)``."""
        adapter._running = True
        adapter._ws_client = MagicMock()
        loop = MagicMock()
        loop.is_closed.return_value = False
        adapter._loop = loop

        adapter._on_ws_thread_exit(_make_resolved_future(RuntimeError("boom")))

        loop.call_soon_threadsafe.assert_called_once()
        # The scheduled callable must be a zero-arg function (matches
        # call_soon_threadsafe contract).
        scheduled = loop.call_soon_threadsafe.call_args.args[0]
        assert callable(scheduled)

    def test_no_op_when_main_loop_closed(self, adapter):
        """A closed main loop means the gateway is shutting down — skip."""
        adapter._running = True
        adapter._ws_client = MagicMock()
        loop = MagicMock()
        loop.is_closed.return_value = True
        adapter._loop = loop

        adapter._on_ws_thread_exit(_make_resolved_future())

        loop.call_soon_threadsafe.assert_not_called()

    def test_no_op_when_main_loop_is_none(self, adapter):
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._loop = None

        adapter._on_ws_thread_exit(_make_resolved_future())  # must not raise


# ---------------------------------------------------------------------------
# _reconnect_websocket_after_unexpected_exit — retry loop semantics
# ---------------------------------------------------------------------------


class TestReconnectAfterUnexpectedExit:
    """Bounded backoff, escalates only after retries exhaust."""

    @pytest.mark.asyncio
    async def test_first_attempt_succeeds_no_escalation(self, adapter, monkeypatch):
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._connect_websocket = AsyncMock()
        adapter._notify_fatal_error = AsyncMock()
        sleep_mock = AsyncMock()
        monkeypatch.setattr("gateway.platforms.feishu.asyncio.sleep", sleep_mock)

        await adapter._reconnect_websocket_after_unexpected_exit()

        assert adapter._connect_websocket.await_count == 1
        sleep_mock.assert_not_awaited()
        adapter._notify_fatal_error.assert_not_awaited()
        assert adapter.has_fatal_error is False
        assert adapter._ws_reconnect_in_progress is False

    @pytest.mark.asyncio
    async def test_succeeds_on_retry_after_transient_failure(self, adapter, monkeypatch):
        """Second attempt wins; backoff sleep was used between."""
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._connect_websocket = AsyncMock(
            side_effect=[ConnectionError("first try fails"), None]
        )
        adapter._notify_fatal_error = AsyncMock()
        sleep_mock = AsyncMock()
        monkeypatch.setattr("gateway.platforms.feishu.asyncio.sleep", sleep_mock)

        await adapter._reconnect_websocket_after_unexpected_exit()

        assert adapter._connect_websocket.await_count == 2
        # Slept at the configured base delay between attempts 1 and 2.
        sleep_mock.assert_awaited_once_with(_FEISHU_WS_RECONNECT_BASE_DELAY_SECONDS)
        adapter._notify_fatal_error.assert_not_awaited()
        assert adapter.has_fatal_error is False

    @pytest.mark.asyncio
    async def test_escalates_after_all_attempts_fail(self, adapter, monkeypatch):
        """Every attempt fails → ``_set_fatal_error(retryable=True)`` +
        ``_notify_fatal_error()`` so the gateway watcher can take over.
        """
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._connect_websocket = AsyncMock(side_effect=ConnectionError("nope"))
        adapter._notify_fatal_error = AsyncMock()
        monkeypatch.setattr(
            "gateway.platforms.feishu.asyncio.sleep", AsyncMock()
        )

        await adapter._reconnect_websocket_after_unexpected_exit()

        assert adapter._connect_websocket.await_count == _FEISHU_WS_RECONNECT_MAX_ATTEMPTS
        adapter._notify_fatal_error.assert_awaited_once()
        assert adapter.has_fatal_error is True
        assert adapter.fatal_error_code == "feishu_ws_reconnect_failed"
        assert adapter.fatal_error_retryable is True

    @pytest.mark.asyncio
    async def test_aborts_when_running_flips_false_mid_retry(self, adapter, monkeypatch):
        """User disconnected mid-retry — stop trying, no escalation."""
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._notify_fatal_error = AsyncMock()

        attempts = {"n": 0}

        async def _flaky_connect():
            attempts["n"] += 1
            if attempts["n"] == 1:
                # Simulate disconnect() racing with the retry loop.
                adapter._running = False
                raise ConnectionError("first attempt failed")

        adapter._connect_websocket = AsyncMock(side_effect=_flaky_connect)
        monkeypatch.setattr(
            "gateway.platforms.feishu.asyncio.sleep", AsyncMock()
        )

        await adapter._reconnect_websocket_after_unexpected_exit()

        # First attempt ran, then the retry loop saw _running=False and bailed.
        assert adapter._connect_websocket.await_count == 1
        adapter._notify_fatal_error.assert_not_awaited()
        assert adapter.has_fatal_error is False

    @pytest.mark.asyncio
    async def test_reentrant_call_is_no_op(self, adapter):
        """Second concurrent call returns immediately — guarded by the flag."""
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._connect_websocket = AsyncMock()
        adapter._notify_fatal_error = AsyncMock()
        adapter._ws_reconnect_in_progress = True  # pretend another run

        await adapter._reconnect_websocket_after_unexpected_exit()

        adapter._connect_websocket.assert_not_awaited()
        adapter._notify_fatal_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clears_dead_executor_state_before_reconnect(self, adapter, monkeypatch):
        """Dead ``_ws_future`` / ``_ws_thread_loop`` must be wiped before
        ``_connect_websocket`` is called so the new run gets fresh refs.
        """
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._ws_future = MagicMock()
        adapter._ws_thread_loop = MagicMock()

        captured = {}

        async def _capture():
            captured["future"] = adapter._ws_future
            captured["thread_loop"] = adapter._ws_thread_loop
            captured["ws_client"] = adapter._ws_client

        adapter._connect_websocket = AsyncMock(side_effect=_capture)
        adapter._notify_fatal_error = AsyncMock()
        monkeypatch.setattr(
            "gateway.platforms.feishu.asyncio.sleep", AsyncMock()
        )

        await adapter._reconnect_websocket_after_unexpected_exit()

        assert captured["future"] is None
        assert captured["thread_loop"] is None
        # ``_disable_websocket_auto_reconnect`` clears _ws_client too.
        assert captured["ws_client"] is None


# ---------------------------------------------------------------------------
# Sanity: the fix doesn't break the happy path through ``disconnect()``.
# ---------------------------------------------------------------------------


class TestCleanDisconnectInteraction:
    """Clean disconnect must not race with the new reconnect callback."""

    def test_clean_disconnect_path_marks_no_reconnect_pending(self, adapter):
        """Simulate the sequence:  disconnect() → _ws_future resolves.

        ``disconnect()`` sets ``_running = False`` and calls
        ``_disable_websocket_auto_reconnect`` which clears ``_ws_client``
        BEFORE awaiting the future.  When the executor-thread callback
        fires after that, both gates trip and the reconnect coroutine
        is never scheduled.
        """
        # Pre-disconnect state
        adapter._running = True
        adapter._ws_client = MagicMock()
        adapter._loop = MagicMock()
        adapter._loop.is_closed.return_value = False

        # disconnect() ran:
        adapter._running = False
        adapter._ws_client = None  # _disable_websocket_auto_reconnect clears it

        adapter._on_ws_thread_exit(_make_resolved_future())

        adapter._loop.call_soon_threadsafe.assert_not_called()
