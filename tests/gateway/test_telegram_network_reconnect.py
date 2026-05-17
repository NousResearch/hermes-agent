"""
Tests for Telegram polling error handling.

After the #3173-series fixes, the adapter delegates ALL polling recovery
to PTB's internal network_retry_loop (max_retries=-1).  The gateway no longer
restarts polling, schedules probes, drains connections, or escalates network
errors to fatal.  The only remaining escalation path is the 409 Conflict
counter (3 conflicts in 60s → non-retryable fatal).
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


# ── Network error handler (trust-PTB) ──────────────────────────────────

@pytest.mark.asyncio
async def test_network_error_handler_logs_and_returns():
    """
    _handle_polling_network_error must only log a warning and return.
    It must NOT restart polling, drain connections, schedule probes,
    or escalate to fatal.  PTB handles recovery autonomously.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    bg_before = len(adapter._background_tasks)

    await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # Must NOT try to restart polling
    mock_updater.stop.assert_not_called()
    mock_updater.start_polling.assert_not_called()

    # Must NOT schedule any background tasks (probes, retries)
    assert len(adapter._background_tasks) == bg_before, (
        "Expected no background tasks after network error handler"
    )

    # Must NOT set a fatal error
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_network_error_handler_skips_when_fatal():
    """
    When a fatal error is already set, the handler should return immediately.
    """
    adapter = _make_adapter()
    adapter._set_fatal_error("telegram_polling_conflict", "already fatal", retryable=False)

    mock_updater = MagicMock()
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    await adapter._handle_polling_network_error(Exception("Timed out"))

    mock_updater.stop.assert_not_called()
    mock_updater.start_polling.assert_not_called()


# ── Conflict handler (count-only, non-retryable fatal) ─────────────────

@pytest.mark.asyncio
async def test_conflict_handler_counts_and_escalates():
    """
    3 consecutive conflicts within 60s must escalate to a non-retryable
    fatal error.  The handler must NOT drain or restart polling.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    fatal_handler = AsyncMock()
    adapter.set_fatal_error_handler(fatal_handler)

    conflict_error = Exception("Conflict: terminated by other getUpdates request")

    # 1st conflict
    await adapter._handle_polling_conflict(conflict_error)
    assert adapter._polling_conflict_count == 1
    assert not adapter.has_fatal_error
    mock_updater.stop.assert_not_called()

    # 2nd conflict
    await adapter._handle_polling_conflict(conflict_error)
    assert adapter._polling_conflict_count == 2
    assert not adapter.has_fatal_error

    # 3rd conflict → fatal
    await adapter._handle_polling_conflict(conflict_error)
    assert adapter._polling_conflict_count == 3
    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "telegram_polling_conflict"
    assert adapter.fatal_error_retryable is False, (
        "polling_conflict fatal must be non-retryable so the reconnect watcher "
        "does NOT auto-restart the adapter"
    )

    # Handler must stop the updater on fatal
    mock_updater.stop.assert_called_once()
    # But must NOT restart polling
    mock_updater.start_polling.assert_not_called()


@pytest.mark.asyncio
async def test_conflict_counter_resets_after_60s_window():
    """
    If more than 60s elapses between conflicts, the counter resets.
    """
    adapter = _make_adapter()

    conflict_error = Exception("Conflict: terminated by other getUpdates request")

    # Two conflicts in quick succession
    await adapter._handle_polling_conflict(conflict_error)
    await adapter._handle_polling_conflict(conflict_error)
    assert adapter._polling_conflict_count == 2

    # Force the window to expire (set it to a time in the past)
    import time
    adapter._conflict_window_until = time.monotonic() - 1

    await adapter._handle_polling_conflict(conflict_error)
    assert adapter._polling_conflict_count == 1, (
        "Counter should reset when 60s window expires"
    )


@pytest.mark.asyncio
async def test_conflict_handler_noop_when_already_fatal():
    """
    Once polling_conflict fatal is set, subsequent conflict callbacks
    are no-ops.
    """
    adapter = _make_adapter()
    adapter._set_fatal_error("telegram_polling_conflict", "already fatal", retryable=False)

    conflict_error = Exception("Conflict: terminated by other getUpdates request")
    await adapter._handle_polling_conflict(conflict_error)

    assert adapter._polling_conflict_count == 0


# ── Drain helper (still exists, used by disconnect path) ───────────────

@pytest.mark.asyncio
async def test_drain_helper_noop_without_app():
    """_drain_polling_connections must be a no-op when _app is None."""
    adapter = _make_adapter()
    adapter._app = None
    # Should not raise
    await adapter._drain_polling_connections()
