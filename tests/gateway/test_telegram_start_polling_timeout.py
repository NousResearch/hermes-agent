"""Regression tests for #59614: start_polling() must be time-bounded.

When both the primary Telegram API server and all fallback IPs are unreachable,
``await app.updater.start_polling(...)`` can block forever inside an exhausted
httpx connection pool — it neither returns nor raises. Unbounded, that wedges:

1. the network-error reconnect ladder (stuck inside attempt 1, never advances),
2. the heartbeat loop (sees the recovery task as alive-but-wedged and skips),
3. the fatal-error escalation (never reached).

The fix wraps every ``start_polling()`` await in ``asyncio.wait_for`` with
``_UPDATER_START_TIMEOUT`` so a hung call raises and feeds the existing retry
ladder. These tests patch the timeout down to keep the suite fast.
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


async def _hang_forever(**kwargs):
    await asyncio.sleep(1000)


def _bare_adapter():
    a = TelegramAdapter.__new__(TelegramAdapter)
    # `name` / `has_fatal_error` are read-only base-class properties; set the
    # backing fields they derive from instead.
    from gateway.config import Platform

    a.platform = Platform.TELEGRAM
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._polling_network_error_count = 0
    a._polling_conflict_count = 0
    a._polling_error_callback_ref = None
    a._background_tasks = set()
    a._send_path_degraded = False
    return a


@pytest.mark.asyncio
async def test_network_ladder_start_polling_hang_does_not_wedge(monkeypatch):
    """A hung start_polling() in _handle_polling_network_error must time out
    and advance the ladder instead of blocking forever (#59614 core repro)."""
    monkeypatch.setattr(tg_adapter, "_UPDATER_START_TIMEOUT", 0.2)
    a = _bare_adapter()
    a._polling_network_error_count = 0  # attempt 1 → 5s backoff before start_polling

    app = MagicMock()
    app.updater = AsyncMock()
    app.updater.start_polling = _hang_forever
    app.updater.running = False
    a._app = app

    with patch.object(a, "_drain_polling_connections", new=AsyncMock()), \
         patch.object(
             tg_adapter.asyncio, "ensure_future",
             side_effect=lambda coro: (coro.close(), asyncio.get_event_loop().create_future())[1],
         ):
        # Unbounded, this await hangs past the 30s wait_for and fails the
        # test; bounded, the handler waits its 5s backoff, times out the hung
        # start_polling() in 0.2s, schedules the chained retry (captured by
        # the ensure_future patch), and returns.
        await asyncio.wait_for(
            a._handle_polling_network_error(Exception("net down")), timeout=30
        )


@pytest.mark.asyncio
async def test_bootstrap_start_polling_hang_schedules_recovery(monkeypatch):
    """_start_polling_resilient: a hung bootstrap start_polling() must raise
    TimeoutError (an OSError → classified as network error) and schedule
    background recovery instead of blocking connect() forever."""
    monkeypatch.setattr(tg_adapter, "_UPDATER_START_TIMEOUT", 0.2)
    a = _bare_adapter()

    app = MagicMock()
    app.updater = AsyncMock()
    app.updater.start_polling = _hang_forever
    a._app = app

    scheduled = []
    monkeypatch.setattr(
        a, "_schedule_polling_recovery",
        lambda err, reason: scheduled.append((err, reason)),
        raising=False,
    )

    ok = await asyncio.wait_for(
        a._start_polling_resilient(drop_pending_updates=False, error_callback=None),
        timeout=10,
    )
    assert ok is False
    assert len(scheduled) == 1
    assert isinstance(scheduled[0][0], (TimeoutError, asyncio.TimeoutError))


@pytest.mark.asyncio
async def test_start_polling_success_path_unaffected(monkeypatch):
    """Sanity: a fast start_polling() still returns True through the wrapper."""
    monkeypatch.setattr(tg_adapter, "_UPDATER_START_TIMEOUT", 5.0)
    a = _bare_adapter()

    app = MagicMock()
    app.updater = AsyncMock()
    app.updater.start_polling = AsyncMock(return_value=None)
    a._app = app

    ok = await a._start_polling_resilient(drop_pending_updates=False, error_callback=None)
    assert ok is True
    app.updater.start_polling.assert_awaited_once()


def test_every_start_polling_call_site_is_time_bounded():
    """Bug-class contract: every `updater.start_polling(` await in the adapter
    must be wrapped in asyncio.wait_for. A new unbounded call site reintroduces
    the #59614 wedge."""
    import inspect
    import re

    src = inspect.getsource(tg_adapter)
    # Find each start_polling( call and check an enclosing wait_for within the
    # preceding 6 lines (the wrapper always sits directly above).
    lines = src.splitlines()
    unbounded = []
    for i, line in enumerate(lines):
        if re.search(r"updater\.start_polling\(", line) and "def " not in line:
            window = "\n".join(lines[max(0, i - 6):i + 1])
            if "wait_for" not in window:
                unbounded.append((i + 1, line.strip()))
    assert not unbounded, f"unbounded start_polling() call sites: {unbounded}"
