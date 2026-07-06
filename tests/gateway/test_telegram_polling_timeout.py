"""
Tests for the start_polling() watchdog bound (#59614).

Background
----------
When both the primary Telegram API server and all fallback IPs are
unreachable simultaneously, ``app.updater.start_polling()`` can hang
indefinitely — neither connecting nor timing out — because the underlying
httpx connection pool is in a degraded state. Without a watchdog the
entire 3-tier reconnect ladder (``_handle_polling_network_error``,
``_polling_heartbeat_loop``, ``_platform_reconnect_watcher``) wedges at
attempt 1 and the gateway process stays alive but silent for hours.

The fix wraps both ``start_polling()`` call sites (bootstrap in
``_start_polling_resilient`` and reconnect in
``_handle_polling_network_error``) in ``asyncio.wait_for()`` with a
configurable bound (``self._polling_start_timeout``, default 30s,
matching the recent cua-driver fix). On timeout a clear ``RuntimeError``
is raised so the ladder advances.

These tests cover:
1. The watchdog fires when ``start_polling()`` blocks forever.
2. The watchdog does NOT fire when ``start_polling()`` returns quickly.
3. The config knob (``telegram.polling_start_timeout`` in config.yaml's
   ``extra`` dict, and ``HERMES_TELEGRAM_POLLING_START_TIMEOUT`` env var)
   is respected.

Refs: NousResearch/hermes-agent#59614
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

from plugins.platforms.telegram.adapter import (  # noqa: E402
    TelegramAdapter,
    _UPDATER_START_TIMEOUT,
)


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.discover_fallback_ips", _noop)


def _make_adapter(
    *, polling_start_timeout=None, env_timeout=None, env_clear=True
) -> TelegramAdapter:
    """Build an adapter with optional per-test config knob + env override."""
    if env_clear:
        # Wipe the env var so it doesn't bleed between tests.
        import os
        for var in ("HERMES_TELEGRAM_POLLING_START_TIMEOUT",):
            os.environ.pop(var, None)
    if env_timeout is not None:
        import os
        os.environ["HERMES_TELEGRAM_POLLING_START_TIMEOUT"] = str(env_timeout)
    extra = {}
    if polling_start_timeout is not None:
        extra["polling_start_timeout"] = polling_start_timeout
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra))


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def test_default_polling_start_timeout_constant():
    """The module-level default must be 30s (matches cua-driver convention)."""
    assert _UPDATER_START_TIMEOUT == 30.0


def test_default_polling_start_timeout_via_instance():
    """Without any config or env override the instance inherits 30s."""
    adapter = _make_adapter()
    assert adapter._polling_start_timeout == 30.0


def test_config_extra_knob_overrides_default():
    """``telegram.polling_start_timeout`` in config.yaml wins over the default."""
    adapter = _make_adapter(polling_start_timeout=12.5)
    assert adapter._polling_start_timeout == 12.5


def test_env_var_overrides_default():
    """``HERMES_TELEGRAM_POLLING_START_TIMEOUT`` env var wins over the default."""
    adapter = _make_adapter(env_timeout=7.0)
    assert adapter._polling_start_timeout == 7.0


def test_config_extra_knob_wins_over_env_var():
    """config.yaml's extra dict wins over the env var (per-platform > process)."""
    adapter = _make_adapter(polling_start_timeout=4.0, env_timeout=99.0)
    assert adapter._polling_start_timeout == 4.0


def test_config_extra_knob_floored_at_one_second():
    """A misconfigured 0 (or negative) cannot insta-timeout every start_polling."""
    adapter = _make_adapter(polling_start_timeout=0)
    assert adapter._polling_start_timeout == 1.0


def test_config_extra_knob_negative_clamped_up_to_one_second():
    """Negative values are floored to 1s (not silently inverted to infinity)."""
    adapter = _make_adapter(polling_start_timeout=-5)
    assert adapter._polling_start_timeout == 1.0


def test_config_extra_knob_capped_at_ten_minutes():
    """Huge values are capped at 600s so the watchdog stays meaningful."""
    adapter = _make_adapter(polling_start_timeout=99999)
    assert adapter._polling_start_timeout == 600.0


def test_env_var_non_finite_falls_back_to_default():
    """An env var set to ``inf`` or ``nan`` falls back to the module default."""
    adapter = _make_adapter(env_timeout="inf")
    assert adapter._polling_start_timeout == _UPDATER_START_TIMEOUT


def test_env_var_unparseable_falls_back_to_default():
    """An env var set to a non-numeric string falls back to the default."""
    adapter = _make_adapter(env_timeout="not-a-number")
    assert adapter._polling_start_timeout == _UPDATER_START_TIMEOUT


def test_config_extra_knob_unparseable_falls_back():
    """An unparseable string in config.extra falls back to the env/default."""
    import os
    os.environ["HERMES_TELEGRAM_POLLING_START_TIMEOUT"] = "11.0"
    try:
        # _make_adapter's default behaviour clears the env var first; we
        # need to bypass that by skipping the env-clear and instead pass
        # the knob in extra (unparseable).
        adapter = TelegramAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"polling_start_timeout": "nope"},
            )
        )
        # config knob unparseable → fall through to env var (11.0).
        assert adapter._polling_start_timeout == 11.0
    finally:
        os.environ.pop("HERMES_TELEGRAM_POLLING_START_TIMEOUT", None)


# ---------------------------------------------------------------------------
# Bootstrap path: _start_polling_resilient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_start_polling_hang_raises_clear_exception():
    """When start_polling() blocks forever, the bootstrap path must NOT wedge.

    The watchdog must fire, _looks_like_network_error(RuntimeError) must
    pass, and the recovery task must be scheduled in the background so the
    gateway stays alive.
    """
    adapter = _make_adapter(polling_start_timeout=0.05)
    adapter._polling_error_task = None

    mock_updater = MagicMock()
    mock_updater.running = True

    async def _hang_forever(**kwargs):
        # Simulate the exact symptom from #59614: never connects, never
        # times out, never raises. asyncio.wait_for must intervene.
        await asyncio.sleep(9999)

    mock_updater.start_polling = _hang_forever
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._schedule_polling_recovery = MagicMock()

    # The call must return False (degraded, not fatal) within a reasonable
    # bound — NOT hang the test. If the watchdog is missing, this test
    # would block until pytest's own timeout (60s+) and fail.
    result = await asyncio.wait_for(
        adapter._start_polling_resilient(
            drop_pending_updates=False,
            error_callback=lambda error: None,
        ),
        timeout=5.0,
    )

    assert result is False, (
        "Expected _start_polling_resilient to return False (degraded) "
        "when start_polling() hangs, not True"
    )
    adapter._schedule_polling_recovery.assert_called_once()
    err = adapter._schedule_polling_recovery.call_args.args[0]
    assert isinstance(err, RuntimeError), (
        f"Expected a clear RuntimeError, got {type(err).__name__}"
    )
    assert "timed out" in str(err).lower()
    assert adapter._schedule_polling_recovery.call_args.kwargs["reason"] == "polling bootstrap timeout"
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_bootstrap_start_polling_fast_completes_normally():
    """When start_polling() returns quickly, the watchdog must NOT fire."""
    adapter = _make_adapter(polling_start_timeout=2.0)

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.start_polling = AsyncMock()  # succeeds
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._schedule_polling_recovery = MagicMock()

    result = await adapter._start_polling_resilient(
        drop_pending_updates=False,
        error_callback=lambda error: None,
    )

    assert result is True
    mock_updater.start_polling.assert_awaited_once_with(
        allowed_updates=mock_updater.start_polling.await_args.kwargs.get(
            "allowed_updates"
        )
        if False else None,  # placeholder; we just want the call to have happened
        drop_pending_updates=False,
        error_callback=adapter._start_polling_resilient.__self__ if False else None,  # placeholder
    ) if False else mock_updater.start_polling.assert_awaited_once()
    adapter._schedule_polling_recovery.assert_not_called()
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_bootstrap_start_polling_propagates_unrelated_exceptions():
    """A non-network, non-conflict exception must propagate (not be swallowed)."""
    adapter = _make_adapter(polling_start_timeout=2.0)

    mock_updater = MagicMock()
    mock_updater.start_polling = AsyncMock(
        side_effect=RuntimeError("totally unrelated bug in PTB")
    )
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with pytest.raises(RuntimeError, match="totally unrelated"):
        await adapter._start_polling_resilient(
            drop_pending_updates=False,
            error_callback=lambda error: None,
        )


# ---------------------------------------------------------------------------
# Reconnect path: _handle_polling_network_error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_start_polling_hang_advances_ladder():
    """When start_polling() hangs on a retry, the chained-retry must fire.

    Regression test for the core #59614 symptom: the 10-retry ladder must
    NOT stall at attempt 1 when the underlying call wedges. Bounding
    start_polling() with asyncio.wait_for() lets the ladder re-enter itself
    so the operator eventually sees the 10-retry escalation path or the
    reconnect watcher takes over.
    """
    adapter = _make_adapter(polling_start_timeout=0.05)
    adapter._polling_network_error_count = 1
    adapter._polling_error_task = None

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()

    async def _hang_forever(**kwargs):
        await asyncio.sleep(9999)

    mock_updater.start_polling = _hang_forever

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    # Drain is irrelevant for this test but must not raise.
    async def _noop_drain():
        return None
    adapter._drain_polling_connections = _noop_drain

    # Spy on the chained-retry handler so we can verify it was scheduled
    # with a clear RuntimeError after the watchdog fires, without letting
    # the chained retry actually run to completion (which under a mocked
    # asyncio.sleep would either burn the rest of the 10-retry budget
    # synchronously and set a fatal error, or hang the test).
    #
    # We do this by wrapping the real method: the wrapper records the
    # argument and then no-ops, so the real ladder still runs through its
    # watchdog → timeout → chained-retry path, but the chained retry
    # itself is captured by the spy instead of entering another attempt.
    chained_calls = []

    async def _spy_handle(error):
        chained_calls.append(error)
        return None

    adapter._handle_polling_network_error = _spy_handle

    # The full handler must complete within a reasonable bound. If the
    # watchdog is missing, _handle_polling_network_error blocks at the
    # ``await start_polling(...)`` line and this test times out.
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await asyncio.wait_for(
            TelegramAdapter._handle_polling_network_error(
                adapter, OSError("primary down")
            ),
            timeout=5.0,
        )

    # Give the asyncio.ensure_future()-scheduled chained-retry coroutine a
    # chance to run up to its first await so the spy captures it.
    await asyncio.sleep(0)
    # Drain any background tasks we created so the spy is fully recorded.
    pending = [t for t in adapter._background_tasks if not t.done()]
    for t in pending:
        try:
            await asyncio.wait_for(t, timeout=1.0)
        except (asyncio.CancelledError, Exception):
            pass

    # The watchdog fired → the chained retry was scheduled with a
    # clear RuntimeError that names the timeout, not the bare
    # asyncio.TimeoutError.
    assert chained_calls, (
        "Expected the watchdog to schedule a chained retry; no call was "
        "captured by the spy"
    )
    err = chained_calls[0]
    # Debug aid: surface the actual error if assertion fails below.
    assert isinstance(err, RuntimeError), (
        f"Expected chained-retry to receive a clear RuntimeError, got "
        f"{type(err).__name__}: {err!r}"
    )
    assert "timed out" in str(err).lower()


@pytest.mark.asyncio
async def test_reconnect_start_polling_fast_completes_normally():
    """When start_polling() succeeds quickly, the watchdog must NOT fire."""
    adapter = _make_adapter(polling_start_timeout=2.0)
    adapter._polling_network_error_count = 3

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()  # succeeds

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(OSError("flap"))

    # Counter resets on a successful start_polling().
    assert adapter._polling_network_error_count == 0
    # Self-scheduled retry must NOT be created (success path).
    assert adapter._polling_error_task is None or adapter._polling_error_task.done()

    # Clean up the heartbeat probe task scheduled after the success.
    pending = [t for t in adapter._background_tasks if not t.done()]
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_reconnect_start_polling_exception_still_chains():
    """A non-timeout exception from start_polling() must still trigger chained retry.

    The watchdog must only catch asyncio.TimeoutError; a regular PTB
    NetworkError raised from start_polling() must continue down the
    existing except branch. This protects against the watchdog accidentally
    swallowing real Telegram errors.
    """
    adapter = _make_adapter(polling_start_timeout=2.0)
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(
        side_effect=Exception("Timed out: PTB caught a TimeoutError")
    )

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(OSError("flap"))

    # Chained retry task must have been created (existing behaviour preserved).
    assert adapter._polling_error_task is not None
    assert not adapter._polling_error_task.done()
    adapter._polling_error_task.cancel()
    try:
        await adapter._polling_error_task
    except (asyncio.CancelledError, Exception):
        pass


# ---------------------------------------------------------------------------
# Watchdog respects the per-instance config knob
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_watchdog_respects_per_instance_timeout():
    """A 0.05s instance timeout must fire faster than the 30s default would."""
    adapter = _make_adapter(polling_start_timeout=0.05)
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()

    call_count = {"n": 0}

    async def _hang(**kwargs):
        call_count["n"] += 1
        await asyncio.sleep(9999)

    mock_updater.start_polling = _hang

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    async def _noop_drain():
        return None
    adapter._drain_polling_connections = _noop_drain

    import time

    t0 = time.monotonic()
    with patch("asyncio.sleep", new_callable=AsyncMock):
        await asyncio.wait_for(
            adapter._handle_polling_network_error(OSError("x")),
            timeout=5.0,
        )
    elapsed = time.monotonic() - t0

    # The watchdog should have fired well under 1s. Without the per-instance
    # knob, the default 30s bound would have made this take ~30s (and the
    # pytest-timeout would still pass, so the test wouldn't catch the bug).
    assert elapsed < 2.0, (
        f"Watchdog took {elapsed:.2f}s; expected well under 2s with a "
        "0.05s per-instance timeout. Did the config knob propagate?"
    )
    # start_polling was called at least once before the timeout fired.
    assert call_count["n"] >= 1

    # Clean up the chained retry.
    if adapter._polling_error_task and not adapter._polling_error_task.done():
        adapter._polling_error_task.cancel()
        try:
            await adapter._polling_error_task
        except (asyncio.CancelledError, Exception):
            pass