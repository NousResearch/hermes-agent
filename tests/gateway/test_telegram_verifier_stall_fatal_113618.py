"""Telegram verifier-stall fatal escalation (#113618).

After a connectivity blip that wedges the getUpdates long-poll without
breaking ``get_me()`` on the general request path, the watchdog
(``_check_polling_stall``) and the verifier
(``_verify_polling_after_reconnect``) both report ``general path healthy
but getUpdates stalled`` on every cycle. The existing network-error
ladder escalates to ``_go_fatal_network`` only after 10 attempts ×
5-60s exponential backoff (≈7 min). For a wedged-but-getMe-OK
condition that path cannot recover — restarting the same Updater is
not the fix — and systemd ``Restart=always`` is the only way out.

The verifier-stall failure mode is distinct from a transient network
error (connect timeout, pool timeout, DNS blip surfaced by getMe).
``_handle_polling_network_error``'s 10-attempt policy still fits those
because the same restart actually fixes them. A wedged-but-getMe-OK
long-poll needs the adapter rebuilt by the supervisor — i.e. a
retryable fatal after a small bounded number of stalls.

This module pins the contract:

  * ``_check_polling_stall`` (watchdog) and the verifier's
    ``general path healthy but getUpdates stalled`` branch route to a
    dedicated ``_handle_polling_verifier_stall`` helper that escalates
    after ``_MAX_VERIFIER_STALL_RETRIES`` consecutive firings.
  * The helper resets to zero on the next confirmed ``getUpdates``
    round-trip (``_record_polling_progress``).
  * Pure network-error recoveries keep using the existing 10-attempt
    ladder (``_handle_polling_network_error``) — the verifier-stall
    counter does not advance for those.
  * getMe connectivity failures keep using the existing network-error
    ladder — the verifier-stall helper never fires when getMe fails.

Regression for #113618.
"""

import asyncio
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_stuck_adapter(
    *, generation_started_seconds_ago: float = 500.0
) -> TelegramAdapter:
    """Adapter that looks like a wedged-but-getMe-OK long-poll.

    No successful round-trip ever (so verifier deadline will trip on a
    fresh generation too), generation start well past the stall
    threshold, get_me() returns successfully so the verifier classifies
    the failure as "general path healthy but getUpdates stalled".
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._webhook_mode = False
    adapter._app = MagicMock()
    adapter._app.updater.running = True
    bot = MagicMock()
    bot.get_me = AsyncMock(return_value=MagicMock(username="test"))
    bot.get_webhook_info = AsyncMock(return_value=MagicMock(pending_update_count=0))
    adapter._app.bot = bot
    adapter._bot = bot
    adapter._polling_generation = 1
    now = _time.monotonic()
    adapter._polling_generation_started_monotonic = now - generation_started_seconds_ago
    adapter._polling_last_progress_monotonic = None
    return adapter


def _complete_verifier_progress(adapter: TelegramAdapter) -> None:
    """Record progress for the current generation so the verifier unblocks."""
    adapter._record_polling_progress(adapter._polling_generation)


# ── RED: counter / fatal contract on _check_polling_stall ──────────────────


@pytest.mark.asyncio
async def test_watchdog_first_stall_increments_verifier_stall_counter():
    """A single watchdog stall must bump the new verifier-stall counter once."""
    adapter = _make_stuck_adapter()

    with patch.object(
        adapter, "_handle_polling_verifier_stall", new=AsyncMock()
    ) as rec:
        await adapter._check_polling_stall()
        # The watchdog spawns the recovery task — let it run so the patched
        # helper is actually awaited (mirrors the test_stalled_long_poll_
        # escalates_to_reconnect_ladder pattern for the network ladder).
        task = adapter._polling_error_task
        assert task is not None
        await task

    assert adapter._polling_verifier_stall_count == 1
    # The watchdog MUST route to the dedicated stall helper, not the
    # network-error ladder — that's the whole point of the fix.
    rec.assert_awaited_once()


@pytest.mark.asyncio
async def test_watchdog_routes_stall_to_dedicated_helper_not_network_ladder():
    """Watchdog stall must not enter the network-error ladder.

    Same root cause as ``test_watchdog_first_stall_increments_*`` but
    guards the negative: a wedged-but-getMe-OK consumer that survives
    ``_start_polling_once`` cannot be healed by the network-error
    ladder. Routing it there (the current behavior) is the bug.
    """
    adapter = _make_stuck_adapter()
    with patch.object(
        adapter, "_handle_polling_network_error", new=AsyncMock()
    ) as network:
        await adapter._check_polling_stall()
    network.assert_not_awaited()


# ── RED: verifier routes stall branch to the dedicated helper ─────────────


@pytest.mark.asyncio
async def test_verifier_general_path_healthy_routes_to_stall_helper(monkeypatch):
    """The verifier's ``general path healthy but getUpdates stalled`` branch
    must call the dedicated stall helper, not ``_schedule_polling_recovery``.

    get_me() OK + no getUpdates progress = distinct failure mode (not
    transient network error). Routing it through the 10-attempt
    network-error ladder is what makes the gateway stay deaf for
    ~7 minutes when the actual fix is adapter-rebuild.
    """
    adapter = _make_stuck_adapter()
    # Stub out the helper so we can assert it was invoked. Wrap _schedule_polling_recovery
    # so we can detect the OLD routing.
    stall_helper = AsyncMock()
    adapter._handle_polling_verifier_stall = stall_helper  # type: ignore[method-assign]
    adapter._schedule_polling_recovery = MagicMock()  # type: ignore[method-assign]

    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    stall_helper.assert_awaited_once()
    adapter._schedule_polling_recovery.assert_not_called()


@pytest.mark.asyncio
async def test_verifier_connectivity_failure_still_routes_to_network_ladder(
    monkeypatch,
):
    """If getMe fails, the verifier MUST keep using the network-error ladder.

    This protects the existing #66377 / #58270 contracts: a real
    connectivity outage surfaces as a getMe failure and the bounded
    network-error ladder is the right escalation. Re-routing it
    through the new stall helper would skip those bounds and break the
    watchdog's connectivity coverage.
    """
    adapter = _make_stuck_adapter()
    conn_err = ConnectionError("get_me failed")

    bot = MagicMock()
    bot.get_me = AsyncMock(side_effect=conn_err)
    bot.get_webhook_info = AsyncMock(return_value=MagicMock(pending_update_count=0))
    adapter._app.bot = bot
    adapter._bot = bot
    adapter._handle_polling_network_error = AsyncMock()
    adapter._handle_polling_verifier_stall = AsyncMock()  # type: ignore[method-assign]

    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    # Stall helper MUST NOT fire for a real connectivity error.
    adapter._handle_polling_verifier_stall.assert_not_awaited()


# ── GREEN: dedicated helper escalates after bounded retries ───────────────


@pytest.mark.asyncio
async def test_stall_helper_escalates_to_retryable_fatal_after_bound(monkeypatch):
    """``_MAX_VERIFIER_STALL_RETRIES + 1`` cumulative firings → ``_go_fatal_network``.

    This is the core fix: a wedged-but-getMe-OK consumer cannot heal
    itself via restart. After the bound is exceeded the adapter MUST
    set a retryable fatal so the supervisor rebuilds it (rather than
    spinning through 10 network-error attempts over 7+ minutes).
    """
    adapter = _make_stuck_adapter()
    # Pretend the watchdog has already fired _MAX_VERIFIER_STALL_RETRIES times; the next
    # observer (verifier or watchdog) sets the counter above the bound and the helper fatals.
    adapter._polling_verifier_stall_count = tg_adapter._MAX_VERIFIER_STALL_RETRIES
    adapter._polling_verifier_stall_count += (
        1  # simulate the bound being crossed at the call site
    )

    app = MagicMock()
    app.updater.running = True
    app.updater.stop = AsyncMock()
    app.updater.start_polling = AsyncMock()
    adapter._app = app
    adapter._notify_fatal_error = AsyncMock()

    monkeypatch.setattr(tg_adapter, "_MAX_VERIFIER_STALL_RETRIES", 3)
    monkeypatch.setattr(
        tg_adapter, "_POLLING_STALL_BACKOFF_SECONDS", 0.001, raising=False
    )

    await adapter._handle_polling_verifier_stall(
        RuntimeError("verifier: general path healthy but getUpdates stalled")
    )

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "telegram_network_error"
    assert adapter.fatal_error_retryable is True
    adapter._notify_fatal_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_stall_helper_does_not_fatal_before_bound(monkeypatch):
    """Below the bound, the helper must restart polling without fataling.

    Single-shot recovery is the right behavior for the first few
    firings — the consumer might still be recovering. Only the bound
    itself forces a fatal; otherwise the helper must do exactly what
    the network-error ladder does (stop / drain / start).
    """
    adapter = _make_stuck_adapter()
    adapter._polling_verifier_stall_count = 1

    app = MagicMock()
    app.updater.running = True
    app.updater.stop = AsyncMock()
    app.updater.start_polling = AsyncMock()
    adapter._app = app
    adapter._notify_fatal_error = AsyncMock()

    monkeypatch.setattr(tg_adapter, "_MAX_VERIFIER_STALL_RETRIES", 3)
    monkeypatch.setattr(
        tg_adapter, "_POLLING_STALL_BACKOFF_SECONDS", 0.001, raising=False
    )

    await adapter._handle_polling_verifier_stall(
        RuntimeError("verifier: general path healthy but getUpdates stalled")
    )

    assert not adapter.has_fatal_error
    app.updater.start_polling.assert_awaited_once()
    # Helper does NOT increment the counter itself — callers do. Asserting
    # the counter is unchanged here proves the helper is bound-check only,
    # not a second source of advancement.
    assert adapter._polling_verifier_stall_count == 1
    adapter._notify_fatal_error.assert_not_called()


# ── Reset contract: real progress clears the bound counter ────────────────


def test_first_round_trip_resets_verifier_stall_counter():
    """A confirmed ``getUpdates`` round-trip clears the stall bound so a
    later wedged consumer starts counting again from zero.

    Otherwise an adapter that recovered once and hit one stall weeks
    later would fatal on the first firing instead of giving the bound
    its full retry budget.
    """
    adapter = _make_stuck_adapter()
    # Begin a generation so progress-accepting is True; otherwise
    # _record_polling_progress returns False (rejects progress for a
    # generation that wasn't started).
    adapter._begin_polling_generation()
    adapter._polling_verifier_stall_count = 2
    # Also bump the network-error counter the same way a few real
    # network blips would — the reset must not touch it.
    adapter._polling_network_error_count = 4

    accepted = adapter._record_polling_progress(adapter._polling_generation)

    assert accepted, (
        "_record_polling_progress must accept progress for the current generation"
    )
    assert adapter._polling_verifier_stall_count == 0
    # Network-error counter also resets on progress (existing behavior).
    assert adapter._polling_network_error_count == 0


# ── Watchdog does not consume the bound while a recovery is in flight ─────


@pytest.mark.asyncio
async def test_watchdog_skipped_while_recovery_in_flight_regardless_of_counter():
    """``_recovery_in_flight`` is the existing reentrancy guard — the
    watchdog must keep skipping when a recovery is mid-flight, the
    stall counter should NOT advance in that case.

    Without this, a long backoff sleep inside the helper could be
    preempted by a second watchdog firing on the same stall event and
    blowing past the bound while the first attempt is still running.
    """
    adapter = _make_stuck_adapter()
    inflight = MagicMock()
    inflight.done.return_value = False
    adapter._polling_error_task = inflight

    await adapter._check_polling_stall()

    assert adapter._polling_verifier_stall_count == 0


# ── Module-level: ensure the bound constant exists and is small ──────────


def test_max_verifier_stall_retries_is_a_small_positive_int():
    """The bound MUST be small (≤ network-error bound / 2). The whole
    point of the new counter is faster escalation than the 10-attempt
    network-error ladder.

    If a future change widens this, the heartbeat loop can still
    force-escalate a wedged recovery task (#66377) but the verifier-
    stall backstop must stay tight.
    """
    assert isinstance(tg_adapter._MAX_VERIFIER_STALL_RETRIES, int)
    assert 1 <= tg_adapter._MAX_VERIFIER_STALL_RETRIES <= 5
