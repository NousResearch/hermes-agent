"""
Tests for Telegram polling-pool wedge detection.

Issue #5729: cold-boot or post-reconnect resolver/network wedge leaves the
long-poll's `_request[0]` connection pool unable to make progress while the
general `_request[1]` pool stays healthy. The pre-existing heartbeat probe
called `bot.get_me()` which routes through `_request[1]` — so a wedged
polling pool falsely passed the probe ("getMe worked but messages weren't
handled" — rblakemesser field report 2026-05-07).

These tests pin the corrected probe behaviour:

  1. The probe routes through `_request[0]` (the polling pool), not the
     general pool. A wedged `_request[0]` is detected even when `_request[1]`
     and `bot.get_me()` are healthy.
  2. The probe is scheduled after cold-boot `start_polling`, not only after
     error-driven reconnects (so silent cold-boot wedges that produce no
     exception are detected at all).
  3. A `bot._request` shape mismatch (custom `HTTPXRequest` injection / PTB
     drift) downgrades to a logged warning and skips the probe — no crash.
  4. The probe and `_drain_polling_connections` mutually exclude via a shared
     `asyncio.Lock` so the probe never hits a half-torn-down pool.
  5. Two probe failures within a 5-minute window escalate to a fatal-retryable
     state so the supervisor restarts the gateway.
  6. A pending probe task is cancelled by `disconnect()`.
  7. The probe does not advance the long-poll's update offset.
  8. Probe-related logs do not contain `bot.base_url` (which embeds the bot
     token in the Telegram URL format).

Council adversarial review session: a8c9534fee5e (2026-05-07).
Local probe primitive validation: 2026-05-07 (separate Bot, getMe via
_request[0], 357.7ms — well under the proposed 65s timeout).
"""

import asyncio
import sys
from types import SimpleNamespace
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
    telegram_mod.error.NetworkError = type("NetworkError", (OSError,), {})
    telegram_mod.error.TimedOut = type("TimedOut", (OSError,), {})
    telegram_mod.error.BadRequest = type("BadRequest", (Exception,), {})

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)
    sys.modules.setdefault("telegram.error", telegram_mod.error)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("gateway.platforms.telegram.discover_fallback_ips", _noop)
    monkeypatch.setattr("gateway.platforms.telegram.HTTPXRequest", lambda **kwargs: MagicMock())


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _wire_adapter_with_pools(
    adapter: TelegramAdapter,
    *,
    pool0_post: AsyncMock,
    pool1_post: AsyncMock,
    base_url: str = "https://api.telegram.org/bot<REDACTED>",
    updater_running: bool = True,
) -> MagicMock:
    """Attach a mock _app whose bot._request is a 2-tuple of distinct pools."""
    pool0 = MagicMock()
    pool0.post = pool0_post
    pool1 = MagicMock()
    pool1.post = pool1_post

    mock_bot = MagicMock()
    mock_bot._request = (pool0, pool1)
    mock_bot.base_url = base_url
    # get_me must be present so a regression to the old probe primitive is
    # observable (we assert it is NOT called by the new probe).
    mock_bot.get_me = AsyncMock(return_value=MagicMock(id=1, is_bot=True))

    mock_updater = MagicMock()
    mock_updater.running = updater_running
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()

    mock_app = MagicMock()
    mock_app.bot = mock_bot
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._bot = mock_bot
    return mock_app


# -----------------------------------------------------------------------------
# Iteration 1 — probe routes through _request[0], not _request[1]
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_routes_through_polling_pool_not_general_pool():
    """
    The corrected heartbeat probe must call `_request[0].post` (polling pool),
    not `bot.get_me()` (which routes through `_request[1]`, the general pool).

    A wedged polling pool with a healthy general pool — exactly the
    rblakemesser failure mode — must be detected.
    """
    adapter = _make_adapter()

    # _request[0]: wedged — never resolves. Use an Event that's never set so
    # the hang survives test-wide asyncio.sleep patches.
    never_set = asyncio.Event()

    async def _hang(*args, **kwargs):
        await never_set.wait()

    pool0_post = AsyncMock(side_effect=_hang)
    # _request[1]: healthy — returns immediately
    pool1_post = AsyncMock(return_value={"id": 1, "is_bot": True})

    _wire_adapter_with_pools(
        adapter,
        pool0_post=pool0_post,
        pool1_post=pool1_post,
    )

    # Skip the 60s heartbeat sleep but keep the probe's wait_for honoring real
    # timeouts so the wedge is detected.
    real_wait_for = asyncio.wait_for

    async def _fast_sleep(seconds):
        return None

    with patch("asyncio.sleep", new=_fast_sleep), \
         patch.object(adapter, "_handle_polling_network_error",
                      new_callable=AsyncMock) as mock_handler:
        # Make the probe's wait_for use a tiny timeout so the test runs fast,
        # but only when called from the probe (preserve the real timeout for
        # any nested wait_fors).
        async def _short_wait_for(coro, timeout):
            return await real_wait_for(coro, timeout=0.05)
        with patch("gateway.platforms.telegram.asyncio.wait_for",
                   side_effect=_short_wait_for):
            await adapter._verify_polling_after_reconnect()

    # The new probe must hit the polling pool.
    assert pool0_post.await_count == 1, (
        "Probe must call _request[0].post exactly once "
        f"(got {pool0_post.await_count} calls)"
    )
    # The new probe must NOT use bot.get_me (which routes through _request[1]).
    assert adapter._bot.get_me.await_count == 0, (
        "Probe must not call bot.get_me() — that routes through the wrong pool. "
        f"get_me was called {adapter._bot.get_me.await_count} times."
    )
    # And on wedge detection it must re-enter the reconnect ladder.
    assert mock_handler.await_count == 1, (
        "Wedge detection must re-enter _handle_polling_network_error "
        f"(was called {mock_handler.await_count} times)"
    )


# -----------------------------------------------------------------------------
# Iteration 2 — probe scheduled on cold-boot (not just error-driven reconnects)
# -----------------------------------------------------------------------------

def _wire_cold_boot_mocks(monkeypatch):
    """Build the minimal mocks needed for connect() to reach start_polling."""
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: None,
    )
    monkeypatch.setattr("asyncio.sleep", AsyncMock())


def _make_cold_boot_app(monkeypatch) -> SimpleNamespace:
    """Construct a minimal Application mock that lets connect() succeed."""
    updater = SimpleNamespace(
        start_polling=AsyncMock(),
        stop=AsyncMock(),
        running=True,
    )
    bot = SimpleNamespace(
        set_my_commands=AsyncMock(),
        delete_webhook=AsyncMock(),
        _request=(MagicMock(post=AsyncMock()), MagicMock(post=AsyncMock())),
        base_url="https://api.telegram.org/bot<REDACTED>",
        get_me=AsyncMock(return_value=MagicMock(id=1, is_bot=True)),
    )
    app = SimpleNamespace(
        bot=bot,
        updater=updater,
        add_handler=MagicMock(),
        initialize=AsyncMock(),
        start=AsyncMock(),
    )
    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.return_value = app
    monkeypatch.setattr(
        "gateway.platforms.telegram.Application",
        SimpleNamespace(builder=MagicMock(return_value=builder)),
    )
    return app


@pytest.mark.asyncio
async def test_cold_boot_schedules_polling_pool_probe(monkeypatch):
    """
    After a successful cold-boot start_polling(), a polling-pool wedge probe
    must be scheduled. Without this, a long-poll that starts wedged (cold-boot
    DNS / TCP race that doesn't raise an exception) is invisible to the
    error-callback path and the gateway sits silent until manual restart
    (issue #5729, rblakemesser field report 2026-05-07).
    """
    adapter = _make_adapter()
    _wire_cold_boot_mocks(monkeypatch)
    _make_cold_boot_app(monkeypatch)

    # Stub the scheduling helper directly. Asserting on it (a regular Mock)
    # avoids the await-ordering subtleties of AsyncMock + ensure_future.
    adapter._schedule_polling_pool_probe = MagicMock()

    ok = await adapter.connect()
    assert ok is True, (
        f"connect() must succeed under the standard cold-boot mocks "
        f"(fatal={adapter.fatal_error_code})"
    )

    assert adapter._schedule_polling_pool_probe.call_count >= 1, (
        "Cold-boot start_polling must call _schedule_polling_pool_probe(); "
        f"got {adapter._schedule_polling_pool_probe.call_count} calls."
    )


# -----------------------------------------------------------------------------
# Iteration 3 — shape-check fallback for bot._request mismatch
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_skips_when_request_shape_mismatch(caplog):
    """
    `bot._request` is PTB-internal API. A custom HTTPXRequest injection
    via ApplicationBuilder.request() / get_updates_request() can replace the
    tuple, and a PTB minor bump may rename or restructure it. The probe must
    log a warning and skip — never crash, never falsely declare wedge.

    Also pins token redaction: with a realistic-looking token in `base_url`,
    no log line in any probe path may contain the token literal.
    """
    import logging
    adapter = _make_adapter()

    # Token sentinel — chosen to NOT match GitHub's secret-scanning regex
    # for Telegram bot tokens (which expects the standard
    # `<digits>:<35-char-base64ish>` format). Test logic only needs a
    # distinctive string to assert is absent from logs; it does not need
    # to look like a real token.
    test_token = "TEST-FAKE-TOKEN-FOR-CI-DO-NOT-SCAN"

    # _request is a non-conforming object: not a sequence with .post on [0].
    mock_bot = MagicMock()
    mock_bot._request = object()  # No __getitem__, no .post
    mock_bot.base_url = f"https://api.telegram.org/bot{test_token}"
    mock_bot.get_me = AsyncMock(return_value=MagicMock(id=1, is_bot=True))

    mock_updater = MagicMock()
    mock_updater.running = True

    mock_app = MagicMock()
    mock_app.bot = mock_bot
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._bot = mock_bot

    with patch("asyncio.sleep", new_callable=AsyncMock), \
         patch.object(adapter, "_handle_polling_network_error",
                      new_callable=AsyncMock) as mock_handler, \
         caplog.at_level(logging.WARNING, logger="gateway.platforms.telegram"):
        # Must not raise.
        await adapter._verify_polling_after_reconnect()

    # Skip semantics: no false-positive wedge declaration on shape mismatch.
    assert mock_handler.await_count == 0, (
        "Shape mismatch must not be treated as a wedge "
        f"(handler called {mock_handler.await_count} times)"
    )
    # Operator-visible signal so the missing protection is not silent.
    assert any(
        "polling pool probe skipped" in rec.message.lower()
        or "request shape" in rec.message.lower()
        or "_request" in rec.message
        for rec in caplog.records
    ), (
        "Expected a WARNING log on shape mismatch; got: "
        f"{[rec.message for rec in caplog.records]}"
    )
    # Token must not leak in ANY captured log line, regardless of message
    # context. Strong assertion — checks for the literal token, not a
    # weaker keyword combination.
    for rec in caplog.records:
        msg = rec.getMessage()
        assert test_token not in msg, (
            f"Probe log leaks bot token (sentinel {test_token!r}): {msg!r}"
        )


# -----------------------------------------------------------------------------
# Iteration 4 — probe and drain mutually exclude via shared asyncio.Lock
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_and_drain_mutually_exclude():
    """
    `_drain_polling_connections` resets `_request[0]` (calls shutdown + init).
    During that window, probing the same pool would hit a half-torn-down
    object. The probe and drain must serialize via a shared `asyncio.Lock`
    so the probe never fires concurrent with drain.
    """
    adapter = _make_adapter()

    # Build an _app with a working _request[0] so the probe would otherwise
    # succeed quickly.
    pool0_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    pool1_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    _wire_adapter_with_pools(
        adapter,
        pool0_post=pool0_post,
        pool1_post=pool1_post,
    )

    # Hold the shared lock as if drain were in its critical section.
    assert hasattr(adapter, "_probe_drain_lock"), (
        "Adapter must expose a shared asyncio.Lock _probe_drain_lock for "
        "probe/drain mutual exclusion"
    )
    assert isinstance(adapter._probe_drain_lock, asyncio.Lock)

    drain_held = adapter._probe_drain_lock
    await drain_held.acquire()
    # Capture real asyncio.sleep BEFORE patching, so our test's scheduling
    # yields actually advance the event loop. Patching asyncio.sleep with
    # `new_callable=AsyncMock` would make `await asyncio.sleep(0)` skip the
    # yield entirely — the probe task would never get CPU and the
    # `pool0_post.await_count == 0` assertion could pass for the wrong
    # reason (Codex review on PR #21548).
    real_sleep = asyncio.sleep

    async def _heartbeat_yield(seconds):
        # Skip the long heartbeat but yield to the loop once.
        await real_sleep(0)

    try:
        with patch("gateway.platforms.telegram.asyncio.sleep",
                   side_effect=_heartbeat_yield):
            probe_task = asyncio.create_task(adapter._verify_polling_after_reconnect())

            # Yield several real event-loop ticks. The probe must reach
            # `async with self._probe_drain_lock` and block there — not
            # call pool0.post.
            for _ in range(20):
                await real_sleep(0)

            assert pool0_post.await_count == 0, (
                "Probe must wait on _probe_drain_lock; called pool0.post "
                f"{pool0_post.await_count} times while drain held the lock"
            )

            # Release the lock — probe should now proceed.
            drain_held.release()

            # Wait briefly for the probe to complete its post.
            await asyncio.wait_for(probe_task, timeout=2.0)

        assert pool0_post.await_count == 1, (
            "Probe must call pool0.post exactly once after lock released; "
            f"got {pool0_post.await_count}"
        )
    finally:
        if drain_held.locked():
            try:
                drain_held.release()
            except RuntimeError:
                pass


@pytest.mark.asyncio
async def test_drain_acquires_shared_probe_drain_lock():
    """
    Symmetric assertion: `_drain_polling_connections` must acquire the same
    `_probe_drain_lock` so a probe in flight blocks drain too. Otherwise the
    invariant ("probe never hits a half-torn-down pool") doesn't hold.
    """
    adapter = _make_adapter()

    # Wire a minimal _request[0] with shutdown + initialize that we can
    # observe.
    polling_req = MagicMock()
    polling_req.shutdown = AsyncMock()
    polling_req.initialize = AsyncMock()
    mock_bot = MagicMock()
    mock_bot._request = (polling_req, MagicMock())

    mock_app = MagicMock()
    mock_app.bot = mock_bot
    adapter._app = mock_app
    adapter._bot = mock_bot

    # Hold the lock from outside, run drain in parallel, observe blocked.
    held = adapter._probe_drain_lock
    await held.acquire()
    try:
        drain_task = asyncio.create_task(adapter._drain_polling_connections())
        for _ in range(20):
            await asyncio.sleep(0)
        assert polling_req.shutdown.await_count == 0, (
            "Drain must wait on _probe_drain_lock; called shutdown "
            f"{polling_req.shutdown.await_count} times while it was held"
        )
        held.release()
        await asyncio.wait_for(drain_task, timeout=2.0)
        assert polling_req.shutdown.await_count == 1
        assert polling_req.initialize.await_count == 1
    finally:
        if held.locked():
            try:
                held.release()
            except RuntimeError:
                pass


# -----------------------------------------------------------------------------
# Iteration 5 — two-strike escalation within 5min window
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_probe_failures_within_window_escalate_to_fatal():
    """
    A single probe failure re-enters the existing reconnect ladder. But two
    failures within 5 minutes mean the previous reconnect's drain+restart
    didn't actually fix the wedge — the slow ladder will just keep trying
    and failing. Escalate to fatal-retryable so the supervisor restarts the
    process (a fresh interpreter clears any stale resolver / connection
    state).
    """
    adapter = _make_adapter()

    never_set = asyncio.Event()

    async def _hang(*args, **kwargs):
        await never_set.wait()

    pool0_post = AsyncMock(side_effect=_hang)
    pool1_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    _wire_adapter_with_pools(
        adapter, pool0_post=pool0_post, pool1_post=pool1_post,
    )

    real_wait_for = asyncio.wait_for

    async def _short_wait_for(coro, timeout):
        return await real_wait_for(coro, timeout=0.05)

    fatal_calls = []

    def _capture_fatal(code, message, *, retryable):
        fatal_calls.append((code, retryable, message))

    with patch("asyncio.sleep", new_callable=AsyncMock), \
         patch("gateway.platforms.telegram.asyncio.wait_for",
               side_effect=_short_wait_for), \
         patch.object(adapter, "_handle_polling_network_error",
                      new_callable=AsyncMock), \
         patch.object(adapter, "_set_fatal_error", side_effect=_capture_fatal), \
         patch.object(adapter, "_notify_fatal_error", new_callable=AsyncMock):
        # First probe failure — handler called, no fatal yet.
        await adapter._verify_polling_after_reconnect()
        assert fatal_calls == [], (
            "Single probe failure must not escalate to fatal "
            f"(got {fatal_calls})"
        )

        # Second probe failure within the same window — must escalate.
        await adapter._verify_polling_after_reconnect()

    assert len(fatal_calls) == 1, (
        "Two probe failures within the 5-minute window must escalate "
        f"exactly once (got {len(fatal_calls)} fatal calls: {fatal_calls})"
    )
    code, retryable, _ = fatal_calls[0]
    assert code == "telegram_network_error" or "telegram" in code, (
        f"Fatal error code must reference telegram (got {code!r})"
    )
    assert retryable is True, (
        "Escalation must be retryable so the supervisor restarts the process"
    )


# -----------------------------------------------------------------------------
# Iteration 6 — probe task cancellation on disconnect()
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_disconnect_cancels_pending_probe_task():
    """
    A probe task in flight at shutdown must be cancelled by disconnect().
    Otherwise asyncio emits "Task was destroyed but it is pending" warnings
    and the long sleep can keep the event loop alive past intended shutdown.
    """
    adapter = _make_adapter()

    # Build minimal _app/_bot so disconnect() doesn't blow up partway through.
    mock_updater = MagicMock()
    mock_updater.running = False  # so disconnect doesn't call stop()
    mock_app = MagicMock()
    mock_app.running = False
    mock_app.updater = mock_updater
    mock_app.stop = AsyncMock()
    mock_app.shutdown = AsyncMock()
    adapter._app = mock_app
    adapter._bot = MagicMock()

    # Schedule a probe — its 60s heartbeat sleep keeps it pending.
    adapter._schedule_polling_pool_probe()
    assert hasattr(adapter, "_latest_probe_task"), (
        "_schedule_polling_pool_probe must record the latest probe task on "
        "the adapter (e.g. self._latest_probe_task) so disconnect() can "
        "cancel it"
    )
    probe_task = adapter._latest_probe_task
    assert probe_task is not None and not probe_task.done()

    await adapter.disconnect()

    # Yield to let the cancellation propagate.
    for _ in range(10):
        await asyncio.sleep(0)

    assert probe_task.done(), (
        "Pending probe task must be done (cancelled or finished) after "
        "disconnect()"
    )
    assert probe_task.cancelled() or probe_task.exception() is not None, (
        "Probe task should have been cancelled by disconnect (got: "
        f"cancelled={probe_task.cancelled()}, "
        f"exception={probe_task.exception() if probe_task.done() and not probe_task.cancelled() else 'n/a'})"
    )


# -----------------------------------------------------------------------------
# Iteration 7 — long-poll offset preservation across the probe
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_does_not_advance_long_poll_offset():
    """
    The probe must not interfere with the long-poll's update offset
    bookkeeping. PTB's ``Updater`` tracks ``last_update_id`` internally and
    advances it as updates are consumed via ``getUpdates``. The probe calls
    ``getMe`` (not ``getUpdates``), so the offset must be untouched. This
    test pins that invariant — a regression toward any ``getUpdates``-based
    probe (forbidden by the council ruling on session a8c9534fee5e) would
    trip this test.
    """
    adapter = _make_adapter()

    pool0_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    pool1_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    mock_app = _wire_adapter_with_pools(
        adapter, pool0_post=pool0_post, pool1_post=pool1_post,
    )
    # Simulate the long-poll's offset bookkeeping. PTB stores this on its
    # internal ``_last_update_id`` (private API; the test intent is the
    # invariant, not the attribute name).
    mock_app.updater._last_update_id = 4242

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()

    assert mock_app.updater._last_update_id == 4242, (
        "Probe must not advance the long-poll's update offset "
        f"(was 4242, became {mock_app.updater._last_update_id})"
    )
    pool0_post.assert_awaited_once()
    # Verify the probe URL is /getMe — never any flavor of getUpdates.
    call_url = pool0_post.await_args.args[0]
    assert "/getMe" in call_url, (
        f"Probe must call /getMe; got URL: {call_url!r}"
    )
    assert "getUpdates" not in call_url, (
        "Probe must NEVER use getUpdates — Telegram's API contract makes "
        f"that destructive (offset=-1 forgets queue). Got: {call_url!r}"
    )


# -----------------------------------------------------------------------------
# Iteration 8 — log distinguishes cold_boot vs reconnect trigger
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wedge_log_distinguishes_cold_boot_vs_reconnect_trigger(caplog):
    """
    Operators triaging a production wedge need to tell at a glance whether
    the probe fired from cold-boot (DNS / TCP failure at startup that didn't
    raise) or from a post-reconnect heartbeat. The log must include the
    trigger label.
    """
    import logging

    adapter = _make_adapter()

    never_set = asyncio.Event()

    async def _hang(*args, **kwargs):
        await never_set.wait()

    pool0_post = AsyncMock(side_effect=_hang)
    pool1_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    _wire_adapter_with_pools(
        adapter, pool0_post=pool0_post, pool1_post=pool1_post,
    )

    real_wait_for = asyncio.wait_for

    async def _short_wait_for(coro, timeout):
        return await real_wait_for(coro, timeout=0.05)

    with patch("asyncio.sleep", new_callable=AsyncMock), \
         patch("gateway.platforms.telegram.asyncio.wait_for",
               side_effect=_short_wait_for), \
         patch.object(adapter, "_handle_polling_network_error",
                      new_callable=AsyncMock), \
         caplog.at_level(logging.WARNING, logger="gateway.platforms.telegram"):
        await adapter._verify_polling_after_reconnect(trigger="cold_boot")

    cold_boot_messages = [
        rec.getMessage() for rec in caplog.records
        if "cold_boot" in rec.getMessage().lower()
        and "wedge detected" in rec.getMessage().lower()
    ]
    assert cold_boot_messages, (
        "Expected at least one WARNING containing both 'cold_boot' and "
        f"'wedge detected'. Got: {[rec.getMessage() for rec in caplog.records]}"
    )


# -----------------------------------------------------------------------------
# Iteration 9 — schedule cancels prior pending probe (PR #21548 review)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schedule_cancels_previously_pending_probe():
    """
    `_schedule_polling_pool_probe` is called from both cold-boot and
    reconnect paths. Without cancelling a prior pending probe, multiple
    60s-sleeping probes can accumulate; `disconnect()` cancels only the
    most recent (`_latest_probe_task`), so older probes wake on a torn-down
    adapter and call `_handle_polling_network_error` against `_app=None`,
    leaking into a recursive reconnect failure.

    Invariant: at most one probe in flight per adapter.
    """
    adapter = _make_adapter()
    adapter._app = MagicMock()  # any truthy value so probes don't bail early

    # First schedule (e.g. cold_boot)
    adapter._schedule_polling_pool_probe(trigger="cold_boot")
    first = adapter._latest_probe_task
    assert first is not None and not first.done()

    # Second schedule (e.g. reconnect after a transient blip)
    adapter._schedule_polling_pool_probe(trigger="reconnect")
    second = adapter._latest_probe_task
    assert second is not None and not second.done()
    assert second is not first, "Second schedule must produce a distinct task"

    # The first probe must have been cancelled — at most one in flight.
    for _ in range(10):
        await asyncio.sleep(0)
    assert first.cancelled() or first.done(), (
        "Previous probe must be cancelled when a new one is scheduled "
        f"(state: cancelled={first.cancelled()}, done={first.done()})"
    )

    # Cleanup
    second.cancel()
    try:
        await second
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_probe_bails_when_app_torn_down_after_sleep():
    """
    If the adapter is disconnected (`self._app = None`) while a probe is
    sleeping its 60s heartbeat, the probe must bail silently after waking —
    not call `_handle_polling_network_error` against a torn-down adapter.
    """
    adapter = _make_adapter()
    adapter._app = None  # simulates a disconnect that already happened

    with patch("asyncio.sleep", new_callable=AsyncMock), \
         patch.object(adapter, "_handle_polling_network_error",
                      new_callable=AsyncMock) as mock_handler:
        await adapter._verify_polling_after_reconnect()

    assert mock_handler.await_count == 0, (
        "Probe must not call _handle_polling_network_error after "
        f"disconnect (called {mock_handler.await_count} times)"
    )


# -----------------------------------------------------------------------------
# Iteration 10 — URL normalization for trailing-slash base_url (PR #21548 review)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_probe_url_normalizes_trailing_slash_in_base_url():
    """
    PTB's `Bot.base_url` is built via `_parse_base_url(base_url, token)`.
    For default config it has no trailing slash, but custom configs
    (callable `base_url`, `format_map`-based, `local_mode=True` self-hosted
    Bot API servers) can produce one. `f"{base}/getMe"` then yields
    `.../bot<TOKEN>//getMe` — most servers normalize, but it's avoidable.
    """
    adapter = _make_adapter()

    pool0_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    pool1_post = AsyncMock(return_value={"id": 1, "is_bot": True})
    _wire_adapter_with_pools(
        adapter,
        pool0_post=pool0_post,
        pool1_post=pool1_post,
        # Trailing slash — the case Copilot flagged on PR #21548.
        base_url="https://api.telegram.org/bot<REDACTED>/",
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._verify_polling_after_reconnect()

    pool0_post.assert_awaited_once()
    call_url = pool0_post.await_args.args[0]
    assert "//getMe" not in call_url, (
        f"Probe URL must not contain double slash; got: {call_url!r}"
    )
    assert call_url.endswith("/getMe"), (
        f"Probe URL must end in '/getMe' (single slash); got: {call_url!r}"
    )
