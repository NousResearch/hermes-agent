"""
Tests for Home Assistant WebSocket gateway network-failure recovery.

Regression coverage for #67470 — the HA adapter could go silently deaf after
transient network failures:
1. A raised ``ws_connect()`` leaked the just-created ``aiohttp.ClientSession``.
2. Teardown awaits (``ws.close()`` / ``session.close()``) had no timeout and
   could block forever on a wedged CLOSE-WAIT socket.
3. The auth-handshake ``receive_json()`` calls had no timeout, so a server
   that accepted the socket but never responded froze ``_ws_connect``.
4. Nothing detected a wedged ``_listen_loop`` task — the gateway stayed
   "running" but silently stopped processing events.
"""

import asyncio
import gc
import time
import weakref
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.homeassistant import adapter as ha_adapter
from plugins.platforms.homeassistant.adapter import HomeAssistantAdapter


def _make_adapter(**extra) -> HomeAssistantAdapter:
    config = PlatformConfig(enabled=True, token="tok", extra=extra)
    return HomeAssistantAdapter(config)


async def _hang_forever(*_args, **_kwargs):
    await asyncio.Event().wait()


async def _await_tracked_teardown(tasks, *, timeout: float = 2) -> None:
    """Deterministically wait for previously snapshotted background
    teardown task(s) (from ``adapter._teardown_tasks``) to fully finish.

    ``_cleanup_ws()`` / ``_cancel_safe_close()`` / ``_full_teardown()``
    shield their close from a caller's cancellation by running it as a
    task tracked in ``self._teardown_tasks``; a *second* cancellation only
    detaches the caller from that task, it does not stop the task itself.
    A signal (``asyncio.Event``) set partway through that task's own body
    only proves ONE step ran -- it does not prove every *later* step in
    the same shielded unit has also run yet on this scheduler. That gap is
    exactly what made
    ``test_cleanup_ws_closes_session_when_cancelled_during_ws_close``
    flaky on Linux CI (failed: session.close awaited 0 times) while
    passing on Windows by scheduling coincidence -- it waited on the
    event from the WS close (step 1) and then immediately asserted on the
    session close (step 2) of the same shielded ``_close_both()`` task.

    Awaiting the task object(s) directly has no such gap: ``gather()``
    only returns once each task's whole coroutine body -- every step in
    it, including nested shielded sub-tasks -- is actually done. Callers
    must snapshot ``tuple(adapter._teardown_tasks)`` at the point where
    the tracked task is known to already exist (e.g. right after an
    ``asyncio.Event`` set from inside that task's first step fires), then
    pass the snapshot here after the caller-side cancellation.
    """
    assert tasks, "expected a background teardown task to already be tracked"
    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)


# ---------------------------------------------------------------------------
# Defect 1: session leak on failed connect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_failed_connect_closes_local_session():
    """A raised ws_connect() must close the just-created local session
    instead of leaking it via a self._session that was assigned before the
    connect attempt (#67470)."""
    adapter = _make_adapter()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(side_effect=ConnectionError("refused"))

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with pytest.raises(ConnectionError):
            await adapter._ws_connect()

    mock_session.close.assert_awaited_once()
    # The failed local session must never have been wired onto the adapter.
    assert adapter._session is None
    assert adapter._ws is None


# ---------------------------------------------------------------------------
# Defect 2: unbounded teardown awaits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_ws_bounds_hanging_close_and_nulls_both(monkeypatch):
    """A wedged ws.close() must not block session.close() from running, and
    both attributes must be nulled regardless (#67470)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    hung_ws = MagicMock()
    hung_ws.closed = False
    hung_ws.close = AsyncMock(side_effect=_hang_forever)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    adapter._ws = hung_ws
    adapter._session = session

    await asyncio.wait_for(adapter._cleanup_ws(), timeout=2)

    assert adapter._ws is None
    assert adapter._session is None
    # The session close must still have run despite the ws close hanging.
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_completes_within_bounds_when_closes_hang(monkeypatch):
    """disconnect() must complete even when every underlying close() hangs."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock(side_effect=_hang_forever)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock(side_effect=_hang_forever)

    rest_session = MagicMock()
    rest_session.closed = False
    rest_session.close = AsyncMock(side_effect=_hang_forever)

    adapter._ws = ws
    adapter._session = session
    adapter._rest_session = rest_session
    adapter._running = True

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)  # let the no-op tasks finish before disconnect() awaits them

    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    assert adapter._ws is None
    assert adapter._session is None
    assert adapter._rest_session is None
    assert adapter._running is False


# ---------------------------------------------------------------------------
# Defect 3: unbounded auth handshake reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_bounds_hanging_auth_handshake(monkeypatch):
    """A server that accepts the socket but never responds to
    receive_json() must not freeze _ws_connect() forever (#67470)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_HANDSHAKE_TIMEOUT", 0.05, raising=False)

    mock_ws = MagicMock()
    mock_ws.closed = False
    mock_ws.receive_json = AsyncMock(side_effect=_hang_forever)
    mock_ws.send_json = AsyncMock()
    mock_ws.close = AsyncMock()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        result = await asyncio.wait_for(adapter._ws_connect(), timeout=2)

    assert result is False
    mock_ws.close.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


# ---------------------------------------------------------------------------
# Defect 4: cause-agnostic watchdog over _listen_loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_respawns_wedged_listen_task(monkeypatch):
    """If _last_progress goes stale past _LISTEN_STUCK_TIMEOUT while running,
    the watchdog must cancel the stuck listen task, force a cleanup, and
    respawn a new _listen_loop task (#67470)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_LISTEN_STUCK_TIMEOUT", 0.01, raising=False)

    respawn_calls = []

    async def _stub_listen_loop():
        respawn_calls.append(1)
        await asyncio.Event().wait()

    adapter._listen_loop = _stub_listen_loop  # type: ignore[method-assign]
    adapter._cleanup_ws = AsyncMock()

    adapter._running = True
    stuck_task = asyncio.ensure_future(_hang_forever())
    adapter._listen_task = stuck_task
    adapter._last_progress = time.monotonic() - 10  # already stale

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())

    for _ in range(100):
        await asyncio.sleep(0.01)
        if respawn_calls and adapter._listen_task is not stuck_task:
            break

    assert respawn_calls, "watchdog must respawn a new _listen_loop task"
    assert adapter._listen_task is not None
    assert adapter._listen_task is not stuck_task
    assert stuck_task.cancelled()
    adapter._cleanup_ws.assert_awaited()

    # Clean up outstanding tasks.
    adapter._running = False
    for t in (watchdog_task, adapter._listen_task):
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_watchdog_stops_when_running_goes_false(monkeypatch):
    """The watchdog loop must exit cleanly once self._running is False."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    adapter._running = True

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())
    await asyncio.sleep(0.03)
    adapter._running = False

    await asyncio.wait_for(watchdog_task, timeout=2)
    assert watchdog_task.done()
    assert not watchdog_task.cancelled()


# ---------------------------------------------------------------------------
# Review follow-ups (#67470): handshake exception cleanup, quiet-vs-wedged
# ping probe, and bounded cancellation of uncancellable tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_cleans_up_when_handshake_send_raises():
    """A send_json() that raises mid-handshake must tear the connection down
    inside _ws_connect() instead of leaking it to a later loop pass."""
    adapter = _make_adapter()

    mock_ws = MagicMock()
    mock_ws.closed = False
    mock_ws.receive_json = AsyncMock(return_value={"type": "auth_required"})
    mock_ws.send_json = AsyncMock(side_effect=ConnectionResetError("peer gone"))
    mock_ws.close = AsyncMock()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(return_value=mock_ws)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        result = await asyncio.wait_for(adapter._ws_connect(), timeout=2)

    assert result is False
    mock_ws.close.assert_awaited_once()
    mock_session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_watchdog_ping_probe_spares_quiet_but_healthy_listener(monkeypatch):
    """aiohttp answers heartbeat PINGs internally, so a healthy-but-quiet HA
    produces no reader frames. The watchdog's HA-protocol ping must detect the
    live listener (pong bumps _last_progress) and skip the respawn."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_LISTEN_STUCK_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_PING_GRACE", 0.01, raising=False)

    async def _pong_arrives(payload):
        # Simulate the reader receiving the pong frame.
        adapter._last_progress = time.monotonic()

    live_ws = MagicMock()
    live_ws.closed = False
    live_ws.send_json = AsyncMock(side_effect=_pong_arrives)

    adapter._ws = live_ws
    adapter._running = True
    listen_task = asyncio.ensure_future(_hang_forever())
    adapter._listen_task = listen_task
    adapter._last_progress = time.monotonic() - 10  # stale by progress alone

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())
    await asyncio.sleep(0.2)

    assert adapter._listen_task is listen_task, \
        "healthy-but-quiet listener must not be respawned"
    assert not listen_task.cancelled()
    live_ws.send_json.assert_awaited()

    adapter._running = False
    for t in (watchdog_task, listen_task):
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_watchdog_ping_probe_failure_respawns(monkeypatch):
    """A ping that cannot even be sent means the socket is wedged — the
    watchdog must proceed with the cancel-and-respawn recovery."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_WATCHDOG_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_LISTEN_STUCK_TIMEOUT", 0.01, raising=False)
    monkeypatch.setattr(ha_adapter, "_PING_GRACE", 0.01, raising=False)

    dead_ws = MagicMock()
    dead_ws.closed = False
    dead_ws.send_json = AsyncMock(side_effect=ConnectionResetError("wedged"))

    respawn_calls = []

    async def _stub_listen_loop():
        respawn_calls.append(1)
        await asyncio.Event().wait()

    adapter._listen_loop = _stub_listen_loop  # type: ignore[method-assign]
    adapter._cleanup_ws = AsyncMock()
    adapter._ws = dead_ws
    adapter._running = True
    stuck_task = asyncio.ensure_future(_hang_forever())
    adapter._listen_task = stuck_task
    adapter._last_progress = time.monotonic() - 10

    watchdog_task = asyncio.ensure_future(adapter._watchdog_loop())

    for _ in range(100):
        await asyncio.sleep(0.01)
        if respawn_calls and adapter._listen_task is not stuck_task:
            break

    assert respawn_calls, "watchdog must respawn after a failed ping probe"
    assert stuck_task.cancelled()

    adapter._running = False
    for t in (watchdog_task, adapter._listen_task):
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_cancel_task_bounded_abandons_uncancellable_task(monkeypatch):
    """A task that swallows CancelledError must not hang the watchdog or
    disconnect(): _cancel_task_bounded gives up after _DRAIN_TIMEOUT."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    started = asyncio.Event()
    stop = asyncio.Event()

    async def _ignores_cancel():
        while not stop.is_set():
            try:
                started.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue  # rude task refuses to die

    zombie = asyncio.ensure_future(_ignores_cancel())
    await started.wait()

    await asyncio.wait_for(
        adapter._cancel_task_bounded(zombie, "zombie"), timeout=2
    )

    assert not zombie.done(), "the zombie survives; the point is WE returned"
    # Final teardown for test hygiene: flip the stop flag, then nudge the
    # sleep with one more cancel so the loop re-checks it and exits normally.
    # (No wait_for on the zombie — wait_for's timeout path awaits cancellation
    # completing, the exact hang the production helper avoids.)
    stop.set()
    zombie.cancel()
    await asyncio.wait({zombie}, timeout=1)


# ---------------------------------------------------------------------------
# Review follow-up (#67470, egilewski): CancelledError bypasses the
# `except Exception` cleanup in _ws_connect(), leaking the freshly created
# session before self._session is ever assigned.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_connect_cancellation_closes_local_session():
    """A cancellation landing inside session.ws_connect() -- e.g. the
    gateway's outer per-platform connect deadline
    (asyncio.wait_for(adapter.connect(...), timeout=...) in
    gateway/run.py's _connect_adapter_with_timeout), or disconnect()/the
    watchdog cancelling an in-flight reconnect attempt -- must still close
    the freshly created ClientSession instead of leaking it.
    asyncio.CancelledError derives from BaseException, not Exception, so a
    plain `except Exception` never sees it and the close is skipped
    entirely (egilewski's probe: session.close observed awaited 0 times)."""
    adapter = _make_adapter()

    mock_session = MagicMock()
    mock_session.closed = False
    mock_session.close = AsyncMock()
    mock_session.ws_connect = AsyncMock(side_effect=_hang_forever)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        with pytest.raises(asyncio.TimeoutError):
            # Mirrors gateway/run.py's outer connect deadline: it wraps the
            # whole connect() (which calls _ws_connect() first) in
            # asyncio.wait_for(..., timeout=...), cancelling it once the
            # deadline passes while ws_connect() is still hung.
            await asyncio.wait_for(adapter._ws_connect(), timeout=0.05)

    # No race here despite the close being shielded internally: there is
    # only ONE cancellation in this test (wait_for's own timeout), and
    # wait_for's cancel-then-wait machinery (_cancel_and_wait) does not
    # raise TimeoutError to us until the cancelled task is fully done --
    # which, since nothing cancels it a second time, requires
    # _cancel_safe_close()'s `await asyncio.shield(task)` to have returned
    # normally, i.e. the close already fully completed. Provable from
    # asyncio.wait_for's documented semantics, not from timing.
    mock_session.close.assert_awaited_once()
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_ws_connect_close_survives_second_cancellation_during_teardown():
    """A naive `except CancelledError: await self._bounded_close(...);
    raise` is not enough: this codebase has a real second cancellation
    source that can race in while that close is still running (the
    watchdog's wedged-listener respawn and disconnect() can both cancel the
    same listen task -- adapter.py's _cancel_task_bounded call sites at
    disconnect() and _watchdog_loop()). If that second cancellation
    interrupts the close before it finishes, the session leaks exactly like
    the original bug, just one frame deeper. The close must still complete
    (detached, tracked in self._teardown_tasks) even under this race."""
    adapter = _make_adapter()

    close_started = asyncio.Event()
    close_completed = asyncio.Event()
    mock_session = MagicMock()
    mock_session.closed = False

    async def _slow_close():
        close_started.set()
        await asyncio.sleep(0.05)
        close_completed.set()

    mock_session.close = _slow_close
    mock_session.ws_connect = AsyncMock(side_effect=_hang_forever)

    with patch("plugins.platforms.homeassistant.adapter.aiohttp") as mock_aiohttp:
        mock_aiohttp.ClientTimeout = lambda total: total
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)

        task = asyncio.ensure_future(adapter._ws_connect())
        await asyncio.sleep(0)
        task.cancel()  # cancel #1: lands inside session.ws_connect()
        # Bounded: on the pre-fix code this never fires (the close is
        # skipped entirely), so an unbounded wait here would hang the
        # suite instead of failing fast.
        await asyncio.wait_for(close_started.wait(), timeout=2)
        # Snapshot the shielded close's tracked task now (it must already
        # exist -- close_started is set from inside it) so we can wait for
        # the WHOLE thing to finish afterward, deterministically.
        tracked = tuple(adapter._teardown_tasks)
        task.cancel()  # cancel #2: races in while that close is in flight
        with pytest.raises(asyncio.CancelledError):
            await task

    # The second cancellation only detaches `task` (the caller) from the
    # close; the close itself keeps running as the tracked background task
    # snapshotted above. See _await_tracked_teardown for why this must be
    # awaited directly rather than close_completed (Linux CI determinism).
    await _await_tracked_teardown(tracked)
    assert close_completed.is_set(), "the session close must have completed"
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_cleanup_ws_close_survives_second_cancellation_during_teardown():
    """_cleanup_ws() clears self._ws/self._session to None before awaiting
    their close -- reached from _ws_connect()'s handshake CancelledError
    handler and the _listen_loop reconnect ladder. If a second cancellation
    lands on the caller while that close is still running, the close must
    still complete instead of being interrupted mid-flight, even though the
    fields are already unreachable by that point (#67470 review,
    egilewski: this is the same leak class as _ws_connect()'s
    pre-assignment session, reached one level deeper)."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock()

    session = MagicMock()
    session.closed = False
    close_started = asyncio.Event()
    close_completed = asyncio.Event()

    async def _slow_close():
        close_started.set()
        await asyncio.sleep(0.05)
        close_completed.set()

    session.close = _slow_close
    adapter._ws = ws
    adapter._session = session

    async def _handshake_cancelled_mid_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await adapter._cleanup_ws()
            raise

    task = asyncio.ensure_future(_handshake_cancelled_mid_cleanup())
    await asyncio.sleep(0)
    task.cancel()  # cancel #1: enters the CancelledError handler
    # Bounded: on the pre-fix code this never fires (the close is skipped
    # entirely), so an unbounded wait here would hang the suite instead of
    # failing fast.
    await asyncio.wait_for(close_started.wait(), timeout=2)
    # Snapshot the shielded close's tracked task now (it must already
    # exist -- close_started is set from inside it) so we can wait for the
    # WHOLE thing to finish afterward, deterministically.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # cancel #2: races in while _cleanup_ws() awaits the close
    with pytest.raises(asyncio.CancelledError):
        await task

    # The second cancellation only detaches `task` (the caller) from the
    # close; the close itself keeps running as the tracked background task
    # snapshotted above. See _await_tracked_teardown for why this must be
    # awaited directly rather than close_completed (Linux CI determinism).
    await _await_tracked_teardown(tracked)
    assert close_completed.is_set(), "the session close must have completed"
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_cleanup_ws_closes_session_when_cancelled_during_ws_close():
    """A cancellation racing in while _cleanup_ws()'s WebSocket close is
    still running must not skip the session close entirely (#67470 review
    round 2, egilewski). Running the WS close and session close as two
    separately shielded steps meant that second cancellation propagated out
    of the WS close's shielded await, so _cleanup_ws() returned before ever
    reaching the code that even attempts the session close -- leaving
    self._session non-None with nothing left running to close it. Both
    closes must run as one shielded unit so the whole thing keeps going."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False
    ws_close_started = asyncio.Event()
    ws_close_completed = asyncio.Event()

    async def _slow_ws_close():
        ws_close_started.set()
        await asyncio.sleep(0.05)
        ws_close_completed.set()

    ws.close = _slow_ws_close

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    adapter._ws = ws
    adapter._session = session

    async def _handshake_cancelled_mid_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await adapter._cleanup_ws()
            raise

    task = asyncio.ensure_future(_handshake_cancelled_mid_cleanup())
    await asyncio.sleep(0)
    task.cancel()  # cancel #1: enters the CancelledError handler
    await asyncio.wait_for(ws_close_started.wait(), timeout=2)
    # Snapshot the shielded _close_both() task now (it must already exist
    # -- ws_close_started is set from inside it) so we can wait for the
    # WHOLE unit -- both the WS close AND the session close after it -- to
    # finish, deterministically, instead of guessing from timing.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # cancel #2: races in while the WS close is still running
    with pytest.raises(asyncio.CancelledError):
        await task

    # _close_both() runs the WS close THEN the session close as one
    # shielded background unit. ws_close_started/ws_close_completed only
    # signal the FIRST step; they do NOT guarantee the scheduler has
    # already run the second step (the session close) by the time control
    # resumes here on every event loop implementation. Waiting on that
    # partial-progress signal and then immediately asserting on the
    # session close is exactly what made this test flaky on Linux CI
    # (observed: session.close awaited 0 times) while passing on Windows
    # by scheduling coincidence. Awaiting the tracked task itself has no
    # such gap -- see _await_tracked_teardown.
    await _await_tracked_teardown(tracked)
    assert ws_close_completed.is_set(), "the WS close must have completed"
    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_disconnect_rest_session_not_double_closed_on_cancellation():
    """disconnect() must detach self._rest_session before awaiting its
    close (#67470 review round 2, egilewski): clearing the field only
    *after* the close returned meant a cancellation racing in on
    disconnect() left the field still assigned to a session whose close was
    already running in the background -- a later disconnect() retry would
    see closed=False and call close() on it a second time."""
    adapter = _make_adapter()
    adapter._running = True

    rest_session = MagicMock()
    rest_session.closed = False
    close_started = asyncio.Event()
    close_completed = asyncio.Event()
    close_calls = 0

    async def _slow_close():
        nonlocal close_calls
        close_calls += 1
        close_started.set()
        await asyncio.sleep(0.05)
        rest_session.closed = True
        close_completed.set()

    rest_session.close = _slow_close
    adapter._rest_session = rest_session

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)

    task = asyncio.ensure_future(adapter.disconnect())
    await asyncio.wait_for(close_started.wait(), timeout=2)
    # Snapshot the shielded _full_teardown() task now (it must already
    # exist -- close_started is set from inside it, via the REST close it
    # runs directly) so we can wait for it to fully finish at the end,
    # deterministically. Deliberately NOT drained here yet -- the whole
    # point of this test is to retry disconnect() while this first close
    # may still be in flight in the background.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # races in while the REST session close is still running
    with pytest.raises(asyncio.CancelledError):
        await task

    # A retried disconnect() (or any other caller) must see the field
    # already cleared, not call close() on the same session a second time.
    # Safe to assert immediately, no race: _full_teardown() clears
    # self._rest_session synchronously before the close is even entered --
    # close_started already firing above proves that already happened.
    assert adapter._rest_session is None

    # Simulate a caller retrying disconnect() right away, before the
    # backgrounded close has even finished.
    adapter._running = True
    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)
    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    # Deterministically wait for the FIRST close (still possibly running in
    # the background from the cancelled call above) to finish before the
    # final assert. See _await_tracked_teardown (Linux CI determinism).
    await _await_tracked_teardown(tracked)
    assert close_completed.is_set(), "the REST session close must have completed"
    assert close_calls == 1, "REST session close() must not run twice"


@pytest.mark.asyncio
async def test_disconnect_closes_rest_session_when_cancelled_during_ws_close():
    """disconnect() has several sequential stages (cancel watchdog, cancel
    listener, close WS/session, close REST session). A cancellation landing
    at an EARLIER stage must not leave a LATER stage's resources untouched
    (#67470 review round 3, egilewski): previously, cancelling disconnect()
    while _cleanup_ws() was still closing the WebSocket aborted the whole
    coroutine before the REST session close was ever attempted, leaving
    self._rest_session assigned and open. The full teardown must run as one
    protected unit so every stage still completes in the background."""
    adapter = _make_adapter()
    adapter._running = True

    ws = MagicMock()
    ws.closed = False
    ws_close_started = asyncio.Event()

    async def _slow_ws_close():
        ws_close_started.set()
        await asyncio.sleep(0.05)

    ws.close = _slow_ws_close
    adapter._ws = ws
    adapter._session = None

    rest_session = MagicMock()
    rest_session.closed = False
    rest_close_completed = asyncio.Event()

    async def _rest_close():
        await asyncio.sleep(0.02)
        rest_close_completed.set()

    rest_session.close = _rest_close
    adapter._rest_session = rest_session

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)

    task = asyncio.ensure_future(adapter.disconnect())
    await asyncio.wait_for(ws_close_started.wait(), timeout=2)
    # Snapshot the shielded task(s) now: disconnect()'s own _full_teardown()
    # task, plus _cleanup_ws()'s nested _close_both() task (both already
    # exist -- ws_close_started fires from inside the innermost one). Wait
    # for ALL of them so we know the whole nested teardown -- WS close,
    # session close, and the REST close after it -- has actually finished,
    # not just that the WS close's own event fired.
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()  # races in while the WS close (an earlier stage) is running
    with pytest.raises(asyncio.CancelledError):
        await task

    # rest_close_completed alone would still be a correct (if slower to
    # fail) bound here -- if the REST close never ran, this would legitimately
    # time out rather than pass early -- but await the tracked tasks
    # directly for the same reason as the other hardened tests: no reliance
    # on scheduling order between nested shielded steps.
    await _await_tracked_teardown(tracked)
    assert rest_close_completed.is_set(), "the REST session close must have completed"
    assert adapter._rest_session is None


@pytest.mark.asyncio
async def test_bounded_close_abandons_cancellation_suppressing_close(monkeypatch):
    """A close() that catches its own cancellation and keeps polling a stop
    signal (rather than actually finishing) must not be AWAITED to
    completion by _bounded_close.

    Pre-fix, `asyncio.wait_for(closeable.close(), timeout=_DRAIN_TIMEOUT)`
    cancels close() on timeout and then WAITS for that cancellation to
    actually finish -- so a close() that swallows the cancellation and
    keeps running left `_bounded_close` pending until close() eventually
    decides to stop on its own, which is exactly the boundedness violation
    the bounded-abandon mechanism removes (#67470 Sol xhigh mechanism
    review, gap 1)."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05, raising=False)

    stop = asyncio.Event()
    close_truly_finished = asyncio.Event()

    async def _suppresses_cancellation_until_stopped():
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                continue  # rude close(): swallows cancellation, keeps going
        close_truly_finished.set()

    closeable = MagicMock()
    closeable.close = AsyncMock(side_effect=_suppresses_cancellation_until_stopped)

    # Run _bounded_close as its OWN task (mirrors the bounded-abandon
    # mechanism itself) and observe it with asyncio.wait(timeout=...)
    # rather than wrapping the bare coroutine in another wait_for.
    # asyncio.wait_for cancels the CURRENT task on timeout, not a separate
    # one, for a bare coroutine argument -- nesting it around another
    # wait_for(bare coroutine) that itself swallows cancellation forever
    # (the exact defect under test) would make the outer wait_for's own
    # cancellation land in that same swallow loop too and never escape,
    # hanging the TEST instead of failing it. asyncio.wait() on an actual
    # Task has no such trap: it only observes, never cancels.
    bounded_close_task = asyncio.ensure_future(
        adapter._bounded_close(closeable, "suppressing")
    )
    try:
        done, pending = await asyncio.wait({bounded_close_task}, timeout=1)

        assert not pending, (
            "_bounded_close must return on its own within a bounded window "
            "instead of hanging behind a cancellation-suppressing close "
            "(pre-fix: asyncio.wait_for waits for the close's actual "
            "completion after cancelling it, which a suppressing close "
            "never delivers)"
        )
        assert not close_truly_finished.is_set(), (
            "_bounded_close must abandon a cancellation-suppressing close "
            "after _DRAIN_TIMEOUT instead of waiting for it to actually "
            "finish"
        )

        # Corroborate durable retention alongside boundedness (Sol xhigh
        # mechanism re-review: the first version of this test "does not
        # prove module-level retention because it never forces GC after
        # abandonment"; the second version kept a strong local reference
        # to the task throughout, so surviving gc.collect() proved
        # nothing about the registry -- it would have survived from the
        # local alone). Find the abandoned inner close task (separate
        # from bounded_close_task, the outer _bounded_close() call, which
        # already completed above) in the module registry, take only a
        # WEAK reference, drop every strong local including the list
        # itself, force a collection, and confirm the weakref is still
        # alive. NOTE (Sol xhigh follow-up, negative-control probe): this
        # is corroborating evidence, not an isolated proof that
        # _TEARDOWN_REGISTRY specifically is what kept it alive -- a task
        # that is still actively scheduled (a live callback pending on
        # its current await) can also survive collection via asyncio's
        # own internal bookkeeping, independent of any registry. The
        # DEFINITIVE proof of the retention mechanism is the direct `in
        # _TEARDOWN_REGISTRY` membership check in
        # test_abandoned_reconnect_does_not_publish_after_disconnect
        # below; this assertion is a secondary sanity check, not the
        # sole evidence.
        abandoned = [t for t in ha_adapter._TEARDOWN_REGISTRY if not t.done()]
        assert abandoned, "the abandoned close task must be rooted in the module registry"
        abandoned_ref = weakref.ref(abandoned[0])
        del abandoned
        gc.collect()
        still_alive = abandoned_ref()
        assert still_alive is not None, (
            "a forced gc.collect() destroyed the abandoned close task after "
            "every local strong reference was dropped"
        )
        assert not still_alive.done(), (
            "a forced gc.collect() must not destroy an abandoned close "
            "that is still rooted in _TEARDOWN_REGISTRY"
        )
    finally:
        # Test hygiene: let the abandoned close actually finish instead of
        # leaving a zombie task behind -- unconditionally, even if an
        # assertion above failed (on pre-fix code the close is genuinely
        # still running at that point; without this in `finally`, the
        # loop's own teardown would gather it, and it never finishes on
        # its own since `stop` was never set, hanging the WHOLE suite
        # instead of just failing this one test).
        stop.set()
        await asyncio.wait_for(close_truly_finished.wait(), timeout=2)


@pytest.mark.asyncio
async def test_cleanup_ws_session_close_runs_when_ws_close_raises_cancellederror():
    """A close() that raises CancelledError ON ITS OWN -- not from an
    external task.cancel() -- must not abort the surrounding cleanup
    sequence: the session close must still run after it (#67470 Sol xhigh
    mechanism review, gap 3). Pre-fix, `_bounded_close` awaited
    `closeable.close()` inline inside `_close_both()`; a self-raised
    CancelledError propagated straight out of `_close_both()`, skipping
    the session close entirely."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False

    async def _raises_cancelled_from_close():
        raise asyncio.CancelledError("close() itself raises, not externally cancelled")

    ws.close = AsyncMock(side_effect=_raises_cancelled_from_close)

    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    adapter._ws = ws
    adapter._session = session

    await asyncio.wait_for(adapter._cleanup_ws(), timeout=2)

    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_disconnect_rest_close_runs_when_ws_close_raises_cancellederror():
    """The same close-originated-CancelledError protection must hold across
    the whole disconnect() sequence, not just _cleanup_ws()'s own two
    closes: the REST session close must still run after a WS close that
    raises CancelledError on its own (#67470 Sol xhigh mechanism review,
    gap 3)."""
    adapter = _make_adapter()
    adapter._running = True

    ws = MagicMock()
    ws.closed = False

    async def _raises_cancelled_from_close():
        raise asyncio.CancelledError("close() itself raises, not externally cancelled")

    ws.close = AsyncMock(side_effect=_raises_cancelled_from_close)
    adapter._ws = ws
    adapter._session = None

    rest_session = MagicMock()
    rest_session.closed = False
    rest_session.close = AsyncMock()
    adapter._rest_session = rest_session

    async def _noop():
        return

    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)

    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    rest_session.close.assert_awaited_once()
    assert adapter._rest_session is None

