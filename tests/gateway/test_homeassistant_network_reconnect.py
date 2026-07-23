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
import time
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
        task.cancel()  # cancel #2: races in while that close is in flight
        with pytest.raises(asyncio.CancelledError):
            await task

    await asyncio.wait_for(close_completed.wait(), timeout=2)
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
    task.cancel()  # cancel #2: races in while _cleanup_ws() awaits the close
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(close_completed.wait(), timeout=2)
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
    task.cancel()  # cancel #2: races in while the WS close is still running
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(ws_close_completed.wait(), timeout=2)
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
    task.cancel()  # races in while the REST session close is still running
    with pytest.raises(asyncio.CancelledError):
        await task

    # A retried disconnect() (or any other caller) must see the field
    # already cleared, not call close() on the same session a second time.
    assert adapter._rest_session is None

    # Simulate a caller retrying disconnect() right away, before the
    # backgrounded close has even finished.
    adapter._running = True
    adapter._listen_task = asyncio.ensure_future(_noop())
    adapter._watchdog_task = asyncio.ensure_future(_noop())
    await asyncio.sleep(0)
    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    await asyncio.wait_for(close_completed.wait(), timeout=2)
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
    task.cancel()  # races in while the WS close (an earlier stage) is running
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(rest_close_completed.wait(), timeout=2)
    assert adapter._rest_session is None
