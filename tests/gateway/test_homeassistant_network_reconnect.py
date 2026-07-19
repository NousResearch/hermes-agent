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
