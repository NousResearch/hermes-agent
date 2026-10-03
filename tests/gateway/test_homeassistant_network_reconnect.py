"""Regression tests for bounded Home Assistant connect and teardown."""

import asyncio
import gc
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


async def _attach_finished_listener(adapter: HomeAssistantAdapter) -> None:
    async def _noop():
        return

    adapter._listen_task = asyncio.create_task(_noop())
    await asyncio.sleep(0)


async def _await_tracked_teardown(tasks, *, timeout: float = 2) -> None:
    assert tasks, "expected a background teardown task"
    await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=timeout)


@pytest.mark.asyncio
async def test_ws_connect_failed_connect_closes_local_session():
    """A failed connect closes the unpublished local session."""
    adapter = _make_adapter()
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(side_effect=ConnectionError("refused"))

    with patch.object(adapter, "_new_session", return_value=session):
        with pytest.raises(ConnectionError):
            await adapter._ws_connect()

    session.close.assert_awaited_once()
    # The failed local session must never have been wired onto the adapter.
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_cleanup_ws_bounds_hanging_close_and_nulls_both(monkeypatch):
    """A wedged WS close cannot prevent the session close."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05)

    ws = MagicMock()
    ws.closed = False
    ws.close = AsyncMock(side_effect=_hang_forever)
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    adapter._ws = ws
    adapter._session = session

    await asyncio.wait_for(adapter._cleanup_ws(), timeout=2)

    assert adapter._ws is None
    assert adapter._session is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_completes_within_bounds_when_closes_hang(
    monkeypatch,
):
    """Disconnect completes when every close wedges."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05)

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
    await _attach_finished_listener(adapter)

    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    assert adapter._ws is None
    assert adapter._session is None
    assert adapter._rest_session is None
    assert adapter._running is False


@pytest.mark.asyncio
async def test_ws_connect_bounds_hanging_auth_handshake(monkeypatch):
    """A silent server cannot freeze the authentication handshake."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_HANDSHAKE_TIMEOUT", 0.05)

    ws = MagicMock()
    ws.closed = False
    ws.receive_json = AsyncMock(side_effect=_hang_forever)
    ws.send_json = AsyncMock()
    ws.close = AsyncMock()
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(return_value=ws)

    with patch.object(adapter, "_new_session", return_value=session):
        result = await asyncio.wait_for(adapter._ws_connect(), timeout=2)

    assert result is False
    ws.close.assert_awaited_once()
    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_ws_connect_cleans_up_when_handshake_send_raises():
    """A failed handshake send closes both unpublished resources."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False
    ws.receive_json = AsyncMock(return_value={"type": "auth_required"})
    ws.send_json = AsyncMock(
        side_effect=ConnectionResetError("peer gone"))
    ws.close = AsyncMock()
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(return_value=ws)

    with patch.object(adapter, "_new_session", return_value=session):
        result = await asyncio.wait_for(adapter._ws_connect(), timeout=2)

    assert result is False
    ws.close.assert_awaited_once()
    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_cancel_task_bounded_abandons_uncancellable_task(monkeypatch):
    """Task cancellation is observed with a deadline, not awaited forever."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05)

    started = asyncio.Event()
    stop = asyncio.Event()

    async def _ignores_cancel():
        while not stop.is_set():
            try:
                started.set()
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue

    zombie = asyncio.create_task(_ignores_cancel())
    await started.wait()

    await asyncio.wait_for(
        adapter._cancel_task_bounded(zombie, "zombie"),
        timeout=2)

    assert not zombie.done()
    stop.set()
    zombie.cancel()
    await asyncio.wait({zombie}, timeout=1)


@pytest.mark.asyncio
async def test_ws_connect_cancellation_closes_local_session():
    """Cancellation during ws_connect still closes the local session."""
    adapter = _make_adapter()
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()
    session.ws_connect = AsyncMock(side_effect=_hang_forever)

    with patch.object(adapter, "_new_session", return_value=session):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                adapter._ws_connect(),
                timeout=0.05)

    session.close.assert_awaited_once()
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_ws_connect_close_survives_second_cancellation_during_teardown():
    """A second cancellation detaches rather than cancelling the close."""
    adapter = _make_adapter()
    close_started = asyncio.Event()
    close_completed = asyncio.Event()
    session = MagicMock()
    session.closed = False

    async def _slow_close():
        close_started.set()
        await asyncio.sleep(0.05)
        close_completed.set()

    session.close = _slow_close
    session.ws_connect = AsyncMock(side_effect=_hang_forever)

    with patch.object(adapter, "_new_session", return_value=session):
        task = asyncio.create_task(adapter._ws_connect())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait_for(close_started.wait(), timeout=2)
        tracked = tuple(adapter._teardown_tasks)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await _await_tracked_teardown(tracked)
    assert close_completed.is_set()
    assert adapter._session is None
    assert adapter._ws is None


@pytest.mark.asyncio
async def test_cleanup_ws_close_survives_second_cancellation_during_teardown():
    """Cleanup keeps closing detached fields after caller cancellation."""
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

    async def _cancelled_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await adapter._cleanup_ws()
            raise

    task = asyncio.create_task(_cancelled_cleanup())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(close_started.wait(), timeout=2)
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _await_tracked_teardown(tracked)
    assert close_completed.is_set()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_cleanup_ws_closes_session_when_cancelled_during_ws_close():
    """Cancellation during WS close cannot skip the later session close."""
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

    async def _cancelled_cleanup():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await adapter._cleanup_ws()
            raise

    task = asyncio.create_task(_cancelled_cleanup())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(ws_close_started.wait(), timeout=2)
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _await_tracked_teardown(tracked)
    assert ws_close_completed.is_set()
    session.close.assert_awaited_once()
    assert adapter._ws is None
    assert adapter._session is None


@pytest.mark.asyncio
async def test_disconnect_rest_session_not_double_closed_on_cancellation():
    """A retry cannot close an already-detached REST session twice."""
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
    await _attach_finished_listener(adapter)

    task = asyncio.create_task(adapter.disconnect())
    await asyncio.wait_for(close_started.wait(), timeout=2)
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert adapter._rest_session is None

    adapter._running = True
    await _attach_finished_listener(adapter)
    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    await _await_tracked_teardown(tracked)
    assert close_completed.is_set()
    assert close_calls == 1


@pytest.mark.asyncio
async def test_disconnect_closes_rest_session_when_cancelled_during_ws_close():
    """Cancellation during WS close cannot skip the REST-session close."""
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
    await _attach_finished_listener(adapter)

    task = asyncio.create_task(adapter.disconnect())
    await asyncio.wait_for(ws_close_started.wait(), timeout=2)
    tracked = tuple(adapter._teardown_tasks)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _await_tracked_teardown(tracked)
    assert rest_close_completed.is_set()
    assert adapter._rest_session is None


@pytest.mark.asyncio
async def test_bounded_close_abandons_cancellation_suppressing_close(
    monkeypatch,
):
    """A cancellation-suppressing close cannot extend the caller's bound."""
    adapter = _make_adapter()
    monkeypatch.setattr(ha_adapter, "_DRAIN_TIMEOUT", 0.05)

    stop = asyncio.Event()
    close_finished = asyncio.Event()

    async def _suppresses_cancellation():
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                continue
        close_finished.set()

    closeable = MagicMock()
    closeable.closed = False
    closeable.close = AsyncMock(side_effect=_suppresses_cancellation)
    before = set(ha_adapter._TEARDOWN_REGISTRY)
    bounded_close = asyncio.create_task(adapter._close(closeable))

    try:
        _, pending = await asyncio.wait({bounded_close}, timeout=1)
        assert not pending
        assert not close_finished.is_set()

        abandoned = [
            task for task in ha_adapter._TEARDOWN_REGISTRY - before
            if not task.done()
        ]
        assert abandoned
        assert abandoned[0] in ha_adapter._TEARDOWN_REGISTRY
        abandoned_ref = weakref.ref(abandoned[0])
        del abandoned
        gc.collect()
        still_alive = abandoned_ref()
        assert still_alive is not None
        assert not still_alive.done()
    finally:
        stop.set()
        await asyncio.wait_for(close_finished.wait(), timeout=2)


@pytest.mark.asyncio
async def test_cleanup_ws_session_close_runs_when_ws_close_raises_cancellederror():
    """A close-originated CancelledError cannot skip the session close."""
    adapter = _make_adapter()

    ws = MagicMock()
    ws.closed = False

    async def _raises_cancelled():
        raise asyncio.CancelledError("raised by close")

    ws.close = AsyncMock(side_effect=_raises_cancelled)
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
    """A close-originated CancelledError cannot skip the REST close."""
    adapter = _make_adapter()
    adapter._running = True

    ws = MagicMock()
    ws.closed = False

    async def _raises_cancelled():
        raise asyncio.CancelledError("raised by close")

    ws.close = AsyncMock(side_effect=_raises_cancelled)
    adapter._ws = ws
    adapter._session = None

    rest_session = MagicMock()
    rest_session.closed = False
    rest_session.close = AsyncMock()
    adapter._rest_session = rest_session
    await _attach_finished_listener(adapter)

    await asyncio.wait_for(adapter.disconnect(), timeout=2)

    rest_session.close.assert_awaited_once()
    assert adapter._rest_session is None
