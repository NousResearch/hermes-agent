"""Regression tests: the shutdown teardown loop must not hang on a wedged adapter.

`GatewayRunner._stop_impl()` tears down every adapter by awaiting
`cancel_background_tasks()` then `disconnect()`. Both calls can block
indefinitely when a platform's network state is half-dead (e.g. a wedged
Feishu/Lark WebSocket thread waiting on I/O). An unbounded await stalls the
whole shutdown past systemd's TimeoutStopSec; the resulting SIGKILL skips
atexit PID-file cleanup, so the next start dies with "PID file race lost"
(#14128).

The fix routes both teardown loops through `_bounded_adapter_teardown`,
which wraps each await in the existing per-adapter timeout budget
(HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT) and always returns.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """A GatewayRunner shell that only needs _bounded_adapter_teardown."""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_teardown_calls_both_methods(bare_runner):
    """Prepare runs with authority live, then revoke, cancel, disconnect."""
    calls = []
    adapter = MagicMock()
    handler_live = True
    adapter.prepare_disconnect = AsyncMock(
        side_effect=lambda: calls.append(f"prepare:{handler_live}")
    )
    def revoke(_handler):
        nonlocal handler_live
        handler_live = False
        calls.append("revoke")
    adapter.set_session_interrupt_handler = MagicMock(side_effect=revoke)
    adapter.cancel_background_tasks = AsyncMock(
        side_effect=lambda: calls.append("cancel")
    )
    adapter.disconnect = AsyncMock(side_effect=lambda: calls.append("disconnect"))

    await bare_runner._bounded_adapter_teardown(adapter, Platform.TELEGRAM)

    adapter.cancel_background_tasks.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()
    assert calls == ["prepare:True", "revoke", "cancel", "disconnect"]


@pytest.mark.asyncio
async def test_teardown_bounds_hanging_disconnect(bare_runner, monkeypatch, caplog):
    """A wedged disconnect() must time out instead of hanging the loop."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()
    adapter.cancel_background_tasks = AsyncMock(return_value=None)

    async def hang():
        await asyncio.sleep(60)

    adapter.disconnect = AsyncMock(side_effect=hang)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await asyncio.wait_for(
            bare_runner._bounded_adapter_teardown(adapter, Platform.FEISHU),
            timeout=5.0,  # the helper itself must return well under this
        )

    adapter.disconnect.assert_awaited_once()
    assert "feishu disconnect timed out" in caplog.text


@pytest.mark.asyncio
async def test_teardown_bounds_hanging_prepare_before_revocation(
    bare_runner, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(60)

    adapter.prepare_disconnect = AsyncMock(side_effect=hang)
    adapter.cancel_background_tasks = AsyncMock(return_value=None)
    adapter.disconnect = AsyncMock(return_value=None)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await bare_runner._bounded_adapter_teardown(adapter, Platform.FEISHU)

    assert "preparing feishu adapter disconnect" in caplog.text
    adapter.set_session_interrupt_handler.assert_called_once_with(None)
    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_teardown_does_not_wait_for_cancellation_resistant_prepare(
    bare_runner, monkeypatch
):
    """A prepare coroutine suppressing cancellation cannot wedge shutdown."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    adapter = MagicMock()

    async def resist_cancellation():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        finally:
            finished.set()

    adapter.prepare_disconnect = AsyncMock(side_effect=resist_cancellation)
    adapter.cancel_background_tasks = AsyncMock(return_value=None)
    adapter.disconnect = AsyncMock(return_value=None)

    try:
        await asyncio.wait_for(
            bare_runner._bounded_adapter_teardown(adapter, Platform.FEISHU),
            timeout=0.5,
        )
        await asyncio.wait_for(cancelled.wait(), timeout=0.5)
        adapter.disconnect.assert_awaited_once()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "fallback", "disconnect"])
async def test_bounded_teardown_finishes_before_reraising_outer_cancel(
    bare_runner, monkeypatch, phase
):
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.2")
    entered = asyncio.Event()
    release = asyncio.Event()
    source = MagicMock()
    adapter = MagicMock()
    adapter.active_session_sources.return_value = (source,) if phase == "fallback" else ()

    async def block_here(*_args):
        entered.set()
        await release.wait()

    if phase == "prepare":
        adapter.prepare_disconnect = AsyncMock(side_effect=block_here)
    elif phase == "fallback":
        failed = asyncio.get_running_loop().create_future()
        failed.cancel()
        adapter.prepare_disconnect = MagicMock(return_value=failed)
        adapter.request_session_interrupt = AsyncMock(side_effect=block_here)
    else:
        adapter.prepare_disconnect = AsyncMock(return_value=None)

    adapter.cancel_background_tasks = AsyncMock(return_value=None)
    adapter.disconnect = AsyncMock(
        side_effect=block_here if phase == "disconnect" else None
    )
    task = asyncio.create_task(
        bare_runner._bounded_adapter_teardown(adapter, Platform.TELEGRAM)
    )
    await entered.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    adapter.cancel_background_tasks.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()
    adapter.set_session_interrupt_handler.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_teardown_bounds_hanging_cancel(bare_runner, monkeypatch, caplog):
    """A wedged cancel_background_tasks() must time out, then disconnect runs."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(60)

    adapter.cancel_background_tasks = AsyncMock(side_effect=hang)
    adapter.disconnect = AsyncMock(return_value=None)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await asyncio.wait_for(
            bare_runner._bounded_adapter_teardown(adapter, Platform.FEISHU),
            timeout=5.0,
        )

    assert "feishu background-task cancel timed out" in caplog.text
    # disconnect still attempted after the cancel timeout — forward progress.
    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_teardown_swallows_exceptions(bare_runner):
    """Errors in either await must not propagate — shutdown continues."""
    adapter = MagicMock()
    adapter.cancel_background_tasks = AsyncMock(side_effect=RuntimeError("bg"))
    adapter.disconnect = AsyncMock(side_effect=RuntimeError("disc"))

    # Must NOT raise.
    await bare_runner._bounded_adapter_teardown(adapter, Platform.TELEGRAM)

    adapter.cancel_background_tasks.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_teardown_profile_suffix_in_logs(bare_runner, caplog):
    """Multiplex (secondary-profile) teardown tags log lines with the profile."""
    adapter = MagicMock()
    adapter.cancel_background_tasks = AsyncMock(return_value=None)
    adapter.disconnect = AsyncMock(return_value=None)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await bare_runner._bounded_adapter_teardown(
            adapter, Platform.TELEGRAM, profile="acct2"
        )

    assert "(profile: acct2)" in caplog.text


@pytest.mark.asyncio
async def test_teardown_timeout_zero_disables_bound(bare_runner, monkeypatch):
    """timeout=0 disables the wait_for wrapper but still calls through."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0")
    adapter = MagicMock()
    adapter.cancel_background_tasks = AsyncMock(return_value=None)
    adapter.disconnect = AsyncMock(return_value=None)

    await bare_runner._bounded_adapter_teardown(adapter, Platform.TELEGRAM)

    adapter.cancel_background_tasks.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()
