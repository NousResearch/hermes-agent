"""Regression tests: failed-connect path must call adapter.disconnect().

When adapter.connect() returns False or raises, the adapter may have
allocated resources (aiohttp.ClientSession, poll tasks, child
subprocesses) before giving up. Without a defensive disconnect() call
these leak and surface as "Unclosed client session" warnings at
process exit (seen on the 2026-04-18 18:08:16 gateway restart).

The fix: gateway/run.py wraps each adapter connect() with a safety-net
call to _safe_adapter_disconnect() in the failure branches.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """A GatewayRunner shell that only needs to support _safe_adapter_disconnect."""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_safe_disconnect_calls_adapter_disconnect(bare_runner):
    """The helper forwards to adapter.disconnect()."""
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(return_value=None)

    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)

    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_swallows_exceptions(bare_runner):
    """An exception in adapter.disconnect() must not propagate — the
    caller is already on an error path."""
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=RuntimeError("partial init"))

    # Must NOT raise
    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)

    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_handles_none_platform(bare_runner):
    """Logging path must tolerate platform=None."""
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=ValueError("nope"))

    await bare_runner._safe_adapter_disconnect(adapter, None)

    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_times_out_and_continues(bare_runner, monkeypatch, caplog):
    """A wedged adapter disconnect must not block gateway shutdown."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.001")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(60)

    adapter.disconnect = AsyncMock(side_effect=hang)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)

    adapter.disconnect.assert_awaited_once()
    assert "Timed out after 0.0s while disconnecting feishu adapter" in caplog.text


@pytest.mark.asyncio
async def test_safe_disconnect_accepts_precreated_and_cancelled_prepare_futures(
    bare_runner
):
    completed = asyncio.get_running_loop().create_future()
    completed.set_result(None)
    adapter = MagicMock()
    adapter.prepare_disconnect = MagicMock(return_value=completed)
    adapter.disconnect = AsyncMock(return_value=None)

    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)

    adapter.disconnect.assert_awaited_once()

    cancelled = asyncio.get_running_loop().create_future()
    cancelled.cancel()
    adapter = MagicMock()
    adapter.prepare_disconnect = MagicMock(return_value=cancelled)
    adapter.active_session_sources.return_value = ()
    adapter.disconnect = AsyncMock(return_value=None)

    # A cancelled child is a failed prepare, not cancellation of the host.
    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)
    adapter.disconnect.assert_awaited_once()
    adapter.set_session_interrupt_handler.assert_called_once_with(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["prepare", "fallback", "disconnect"])
async def test_safe_disconnect_finishes_cleanup_before_reraising_outer_cancel(
    bare_runner, monkeypatch, phase
):
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.2")
    entered = asyncio.Event()
    release = asyncio.Event()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        chat_type="dm",
        user_id="user",
    )
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

    adapter.disconnect = AsyncMock(
        side_effect=block_here if phase == "disconnect" else None
    )
    task = asyncio.create_task(
        bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)
    )
    await entered.wait()
    task.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    adapter.disconnect.assert_awaited_once()
    adapter.set_session_interrupt_handler.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_timed_prepare_keeps_runner_authority_for_canonical_fallback(
    bare_runner, monkeypatch
):
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat",
        chat_type="dm",
        user_id="user",
        message_id="message",
        profile="default",
    )
    adapter = MagicMock()
    adapter.platform = Platform.TELEGRAM
    adapter._session_interrupt_handler = None
    adapter.active_session_sources.return_value = (source,)

    async def hang_prepare():
        await asyncio.sleep(60)

    async def disconnect_with_authority():
        assert adapter._session_interrupt_handler is not None

    def install(handler):
        adapter._session_interrupt_handler = handler

    async def request_interrupt(active):
        return await adapter._session_interrupt_handler(
            active,
            interrupt_reason="fallback",
            invalidation_reason="teardown",
        )

    adapter.prepare_disconnect = AsyncMock(side_effect=hang_prepare)
    adapter.disconnect = AsyncMock(side_effect=disconnect_with_authority)
    adapter.set_session_interrupt_handler = MagicMock(side_effect=install)
    adapter.request_session_interrupt = AsyncMock(side_effect=request_interrupt)
    bare_runner.config = SimpleNamespace(multiplex_profiles=False)
    bare_runner._session_key_for_source = MagicMock(return_value="telegram:chat")
    bare_runner._interrupt_and_clear_session = AsyncMock()
    handler = bare_runner._make_adapter_session_interrupt_handler(
        adapter, profile_name="default"
    )
    adapter.set_session_interrupt_handler(handler)

    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)

    bare_runner._interrupt_and_clear_session.assert_awaited_once()
    adapter.disconnect.assert_awaited_once()
    assert adapter._session_interrupt_handler is None
