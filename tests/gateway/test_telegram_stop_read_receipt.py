"""Network recovery must not wedge ``Updater.stop()`` on PTB's read-receipt getUpdates (#131783)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


class _FakePTBUpdater:
    """Mimic PTB 22.x ``Updater.stop()``: after the polling task quiesces, run the private
    read-receipt callback (a ``get_updates(timeout=0)`` on the same request pool)."""

    def __init__(self, cleanup_cb):
        self.running = True
        if cleanup_cb is not None:
            self._Updater__polling_cleanup_cb = cleanup_cb

    async def stop(self):
        self.running = False
        cb = getattr(self, "_Updater__polling_cleanup_cb", None)
        if cb is not None:
            await cb()
            self._Updater__polling_cleanup_cb = None


@pytest.mark.asyncio
async def test_network_recovery_skips_read_receipt_cleanup_wedged_in_stop():
    """A transient network error must not wedge stop() on PTB's read-receipt getUpdates.

    PTB's ``Updater.stop()`` ends with ``_get_updates_cleanup()`` — a ``get_updates(timeout=0)``
    confirming the last fetched offset. After a network error that request re-uses the broken
    pool, and its getUpdates read timeout outlives the stop deadline, so stop() hung for the
    full budget on every transient error and escalated to a full adapter rebuild (#131783).
    Recovery must drop the receipt (offset resume + admission receipts already cover it) so
    stop() completes and the same Updater is reused.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    released = asyncio.Event()

    async def _wedged_cleanup():
        await released.wait()

    updater = _FakePTBUpdater(_wedged_cleanup)
    app = MagicMock()
    app.updater = updater
    app.bot._request = ()
    adapter._app = app
    adapter._notify_fatal_error = AsyncMock()

    drain_called = []
    start_called = []

    async def _fake_drain():
        drain_called.append(True)

    async def _fake_start(*args, **kwargs):
        start_called.append(True)

    adapter._drain_polling_connections = _fake_drain
    adapter._start_polling_once = _fake_start

    import plugins.platforms.telegram.adapter as _mod

    try:
        with patch.object(_mod, "_UPDATER_STOP_TIMEOUT", 0.05), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            await adapter._handle_polling_network_error(Exception("Bad Gateway"))

        assert not adapter.has_fatal_error, (
            "stop() wedged on the read-receipt cleanup and escalated to a fatal rebuild"
        )
        adapter._notify_fatal_error.assert_not_awaited()
        assert getattr(updater, "_Updater__polling_cleanup_cb", None) is None
        assert drain_called and start_called, "recovery must reuse the stopped Updater"
    finally:
        released.set()


@pytest.mark.asyncio
async def test_stop_skips_cleanup_only_when_callback_present():
    """The cleanup skip must be inert when PTB exposes no cleanup callback (renamed internals)."""
    adapter = _make_adapter()
    updater = _FakePTBUpdater(None)
    app = MagicMock()
    app.updater = updater
    adapter._app = app

    assert await adapter._stop_updater_or_go_fatal(
        app, "network-recovery", skip_read_receipt_cleanup=True
    )
    assert not updater.running
    assert not hasattr(updater, "_Updater__polling_cleanup_cb")
    assert not adapter.has_fatal_error
