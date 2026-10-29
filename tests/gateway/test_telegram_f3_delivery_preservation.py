"""F3 — an exhausted Telegram polling conflict must not lose queued messages.

On 2026-10-23 the gateway received Bryan's DMs and then destroyed them itself.  The
polling stall watchdog rebuilt the adapter (00:01:27), the replacement's first poll was
409'd (00:01:43 — two seconds after the connect, far inside Telegram's ~30 s session
window), and the conflict ladder ended in a fatal that was reported NON-retryable.

The server-side queue drop is already fixed upstream by t_ab25bd5a
(``drop_pending_updates`` now follows the ``drop_pending_on_cold_boot`` knob, default
preserving) — see ``tests/gateway/test_telegram_conflict.py``.  What is NOT covered
there is the data loss on the LOCAL side, which is what this file pins:

1. an exhausted conflict must stay RETRYABLE, because the non-retryable branch of
   ``_set_fatal_error`` clears the adapter's own held-inbound queue and
   ``_queue_retryable_fatal_platform`` refuses to queue a non-retryable platform for
   reconnection — so nothing would ever retry it, and the runner's stranded check is
   gated on the same flag, so the gateway would exit cleanly (systemd would not restart
   it either);
2. a rebuilt adapter must not start polling inside its predecessor's getUpdates session
   window, which is what turned a normal rebuild into a 409 fight.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter.set_fatal_error_handler(AsyncMock())
    adapter._drain_polling_connections = AsyncMock()
    return adapter


def _conflict() -> Exception:
    return type("Conflict", (Exception,), {})(
        "Conflict: terminated by other getUpdates request"
    )


class TestExhaustedConflictStaysRetryable:
    """A conflict is transient by construction; a permanent fatal destroys the queue."""

    @pytest.mark.asyncio
    async def test_exhausted_conflict_is_retryable(self, monkeypatch):
        """A non-retryable conflict clears the held inbound queue and blocks reconnection."""
        adapter = _adapter()
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None)
        )
        monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

        async def failing_start_polling(**kwargs):
            raise Exception("Connection refused")

        adapter._app = SimpleNamespace(
            updater=SimpleNamespace(
                start_polling=AsyncMock(side_effect=failing_start_polling),
                stop=AsyncMock(),
                running=True,
            )
        )
        for _ in range(7):
            await adapter._handle_polling_conflict(_conflict())

        assert adapter.fatal_error_code == "telegram_polling_conflict"
        assert adapter.has_fatal_error is True
        # Retryable — so the platform is queued for background reconnection and the
        # gateway exits with FAILURE (systemd restarts it) instead of exiting cleanly.
        assert adapter.fatal_error_retryable is True

    @pytest.mark.asyncio
    async def test_held_inbound_queue_survives_an_exhausted_conflict(self, monkeypatch):
        """The held inbound queue must not be discarded by the conflict path."""
        adapter = _adapter()
        monkeypatch.setattr("asyncio.sleep", AsyncMock())
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock", lambda scope, identity, metadata=None: (True, None)
        )
        monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)

        async def failing_start_polling(**kwargs):
            raise Exception("Connection refused")

        adapter._app = SimpleNamespace(
            updater=SimpleNamespace(
                start_polling=AsyncMock(side_effect=failing_start_polling),
                stop=AsyncMock(),
                running=True,
            )
        )
        # A message that arrived during the outage and is waiting to be delivered.
        adapter._held_inbound_events = [SimpleNamespace(chat_id=1, text="hello")]

        for _ in range(7):
            await adapter._handle_polling_conflict(_conflict())

        assert adapter.fatal_error_code == "telegram_polling_conflict"
        assert len(adapter._held_inbound_events) == 1, (
            "the held inbound queue must survive a polling conflict — it holds the messages "
            "that arrived during the outage"
        )


class TestReconnectWaitsForPreviousSession:
    """A rebuilt adapter must not start polling inside the predecessor's session window."""

    @pytest.mark.asyncio
    async def test_reconnect_waits_before_polling(self, monkeypatch):
        adapter = _adapter()
        adapter._polling_generation_started_monotonic = 1.0  # a predecessor polled
        waited = []

        async def fake_sleep(seconds):
            waited.append(seconds)

        monkeypatch.setattr("asyncio.sleep", fake_sleep)
        monkeypatch.setattr(adapter, "_drain_polling_connections", AsyncMock())
        monkeypatch.setattr(adapter, "_arm_queued_update_redelivery", lambda: None)

        async def fake_stop_updater(app, reason):
            return True

        monkeypatch.setattr(adapter, "_stop_updater_or_go_fatal", fake_stop_updater)

        async def fake_start_polling_once(app, *, drop_pending_updates, error_callback):
            return None

        monkeypatch.setattr(adapter, "_start_polling_once", fake_start_polling_once)
        adapter._app = SimpleNamespace(updater=SimpleNamespace(stop=AsyncMock(), running=True))

        await adapter._handle_polling_network_error(Exception("network error"))
        assert waited, "a reconnect must wait for the previous getUpdates session to be released"

    @pytest.mark.asyncio
    async def test_cold_boot_does_not_wait(self, monkeypatch):
        """A cold boot has no predecessor session to wait for."""
        adapter = _adapter()
        waited = []

        async def fake_sleep(seconds):
            waited.append(seconds)

        monkeypatch.setattr("asyncio.sleep", fake_sleep)
        adapter._polling_generation_started_monotonic = None

        await adapter._await_previous_session_release()
        assert not waited, "a cold boot must not wait for a previous session"

    @pytest.mark.asyncio
    async def test_teardown_short_circuits_the_wait(self, monkeypatch):
        adapter = _adapter()
        adapter._polling_generation_started_monotonic = 1.0
        adapter._polling_teardown_started = True
        waited = []

        async def fake_sleep(seconds):
            waited.append(seconds)

        monkeypatch.setattr("asyncio.sleep", fake_sleep)
        await adapter._await_previous_session_release()
        assert not waited
