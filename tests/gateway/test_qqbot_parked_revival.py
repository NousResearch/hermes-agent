"""QQBot adapter parks (typed state + revival probe) instead of terminal silence.

After ``Max reconnect attempts reached``, the adapter previously performed
zero reconnect action in-process with no typed alert; only a new process
revived it. This borrows the MCP parking pattern: the ladder topping out
flips the adapter to ``parked`` (typed, observable), emits a typed
``PLATFORM_ADAPTER_PARKED`` event line, and keeps a bounded slow revival
probe.

Zero-network criteria: mock DNS failure until the
ladder tops out and assert (a) state is parked, (b) a callable revive entry
exists, (c) a typed alert event was emitted. No real connection to
bots.qq.com is made.
"""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest

from gateway.platforms.qqbot.adapter import QQAdapter as QQBotAdapter
from gateway.platforms.qqbot.constants import PARKED_PROBE_INTERVAL_SECONDS


class _FakeConfig:
    def __init__(self):
        from gateway.platforms.base import Platform
        self.platform = Platform.QQBOT
        self.extra = {}


def _make_adapter() -> QQBotAdapter:
    adapter = QQBotAdapter.__new__(QQBotAdapter)
    adapter._app_id = "test"
    adapter._parked = False
    adapter._parked_since = None
    adapter._first_failure_at = None
    adapter._revival_task = None
    adapter._listen_task = None
    adapter._heartbeat_task = None
    adapter._running = True
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = None
    adapter._fatal_error_handler = None
    adapter._status_write_logged = set()
    return adapter


class TestParkedRevival:
    @pytest.mark.asyncio
    async def test_park_sets_typed_state_and_event(self, caplog):
        """Ladder top-out → parked state + typed PLATFORM_ADAPTER_PARKED event."""
        adapter = _make_adapter()
        statuses = []
        with patch.object(adapter, "_write_runtime_status_safe", side_effect=lambda ctx, **kw: statuses.append((ctx, kw))), \
             patch.object(adapter, "_revival_probe_loop", new=AsyncMock()) as probe, \
             patch("asyncio.create_task") as create_task:
            create_task.return_value = object()
            with caplog.at_level(logging.ERROR):
                await adapter._park_for_revival("DNS-resolution-failed")

        assert adapter._parked is True
        assert adapter._parked_since is not None
        assert adapter._first_failure_at is not None
        # (c) typed alert event emitted
        typed = [r for r in caplog.records if "PLATFORM_ADAPTER_PARKED" in r.getMessage()]
        assert typed, "typed PLATFORM_ADAPTER_PARKED event must be logged at ERROR level"
        msg = typed[0].getMessage()
        assert "platform=qqbot" in msg
        assert "attempts=%d" % 100 in msg or "attempts=" in msg
        assert "first_failure_at=" in msg
        # (a) runtime status published as parked with typed error code
        assert statuses and statuses[0][1].get("platform_state") == "parked"
        assert statuses[0][1].get("error_code") == "PLATFORM_ADAPTER_PARKED"
        # (b) revival entry scheduled
        probe.assert_not_awaited()  # the loop itself is mocked; only its task creation is asserted
        create_task.assert_called()

    @pytest.mark.asyncio
    async def test_revival_probe_recovers_without_process_restart(self, caplog):
        """The probe loop revives the adapter on platform recovery."""
        adapter = _make_adapter()
        adapter._parked = True
        adapter._parked_since = time.time() - 60
        adapter._first_failure_at = time.time() - 120

        calls = {"open": 0}

        async def _fake_open():
            calls["open"] += 1
            if calls["open"] < 2:
                raise OSError("probe fails while platform down")
            return None

        with patch.object(adapter, "_open_gateway_ws", side_effect=_fake_open), \
             patch.object(adapter, "_mark_connected") as mark_c, \
             patch.object(QQBotAdapter, "_listen_loop", new=AsyncMock()) as listen, \
             patch.object(QQBotAdapter, "_heartbeat_loop", new=AsyncMock()) as heart, \
             patch("asyncio.create_task", side_effect=lambda coro: asyncio.ensure_future(coro)), \
             patch("asyncio.sleep", new=AsyncMock()) as asleep:
            await adapter._revival_probe_loop()

        assert adapter._parked is False
        # A revived adapter must forget the prior outage's first-failure stamp so a
        # later re-park times from its own first failure, not a stale timestamp.
        assert adapter._first_failure_at is None
        mark_c.assert_called_once()
        revived = [r for r in caplog.records if "PLATFORM_ADAPTER_REVIVED" in r.getMessage()]
        assert revived, "typed PLATFORM_ADAPTER_REVIVED event must be logged"

    @pytest.mark.asyncio
    async def test_park_cancels_stale_heartbeat_task(self):
        """Blocking-review regression: park must stop the pre-park heartbeat.

        Park does not clear ``_running``, and ``_heartbeat_loop`` loops on
        ``while self._running`` while re-reading ``self._ws`` each pass — so a
        surviving heartbeat task sends op-1 frames the moment a revival probe
        reopens the socket. Park must cancel it (``disconnect()`` only ever
        holds the newest handle, so an uncancelled one is orphaned)."""
        adapter = _make_adapter()
        stale_heartbeat = asyncio.ensure_future(asyncio.sleep(3600))
        adapter._heartbeat_task = stale_heartbeat
        await asyncio.sleep(0)  # let the stale task actually start

        with patch.object(adapter, "_write_runtime_status_safe"), \
             patch.object(adapter, "_revival_probe_loop", new=AsyncMock()), \
             patch("asyncio.create_task", return_value=object()):
            await adapter._park_for_revival("generic")

        assert stale_heartbeat.cancelled(), "pre-park heartbeat task must be cancelled on park"
        assert adapter._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_revive_cancels_old_tasks_before_reassign_no_stacking(self):
        """Blocking-review regression: revival must cancel the old listen/heartbeat
        handles before overwriting them (the disconnect() discipline), so a
        re-park/revive cycle cannot stack orphaned loops on one socket."""
        adapter = _make_adapter()
        adapter._parked = True
        adapter._parked_since = time.time() - 60
        old_listen = asyncio.ensure_future(asyncio.sleep(3600))
        old_heart = asyncio.ensure_future(asyncio.sleep(3600))
        adapter._listen_task = old_listen
        adapter._heartbeat_task = old_heart
        await asyncio.sleep(0)

        async def _ok_open():
            return None

        new_tasks = []

        def _track_create(coro):
            task = asyncio.ensure_future(coro)
            new_tasks.append(task)
            return task

        with patch.object(adapter, "_open_gateway_ws", side_effect=_ok_open), \
             patch.object(adapter, "_mark_connected"), \
             patch.object(QQBotAdapter, "_listen_loop", new=AsyncMock()), \
             patch.object(QQBotAdapter, "_heartbeat_loop", new=AsyncMock()), \
             patch("asyncio.create_task", side_effect=_track_create), \
             patch("asyncio.sleep", new=AsyncMock()):
            await adapter._revival_probe_loop()

        assert old_listen.cancelled(), "pre-revive listen task must be cancelled, not orphaned"
        assert old_heart.cancelled(), "pre-revive heartbeat task must be cancelled, not orphaned"
        assert adapter._listen_task is not old_listen
        assert adapter._heartbeat_task is not old_heart
        # Exactly one fresh listen + one fresh heartbeat — no stacking.
        assert adapter._listen_task in new_tasks and adapter._heartbeat_task in new_tasks
        for t in new_tasks:
            t.cancel()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_probe(self):
        """disconnect() must cancel the revival task so it cannot outlive the adapter."""
        adapter = _make_adapter()
        probe_task = asyncio.ensure_future(asyncio.sleep(3600))
        adapter._revival_task = probe_task
        adapter._listen_task = None
        adapter._heartbeat_task = None

        with patch.object(adapter, "_mark_disconnected"), \
             patch.object(adapter, "_cleanup", new=AsyncMock()), \
             patch.object(adapter, "_release_platform_lock"), \
             patch("gateway.platforms.qqbot.adapter.cancel_task", new=AsyncMock()) as cancel:
            await adapter.disconnect()

        cancel.assert_any_call(probe_task)
        assert adapter._parked is False

    def test_probe_interval_is_bounded_and_low_frequency(self):
        """The probe must not be a hot loop: interval in minutes, not seconds."""
        assert 60 <= PARKED_PROBE_INTERVAL_SECONDS <= 900, \
            "revival probe interval should stay in the 1-15 minute band"
