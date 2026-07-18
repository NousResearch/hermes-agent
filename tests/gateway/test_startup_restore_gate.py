"""Tests for the startup-restore queue ACK.

Covers the user-facing half of the "slash command silently queued after a
restart" bug. The *bounded wait* half (a slow boot-resume turn holding the
inbound gate) shipped separately in PR #256; this adds the missing feedback:
when a fresh inbound message is queued during startup restore, the user gets a
one-time "still starting up" ack per chat instead of silence.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform
from gateway.run import GatewayRunner
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


def _bind(runner):
    """Bind the ack methods-under-test onto an object.__new__ runner."""
    for name in (
        "_queue_startup_restore_event",
        "_maybe_ack_startup_restore_queue",
        "_send_startup_restore_ack",
    ):
        setattr(runner, name, getattr(GatewayRunner, name).__get__(runner, GatewayRunner))
    return runner


def _make_event(source):
    ev = MagicMock()
    ev.source = source
    ev.internal = False
    return ev


class TestQueueAck:
    def test_ack_sent_once_per_chat(self):
        adapter = AsyncMock()
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._startup_restore_in_progress = True

        async def _drive():
            src = make_restart_source(chat_id="c1")
            runner._queue_startup_restore_event(_make_event(src))
            runner._queue_startup_restore_event(_make_event(src))
            runner._queue_startup_restore_event(_make_event(src))
            await asyncio.sleep(0.05)  # let fire-and-forget ack tasks run

        asyncio.run(_drive())
        # Three queued messages, ONE ack; all three still queued for the drain.
        assert adapter.send.await_count == 1
        assert len(runner._startup_restore_queue) == 3

    def test_ack_per_distinct_chat(self):
        adapter = AsyncMock()
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._startup_restore_in_progress = True

        async def _drive():
            runner._queue_startup_restore_event(_make_event(make_restart_source(chat_id="a")))
            runner._queue_startup_restore_event(_make_event(make_restart_source(chat_id="b")))
            await asyncio.sleep(0.05)

        asyncio.run(_drive())
        assert adapter.send.await_count == 2

    def test_ack_reset_between_restore_cycles(self):
        adapter = AsyncMock()
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._startup_restore_in_progress = True

        async def _drive():
            src = make_restart_source(chat_id="c1")
            runner._queue_startup_restore_event(_make_event(src))
            await asyncio.sleep(0.02)
            # Simulate a NEW restart cycle resetting the dedupe set.
            runner._startup_restore_acked_chats = set()
            runner._queue_startup_restore_event(_make_event(src))
            await asyncio.sleep(0.02)

        asyncio.run(_drive())
        assert adapter.send.await_count == 2

    def test_no_ack_when_adapter_missing(self):
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {}  # no adapter for the platform
        runner._startup_restore_in_progress = True

        async def _drive():
            runner._queue_startup_restore_event(_make_event(make_restart_source(chat_id="c1")))
            await asyncio.sleep(0.02)

        asyncio.run(_drive())  # must not raise; simply no ack
        assert len(runner._startup_restore_queue) == 1

    def test_ack_send_failure_is_swallowed(self):
        adapter = AsyncMock()
        adapter.send.side_effect = RuntimeError("boom")
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._startup_restore_in_progress = True

        async def _drive():
            runner._queue_startup_restore_event(_make_event(make_restart_source(chat_id="c1")))
            await asyncio.sleep(0.05)

        asyncio.run(_drive())  # a failing ack must not break queuing
        assert len(runner._startup_restore_queue) == 1


class TestGreptileRaces:
    """The three #258 review races, each pinned."""

    def test_missing_adapter_does_not_burn_dedupe_key(self):
        """Greptile: adapter absent at queue time must NOT consume the one ack
        slot — a later queued message (adapter now registered) still acks."""
        adapter = AsyncMock()
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {}  # not registered yet
        runner._startup_restore_in_progress = True

        async def _drive():
            src = make_restart_source(chat_id="c1")
            runner._queue_startup_restore_event(_make_event(src))
            await asyncio.sleep(0.02)
            runner.adapters = {Platform.TELEGRAM: adapter}  # adapter comes up
            runner._queue_startup_restore_event(_make_event(src))
            await asyncio.sleep(0.05)

        asyncio.run(_drive())
        assert adapter.send.await_count == 1  # the ack still fired

    def test_no_ack_after_restore_gate_released(self):
        """Greptile: the fire-and-forget ack must not land AFTER the gate
        released (it would arrive after the real response as noise)."""
        adapter = AsyncMock()
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._startup_restore_in_progress = True

        async def _drive():
            src = make_restart_source(chat_id="c1")
            runner._queue_startup_restore_event(_make_event(src))
            # Gate releases BEFORE the fire-and-forget task runs.
            runner._startup_restore_in_progress = False
            await asyncio.sleep(0.05)

        asyncio.run(_drive())
        assert adapter.send.await_count == 0  # stale ack suppressed

    def test_ack_scheduling_error_never_breaks_queuing(self):
        """Greptile: an unexpected exception inside the ack helper must not
        propagate out of _queue_startup_restore_event (queuing is load-bearing)."""
        runner, _ = make_restart_runner()
        _bind(runner)
        runner._startup_restore_in_progress = True
        # Poison adapters so .get raises TypeError (not a dict).
        runner.adapters = object()

        async def _drive():
            runner._queue_startup_restore_event(_make_event(make_restart_source(chat_id="c1")))

        asyncio.run(_drive())  # must not raise
        assert len(runner._startup_restore_queue) == 1

    def test_no_loop_unburns_dedupe_key(self):
        """Greptile #396: a RuntimeError from create_task (no running loop) must
        un-burn the dedupe key so the chat can still ack later in the cycle."""
        adapter = AsyncMock()
        runner, _ = make_restart_runner()
        _bind(runner)
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._startup_restore_in_progress = True

        src = make_restart_source(chat_id="c1")
        # No running loop here (sync context) -> create_task raises RuntimeError.
        runner._queue_startup_restore_event(_make_event(src))
        acked = getattr(runner, "_startup_restore_acked_chats", set())
        assert len(acked) == 0  # key un-burned

        async def _drive():
            runner._queue_startup_restore_event(_make_event(src))
            await asyncio.sleep(0.05)

        asyncio.run(_drive())
        assert adapter.send.await_count == 1  # ack still fired once a loop exists
