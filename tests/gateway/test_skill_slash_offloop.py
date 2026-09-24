"""Skill-slash fallthrough must not hold the gateway event loop (#111091)."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.session import SessionSource


def _runner(command):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)

    async def _resolve(event, source, qk):
        return False, None, command, None

    async def _canonical(event, source, qk, canonical):
        return False, None

    async def _quick(event, source, cmd):
        return False, None, cmd

    runner._hm_resolve_command = _resolve
    runner._hm_dispatch_canonical_command = _canonical
    runner._hm_dispatch_quick_and_plugin_commands = _quick
    return runner


@pytest.mark.asyncio
async def test_unavailable_skill_scan_skips_known_commands_and_runs_off_loop():
    import gateway.run as gateway_run

    source = SessionSource(platform=Platform.DISCORD, chat_id="c1")
    event = SimpleNamespace(text="/x", get_command_args=lambda: "")
    scanned = []
    loop_was_free = []
    scan_started = threading.Event()
    loop_ticked = threading.Event()

    def _slow_scan(command):
        scanned.append(command)
        scan_started.set()
        # True only if the loop ran while this scan was in flight, i.e. the scan is off-loop.
        loop_was_free.append(loop_ticked.wait(timeout=1))
        return None

    with patch.object(gateway_run, "_check_unavailable_skill", _slow_scan):
        # A registered gateway command returns before any filesystem walk.
        handled, reply = await _runner("steer")._hm_dispatch_idle_commands(event, source, "qk")
        assert (handled, reply, scanned) == (False, None, [])

        # An unknown command still consults the hint, but the scan runs off the loop.
        task = asyncio.create_task(_runner("no-such-skill")._hm_dispatch_idle_commands(event, source, "qk"))
        await asyncio.to_thread(scan_started.wait, 1)
        loop_ticked.set()  # only reachable mid-scan when the loop is free
        handled, reply = await task
    assert scanned == ["no-such-skill"]
    assert loop_was_free == [True]
    assert handled and "Unknown command" in reply
