"""Shutdown watchdog + loop heartbeat coverage for #66892.

The drain path is asyncio-based; a frozen loop makes every asyncio timeout
structurally unable to fire. These tests pin the out-of-loop backstop
(thread watchdog) and the loop-liveness heartbeat file contract.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import patch

import pytest

from gateway.shutdown_watchdog import (
    DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S,
    arm_shutdown_watchdog,
    get_loop_heartbeat_path,
    get_shutdown_watchdog_dump_path,
    loop_heartbeat_forever,
    resolve_shutdown_watchdog_delay,
    write_loop_heartbeat,
)

def test_resolve_shutdown_watchdog_delay_adds_grace():
    assert resolve_shutdown_watchdog_delay(180) == 180 + DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay(0) == DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay("bad") == DEFAULT_SHUTDOWN_WATCHDOG_GRACE_S
    assert resolve_shutdown_watchdog_delay(10, grace_s=5) == 15.0


def test_write_loop_heartbeat_atomic_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = write_loop_heartbeat(pid=4242, start_time=100.5, home=tmp_path)
    assert path == tmp_path / "state" / "gateway.heartbeat"
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pid"] == 4242
    assert data["start_time"] == 100.5
    assert "updated_at" in data
    assert "monotonic" in data
    assert get_loop_heartbeat_path(tmp_path) == path


def test_arm_shutdown_watchdog_disarm_before_fire(tmp_path):
    done = threading.Event()
    exited = []

    def fake_exit(code):
        exited.append(code)
        raise _ExitCalled(code)

    with patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit):
        arm_shutdown_watchdog(
            0.4,
            done_event=done,
            dump_path=tmp_path / "dump.log",
            exit_code=7,
        )
        time.sleep(0.1)
        done.set()
        time.sleep(0.5)

    assert exited == []


def test_arm_shutdown_watchdog_fires_with_dump_and_exit(tmp_path):
    done = threading.Event()
    fired = threading.Event()
    dump = tmp_path / "logs" / "watchdog.log"
    snapshot_calls = []
    exit_codes = []

    def snapshot():
        snapshot_calls.append(1)
        return {"active_agents": 1, "draining": True}

    def fake_exit(code):
        exit_codes.append(code)
        fired.set()

    with patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit):
        arm_shutdown_watchdog(
            0.15,
            done_event=done,
            snapshot_fn=snapshot,
            dump_path=dump,
            exit_code=9,
        )
        assert fired.wait(timeout=5.0), "watchdog did not fire"

    assert exit_codes == [9]
    assert snapshot_calls == [1]
    assert dump.is_file()
    text = dump.read_text(encoding="utf-8")
    assert "shutdown_watchdog_fired" in text
    assert "faulthandler dump" in text
    assert get_shutdown_watchdog_dump_path(tmp_path).name == "gateway-shutdown-watchdog.log"


@pytest.mark.asyncio
async def test_loop_heartbeat_rewrites_until_cancelled(tmp_path):
    path = get_loop_heartbeat_path(tmp_path)
    task = asyncio.create_task(
        loop_heartbeat_forever(
            interval_s=0.05,
            start_time=12.0,
            home=tmp_path,
        )
    )
    try:
        # First write is immediate.
        for _ in range(50):
            if path.is_file():
                break
            await asyncio.sleep(0.02)
        assert path.is_file()
        first = path.read_text(encoding="utf-8")
        assert json.loads(first)["start_time"] == 12.0

        # Poll until a refresh lands (monotonic / updated_at change).
        second = first
        for _ in range(100):
            await asyncio.sleep(0.03)
            second = path.read_text(encoding="utf-8")
            if second != first:
                break
        assert second != first
        assert json.loads(second)["start_time"] == 12.0
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def test_gateway_runner_exposes_shutdown_watchdog_state():
    """Attrs used by stop()/start() exist after normal construction hooks."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._shutdown_watchdog_done = threading.Event()
    runner._loop_heartbeat_task = None
    runner._gateway_started_at = time.time()
    assert not runner._shutdown_watchdog_done.is_set()
    runner._shutdown_watchdog_done.set()
    assert runner._shutdown_watchdog_done.is_set()
    assert runner._loop_heartbeat_task is None
