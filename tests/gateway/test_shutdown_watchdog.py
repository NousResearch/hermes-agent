"""Shutdown watchdog + loop heartbeat coverage for #66892.

The drain path is asyncio-based; a frozen loop makes every asyncio timeout
structurally unable to fire. These tests pin the out-of-loop backstop
(thread watchdog) and the loop-liveness heartbeat file contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import gateway.shutdown_watchdog as shutdown_watchdog_module
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




async def _run_heartbeat_until_payload(tmp_path, timeout_s=10.0):
    """Run loop_heartbeat_forever as a task until a heartbeat payload exists.

    Returns (task, payload). Cancels the task and awaits it (suppressing
    CancelledError) before returning so the tick server is closed cleanly.
    """
    task = asyncio.ensure_future(
        loop_heartbeat_forever(interval_s=1.0, home=tmp_path)
    )
    heartbeat_path = get_loop_heartbeat_path(tmp_path)
    deadline = time.monotonic() + timeout_s
    payload = None
    while time.monotonic() < deadline:
        if heartbeat_path.is_file():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                if payload:
                    break
                payload = None
        await asyncio.sleep(0.05)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    if payload is None:
        pytest.fail(
            f"heartbeat payload did not appear at {heartbeat_path} within "
            f"{timeout_s}s"
        )
    return payload


@pytest.fixture()
def short_home():
    """Short HERMES_HOME for tests that bind a real AF_UNIX socket.

    pytest's tmp_path nests deep enough on CI runners / macOS that
    ``state/gateway.loop-tick.<pid>.sock`` exceeds the sockaddr_un limit and
    bind() raises ``OSError: AF_UNIX path too long`` — which the producer
    swallows into ``loop_tick_socket=False``, falsely failing the POSIX arm
    test. Same pattern as tests/hermes_cli/test_update_wedged_gateway.py.
    """
    path = Path(tempfile.mkdtemp(prefix="hsw-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_loop_tick_witness_skipped_intentionally_on_windows(
    short_home, caplog, monkeypatch
):
    tmp_path = short_home
    # Pretend the platform is Windows as seen from the module under test.
    # A plain monkeypatch of the global os.name would flip pathlib.Path
    # dispatch (Path.__new__ reads os.name at runtime) and crash pytest's
    # own tmp-dir machinery, so swap the module's `os` binding for a proxy
    # whose `.name` is "nt" and which delegates everything else to real os.
    class _WindowsOsProxy:
        name = "nt"

        def __getattr__(self, item):
            return getattr(os, item)

    monkeypatch.setattr(shutdown_watchdog_module, "os", _WindowsOsProxy())

    start_unix_server_calls = []

    def _forbid_start_unix_server(*args, **kwargs):
        start_unix_server_calls.append((args, kwargs))
        raise AssertionError("start_unix_server must not be called on non-POSIX")

    with patch.object(
        shutdown_watchdog_module.asyncio,
        "start_unix_server",
        side_effect=_forbid_start_unix_server,
    ), caplog.at_level(logging.DEBUG, logger="gateway.shutdown_watchdog"):
        payload = await _run_heartbeat_until_payload(tmp_path)

    # (a) the witness server was never attempted
    assert start_unix_server_calls == []
    # (b) no warning about an unavailable tick socket
    assert not [
        r
        for r in caplog.records
        if r.levelname == "WARNING"
        and "Loop tick socket unavailable" in r.getMessage()
    ]
    # (c) the fail-safe flag is still recorded
    assert payload["loop_tick_socket"] is False


@pytest.mark.asyncio
async def test_loop_tick_witness_arms_on_posix(short_home):
    payload = await _run_heartbeat_until_payload(short_home)
    assert payload["loop_tick_socket"] is True
