"""Tests for OpenViking memory-provider shutdown teardown.

The runtime-autostart waiter is a tracked ``daemon=True`` thread that blocks
on network health probes. If ``shutdown()`` doesn't join it (and the waiter
doesn't bail on the shutdown flag), it can be left alive at interpreter exit,
which crashes CPython with SIGABRT at ``Py_FinalizeEx``. These tests assert
the waiter short-circuits on shutdown and that ``shutdown()`` waits for the
runtime-start thread.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import plugins.memory.openviking as openviking_module
from plugins.memory.openviking import OpenVikingMemoryProvider


def test_wait_for_health_short_circuits_on_should_stop():
    """The health waiter returns False without probing when should_stop is set,
    so the daemon thread running it can be join()ed promptly at shutdown."""
    probes: list[str] = []

    def _reach(endpoint):
        probes.append(endpoint)
        return (False, "down")

    with patch.object(
        openviking_module, "_validate_openviking_reachability", _reach
    ):
        result = openviking_module._wait_for_openviking_health(
            "http://example.invalid",
            timeout_seconds=60.0,
            should_stop=lambda: True,
        )

    assert result is False
    assert probes == []  # bailed before the first network probe


def test_shutdown_waits_for_runtime_start_thread():
    """shutdown() must join the runtime-autostart waiter thread.

    The fake waiter does post-stop work (a short sleep) once it observes the
    shutdown flag. If shutdown() joins it, that work has completed by the time
    shutdown() returns; without the join, shutdown() returns early and the
    thread is still running (the SIGABRT-at-exit failure mode).
    """
    provider = OpenVikingMemoryProvider()
    started = threading.Event()
    finished = threading.Event()

    def _runtime():
        started.set()
        while not provider._shutting_down:
            time.sleep(0.01)
        time.sleep(0.2)  # work that must finish during shutdown's join
        finished.set()

    t = threading.Thread(target=_runtime, daemon=True, name="openviking-runtime-start")
    provider._runtime_start_thread = t
    t.start()
    assert started.wait(2.0)

    provider.shutdown()

    assert finished.is_set()
    assert not t.is_alive()
