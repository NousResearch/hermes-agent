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


def test_shutdown_during_pending_runtime_start_does_not_launch_waiter(monkeypatch):
    """A waiter reserved before shutdown must not start after shutdown returns."""
    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1934"
    status_entered = threading.Event()
    release_status = threading.Event()
    waiter_calls = []

    monkeypatch.setattr(
        openviking_module,
        "_start_local_openviking_server",
        lambda endpoint: (True, "started"),
    )
    monkeypatch.setattr(
        provider,
        "_start_runtime_openviking_waiter",
        lambda **kwargs: waiter_calls.append(kwargs),
    )

    def status_callback(_message):
        status_entered.set()
        assert release_status.wait(2.0)

    starter = threading.Thread(
        target=provider._handle_runtime_openviking_unreachable,
        kwargs={"status_callback": status_callback},
        name="openviking-pending-start",
    )
    starter.start()
    assert status_entered.wait(2.0)

    provider.shutdown()
    release_status.set()
    starter.join(timeout=2.0)

    assert not starter.is_alive()
    assert provider._runtime_start_pending is False
    assert waiter_calls == []


def test_shutdown_during_final_health_probe_does_not_publish_client(monkeypatch):
    """A successful final probe must not reactivate a provider being torn down."""
    provider = OpenVikingMemoryProvider()
    provider._endpoint = "http://127.0.0.1:1934"
    provider._api_key = ""
    provider._account = ""
    provider._user = ""
    provider._agent = "hermes"
    health_entered = threading.Event()
    release_health = threading.Event()

    monkeypatch.setattr(
        openviking_module,
        "_wait_for_openviking_health",
        lambda endpoint, **kwargs: True,
    )

    class FakeVikingClient:
        def __init__(self, endpoint, api_key="", account="", user="", agent=""):
            self.endpoint = endpoint

        def health(self):
            health_entered.set()
            assert release_health.wait(2.0)
            return True

    monkeypatch.setattr(openviking_module, "_VikingClient", FakeVikingClient)

    waiter = threading.Thread(
        target=provider._finish_runtime_openviking_start,
        name="openviking-final-health",
    )
    provider._runtime_start_thread = waiter
    waiter.start()
    assert health_entered.wait(2.0)

    shutdown = threading.Thread(target=provider.shutdown, name="openviking-shutdown")
    shutdown.start()
    for _ in range(100):
        if provider._shutting_down:
            break
        time.sleep(0.01)
    assert provider._shutting_down is True

    release_health.set()
    waiter.join(timeout=2.0)
    shutdown.join(timeout=2.0)

    assert not waiter.is_alive()
    assert not shutdown.is_alive()
    assert provider._client is None
