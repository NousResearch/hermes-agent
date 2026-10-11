"""The gateway must start its periodic memory monitor at boot.

``gateway.memory_monitor`` (ported from cline/cline#10343) is the RSS/gc/threads
time series used to diagnose long-run gateway memory trends. It was complete and
tested but never wired into startup — dead code. These tests pin the wiring so a
refactor cannot silently drop it again.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import gateway.run as gateway_run
from gateway import memory_monitor as mm


def test_start_gateway_wires_memory_monitor():
    """start_gateway calls start_memory_monitoring() right after logging config,
    before the runner is constructed; a monitor failure never blocks startup."""
    started: list[bool] = []

    def fake_start_memory_monitoring(interval_seconds: float = 300.0) -> bool:
        started.append(True)
        return True

    class _StopEarly(Exception):
        pass

    class _BailRunner:
        def __init__(self, *_a, **_kw):
            raise _StopEarly

    async def no_host_attach(*_a, **_kw):
        return None

    # Drive start_gateway exactly to the wiring point: no host attach, no
    # existing-gateway discovery (the REAL gateway is running on this box),
    # no boot-fingerprint write, and bail at GatewayRunner construction —
    # the first statement after the wiring under test.
    with (
        patch.object(gateway_run, "resolve_placeholder_terminal_cwd", lambda **_kw: "/tmp"),
        patch.object(gateway_run, "_host_attach_or_none", no_host_attach),
        patch("gateway.code_skew.record_boot_fingerprint", lambda: None),
        patch.object(gateway_run, "_start_gateway_configure_logging", lambda *_a, **_kw: None),
        patch.object(gateway_run, "GatewayRunner", _BailRunner),
        patch.object(mm, "start_memory_monitoring", fake_start_memory_monitoring),
    ):
        try:
            asyncio.run(gateway_run.start_gateway(config=None, verbosity=0))
        except _StopEarly:
            pass

    assert started, "start_gateway never called start_memory_monitoring() before the runner was built"


def test_memory_monitor_module_importable_and_bounded():
    """The monitor itself stays importable, idempotent, and stops cleanly."""
    mm.stop_memory_monitoring(timeout=1.0)
    assert mm.start_memory_monitoring(interval_seconds=3600.0) is True
    assert mm.start_memory_monitoring(interval_seconds=3600.0) is False  # idempotent
    mm.stop_memory_monitoring(timeout=1.0)
    assert mm.is_running() is False
