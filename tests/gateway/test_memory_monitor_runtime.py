"""Regression tests for gateway runtime memory-monitor wiring.

Issue #49773: ``start_memory_monitoring()`` was defined in
``gateway/memory_monitor.py`` and its own unit tests passed, but the gateway
runtime (``gateway/run.py``) never imported or invoked it.  Result: no
``[MEMORY]`` heartbeat in ``gateway.log``, so external log-freshness watchdogs
false-triggered and killed healthy idle gateways.

These tests exercise the **real production path**: they write a real
``config.yaml``, call the real gateway-runtime helpers, and verify the real
background monitor thread actually starts/stops with the configured interval.
No inline recomputation of the expected behaviour.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from gateway import memory_monitor as mm


@pytest.fixture(autouse=True)
def _clean_monitor_state():
    """Every test starts and ends with the monitor stopped."""
    mm.stop_memory_monitoring(timeout=1.0)
    yield
    mm.stop_memory_monitoring(timeout=1.0)


# ---------------------------------------------------------------------------
# Helpers — these live in gateway.run and are the production wiring that was
# missing (issue #49773).  They read config.yaml and drive the real monitor.
# ---------------------------------------------------------------------------


def test_start_helper_starts_real_monitor_with_config_interval(monkeypatch, tmp_path):
    """_start_runtime_memory_monitoring() starts the background thread using
    the interval from config.yaml."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n    interval_seconds: 99\n",
        encoding="utf-8",
    )

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is True
    assert mm._interval_seconds == 99.0


def test_start_helper_defaults_interval_when_unset(monkeypatch, tmp_path):
    """When interval_seconds is omitted, the default 300s applies."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n",
        encoding="utf-8",
    )

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is True
    assert mm._interval_seconds == 300.0


def test_start_helper_skips_when_disabled(monkeypatch, tmp_path):
    """enabled: false must NOT start the monitor."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: false\n",
        encoding="utf-8",
    )

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is False


def test_start_helper_starts_with_defaults_when_no_config(monkeypatch, tmp_path):
    """No config.yaml at all → monitoring starts enabled with the 300s default
    (the heartbeat is the safe default; users opt out via config)."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    # Deliberately do NOT write a config.yaml.

    gr._start_runtime_memory_monitoring()

    assert mm.is_running() is True
    assert mm._interval_seconds == 300.0


def test_stop_helper_stops_real_monitor():
    """_stop_runtime_memory_monitoring() stops the background thread."""
    import gateway.run as gr

    mm.start_memory_monitoring(interval_seconds=3600.0)
    assert mm.is_running() is True

    gr._stop_runtime_memory_monitoring()

    assert mm.is_running() is False


def test_stop_helper_is_noop_when_not_started():
    """Calling stop without a prior start must not raise."""
    import gateway.run as gr

    # _clean_monitor_state fixture already stopped it; call again.
    gr._stop_runtime_memory_monitoring()
    assert mm.is_running() is False


# ---------------------------------------------------------------------------
# Integration: the helpers are actually wired into the start_gateway() lifecycle.
# We mock the heavyweight external dependencies (PID locks, MCP discovery, the
# platform adapters) so we can drive start_gateway() to completion and observe
# that start/stop_memory_monitoring fire through the real code path.
# ---------------------------------------------------------------------------


class _StubRunner:
    """Minimal stand-in for GatewayRunner — start succeeds, shutdown returns."""

    should_exit_cleanly = False
    should_exit_with_failure = False
    exit_code = None
    exit_reason = None
    adapters: dict = {}
    _signal_initiated_shutdown = False
    _restart_requested = False
    _restart_via_service = False

    async def start(self) -> bool:
        return True

    async def wait_for_shutdown(self) -> None:
        return None

    async def stop(self) -> None:
        return None


@pytest.mark.asyncio
async def test_start_gateway_lifecycle_invokes_memory_monitor(monkeypatch, tmp_path):
    """A successful start_gateway() run must start the memory monitor on boot
    and stop it on teardown."""
    import gateway.run as gr

    monkeypatch.setattr(gr, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "logging:\n  memory_monitor:\n    enabled: true\n    interval_seconds: 42\n",
        encoding="utf-8",
    )

    start_calls: list[float] = []
    stop_calls: list[bool] = []

    def _fake_start(interval_seconds: float = 300.0) -> bool:
        start_calls.append(interval_seconds)
        return True

    def _fake_stop(timeout: float = 2.0) -> None:
        stop_calls.append(True)

    # Patch the memory_monitor functions at their source so the lazy imports
    # inside the helpers resolve to our fakes.
    monkeypatch.setattr(mm, "start_memory_monitoring", _fake_start)
    monkeypatch.setattr(mm, "stop_memory_monitoring", _fake_stop)

    # Stub out the heavyweight external dependencies of start_gateway().
    # These are imported *inside* start_gateway() (function-local), so we
    # patch them at their source modules.
    import gateway.status as gstatus

    monkeypatch.setattr(gstatus, "get_running_pid", lambda: None)
    monkeypatch.setattr(gstatus, "acquire_gateway_runtime_lock", lambda: True)
    monkeypatch.setattr(gstatus, "write_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(gstatus, "remove_pid_file", lambda *a, **k: None)
    monkeypatch.setattr(gstatus, "release_gateway_runtime_lock", lambda *a, **k: None)

    monkeypatch.setattr(gr, "GatewayRunner", lambda config=None: _StubRunner())

    import tools.skills_sync as ssync

    monkeypatch.setattr(ssync, "sync_skills", lambda *a, **k: None)

    import hermes_logging as hlog

    monkeypatch.setattr(hlog, "setup_logging", lambda *a, **k: None)

    import tools.mcp_tool as mcp

    monkeypatch.setattr(mcp, "discover_mcp_tools", lambda *a, **k: None)

    # The cron provider must expose .start() and .stop().
    _cron_stop_holder: dict = {}

    class _FakeCronProvider:
        def start(self, stop_event, adapters=None, loop=None):
            _cron_stop_holder["event"] = stop_event
            stop_event.wait()  # block until teardown signals

        def stop(self):
            pass

    import cron.scheduler_provider as csp

    monkeypatch.setattr(csp, "resolve_cron_scheduler", lambda: _FakeCronProvider())

    result = await gr.start_gateway(verbosity=None)

    assert result is True
    assert start_calls == [42.0], f"expected start_memory_monitoring(42.0), got {start_calls}"
    assert stop_calls == [True], f"expected stop_memory_monitoring called once, got {stop_calls}"
