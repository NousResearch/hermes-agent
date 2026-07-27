"""Tests that the gateway runtime actually starts the [MEMORY] heartbeat.

Regression for #49773: ``gateway/memory_monitor.start_memory_monitoring()``
existed and was tested in isolation, but nothing in the gateway runtime called
it, so the heartbeat never ran in production and idle gateways went silent
(false-tripping external log-freshness watchdogs).

These tests exercise the wiring helper ``gateway.run._start_gateway_memory_monitor``
and the config default block that backs it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import gateway.run as run
from gateway.run import GatewayRunner
from hermes_cli.config import DEFAULT_CONFIG, cfg_get


def test_memory_monitor_default_config_present():
    """The defaults expose logging.memory_monitor so the heartbeat is on by default."""
    assert cfg_get(DEFAULT_CONFIG, "logging", "memory_monitor", "enabled") is True
    assert (
        cfg_get(DEFAULT_CONFIG, "logging", "memory_monitor", "interval_seconds") == 300
    )


def test_start_helper_starts_monitor_with_configured_interval():
    with patch.object(run, "_load_gateway_config", return_value={
        "logging": {"memory_monitor": {"enabled": True, "interval_seconds": 123}}
    }), patch("gateway.memory_monitor.start_memory_monitoring") as mock_start:
        run._start_gateway_memory_monitor()
    mock_start.assert_called_once_with(interval_seconds=123.0)


def test_start_helper_defaults_to_300_when_unconfigured():
    with patch.object(run, "_load_gateway_config", return_value={}), patch(
        "gateway.memory_monitor.start_memory_monitoring"
    ) as mock_start:
        run._start_gateway_memory_monitor()
    mock_start.assert_called_once_with(interval_seconds=300.0)


def test_start_helper_skips_when_disabled():
    with patch.object(run, "_load_gateway_config", return_value={
        "logging": {"memory_monitor": {"enabled": False}}
    }), patch("gateway.memory_monitor.start_memory_monitoring") as mock_start:
        run._start_gateway_memory_monitor()
    mock_start.assert_not_called()


def test_start_helper_never_raises_on_bad_config():
    """A broken config loader must not abort gateway startup."""
    with patch.object(run, "_load_gateway_config", side_effect=RuntimeError("boom")), patch(
        "gateway.memory_monitor.start_memory_monitoring"
    ) as mock_start:
        run._start_gateway_memory_monitor()  # should swallow and return
    mock_start.assert_not_called()


def test_start_helper_logs_warning_on_failure(caplog):
    """Failure must be visible at WARNING level, not silently debug-logged.

    The entire purpose of the heartbeat is observability. If it fails silently
    (debug level), the gateway goes dark — the same symptom as the bug (#49773).
    """
    import logging

    with patch.object(run, "_load_gateway_config", side_effect=RuntimeError("boom")), patch(
        "gateway.memory_monitor.start_memory_monitoring"
    ):
        with caplog.at_level(logging.WARNING):
            run._start_gateway_memory_monitor()
    assert any("Memory monitor start skipped" in r.message for r in caplog.records), (
        "Failure must be logged at WARNING level, not debug"
    )


def _make_partial_runner(tmp_path):
    """Partially-constructed GatewayRunner, enough for the start() prologue.

    Follows the established pattern (see test_teams_pipeline_runtime_wiring):
    ``__new__`` + only the attributes the exercised code path touches. The
    stubbed ``_abort_startup_if_shutdown_requested`` makes start() return at
    its first checkpoint — which sits *after* the memory-monitor wiring — so
    no adapters, servers, or background loops are ever started.
    """
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = SimpleNamespace(sessions_dir=str(tmp_path))
    runner._start_loop_liveness_guards = lambda loop: None

    async def _abort(*_args, **_kwargs) -> bool:
        return True

    runner._abort_startup_if_shutdown_requested = _abort
    return runner


def test_gateway_runner_start_starts_memory_monitor(monkeypatch, tmp_path):
    """Regression for #49773: starting the runner must start the heartbeat.

    This is the exact failure class of the bug: ``start_memory_monitoring()``
    existed and was tested in isolation, but nothing in the gateway runtime
    called it. This test drives ``GatewayRunner.start()`` itself (up to its
    first shutdown checkpoint) and asserts the monitor is actually started
    with the configured interval.
    """
    runner = _make_partial_runner(tmp_path)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"logging": {"memory_monitor": {"enabled": True, "interval_seconds": 60}}},
    )
    # Keep the test from touching the real ~/.hermes runtime-status file.
    monkeypatch.setattr(
        "gateway.status.write_runtime_status", lambda **_kw: None
    )

    with patch("gateway.memory_monitor.start_memory_monitoring") as mock_start:
        result = asyncio.run(GatewayRunner.start(runner))

    assert result is True
    mock_start.assert_called_once_with(interval_seconds=60.0)


def test_gateway_runner_start_respects_disabled_config(monkeypatch, tmp_path):
    """start() must not spin up the heartbeat when the operator disabled it."""
    runner = _make_partial_runner(tmp_path)
    monkeypatch.setattr(
        "gateway.run._load_gateway_config",
        lambda: {"logging": {"memory_monitor": {"enabled": False}}},
    )
    monkeypatch.setattr(
        "gateway.status.write_runtime_status", lambda **_kw: None
    )

    with patch("gateway.memory_monitor.start_memory_monitoring") as mock_start:
        result = asyncio.run(GatewayRunner.start(runner))

    assert result is True
    mock_start.assert_not_called()
