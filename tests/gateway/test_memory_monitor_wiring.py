"""Tests that the gateway runtime actually starts the [MEMORY] heartbeat.

Regression for #49773: ``gateway/memory_monitor.start_memory_monitoring()``
existed and was tested in isolation, but nothing in the gateway runtime called
it, so the heartbeat never ran in production and idle gateways went silent
(false-tripping external log-freshness watchdogs).

These tests exercise the wiring helper ``gateway.run._start_gateway_memory_monitor``
and the config default block that backs it.
"""

from __future__ import annotations

from unittest.mock import patch

import gateway.run as run
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
