"""Regression tests for agent.gateway_* knob bridging into gateway readers.

These tests exercise the production gateway bootstrap helper against temp
config/env state. They avoid source-shape assertions so they fail only when a
runtime bridge contract regresses.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

import gateway.run as gateway_run
from gateway.restart import parse_restart_drain_timeout
from hermes_cli import config as hermes_config


AGENT_GATEWAY_KNOBS = (
    ("gateway_timeout", "HERMES_AGENT_TIMEOUT"),
    ("gateway_timeout_warning", "HERMES_AGENT_TIMEOUT_WARNING"),
    ("gateway_notify_interval", "HERMES_AGENT_NOTIFY_INTERVAL"),
    ("gateway_auto_continue_freshness", "HERMES_AUTO_CONTINUE_FRESHNESS"),
    ("restart_drain_timeout", "HERMES_RESTART_DRAIN_TIMEOUT"),
)


def _write_temp_config(home: Path, values: dict[str, Any]) -> None:
    body = "agent:\n" + "".join(f"  {key}: {value}\n" for key, value in values.items())
    (home / "config.yaml").write_text(body, encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated_gateway_knob_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for _, env_name in AGENT_GATEWAY_KNOBS:
        monkeypatch.delenv(env_name, raising=False)


def test_agent_gateway_knob_overrides_reach_runtime_readers_with_temp_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    hermes_config._RAW_CONFIG_CACHE.clear()
    hermes_config._LOAD_CONFIG_CACHE.clear()

    values = {
        "gateway_timeout": 11,
        "gateway_timeout_warning": 7,
        "gateway_notify_interval": 5,
        "gateway_auto_continue_freshness": 17,
        "restart_drain_timeout": 19,
    }
    _write_temp_config(tmp_path, values)

    raw = hermes_config.read_raw_config()
    assert raw["agent"] == values

    gateway_run._bridge_agent_config_to_env(raw["agent"])

    assert os.environ["HERMES_AGENT_TIMEOUT"] == "11"
    assert gateway_run._float_env("HERMES_AGENT_TIMEOUT", 1800) == 11.0
    assert os.environ["HERMES_AGENT_TIMEOUT_WARNING"] == "7"
    assert gateway_run._float_env("HERMES_AGENT_TIMEOUT_WARNING", 900) == 7.0
    assert os.environ["HERMES_AGENT_NOTIFY_INTERVAL"] == "5"
    assert gateway_run._float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180) == 5.0
    assert os.environ["HERMES_AUTO_CONTINUE_FRESHNESS"] == "17"
    assert gateway_run._auto_continue_freshness_window() == 17.0
    assert os.environ["HERMES_RESTART_DRAIN_TIMEOUT"] == "19"
    assert parse_restart_drain_timeout(os.getenv("HERMES_RESTART_DRAIN_TIMEOUT")) == 19.0


def test_agent_gateway_bridge_does_not_publish_unimplemented_wall_timeout_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_AGENT_WALL_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_AGENT_WALL_TIMEOUT_GROUPS_ONLY", raising=False)

    gateway_run._bridge_agent_config_to_env(
        {
            "gateway_wall_timeout": 13,
            "gateway_wall_timeout_groups_only": False,
        }
    )

    assert "HERMES_AGENT_WALL_TIMEOUT" not in os.environ
    assert "HERMES_AGENT_WALL_TIMEOUT_GROUPS_ONLY" not in os.environ
