"""Regression tests for agent.gateway_* knob bridging into gateway readers.

These tests stay non-runtime: they inspect the static startup bridge in
``gateway.run`` and exercise only temp-config/env reader paths.  They do not
start the gateway, agents, cron, providers, or platform adapters.
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from typing import Any

import pytest

import gateway.run as gateway_run
from gateway.restart import parse_restart_drain_timeout
from hermes_cli import config as hermes_config


AGENT_GATEWAY_KNOBS = (
    # agent config key, gateway bridge env var, representative gateway reader
    ("gateway_timeout", "HERMES_AGENT_TIMEOUT", "_float_env"),
    ("gateway_timeout_warning", "HERMES_AGENT_TIMEOUT_WARNING", "_float_env"),
    ("gateway_wall_timeout", "HERMES_AGENT_WALL_TIMEOUT", "_float_env"),
    (
        "gateway_wall_timeout_groups_only",
        "HERMES_AGENT_WALL_TIMEOUT_GROUPS_ONLY",
        "os.getenv",
    ),
    ("gateway_notify_interval", "HERMES_AGENT_NOTIFY_INTERVAL", "_float_env"),
    (
        "gateway_auto_continue_freshness",
        "HERMES_AUTO_CONTINUE_FRESHNESS",
        "_auto_continue_freshness_window",
    ),
    (
        "restart_drain_timeout",
        "HERMES_RESTART_DRAIN_TIMEOUT",
        "parse_restart_drain_timeout",
    ),
)


def _agent_cfg_key_from_subscript(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "_agent_cfg":
        return None
    index = node.slice
    if isinstance(index, ast.Constant) and isinstance(index.value, str):
        return index.value
    return None


def _find_agent_cfg_key(node: ast.AST) -> str | None:
    direct = _agent_cfg_key_from_subscript(node)
    if direct:
        return direct
    for child in ast.walk(node):
        key = _agent_cfg_key_from_subscript(child)
        if key:
            return key
    return None


def _gateway_run_agent_bridge_map() -> dict[str, str]:
    """Return ``agent`` config key -> env var from gateway.run's source bridge."""

    tree = ast.parse(inspect.getsource(gateway_run))
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        if not (
            isinstance(target.value, ast.Attribute)
            and target.value.attr == "environ"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "os"
        ):
            continue
        env_index = target.slice
        if not (isinstance(env_index, ast.Constant) and isinstance(env_index.value, str)):
            continue
        cfg_key = _find_agent_cfg_key(node.value)
        if cfg_key:
            mapping[cfg_key] = env_index.value
    return mapping


def _write_temp_config(home: Path, values: dict[str, Any]) -> None:
    body = "agent:\n" + "".join(f"  {key}: {value}\n" for key, value in values.items())
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _apply_gateway_run_agent_bridge_slice(agent_cfg: dict[str, Any]) -> None:
    """Bounded source-slice equivalent of gateway.run's startup env bridge."""

    bridge = _gateway_run_agent_bridge_map()
    for cfg_key, env_name in bridge.items():
        if cfg_key not in agent_cfg:
            continue
        value = agent_cfg[cfg_key]
        if cfg_key == "gateway_wall_timeout_groups_only":
            os.environ[env_name] = str(value).lower()
        else:
            os.environ[env_name] = str(value)


@pytest.fixture(autouse=True)
def isolated_gateway_knob_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for _, env_name, _ in AGENT_GATEWAY_KNOBS:
        monkeypatch.delenv(env_name, raising=False)


def test_agent_gateway_knobs_have_env_bridge_entries() -> None:
    bridge = _gateway_run_agent_bridge_map()

    for cfg_key, env_name, _ in AGENT_GATEWAY_KNOBS:
        assert bridge.get(cfg_key) == env_name


def test_agent_gateway_knobs_have_runtime_reader_paths() -> None:
    gateway_run_source = inspect.getsource(gateway_run)

    for _, env_name, reader in AGENT_GATEWAY_KNOBS:
        if reader in {"_float_env", "_auto_continue_freshness_window", "os.getenv"}:
            assert env_name in gateway_run_source

    assert "HERMES_AGENT_TIMEOUT" in gateway_run_source
    assert "HERMES_AGENT_TIMEOUT_WARNING" in gateway_run_source
    assert "HERMES_AGENT_WALL_TIMEOUT" in gateway_run_source
    assert "HERMES_AGENT_WALL_TIMEOUT_GROUPS_ONLY" in gateway_run_source
    assert "HERMES_AGENT_NOTIFY_INTERVAL" in gateway_run_source
    assert "HERMES_AUTO_CONTINUE_FRESHNESS" in inspect.getsource(
        gateway_run._auto_continue_freshness_window
    )

    from hermes_cli.gateway import _get_restart_drain_timeout

    restart_reader_source = inspect.getsource(_get_restart_drain_timeout)
    assert "HERMES_RESTART_DRAIN_TIMEOUT" in restart_reader_source
    assert "restart_drain_timeout" in restart_reader_source
    assert "parse_restart_drain_timeout" in restart_reader_source


def test_agent_gateway_knob_overrides_are_visible_to_gateway_runtime_readers_with_temp_config(
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
        "gateway_wall_timeout": 13,
        "gateway_wall_timeout_groups_only": False,
        "gateway_notify_interval": 5,
        "gateway_auto_continue_freshness": 17,
        "restart_drain_timeout": 19,
    }
    _write_temp_config(tmp_path, values)

    raw = hermes_config.read_raw_config()
    assert raw["agent"] == values

    _apply_gateway_run_agent_bridge_slice(raw["agent"])

    assert gateway_run._float_env("HERMES_AGENT_TIMEOUT", 1800) == 11.0
    assert gateway_run._float_env("HERMES_AGENT_TIMEOUT_WARNING", 900) == 7.0
    assert gateway_run._float_env("HERMES_AGENT_WALL_TIMEOUT", 900) == 13.0
    assert os.getenv("HERMES_AGENT_WALL_TIMEOUT_GROUPS_ONLY") == "false"
    assert gateway_run._float_env("HERMES_AGENT_NOTIFY_INTERVAL", 180) == 5.0
    assert gateway_run._auto_continue_freshness_window() == 17.0
    assert parse_restart_drain_timeout(os.getenv("HERMES_RESTART_DRAIN_TIMEOUT")) == 19.0
