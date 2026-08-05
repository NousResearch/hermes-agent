"""Regression tests for routed-profile terminal configuration scope (#68559)."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from unittest import mock

import pytest


_TERMINAL_ENV_PREFIX = "TERMINAL_"


@pytest.fixture(autouse=True)
def _reset_terminal_scope_and_environment():
    from tools import terminal_tool

    original_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_TERMINAL_ENV_PREFIX)
    }
    for key in list(original_env):
        os.environ.pop(key, None)
    bridge_attempted = terminal_tool._terminal_config_bridge_attempted
    terminal_tool._terminal_config_bridge_attempted = False
    scope_setter = getattr(terminal_tool, "set_terminal_config_scope", None)
    token = scope_setter(None) if scope_setter is not None else None
    try:
        yield
    finally:
        if token is not None:
            terminal_tool.reset_terminal_config_scope(token)
        terminal_tool._terminal_config_bridge_attempted = bridge_attempted
        for key in list(os.environ):
            if key.startswith(_TERMINAL_ENV_PREFIX):
                os.environ.pop(key, None)
        os.environ.update(original_env)


def _write_profile(home: Path, terminal: dict | None = None) -> None:
    home.mkdir()
    config = {} if terminal is None else {"terminal": terminal}
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


def _gateway_and_routed_homes(tmp_path, gateway_terminal, routed_terminal=None):
    gateway_home = tmp_path / "gateway"
    routed_home = tmp_path / "routed"
    _write_profile(gateway_home, gateway_terminal)
    _write_profile(routed_home, routed_terminal)
    return gateway_home, routed_home


def _seed_gateway_terminal_config(monkeypatch, gateway_home):
    from tools import terminal_tool

    monkeypatch.setenv("HERMES_HOME", str(gateway_home))
    gateway_config = terminal_tool._get_env_config()
    assert terminal_tool._terminal_config_bridge_attempted is True
    return gateway_config


def test_routed_profile_uses_its_terminal_backend(tmp_path, monkeypatch):
    """A routed docker profile must not inherit the gateway's local backend."""
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        {"backend": "docker", "docker_image": "routed-image"},
    )
    gateway_config = _seed_gateway_terminal_config(monkeypatch, gateway_home)
    assert gateway_config["env_type"] == "local"

    with _profile_runtime_scope(routed_home):
        routed_config = terminal_tool._get_env_config()

    assert routed_config["env_type"] == "docker"
    assert routed_config["docker_image"] == "routed-image"


def test_gateway_terminal_config_is_restored_after_scope(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        {"backend": "docker"},
    )
    before = _seed_gateway_terminal_config(monkeypatch, gateway_home)

    with _profile_runtime_scope(routed_home):
        assert terminal_tool._get_env_config()["env_type"] == "docker"

    assert terminal_tool._get_env_config() == before


def test_routed_local_profile_does_not_inherit_docker_settings(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {
            "backend": "docker",
            "docker_image": "gateway-image",
            "docker_volumes": ["/host:/container"],
            "docker_network": False,
        },
        {"backend": "local"},
    )
    gateway_config = _seed_gateway_terminal_config(monkeypatch, gateway_home)
    assert gateway_config["env_type"] == "docker"
    assert gateway_config["docker_image"] == "gateway-image"

    with _profile_runtime_scope(routed_home):
        routed_config = terminal_tool._get_env_config()

    assert routed_config["env_type"] == "local"
    assert routed_config["docker_image"] != "gateway-image"
    assert routed_config["docker_volumes"] == []
    assert routed_config["docker_network"] is True


def test_single_profile_path_still_uses_process_environment(tmp_path, monkeypatch):
    from tools import terminal_tool

    home = tmp_path / "single"
    _write_profile(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "37")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    with mock.patch.object(
        terminal_tool,
        "_ensure_terminal_env_bridged",
        wraps=terminal_tool._ensure_terminal_env_bridged,
    ) as bridge:
        config = terminal_tool._get_env_config()

    bridge.assert_called_once_with()
    assert config["env_type"] == "local"
    assert config["timeout"] == 37
    assert config["cwd"] == str(tmp_path)


def test_terminal_scope_propagates_through_copy_context(tmp_path):
    from tools import terminal_tool

    scope = {
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_IMAGE": "thread-image",
    }
    token = terminal_tool.set_terminal_config_scope(scope)
    try:
        context = copy_context()
        with mock.patch.object(
            terminal_tool,
            "_ensure_terminal_env_bridged",
            side_effect=AssertionError("active terminal scope used the global bridge"),
        ):
            with ThreadPoolExecutor(max_workers=1) as executor:
                config = executor.submit(context.run, terminal_tool._get_env_config).result()
    finally:
        terminal_tool.reset_terminal_config_scope(token)

    assert config["env_type"] == "docker"
    assert config["docker_image"] == "thread-image"


def test_missing_terminal_section_fails_open(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from tools import terminal_tool

    gateway_home, routed_home = _gateway_and_routed_homes(
        tmp_path,
        {"backend": "local"},
        None,
    )
    gateway_config = _seed_gateway_terminal_config(monkeypatch, gateway_home)

    with _profile_runtime_scope(routed_home):
        routed_config = terminal_tool._get_env_config()

    assert routed_config == gateway_config
