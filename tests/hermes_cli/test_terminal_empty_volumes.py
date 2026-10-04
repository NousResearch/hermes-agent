"""Tests for empty-collection guard in apply_terminal_config_to_env."""

import os
from unittest.mock import patch

from hermes_cli import config as config_mod


def test_empty_docker_volumes_does_not_override_env():
    """An empty docker_volumes list in config.yaml must not clobber
    a non-empty TERMINAL_DOCKER_VOLUMES env var."""
    raw_config = {
        "terminal": {
            "backend": "docker",
            "docker_volumes": [],
        }
    }
    env = {
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_VOLUMES": '["/host/data:/data"]',
    }

    with patch.object(config_mod, "read_raw_config", return_value=raw_config):
        with patch.object(config_mod, "load_config_readonly", return_value=raw_config):
            result = config_mod.apply_terminal_config_to_env(env=dict(env))

    assert result["TERMINAL_DOCKER_VOLUMES"] == '["/host/data:/data"]'


def test_non_empty_does_override_env():
    """A non-empty docker_volumes list in config.yaml should override env."""
    raw_config = {
        "terminal": {
            "backend": "docker",
            "docker_volumes": ["/custom:/path"],
        }
    }
    env = {
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_VOLUMES": '["/host/data:/data"]',
    }

    with patch.object(config_mod, "read_raw_config", return_value=raw_config):
        with patch.object(config_mod, "load_config_readonly", return_value=raw_config):
            result = config_mod.apply_terminal_config_to_env(env=dict(env))

    assert result["TERMINAL_DOCKER_VOLUMES"] == '["/custom:/path"]'


def test_empty_env_not_overwritten_by_empty_config():
    """If env is also empty, the empty config value is fine (applied)."""
    raw_config = {
        "terminal": {
            "backend": "docker",
            "docker_volumes": [],
        }
    }
    env = {
        "TERMINAL_ENV": "docker",
        "TERMINAL_DOCKER_VOLUMES": "",
    }

    with patch.object(config_mod, "read_raw_config", return_value=raw_config):
        with patch.object(config_mod, "load_config_readonly", return_value=raw_config):
            result = config_mod.apply_terminal_config_to_env(env=dict(env))

    # Empty config value is applied when env is empty (no clobbering concern)
    assert result["TERMINAL_DOCKER_VOLUMES"] == "[]"


def test_no_env_var_gets_empty_config():
    """If env var is not set, empty config value is applied."""
    raw_config = {
        "terminal": {
            "backend": "docker",
            "docker_volumes": [],
        }
    }
    env = {
        "TERMINAL_ENV": "docker",
    }

    with patch.object(config_mod, "read_raw_config", return_value=raw_config):
        with patch.object(config_mod, "load_config_readonly", return_value=raw_config):
            result = config_mod.apply_terminal_config_to_env(env=dict(env))

    assert result["TERMINAL_DOCKER_VOLUMES"] == "[]"
