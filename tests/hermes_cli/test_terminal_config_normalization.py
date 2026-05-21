import json
from pathlib import Path

import pytest

from hermes_cli.terminal_config import (
    CWD_PLACEHOLDERS,
    DEFAULT_TERMINAL_CONFIG,
    normalize_terminal_config,
    resolve_cli_terminal_cwd,
    resolve_gateway_terminal_cwd,
    terminal_env_values,
)


@pytest.mark.parametrize("raw", [None, "local", [], 123, False])
def test_non_mapping_terminal_config_uses_defaults(raw):
    config = normalize_terminal_config(raw)

    assert config["backend"] == "local"
    assert config["env_type"] == "local"
    assert config["timeout"] == 180
    assert config["lifetime_seconds"] == 300


def test_partial_terminal_config_merges_with_defaults():
    config = normalize_terminal_config({"timeout": 45})

    assert config["timeout"] == 45
    assert config["cwd"] == "."
    assert config["lifetime_seconds"] == 300
    assert config["backend"] == "local"
    assert config["env_type"] == "local"


@pytest.mark.parametrize("backend", ["docker", "ssh", "modal"])
def test_backend_canonical_sets_env_type(backend):
    config = normalize_terminal_config({"backend": backend})

    assert config["backend"] == backend
    assert config["env_type"] == backend


def test_legacy_env_type_sets_backend_when_backend_missing():
    config = normalize_terminal_config({"env_type": "docker"})

    assert config["backend"] == "docker"
    assert config["env_type"] == "docker"


def test_backend_wins_over_env_type_when_both_are_present():
    config = normalize_terminal_config({"backend": "ssh", "env_type": "docker"})

    assert config["backend"] == "ssh"
    assert config["env_type"] == "ssh"


def test_default_terminal_config_is_deep_copied_for_docker_forward_env():
    first = normalize_terminal_config(None)
    second = normalize_terminal_config(None)

    first["docker_forward_env"].append("TOKEN")

    assert second["docker_forward_env"] == []
    assert DEFAULT_TERMINAL_CONFIG["docker_forward_env"] == []


@pytest.mark.parametrize("placeholder", sorted(CWD_PLACEHOLDERS))
def test_cli_placeholder_cwd_resolves_to_invocation_cwd(placeholder):
    config = normalize_terminal_config({"cwd": placeholder})

    assert resolve_cli_terminal_cwd(config, invocation_cwd="/tmp/invoked") == "/tmp/invoked"


def test_cli_missing_cwd_resolves_to_invocation_cwd():
    config = normalize_terminal_config({})
    config.pop("cwd")

    assert resolve_cli_terminal_cwd(config, invocation_cwd="/tmp/invoked") == "/tmp/invoked"


def test_explicit_cli_cwd_is_preserved_and_expanded():
    config = normalize_terminal_config({"cwd": "~/project"})

    assert resolve_cli_terminal_cwd(config) == str(Path("~/project").expanduser())


def test_gateway_placeholder_cwd_uses_messaging_cwd():
    config = normalize_terminal_config({"cwd": "auto"})

    assert (
        resolve_gateway_terminal_cwd(config, existing_env={}, messaging_cwd="/chat/workspace")
        == "/chat/workspace"
    )


def test_gateway_placeholder_cwd_uses_existing_terminal_cwd_first():
    config = normalize_terminal_config({"cwd": "."})

    assert (
        resolve_gateway_terminal_cwd(
            config,
            existing_env={"TERMINAL_CWD": "/existing/workspace"},
            messaging_cwd="/chat/workspace",
        )
        == "/existing/workspace"
    )


def test_gateway_explicit_cwd_expands_home():
    config = normalize_terminal_config({"cwd": "~/project"})

    assert (
        resolve_gateway_terminal_cwd(
            config,
            existing_env={"TERMINAL_CWD": "/existing/workspace"},
            messaging_cwd="/chat/workspace",
            home="/home/example",
        )
        == "/home/example/project"
    )


def test_terminal_env_values_serializes_config_to_environment_strings():
    config = normalize_terminal_config(
        {
            "backend": "docker",
            "cwd": "/workspace",
            "timeout": 45,
            "docker_forward_env": ["API_KEY", "TOKEN"],
            "docker_env": {"DEBUG": "1"},
        }
    )

    env = terminal_env_values(config)

    assert env["TERMINAL_ENV"] == "docker"
    assert env["TERMINAL_CWD"] == "/workspace"
    assert env["TERMINAL_TIMEOUT"] == "45"
    assert env["TERMINAL_DOCKER_FORWARD_ENV"] == json.dumps(["API_KEY", "TOKEN"])
    assert env["TERMINAL_DOCKER_ENV"] == json.dumps({"DEBUG": "1"})
