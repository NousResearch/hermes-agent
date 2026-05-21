import copy
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


@pytest.mark.parametrize(
    ("key", "mutated_value"),
    [
        ("env_passthrough", "TOKEN"),
        ("shell_init_files", "~/.bashrc"),
        ("docker_forward_env", "TOKEN"),
        ("docker_env", {"DEBUG": "1"}),
        ("docker_volumes", "/host:/container"),
        ("docker_extra_args", "--network=host"),
    ],
)
def test_default_terminal_config_mutable_values_are_deep_copied(key, mutated_value):
    first = normalize_terminal_config(None)
    second = normalize_terminal_config(None)

    if isinstance(first[key], dict):
        first[key].update(mutated_value)
    else:
        first[key].append(mutated_value)

    assert second[key] == DEFAULT_TERMINAL_CONFIG[key]
    assert DEFAULT_TERMINAL_CONFIG[key] == ([] if isinstance(DEFAULT_TERMINAL_CONFIG[key], list) else {})


@pytest.mark.parametrize(
    ("key", "raw_value", "mutate_normalized"),
    [
        ("env_passthrough", ["TOKEN"], lambda value: value.append("EXTRA")),
        ("shell_init_files", ["~/.bashrc"], lambda value: value.append("~/.profile")),
        ("docker_forward_env", ["TOKEN"], lambda value: value.append("EXTRA")),
        ("docker_env", {"DEBUG": "1"}, lambda value: value.update({"TRACE": "1"})),
        ("docker_volumes", ["/host:/container"], lambda value: value.append("/tmp:/tmp")),
        ("docker_extra_args", ["--network=host"], lambda value: value.append("--rm")),
    ],
)
def test_normalized_mutable_values_do_not_alias_raw_input(key, raw_value, mutate_normalized):
    raw = {key: raw_value}
    expected_raw_value = copy.deepcopy(raw_value)
    config = normalize_terminal_config(raw)

    mutate_normalized(config[key])

    assert raw[key] == expected_raw_value


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


@pytest.mark.parametrize("backend", ["ssh", "docker"])
@pytest.mark.parametrize("placeholder", sorted(CWD_PLACEHOLDERS))
def test_non_local_cli_placeholder_cwd_resolves_to_none(backend, placeholder):
    config = normalize_terminal_config({"backend": backend, "cwd": placeholder})

    assert resolve_cli_terminal_cwd(config, invocation_cwd="/tmp/invoked") is None


@pytest.mark.parametrize("backend", ["ssh", "docker"])
def test_non_local_cli_missing_cwd_resolves_to_none(backend):
    config = normalize_terminal_config({"backend": backend})
    config.pop("cwd")

    assert resolve_cli_terminal_cwd(config, invocation_cwd="/tmp/invoked") is None


@pytest.mark.parametrize("backend", ["ssh", "docker"])
def test_non_local_cli_explicit_cwd_is_preserved(backend):
    config = normalize_terminal_config({"backend": backend, "cwd": "/workspace"})

    assert resolve_cli_terminal_cwd(config, invocation_cwd="/tmp/invoked") == "/workspace"


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


def test_terminal_env_values_omits_sudo_password_by_default():
    env = terminal_env_values({"sudo_password": "secret"})

    assert "SUDO_PASSWORD" not in env


def test_terminal_env_values_includes_sudo_password_when_requested():
    env = terminal_env_values({"sudo_password": "secret"}, include_secrets=True)

    assert env["SUDO_PASSWORD"] == "secret"


def test_terminal_env_values_uses_backend_for_terminal_env_when_env_type_absent():
    env = terminal_env_values({"backend": "docker", "cwd": "/workspace"})

    assert env["TERMINAL_ENV"] == "docker"
    assert env["TERMINAL_CWD"] == "/workspace"


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
