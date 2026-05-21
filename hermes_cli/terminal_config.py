"""Shared terminal configuration normalization helpers.

This module centralizes terminal runtime defaults and the small differences in
how CLI and gateway entry points resolve placeholder working directories.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CWD_PLACEHOLDERS = {".", "auto", "cwd"}

_DEFAULT_CONTAINER_IMAGE = "nikolaik/python-nodejs:python3.11-nodejs20"

DEFAULT_TERMINAL_CONFIG: dict[str, Any] = {
    "backend": "local",
    "env_type": "local",
    "modal_mode": "auto",
    "cwd": ".",
    "timeout": 180,
    "lifetime_seconds": 300,
    "env_passthrough": [],
    "shell_init_files": [],
    "auto_source_bashrc": True,
    "docker_image": _DEFAULT_CONTAINER_IMAGE,
    "docker_forward_env": [],
    "docker_env": {},
    "singularity_image": f"docker://{_DEFAULT_CONTAINER_IMAGE}",
    "modal_image": _DEFAULT_CONTAINER_IMAGE,
    "daytona_image": _DEFAULT_CONTAINER_IMAGE,
    "vercel_runtime": "node24",
    "container_cpu": 1,
    "container_memory": 5120,
    "container_disk": 51200,
    "container_persistent": True,
    "docker_volumes": [],
    "docker_mount_cwd_to_workspace": False,
    "docker_extra_args": [],
    "docker_run_as_host_user": False,
    "persistent_shell": True,
}

TERMINAL_ENV_MAPPINGS: dict[str, str] = {
    "env_type": "TERMINAL_ENV",
    "cwd": "TERMINAL_CWD",
    "timeout": "TERMINAL_TIMEOUT",
    "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
    "modal_mode": "TERMINAL_MODAL_MODE",
    "docker_image": "TERMINAL_DOCKER_IMAGE",
    "docker_forward_env": "TERMINAL_DOCKER_FORWARD_ENV",
    "docker_env": "TERMINAL_DOCKER_ENV",
    "singularity_image": "TERMINAL_SINGULARITY_IMAGE",
    "modal_image": "TERMINAL_MODAL_IMAGE",
    "daytona_image": "TERMINAL_DAYTONA_IMAGE",
    "vercel_runtime": "TERMINAL_VERCEL_RUNTIME",
    "container_cpu": "TERMINAL_CONTAINER_CPU",
    "container_memory": "TERMINAL_CONTAINER_MEMORY",
    "container_disk": "TERMINAL_CONTAINER_DISK",
    "container_persistent": "TERMINAL_CONTAINER_PERSISTENT",
    "docker_volumes": "TERMINAL_DOCKER_VOLUMES",
    "docker_mount_cwd_to_workspace": "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE",
    "docker_extra_args": "TERMINAL_DOCKER_EXTRA_ARGS",
    "docker_run_as_host_user": "TERMINAL_DOCKER_RUN_AS_HOST_USER",
    "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
    "sandbox_dir": "TERMINAL_SANDBOX_DIR",
    "ssh_host": "TERMINAL_SSH_HOST",
    "ssh_user": "TERMINAL_SSH_USER",
    "ssh_port": "TERMINAL_SSH_PORT",
    "ssh_key": "TERMINAL_SSH_KEY",
}

SENSITIVE_TERMINAL_ENV_MAPPINGS: dict[str, str] = {
    "sudo_password": "SUDO_PASSWORD",
}


def default_terminal_config() -> dict[str, Any]:
    """Return a fresh copy of the default terminal configuration."""

    return copy.deepcopy(DEFAULT_TERMINAL_CONFIG)


def _stripped_nonempty(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_terminal_config(raw: Any) -> dict[str, Any]:
    """Normalize user terminal config while accepting legacy ``env_type``.

    ``backend`` is the canonical key. Legacy configs may set only ``env_type``;
    when both keys are present and non-empty, ``backend`` wins. The returned
    config always contains matching ``backend`` and ``env_type`` strings.
    """

    config = default_terminal_config()
    raw_backend = None
    raw_env_type = None

    if isinstance(raw, Mapping):
        raw_config = copy.deepcopy(raw)
        config.update(raw_config)
        raw_backend = raw_config.get("backend")
        raw_env_type = raw_config.get("env_type")

    backend = _stripped_nonempty(raw_backend)
    env_type = _stripped_nonempty(raw_env_type)
    canonical = backend or env_type or _stripped_nonempty(config.get("backend")) or "local"

    config["backend"] = canonical
    config["env_type"] = canonical
    return config


def _is_placeholder_cwd(value: Any) -> bool:
    text = _stripped_nonempty(value)
    return text is None or text.lower() in CWD_PLACEHOLDERS


def _expand_user(value: Any, home: str | os.PathLike[str] | None = None) -> str:
    text = str(value)
    if home is not None and (text == "~" or text.startswith("~/")):
        home_text = str(home)
        return home_text if text == "~" else os.path.join(home_text, text[2:])
    return str(Path(text).expanduser())


def resolve_cli_terminal_cwd(
    config: Mapping[str, Any], invocation_cwd: str | os.PathLike[str] | None = None
) -> str | None:
    """Resolve CLI cwd placeholders against the process invocation cwd for local backends."""

    cwd = config.get("cwd")
    if _is_placeholder_cwd(cwd):
        backend = _stripped_nonempty(config.get("backend")) or _stripped_nonempty(config.get("env_type")) or "local"
        if backend != "local":
            return None
        return _expand_user(invocation_cwd or os.getcwd())
    return _expand_user(cwd)


def resolve_gateway_terminal_cwd(
    config: Mapping[str, Any],
    existing_env: Mapping[str, str] | None = None,
    messaging_cwd: str | os.PathLike[str] | None = None,
    home: str | os.PathLike[str] | None = None,
) -> str:
    """Resolve gateway cwd placeholders using env, message cwd, then home."""

    cwd = config.get("cwd")
    if not _is_placeholder_cwd(cwd):
        return _expand_user(cwd, home=home)

    existing_cwd = (existing_env or {}).get("TERMINAL_CWD")
    fallback = existing_cwd or messaging_cwd or home or Path.home()
    return _expand_user(fallback, home=home)


def terminal_env_values(config: Mapping[str, Any], *, include_secrets: bool = False) -> dict[str, str]:
    """Serialize terminal config values for process environment variables."""

    env: dict[str, str] = {}
    if "env_type" not in config and config.get("backend") is not None:
        env["TERMINAL_ENV"] = str(config["backend"])

    mappings = TERMINAL_ENV_MAPPINGS
    if include_secrets:
        mappings = {**TERMINAL_ENV_MAPPINGS, **SENSITIVE_TERMINAL_ENV_MAPPINGS}

    for config_key, env_key in mappings.items():
        if config_key not in config:
            continue
        value = config[config_key]
        if value is None:
            continue
        if isinstance(value, (list, dict)):
            env[env_key] = json.dumps(value)
        else:
            env[env_key] = str(value)
    return env


__all__ = [
    "CWD_PLACEHOLDERS",
    "DEFAULT_TERMINAL_CONFIG",
    "SENSITIVE_TERMINAL_ENV_MAPPINGS",
    "TERMINAL_ENV_MAPPINGS",
    "default_terminal_config",
    "normalize_terminal_config",
    "resolve_cli_terminal_cwd",
    "resolve_gateway_terminal_cwd",
    "terminal_env_values",
]
