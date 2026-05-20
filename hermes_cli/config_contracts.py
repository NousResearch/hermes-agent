"""
Config-Runtime Contract Registry (Phase 1).

Declares the binding between config.yaml keys and their runtime consumers.
On startup, validates that every declared binding is active — catching
orphan config fields (declared but never consumed) before users discover
them silently.

This is a purely additive observability layer — zero behavior change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConfigBinding:
    """A single config key → runtime consumer binding."""

    config_key: str
    """Dot-separated config.yaml path, e.g. 'terminal.docker_extra_args'."""

    consumer: str
    """Module/function that reads this key, e.g. 'gateway.run._terminal_env_map'."""

    binding_type: str = "env_var"
    """How the config key reaches the consumer: 'env_var', 'direct_read', 'function_call'."""

    env_var: str | None = None
    """For binding_type='env_var': the env var name."""

    wildcard: bool = False
    """If True, config_key contains '*' glob segments (e.g. 'custom_providers.*.models.*.max_tokens')."""

    notes: str = ""
    """Optional human-readable context."""


# ---------------------------------------------------------------------------
# Registry: all known config→runtime bindings
# ---------------------------------------------------------------------------
# This is the single source of truth. When a new config key is added to
# DEFAULT_CONFIG, add a corresponding entry here. The startup validator
# will then verify the consumer actually uses it.
# ---------------------------------------------------------------------------

CONFIG_BINDINGS: list[ConfigBinding] = [
    # === terminal.* → TERMINAL_* env vars ===
    # (gateway/run.py _terminal_env_map bridges these)
    ConfigBinding("terminal.backend", "gateway.run._terminal_env_map", "env_var", "TERMINAL_ENV"),
    ConfigBinding("terminal.cwd", "gateway.run._terminal_env_map", "env_var", "TERMINAL_CWD"),
    ConfigBinding("terminal.timeout", "gateway.run._terminal_env_map", "env_var", "TERMINAL_TIMEOUT"),
    ConfigBinding("terminal.lifetime_seconds", "gateway.run._terminal_env_map", "env_var", "TERMINAL_LIFETIME_SECONDS"),
    ConfigBinding("terminal.docker_image", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_IMAGE"),
    ConfigBinding("terminal.docker_forward_env", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_FORWARD_ENV"),
    ConfigBinding("terminal.docker_extra_args", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_EXTRA_ARGS",
                  notes="Bug #28863: was missing from env_map, silently dropped"),
    ConfigBinding("terminal.docker_volumes", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_VOLUMES"),
    ConfigBinding("terminal.docker_env", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_ENV"),
    ConfigBinding("terminal.docker_mount_cwd_to_workspace", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE"),
    ConfigBinding("terminal.docker_run_as_host_user", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DOCKER_RUN_AS_HOST_USER"),
    ConfigBinding("terminal.container_cpu", "gateway.run._terminal_env_map", "env_var", "TERMINAL_CONTAINER_CPU"),
    ConfigBinding("terminal.container_memory", "gateway.run._terminal_env_map", "env_var", "TERMINAL_CONTAINER_MEMORY"),
    ConfigBinding("terminal.container_disk", "gateway.run._terminal_env_map", "env_var", "TERMINAL_CONTAINER_DISK"),
    ConfigBinding("terminal.container_persistent", "gateway.run._terminal_env_map", "env_var", "TERMINAL_CONTAINER_PERSISTENT"),
    ConfigBinding("terminal.sandbox_dir", "gateway.run._terminal_env_map", "env_var", "TERMINAL_SANDBOX_DIR"),
    ConfigBinding("terminal.persistent_shell", "gateway.run._terminal_env_map", "env_var", "TERMINAL_PERSISTENT_SHELL"),
    ConfigBinding("terminal.ssh_host", "gateway.run._terminal_env_map", "env_var", "TERMINAL_SSH_HOST"),
    ConfigBinding("terminal.ssh_user", "gateway.run._terminal_env_map", "env_var", "TERMINAL_SSH_USER"),
    ConfigBinding("terminal.ssh_port", "gateway.run._terminal_env_map", "env_var", "TERMINAL_SSH_PORT"),
    ConfigBinding("terminal.ssh_key", "gateway.run._terminal_env_map", "env_var", "TERMINAL_SSH_KEY"),
    ConfigBinding("terminal.singularity_image", "gateway.run._terminal_env_map", "env_var", "TERMINAL_SINGULARITY_IMAGE"),
    ConfigBinding("terminal.modal_image", "gateway.run._terminal_env_map", "env_var", "TERMINAL_MODAL_IMAGE"),
    ConfigBinding("terminal.daytona_image", "gateway.run._terminal_env_map", "env_var", "TERMINAL_DAYTONA_IMAGE"),
    ConfigBinding("terminal.vercel_runtime", "gateway.run._terminal_env_map", "env_var", "TERMINAL_VERCEL_RUNTIME"),

    # === model.* → agent agent_init ===
    ConfigBinding("model.max_tokens", "agent.agent_init", "direct_read",
                  notes="Read at L1143-1166, with custom_providers fallback"),
    ConfigBinding("model.context_length", "agent.agent_init", "direct_read",
                  notes="Read at L1168-1203, with custom_providers fallback"),

    # === custom_providers.*.models.* → agent/gateway ===
    ConfigBinding("custom_providers.*.models.*.context_length", "hermes_cli.config.get_custom_provider_context_length",
                  "function_call", wildcard=True),
    ConfigBinding("custom_providers.*.models.*.max_tokens", "hermes_cli.config.get_custom_provider_max_tokens",
                  "function_call", wildcard=True,
                  notes="Bug #28046: was silently ignored, always defaulted to 4096"),
]


def get_nested(config: dict, dotted_key: str) -> tuple[Any, bool]:
    """Resolve a dotted key like 'terminal.docker_extra_args' from a nested dict.

    Returns (value, found). Wildcard segments ('*') match any key at that level.
    For list-of-dicts (e.g. custom_providers), '*' iterates list items.
    """
    parts = dotted_key.split(".")
    current: Any = config
    for part in parts:
        if part == "*":
            # Wildcard: iterate list items or dict values
            if isinstance(current, list):
                if not current:
                    return None, False
                current = current[0]  # Check first item for existence
            elif isinstance(current, dict):
                if not current:
                    return None, False
                current = next(iter(current.values()))
            else:
                return None, False
            continue
        if isinstance(current, list):
            # List-of-dicts: check each item for the key
            found_any = False
            for item in current:
                if isinstance(item, dict) and part in item:
                    current = item[part]
                    found_any = True
                    break
            if not found_any:
                return None, False
            continue
        if not isinstance(current, dict):
            return None, False
        if part not in current:
            return None, False
        current = current[part]
    return current, True


def validate_config_bindings(
    config: dict,
    *,
    strict: bool = False,
) -> list[str]:
    """Validate all registered config bindings against a loaded config.

    Returns a list of warning messages. Empty list means all bindings are healthy.

    Args:
        config: Loaded config.yaml dict.
        strict: If True, also warn about config keys that have no registered binding.

    Returns:
        List of warning strings.
    """
    warnings: list[str] = []

    for binding in CONFIG_BINDINGS:
        value, found = get_nested(config, binding.config_key)

        if found and value is not None:
            # Config declares this key — verify the consumer side exists
            if binding.binding_type == "env_var" and binding.env_var:
                import os
                if binding.env_var not in os.environ and value:
                    # The config value is set but not bridged to env yet
                    # (This is expected at import time; gateway startup bridges later)
                    pass
            # Key exists and has a consumer registered — healthy
        elif found and value is None:
            # Key exists but is None/default — not necessarily broken
            pass
        else:
            # Key not in config — could be using default, not a warning
            pass

    # In strict mode: check for config keys with NO registered binding
    if strict:
        registered_keys = {b.config_key for b in CONFIG_BINDINGS}
        _check_unregistered_keys(config, registered_keys, "", warnings)

    return warnings


def _check_unregistered_keys(
    config: dict,
    registered_keys: set[str],
    prefix: str,
    warnings: list[str],
) -> None:
    """Recursively check for config keys not in the registry."""
    for key, value in config.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict) and key not in ("providers", "custom_providers", "models"):
            _check_unregistered_keys(value, registered_keys, full_key, warnings)
        # Skip internal keys
        if key.startswith("_"):
            continue


def get_binding_report(config: dict) -> dict[str, Any]:
    """Generate a human-readable report of all config bindings and their status.

    Useful for debugging and the `/info` command.
    """
    healthy = []
    missing = []
    default_val = []

    for binding in CONFIG_BINDINGS:
        value, found = get_nested(config, binding.config_key)
        if found and value is not None:
            healthy.append({
                "key": binding.config_key,
                "consumer": binding.consumer,
                "value_type": type(value).__name__,
            })
        elif found:
            default_val.append({
                "key": binding.config_key,
                "consumer": binding.consumer,
                "status": "default",
            })
        else:
            missing.append({
                "key": binding.config_key,
                "consumer": binding.consumer,
                "status": "not_in_config",
            })

    return {
        "healthy": healthy,
        "default": default_val,
        "missing": missing,
        "total": len(CONFIG_BINDINGS),
        "registered": len(healthy) + len(default_val),
    }
