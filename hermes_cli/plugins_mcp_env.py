"""Core-owned, profile-scoped environment settings for portable stdio MCP servers."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from typing import Any

from hermes_cli.plugins_state import _plugin_settings_entry

logger = logging.getLogger("hermes_cli.plugins")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*", re.ASCII)
_RESERVED_ENV = frozenset({"PLUGIN_ROOT", "PLUGIN_DATA"})


def _validate_env(override: object) -> Mapping[str, str]:
    """Validate the entire target before merging; never include values in errors."""
    if not isinstance(override, Mapping):
        raise ValueError("must map environment variable names to strings")
    seen = set()
    for name, value in override.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            raise ValueError("variable names must match [A-Za-z_][A-Za-z0-9_]*")
        normalized = name.upper() if os.name == "nt" else name
        if normalized in seen:
            raise ValueError("variable names must be unique (case-insensitive on Windows)")
        seen.add(normalized)
        if normalized in _RESERVED_ENV:
            raise ValueError("PLUGIN_ROOT and PLUGIN_DATA are reserved")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError("values must be strings without NUL bytes")
    return override


def portable_mcp_configs_with_env(
    plugin_id: str, servers: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Overlay only this owner's declared stdio env, under the caller's profile scope.

    A bad target is withheld, not launched with defaults or a partial override. A malformed
    outer map withholds this plugin's MCP registrations because no target can be resolved.
    Unknown/remote targets warn and cannot create or change a server. No input is mutated.
    """
    from hermes_cli.config import load_config_readonly

    entry = _plugin_settings_entry(load_config_readonly(), plugin_id)
    overrides = entry.get("mcp_env", {}) if entry is not None else {}
    path = f"plugins.entries.{plugin_id}.mcp_env"
    if not isinstance(overrides, Mapping):
        logger.warning("%s must be a mapping; this plugin's MCP servers skipped", path)
        return {}
    for target in overrides:
        if target not in servers:
            logger.warning("%s target %r ignored: not a declared server of this plugin", path, target)
    result = {}
    for name, config in servers.items():
        merged = dict(config)
        if name in overrides:
            if not config.get("command") or config.get("url"):
                logger.warning("%s.%s ignored: environment overrides require a stdio server", path, name)
            else:
                try:
                    env = _validate_env(overrides[name])
                except ValueError as exc:
                    logger.warning("%s.%s invalid: %s; MCP server skipped", path, name, exc)
                    continue
                defaults = config.get("env", {})
                if os.name == "nt":
                    replaced = {key.upper() for key in env}
                    defaults = {key: value for key, value in defaults.items() if key.upper() not in replaced}
                merged["env"] = {**defaults, **env}
        result[name] = merged
    return result
