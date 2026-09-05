"""Strict, read-only install-level Bot Mode policy loading."""

from pathlib import Path

from hermes_cli.config import (
    InvalidUserConfigError,
    _deep_merge,
    _expand_env_vars,
    read_user_config_raw,
)
from hermes_cli.managed_scope import get_managed_dir


def _bot_mode_section(path: Path) -> dict:
    data = read_user_config_raw(path, strict=True)
    agent = data.get("agent", {})
    if not isinstance(agent, dict):
        raise InvalidUserConfigError("agent must be a mapping")
    policy = agent.get("bot_mode", {})
    if not isinstance(policy, dict):
        raise InvalidUserConfigError("agent.bot_mode must be a mapping")
    if "enabled" in policy and not isinstance(policy["enabled"], bool):
        raise InvalidUserConfigError("agent.bot_mode.enabled must be a boolean")
    return policy


def load_bot_mode_config(root: Path) -> dict:
    """Read the explicit install root, expand refs, then apply admin-pinned leaves.

    No home/env mutation, cache, migration, or last-known-good fallback: revoked
    authority must be visible on the next dispatch. Missing files remain legacy.
    """
    policy = _expand_env_vars(_bot_mode_section(root / "config.yaml"))
    managed = get_managed_dir()
    if managed is not None:
        policy = _deep_merge(
            policy, _expand_env_vars(_bot_mode_section(managed / "config.yaml")),
        )
    return policy
