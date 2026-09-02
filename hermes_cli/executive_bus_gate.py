"""Availability checks for the Executive Capability Bus toolset."""

from __future__ import annotations

import os
from typing import Any


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def _current_platform() -> str:
    """Return the platform scope used for per-platform tool config."""

    try:
        from agent.runtime_env import get_session_env

        platform = os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM")
    except Exception:
        platform = os.getenv("HERMES_PLATFORM")
    return str(platform or "cli").strip() or "cli"


def _executive_bus_configured(config: dict[str, Any], platform: str) -> bool:
    """Return True when config explicitly enables ``executive_bus``.

    The bus is an inter-profile coordination surface, so it should never be
    exposed merely because its Python modules import successfully. Users or
    profile owners must deliberately opt in via ``platform_toolsets`` (normal
    gateway/CLI path) or the legacy top-level ``toolsets`` list.
    """

    platform_toolsets = config.get("platform_toolsets")
    if isinstance(platform_toolsets, dict):
        for key in (platform, "default"):
            if "executive_bus" in _as_list(platform_toolsets.get(key)):
                return True
    if "executive_bus" in _as_list(config.get("toolsets")):
        return True
    tools_cfg = config.get("tools")
    if isinstance(tools_cfg, dict) and "executive_bus" in _as_list(tools_cfg.get("toolsets")):
        return True
    return False


def executive_bus_prerequisites_available() -> bool:
    """Return True when Bus storage/dispatch modules can be imported.

    This check is intentionally local and side-effect free: no network calls, no
    credential reads beyond the normal config loader, and no DB writes. Runtime
    credential usability remains the executor profile's responsibility.
    """

    try:
        from hermes_cli import kanban_db as _kb  # noqa: F401
        from hermes_cli.capability_registry import find_capability as _find  # noqa: F401
        from hermes_cli.profile_delegation import delegate_to_profile as _delegate  # noqa: F401
        from tools.process_registry import process_registry as _registry  # noqa: F401
    except Exception:
        return False
    return True


def executive_bus_enabled_for_current_context() -> bool:
    """Tool ``check_fn`` for the Executive Capability Bus.

    The toolset is available only when the current profile/platform explicitly
    enables ``executive_bus`` and the local Kanban/profile-delegation modules are
    importable. This keeps the two model-facing Bus tools out of ordinary
    sessions while preserving deliberate profile-level opt-in.
    """

    if not executive_bus_prerequisites_available():
        return False
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return False
    return _executive_bus_configured(config if isinstance(config, dict) else {}, _current_platform())
