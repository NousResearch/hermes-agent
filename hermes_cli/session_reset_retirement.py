"""Point configs that still declare an idle/daily ``session_reset`` at the plugin that restores it.

Core stopped rotating gateway conversations on timers (1d5d059410) and reads nothing under
``session_reset``. A household that relied on it silently gets conversations that never end and
a bill that grows with them, so gateway startup and ``hermes doctor`` say so. The
``hermes-session-reset-policy`` catalog plugin reads the top-level block unchanged, which is why
this module reports and never rewrites or drops the key.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

PLUGIN_NAME = "hermes-session-reset-policy"
_TIMED_MODES = frozenset({"idle", "daily", "both"})


def retired_reset_policy(config: Any) -> Optional[Tuple[str, str]]:
    """``(config_path, mode)`` for a timed ``session_reset`` the config still declares, else None.

    Looks at the top-level block and the ``gateway:`` form the pre-removal loader also accepted.
    """
    if not isinstance(config, dict):
        return None
    gateway = config.get("gateway")
    for path, block in (
        ("session_reset", config.get("session_reset")),
        ("gateway.session_reset", gateway.get("session_reset") if isinstance(gateway, dict) else None),
    ):
        if isinstance(block, dict):
            mode = str(block.get("mode") or "").strip().lower()
            if mode in _TIMED_MODES:
                return path, mode
    return None


def retired_legacy_reset_policies(config: Any) -> list[Tuple[str, str]]:
    """Timed policies in the retired gateway.json schema (not the plugin's YAML schema)."""
    if not isinstance(config, dict):
        return []
    blocks = [("default_reset_policy", config.get("default_reset_policy"))]
    for key in ("reset_by_platform", "reset_by_type"):
        overrides = config.get(key)
        if isinstance(overrides, dict):
            blocks.extend((f"{key}.{name}", block) for name, block in overrides.items())
    return [
        (f"gateway.json:{path}", str(block["mode"]).strip().lower())
        for path, block in blocks
        if isinstance(block, dict) and str(block.get("mode") or "").strip().lower() in _TIMED_MODES
    ]


def reset_plugin_enabled() -> bool:
    """Whether the restoring plugin is installed and enabled in this process's plugin set."""
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    discover_plugins()
    return any(p["name"] == PLUGIN_NAME and p["enabled"] for p in get_plugin_manager().list_plugins())


def format_notice(path: str, mode: str) -> str:
    move = "" if path == "session_reset" else " (move the block to top-level session_reset first)"
    return (
        f"{path}.mode: {mode} is no longer applied: gateway conversations reset only on /new or "
        f"/reset. To keep idle/daily resets, run `hermes plugins install {PLUGIN_NAME}`{move}."
    )
