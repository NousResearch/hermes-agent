"""Enable unified OSINT agent stack (all sibling plugins + toolsets)."""

from __future__ import annotations

from typing import Any

from . import plugin_loader

PLUGIN_NAME = "osint-agent"
SIBLING_PLUGINS = (
    "osint-agent",
    "sitdeck-osint",
    "worldmonitor-osint",
    "scrapling-feeds",
    "shinka-osint",
)
TOOLSETS = frozenset(
    {
        "osint_agent",
        "sitdeck_osint",
        "worldmonitor_osint",
        "scrapling_feeds",
        "shinka_osint",
        "web",
        "search",
    }
)


def enable_osint_agent_stack(
    *,
    platforms: tuple[str, ...] = ("cli", "messaging"),
    disable_wm_mcp: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    from hermes_cli.config import load_config
    from hermes_cli.plugins_cmd import _get_enabled_set, _resolve_plugin_key, _save_enabled_set
    from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools

    result: dict[str, Any] = {
        "success": True,
        "dry_run": dry_run,
        "plugins": {},
        "toolsets": {},
        "worldmonitor_mcp": {},
        "next_steps": [
            "hermes osint-agent status",
            "hermes osint-agent brief --slot morning",
            "hermes osint-agent cron install --deliver telegram,discord",
        ],
    }

    enabled = _get_enabled_set()
    for name in SIBLING_PLUGINS:
        key = _resolve_plugin_key(name)
        if key is None:
            result["plugins"][name] = "not_found"
            if name == PLUGIN_NAME:
                result["success"] = False
            continue
        if dry_run:
            result["plugins"][name] = "would_enable" if key not in enabled else "already_enabled"
        else:
            enabled.add(key)
            result["plugins"][name] = "enabled"
    if not dry_run:
        _save_enabled_set(enabled)

    config = load_config()
    for platform in platforms:
        current = _get_platform_tools(config, platform)
        merged = set(current) | set(TOOLSETS)
        if dry_run:
            result["toolsets"][platform] = sorted(merged)
        else:
            _save_platform_tools(config, platform, merged)
            result["toolsets"][platform] = sorted(merged)
        config = load_config()

    if disable_wm_mcp:
        try:
            plugin_loader.load_plugin_modules("sitdeck-osint", ("credentials", "stack"))
            stack_mod = __import__(
                "hermes_sitdeck_osint.stack",
                fromlist=["disable_worldmonitor_mcp"],
            )
            result["worldmonitor_mcp"] = stack_mod.disable_worldmonitor_mcp(dry_run=dry_run)
        except Exception as exc:
            result["worldmonitor_mcp"] = {"skipped": True, "error": str(exc)[:120]}

    return result
