"""Application tool-setting operations using the existing configuration backend.

Callers bind the target profile before invoking these operations. Runtime selection
remains in tools.platform_policy; interactive configuration remains in tools_config.
"""

from typing import List, Set

from hermes_cli.config import cfg_get, save_config
from tools import platform_policy as _tool_policy
from tools.toolset_scope import (
    _TOOLSET_PLATFORM_RESTRICTIONS, toolset_allowed_for_platform as _toolset_allowed_for_platform)

# Capabilities with provider configuration but no model-facing tool schemas.
CONFIG_ONLY_TOOLSETS = frozenset({"stt"})


def config_section(config: dict, key: str) -> dict:
    """Return a mutable dict section, normalizing missing and non-dict values."""
    section = config.setdefault(key, {})
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def toolset_configuration_platform(ts_key: str, default: str = "cli") -> str:
    """Platform a platform-less configuration UI should target: a toolset restricted away from ``default``
    must be configured on a supported platform, else the save helper drops it and the UI reports a no-op."""
    allowed = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    return default if not allowed or default in allowed else sorted(allowed)[0]


def save_platform_tools(config: dict, platform: str, enabled_toolset_keys: Set[str]):
    """Save the selected toolset keys for a platform to config."""
    config.setdefault("platform_toolsets", {})
    # Drop platform-scoped toolsets that don't apply here, so the "Configure all platforms" checklist (or a
    # hand-edited config.yaml) can't turn on `discord` for Telegram.
    enabled_toolset_keys = {ts for ts in enabled_toolset_keys if _toolset_allowed_for_platform(ts, platform)}
    plugin_keys = _tool_policy.get_plugin_toolset_keys()
    # Preserve only existing entries that are neither configurable nor platform defaults (i.e. MCP server
    # names): platform defaults (hermes-cli, ...) resolve to ALL tools and would silently override the user's
    # unchecked selections on the next read. Saving from the picker is consent to clear the "no_mcp" sentinel
    # (no checkbox for it; users who once set it by hand could otherwise never re-enable MCP via the UI).
    drop = _tool_policy.configurable_toolset_keys() | plugin_keys | _tool_policy._platform_default_keys() | {"no_mcp"}
    existing_toolsets = _tool_policy.coerce_platform_toolsets_value(
        cfg_get(config, "platform_toolsets", platform, default=[]), platform
    )
    preserved_entries = {str(e) for e in (existing_toolsets if isinstance(existing_toolsets, list) else [])
                         if str(e) not in drop}
    config["platform_toolsets"][platform] = sorted(enabled_toolset_keys | preserved_entries)
    # Record which plugin toolsets this platform "knows" (distinguishes "new plugin, default enabled" from
    # "user disabled it"). config_section normalizes a present-but-null key that setdefault alone would not replace.
    if plugin_keys:
        config_section(config, "known_plugin_toolsets")[platform] = sorted(plugin_keys)
    # Same record for builtin toolsets the checklist offered; without it an unchecked toolset is
    # indistinguishable from one shipped after the save and _tool_policy._enable_recently_shipped_toolsets re-enables it.
    config_section(config, "known_builtin_toolsets")[platform] = sorted(_tool_policy.configurable_toolset_keys())
    # Reconcile with agent.disabled_toolsets, which _tool_policy.get_platform_tools applies as a final override: a toolset
    # listed there stays OFF no matter what this writes (Blank Slate installs pre-populate ~27 entries, making
    # the desktop Toolsets UI unable to re-enable anything). Only toolsets just explicitly enabled FOR THIS
    # PLATFORM are cleared, so the list keeps working as a cross-platform suppression list for everything else.
    # See #49995.
    agent_cfg = config.get("agent")
    newly_enabled = enabled_toolset_keys - preserved_entries
    if isinstance(agent_cfg, dict) and agent_cfg.get("disabled_toolsets") and newly_enabled:
        from agent.skill_utils import parse_config_string_list
        parsed_disabled = parse_config_string_list(agent_cfg["disabled_toolsets"])
        remaining = [ts for ts in parsed_disabled if ts not in newly_enabled]
        if remaining != parsed_disabled:
            agent_cfg["disabled_toolsets"] = remaining
    save_config(config)


def apply_toolset_change(config: dict, platform: str, toolset_names: List[str], action: str):
    """Add or remove built-in toolsets for a platform."""
    from hermes_cli.config import has_xai_tool_credentials
    from tools.platform_policy import get_platform_tools

    enabled = get_platform_tools(config, platform, include_default_mcp_servers=False, xai_credentials_present=has_xai_tool_credentials)
    updated = enabled - set(toolset_names) if action == "disable" else enabled | set(toolset_names)
    save_platform_tools(config, platform, updated)


def apply_mcp_change(config: dict, targets: List[str], action: str) -> Set[str]:
    """Add or remove specific MCP tools from a server's exclude list."""
    failed_servers: Set[str] = set()
    mcp_servers = config.get("mcp_servers") or {}

    for target in targets:
        server_name, tool_name = target.split(":", 1)
        if server_name not in mcp_servers:
            failed_servers.add(server_name)
            continue
        tools_cfg = mcp_servers[server_name].setdefault("tools", {})
        exclude = list(tools_cfg.get("exclude") or [])
        if action != "disable":
            exclude = [t for t in exclude if t != tool_name]
        elif tool_name not in exclude:
            exclude.append(tool_name)
        tools_cfg["exclude"] = exclude

    return failed_servers


def parse_enabled_flag(value, default: bool = True) -> bool:
    """Parse bool-like config values used by tool/platform settings."""
    if isinstance(value, (bool, int)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on", "false", "0", "no", "off"}:
            return lowered in {"true", "1", "yes", "on"}
    return default
