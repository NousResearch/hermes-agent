"""MCP tool checklists and the non-interactive ``hermes tools enable|disable|list`` command."""

from __future__ import annotations

from typing import List, Set

from hermes_cli.cli_output import (
    print_error as _print_error, print_info as _print_info, print_success as _print_success,
    print_warning as _print_warning)
from hermes_cli.colors import Colors, color
from hermes_cli.toolset_scope import (
    _TOOLSET_PLATFORM_RESTRICTIONS, toolset_allowed_for_platform as _toolset_allowed_for_platform)


def _mcp_match_filter():
    """Runtime name-filter matcher (exact names or fnmatch globs), with a literal fallback.
    Must use the SAME semantics as tools/mcp_tool.py registration — a literal ``in`` check renders glob
    excludes (e.g. ``*team_member*`` from catalog default_excluded manifests) as if nothing were
    excluded."""
    try:
        from tools.mcp_tool_schema import matches_name_filter
        return matches_name_filter
    except ImportError:  # pragma: no cover — defensive fallback
        return lambda tool_name, patterns: tool_name in patterns


def _mcp_preselected(tool_names: List[str], include_set, exclude_set, match) -> Set[int]:
    """Indices of tools currently enabled: include mode, exclude mode, or all when unfiltered."""
    if include_set:
        return {i for i, tn in enumerate(tool_names) if match(tn, include_set)}
    if exclude_set:
        return {i for i, tn in enumerate(tool_names) if not match(tn, exclude_set)}
    return set(range(len(tool_names)))


def _apply_mcp_checklist(server_name: str, tools_cfg: dict, tool_names: List[str], chosen: Set[int],
                         include_set, exclude_set, match) -> None:
    """Write a checklist result back as ``tools.include`` / ``tools.exclude``."""
    exclude_mode = bool(exclude_set) and not include_set

    if len(chosen) == len(tool_names) and not exclude_mode:
        # All tools enabled — clear filters so tools the server adds later are auto-enabled.
        tools_cfg.pop("exclude", None)
        tools_cfg.pop("include", None)
    elif exclude_mode:
        # Exclude-mode server (catalog default_excluded / hand-written tools.exclude): stay in exclude
        # mode — do NOT demote the dynamic filter to a frozen include list. Unchecked tools become literal
        # excludes; re-checked literals are dropped; glob patterns are preserved (they intentionally keep
        # matching tools the vendor ships later).
        old_exclude = sorted(exclude_set or set())
        glob_entries = [p for p in old_exclude if "*" in p or "?" in p or "[" in p]
        literal_entries = {p for p in old_exclude if p not in glob_entries}
        unchecked = {tn for i, tn in enumerate(tool_names) if i not in chosen}
        checked = {tool_names[i] for i in chosen}
        new_literals = (literal_entries - checked) | {tn for tn in unchecked if not match(tn, set(old_exclude))}
        new_exclude = glob_entries + sorted(new_literals)
        glob_shadowed = sorted(tn for tn in checked if glob_entries and match(tn, set(glob_entries)))
        if glob_shadowed:
            _print_warning(
                f"  {server_name}: {len(glob_shadowed)} re-enabled "
                f"tool(s) still match glob exclude pattern(s) "
                f"{glob_entries} and stay excluded — edit "
                f"mcp_servers.{server_name}.tools.exclude in config.yaml "
                "to enable them.")
        if new_exclude:
            tools_cfg["exclude"] = new_exclude
        else:
            tools_cfg.pop("exclude", None)
        tools_cfg.pop("include", None)
    else:
        tools_cfg["include"] = [tool_names[i] for i in sorted(chosen)]
        tools_cfg.pop("exclude", None)  # include-mode now; drop any legacy exclude block


def _configure_mcp_tools_interactive(config: dict):
    """Probe each MCP server for its tools, show a per-server curses checklist, and write the result
    back as ``tools.exclude`` / ``tools.include`` entries in config.yaml."""
    from hermes_cli.curses_ui import curses_checklist
    from hermes_cli.tools_config import save_config

    mcp_servers = config.get("mcp_servers") or {}
    if not mcp_servers:
        _print_info("No MCP servers configured.")
        return

    enabled_names = [k for k, v in mcp_servers.items() if v.get("enabled", True) not in {False, "false", "0", "no", "off"}]
    if not enabled_names:
        _print_info("All MCP servers are disabled.")
        return

    print()
    print(color("  Discovering tools from MCP servers...", Colors.YELLOW))
    print(color(f"  Connecting to {len(enabled_names)} server(s): {', '.join(enabled_names)}", Colors.DIM))

    try:
        from tools.mcp_tool_discovery import probe_mcp_server_tools
        server_tools = probe_mcp_server_tools()
    except Exception as exc:
        _print_error(f"Failed to probe MCP servers: {exc}")
        return

    if not server_tools:
        _print_warning("Could not discover tools from any MCP server.")
        _print_info("Check that server commands/URLs are correct and dependencies are installed.")
        return

    for name in (n for n in enabled_names if n not in server_tools):
        _print_warning(f"  Could not connect to '{name}'")

    total_tools = sum(len(tools) for tools in server_tools.values())
    print(color(f"  Found {total_tools} tool(s) across {len(server_tools)} server(s)", Colors.GREEN))
    print()

    any_changes = False
    for server_name, tools in server_tools.items():
        if not tools:
            _print_info(f"  {server_name}: no tools found")
            continue

        tools_cfg = mcp_servers.get(server_name, {}).get("tools") or {}
        include_set = {str(p) for p in tools_cfg.get("include") or []} or None
        exclude_set = {str(p) for p in tools_cfg.get("exclude") or []} or None

        labels = []
        for tool_name, description in tools:
            desc_short = description[:70] + "..." if len(description) > 70 else description
            labels.append(f"{tool_name}  ({desc_short})" if desc_short else tool_name)
        match = _mcp_match_filter()
        tool_names = [t[0] for t in tools]
        pre_selected = _mcp_preselected(tool_names, include_set, exclude_set, match)

        chosen = curses_checklist(
            f"MCP Server: {server_name}  ({len(tools)} tools)", labels, pre_selected, cancel_returns=pre_selected)

        if chosen == pre_selected:
            _print_info(f"  {server_name}: no changes")
            continue

        tools_cfg = mcp_servers.setdefault(server_name, {}).setdefault("tools", {})
        _apply_mcp_checklist(server_name, tools_cfg, tool_names, chosen, include_set, exclude_set, match)

        _print_success(f"  {server_name}: {len(chosen)} enabled, {len(tools) - len(chosen)} disabled")
        any_changes = True

    if any_changes:
        save_config(config)
        print()
        print(color("  ✓ MCP tool configuration saved", Colors.GREEN))
    else:
        print(color("  No changes to MCP tools", Colors.DIM))


def _apply_toolset_change(config: dict, platform: str, toolset_names: List[str], action: str):
    """Add or remove built-in toolsets for a platform."""
    from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools

    enabled = _get_platform_tools(config, platform, include_default_mcp_servers=False)
    updated = enabled - set(toolset_names) if action == "disable" else enabled | set(toolset_names)
    _save_platform_tools(config, platform, updated)


def _apply_mcp_change(config: dict, targets: List[str], action: str) -> Set[str]:
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


def _print_tools_list(enabled_toolsets: set, mcp_servers: dict, platform: str = "cli"):
    """Print a summary of enabled/disabled toolsets and MCP tool filters."""
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_effective_configurable_toolsets

    effective_all = _get_effective_configurable_toolsets()
    effective = [(k, l, d) for (k, l, d) in effective_all if _toolset_allowed_for_platform(k, platform)]
    builtin_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}

    def _print_rows(entries):
        for ts_key, label in entries:
            status = color("✓ enabled", Colors.GREEN) if ts_key in enabled_toolsets else color("✗ disabled", Colors.RED)
            print(f"  {status}  {ts_key}  {color(label, Colors.DIM)}")

    print(f"Built-in toolsets ({platform}):")
    _print_rows((k, l) for k, l, _ in effective if k in builtin_keys)

    plugin_entries = [(k, l) for k, l, _ in effective if k not in builtin_keys]
    if plugin_entries:
        print()
        print(f"Plugin toolsets ({platform}):")
        _print_rows(plugin_entries)

    if mcp_servers:
        print()
        print("MCP servers:")
        for srv_name, srv_cfg in mcp_servers.items():
            tools_cfg = srv_cfg.get("tools") or {}
            exclude, include = tools_cfg.get("exclude") or [], tools_cfg.get("include") or []
            if include:
                _print_info(f"{srv_name}  [include only: {', '.join(include)}]")
            elif exclude:
                _print_info(f"{srv_name}  [excluded: {color(', '.join(exclude), Colors.YELLOW)}]")
            else:
                _print_info(f"{srv_name}  {color('all tools enabled', Colors.DIM)}")


def _known_tool_platforms() -> set[str]:
    """Return built-in plus discovered plugin platform names. Plugin platforms register at runtime, not
    in the static CLI display registry, and must be recognized so an active plugin platform can audit
    its authority."""
    from hermes_cli.tools_config import PLATFORMS

    known = set(PLATFORMS)
    try:
        from hermes_cli.plugins import discover_plugins
        from gateway.platform_registry import platform_registry
        discover_plugins()  # idempotent
        known.update(platform_registry.registered_names())
    except Exception:
        # Plugin discovery is optional: keep the built-in path when a plugin is malformed or deps are missing.
        pass
    return known


def tools_disable_enable_command(args):
    """Enable, disable, or list tools for a platform."""
    from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_platform_tools, _get_plugin_toolset_keys, load_config, save_config

    action = args.tools_action
    platform = getattr(args, "platform", "cli")
    config = load_config()

    valid_platforms = _known_tool_platforms()
    if platform not in valid_platforms:
        _print_error(f"Unknown platform '{platform}'. Valid: {', '.join(sorted(valid_platforms))}")
        return

    if action == "list":
        _print_tools_list(_get_platform_tools(config, platform, include_default_mcp_servers=False),
                          config.get("mcp_servers") or {}, platform)
        return

    targets: List[str] = args.names
    toolset_targets = [t for t in targets if ":" not in t]
    mcp_targets = [t for t in targets if ":" in t]

    valid_toolsets = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS} | _get_plugin_toolset_keys()
    unknown_toolsets = [t for t in toolset_targets if t not in valid_toolsets]
    for name in unknown_toolsets:
        _print_error(f"Unknown toolset '{name}'")
    # Reject platform-scoped toolsets on platforms that don't allow them.
    restricted_targets = [t for t in toolset_targets
                          if t in valid_toolsets and not _toolset_allowed_for_platform(t, platform)]
    for name in restricted_targets:
        allowed = sorted(_TOOLSET_PLATFORM_RESTRICTIONS.get(name) or set())
        _print_error(f"Toolset '{name}' is not available on platform '{platform}' (only: {', '.join(allowed)})")
    rejected = set(unknown_toolsets) | set(restricted_targets)
    toolset_targets = [t for t in toolset_targets if t not in rejected]
    if toolset_targets:
        _apply_toolset_change(config, platform, toolset_targets, action)

    failed_servers: Set[str] = set()
    if mcp_targets:
        failed_servers = _apply_mcp_change(config, mcp_targets, action)
        for srv in failed_servers:
            _print_error(f"MCP server '{srv}' not found in config")
    save_config(config)

    successful = [t for t in targets
                  if t not in rejected and (":" not in t or t.split(":")[0] not in failed_servers)]
    if successful:
        verb = "Disabled" if action == "disable" else "Enabled"
        _print_success(f"{verb}: {', '.join(successful)}")
