"""Inspect the assembled tool surface of a fresh agent, not a live session (#47404)."""

from __future__ import annotations

import io
import json
from collections import Counter
from contextlib import redirect_stdout


class UnknownPlatformError(ValueError):
    pass


def _names(definitions: list[dict]) -> list[str]:
    from agent.memory_manager_tools import normalize_tool_schema

    return [schema["name"] for item in definitions if (schema := normalize_tool_schema(item))]


def _provider_summary(agent, visible: set[str], toolsets: dict[str, str]) -> dict:
    manager = getattr(agent, "_memory_manager", None)
    memory_names = set(manager.get_all_tool_names()) if manager else set()
    context_names = set(getattr(agent, "_context_engine_tool_names", ()))
    mcp_names = {name for name, toolset in toolsets.items() if toolset.startswith("mcp-")}
    from hermes_cli.tools_config import _get_plugin_toolset_keys

    plugin_toolsets = _get_plugin_toolset_keys()
    plugin_names = {name for name, toolset in toolsets.items() if toolset in plugin_toolsets}
    result = {}
    for family, names in (("memory", memory_names), ("context_engine", context_names),
                          ("mcp", mcp_names), ("plugins", plugin_names)):
        result[family] = {"known_tools": sorted(names), "visible_tools": sorted(names & visible)}
    result["memory"]["initialized_providers"] = [p.name for p in manager.providers] if manager else []
    result["context_engine"]["inspection"] = "injected_names_only"
    result["mcp"]["inspection"] = "registered_tools_only; no MCP discovery started by this command"
    result["plugins"]["inspection"] = "registered_tools_only"
    return result


def _filtered_tools(agent, visible: set[str], toolsets: dict[str, str]) -> list[dict]:
    from model_tools import _select_tool_names

    selected = _select_tool_names(agent.enabled_toolsets, agent.disabled_toolsets, quiet_mode=True)
    # Selection can explain a config gate. It cannot prove whether a selected-but-absent
    # schema was deferred, failed a requirement, or was removed by a later session gate.
    return [{"tool": name, "toolset": toolsets[name],
             "reason": "not_selected_by_toolsets" if name not in selected else "not_in_final_surface"}
            for name in sorted(toolsets.keys() - visible)]


def compute_tool_diagnostics(platform: str = "cli") -> dict:
    from hermes_cli.config import load_config
    from hermes_cli.platforms import get_all_platforms
    from hermes_cli.prompt_size import _build_inspection_agent
    from tools.registry import registry

    if platform not in get_all_platforms():
        raise UnknownPlatformError(platform)
    agent = _build_inspection_agent(platform)
    try:
        names = _names(agent.tools or [])
        visible = set(names)
        cfg = load_config()
        toolsets = registry.get_tool_to_toolset_map()
        platform_key = f"platform_toolsets.{platform}"
        configured = (cfg.get("platform_toolsets") or {}).get(platform)
        return {
            "platform": platform,
            "scope": "fresh_inspection_agent",
            "enabled_toolsets": sorted(agent.enabled_toolsets or []),
            "disabled_toolsets": sorted(agent.disabled_toolsets or []),
            "config_sources": {
                "platform_toolsets": platform_key if configured is not None else "platform default",
                "disabled_toolsets": "agent.disabled_toolsets (plus runtime session gates)",
                "tool_search": "tools.tool_search",
            },
            "tools_visible": sorted(visible),
            "tool_search": {"bridge_visible": "tool_search" in visible},
            "provider_tools": _provider_summary(agent, visible, toolsets),
            "filtered": _filtered_tools(agent, visible, toolsets),
            "conflicts": [{"tool": name, "reason": "duplicate_final_schema", "count": count}
                          for name, count in sorted(Counter(names).items()) if count > 1],
            "limitations": [
                "Fresh-agent snapshot, not an existing gateway session or a provider-specific wire payload.",
                "Initialization hooks run normally; this command is not a read-only sandbox.",
                "MCP discovery is not started; disconnected server schemas are unknown, not absent.",
                "Selected but invisible tools may be deferred, unavailable, or session-gated.",
                "Conflicts cover final schemas only; earlier registry/provider deduplication is not observable.",
            ],
        }
    finally:
        agent.close()


def _terminal_text(value: str) -> str:
    return "".join(char if char.isprintable() else repr(char)[1:-1] for char in value)


def render_tool_diagnostics(report: dict) -> str:
    lines = [f"Tool diagnostics ({report['platform']}; fresh inspection agent)",
             "Enabled toolsets: " + ", ".join(report["enabled_toolsets"]),
             "Disabled toolsets: " + ", ".join(report["disabled_toolsets"]),
             "Configuration: " + report["config_sources"]["platform_toolsets"],
             f"Model-visible tools ({len(report['tools_visible'])}): " + ", ".join(report["tools_visible"]),
             f"Tool Search bridge visible: {report['tool_search']['bridge_visible']}"]
    for family, data in report["provider_tools"].items():
        lines.append(f"{family}: {len(data['visible_tools'])} visible / {len(data['known_tools'])} known")
    lines.append(f"Final-schema conflicts: {len(report['conflicts'])}")
    lines.append(f"Registered tools not visible: {len(report['filtered'])} (use --json for names/reasons)")
    lines.extend("Note: " + limitation for limitation in report["limitations"])
    return "\n".join(_terminal_text(line) for line in lines)


def tools_diagnose_command(args) -> int:
    try:
        # Normal initialization can print progress. It is not part of this report, and may
        # contain provider details, so do not echo the captured text on failure either.
        with redirect_stdout(io.StringIO()):
            report = compute_tool_diagnostics(getattr(args, "platform", "cli"))
    except UnknownPlatformError:
        report = {"error": "unknown_platform"}
        code = 2
    except Exception as exc:  # health: allow BLE001 -- CLI boundary redacts arbitrary provider exceptions; tracebacks may contain credentials.
        report = {"error": "inspection_failed", "error_type": type(exc).__name__}
        code = 1
    else:
        code = 0
    if getattr(args, "json", False):
        print(json.dumps(report, ensure_ascii=True, indent=2))
    elif code:
        print(f"Tool diagnostics failed: {report['error']}. Check platform/configuration and Hermes logs.")
    else:
        print(render_tool_diagnostics(report))
    return code
