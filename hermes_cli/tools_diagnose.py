"""
Tool diagnostics for `hermes tools diagnose`.

Provides a read-only snapshot of tool/toolset health:
- Which toolsets are enabled and available for a platform
- Per-tool: enabled, available, filtered, reason
- Lazy dependency status
- Configuration issues (empty toolsets, conflicts, shadowing)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

from hermes_cli.colors import Colors, color
from hermes_cli.tools_config import (
    PLATFORMS,
    _get_effective_configurable_toolsets,
    _toolset_allowed_for_platform,
    load_config,
)
from hermes_cli.cli_output import print_error as _print_error

# Import the tool registry. This triggers tool discovery as a side-effect.
from model_tools import registry as _registry
from model_tools import get_all_tool_names as _get_all_tool_names
from model_tools import check_toolset_requirements as _check_toolset_requirements
from tools.lazy_deps import LAZY_DEPS as _LAZY_DEPS
from tools.lazy_deps import feature_missing as _feature_missing
from tools.lazy_deps import is_available as _is_available
from tools.lazy_deps import active_features as _active_features


def tools_diagnose_command(args) -> None:
    """Diagnose tool and toolset health for a platform.

    Entry point wired from ``cmd_tools`` in ``main.py``.
    """
    platform = getattr(args, "platform", "cli")
    as_json = getattr(args, "json", False)

    if platform not in PLATFORMS:
        _print_error(f"Unknown platform '{platform}'. Valid: {', '.join(PLATFORMS)}")
        return

    # Load current config
    config = load_config()
    platform_toolsets = config.get("platform_toolsets") or {}
    enabled_toolsets: set = set(platform_toolsets.get(platform, []))

    # Platform-default toolsets when nothing is explicitly configured
    default_toolsets = set(PLATFORMS[platform].get("default_toolset") or [])
    if not enabled_toolsets:
        enabled_toolsets = default_toolsets

    # Build diagnostic snapshot
    report = _build_diagnose_report(platform, enabled_toolsets, config)

    if as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_diagnose_human(report, platform)


def _build_diagnose_report(platform: str, enabled_toolsets: set, config: dict) -> dict:
    """Build the full diagnostic report as a dict."""
    # ── 1. Toolset availability ────────────────────────────────────────────
    toolset_availability = _check_toolset_requirements()

    # ── 2. All registered tools ───────────────────────────────────────────
    all_tool_names = _get_all_tool_names()
    tools: List[dict] = []
    issues: List[dict] = []

    for tool_name in sorted(all_tool_names):
        entry = _registry.get_entry(tool_name)
        if entry is None:
            continue

        toolset = entry.toolset
        is_enabled = toolset in enabled_toolsets
        is_available = toolset_availability.get(toolset, True)

        # Evaluate check_fn if toolset is enabled but reported unavailable
        check_detail = None
        if not is_available and entry.check_fn:
            try:
                is_available = _registry._evaluate_toolset_check(toolset, entry.check_fn)
            except Exception as e:
                check_detail = str(e)
                is_available = False

        # Determine why tool is not visible to the model
        filtered_reason = None
        if not is_enabled:
            filtered_reason = "toolset disabled"
        elif not is_available:
            filtered_reason = check_detail or "toolset requirements not met"
        elif entry.requires_env:
            missing_env = [e for e in entry.requires_env if not _has_env_var(e)]
            if missing_env:
                filtered_reason = f"missing env: {', '.join(missing_env)}"

        tool_record = {
            "name": tool_name,
            "toolset": toolset,
            "enabled": is_enabled,
            "available": is_available,
            "visible": is_enabled and is_available and not filtered_reason,
            "filtered_reason": filtered_reason,
            "requires_env": entry.requires_env or [],
            "description": entry.description or "",
            "emoji": entry.emoji or "",
        }
        tools.append(tool_record)

        # Aggregate issues
        if is_enabled and not is_available:
            issues.append({
                "severity": "error",
                "component": toolset,
                "tool": tool_name,
                "message": f"toolset '{toolset}' requirements not met — '{tool_name}' unavailable",
                "detail": check_detail,
                "fix": f"Run: hermes tools post-setup {toolset}",
            })
        elif is_enabled and filtered_reason and not filtered_reason.startswith("toolset"):
            issues.append({
                "severity": "warning",
                "component": toolset,
                "tool": tool_name,
                "message": filtered_reason,
                "fix": None,
            })

    # ── 3. Lazy deps status ────────────────────────────────────────────────
    lazy_deps: Dict[str, dict] = {}
    for feature, specs in sorted(_LAZY_DEPS.items()):
        missing = _feature_missing(feature)
        lazy_deps[feature] = {
            "installed": not missing,
            "missing_specs": list(missing) if missing else [],
            "is_active": _is_available(feature),
            "active_in_session": feature in _active_features(),
        }

    # ── 4. Config-level issues ─────────────────────────────────────────────
    configurable = _get_effective_configurable_toolsets()
    configured_keys = {ts_key for ts_key, _, _ in configurable}

    for ts_key in enabled_toolsets:
        if ts_key not in configured_keys:
            mcp_servers = config.get("mcp_servers") or {}
            if ts_key in mcp_servers:
                continue
            issues.append({
                "severity": "warning",
                "component": ts_key,
                "tool": None,
                "message": f"toolset '{ts_key}' enabled in config but not in registered toolsets",
                "fix": "Check for typos or remove from platform_toolsets in config.yaml",
            })

    # ── 5. Tool conflicts / shadowing ─────────────────────────────────────
    name_to_tools: Dict[str, List[dict]] = {}
    for tool in tools:
        key = tool["name"].rstrip("0123456789_")
        name_to_tools.setdefault(key, []).append(tool)

    for key, group in name_to_tools.items():
        if len(group) > 1 and key:
            visible = [t for t in group if t["visible"]]
            if len(visible) > 1:
                issues.append({
                    "severity": "warning",
                    "component": group[0]["toolset"],
                    "tool": key,
                    "message": f"tool name conflict: {', '.join(t['name'] for t in visible)}",
                    "fix": "Multiple tools with the same base name are visible to the model",
                })

    # ── 6. Overall status ─────────────────────────────────────────────────
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    overall_status = "unhealthy" if errors else "degraded" if warnings else "healthy"

    # ── 7. Toolsets summary ────────────────────────────────────────────────
    toolsets_summary: Dict[str, dict] = {}
    for ts_key, label, desc in configurable:
        if not _toolset_allowed_for_platform(ts_key, platform):
            continue
        avail = toolset_availability.get(ts_key, False)
        tools_in_ts = [t for t in tools if t["toolset"] == ts_key]
        toolsets_summary[ts_key] = {
            "label": label,
            "enabled": ts_key in enabled_toolsets,
            "available": avail,
            "tool_count": len(tools_in_ts),
            "visible_count": len([t for t in tools_in_ts if t["visible"]]),
        }

    # ── 8. Hermes version ─────────────────────────────────────────────────
    hermes_version = "unknown"
    try:
        from hermes_constants import __version__ as _v

        hermes_version = _v
    except Exception:
        pass

    return {
        "version": hermes_version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform,
        "status": overall_status,
        "errors": len(errors),
        "warnings": len(warnings),
        "enabled_toolsets": sorted(enabled_toolsets),
        "disabled_toolsets": sorted(set(k for k, _, _ in configurable) - enabled_toolsets),
        "toolsets": toolsets_summary,
        "tools": tools,
        "lazy_deps": lazy_deps,
        "issues": issues,
    }


def _print_diagnose_human(report: dict, platform: str) -> None:
    """Print a human-readable diagnostic report."""
    status = report["status"]
    status_color = {
        "healthy": Colors.GREEN,
        "degraded": Colors.YELLOW,
        "unhealthy": Colors.RED,
    }.get(status, Colors.RESET)

    print()
    print(f"  Hermes tool diagnostics — {platform}")
    print(f"  {'─' * 50}")
    print(f"  Status:   {color(status.upper(), status_color)}")
    print(f"  Errors:   {report['errors']}   Warnings: {report['warnings']}")
    print(f"  Version:  {report['version']}")
    print(f"  Time:     {report['timestamp']}")
    print()

    # ── Toolsets ─────────────────────────────────────────────────────────
    toolsets = report.get("toolsets") or {}
    if toolsets:
        print(f"  {'Toolset':<22} {'Enabled':<10} {'Available':<10} {'Visible':<10}")
        print(f"  {'─' * 52}")
        for ts_key, info in sorted(toolsets.items()):
            enabled = color("✓", Colors.GREEN) if info["enabled"] else color("✗", Colors.RED)
            avail = color("✓", Colors.GREEN) if info["available"] else color("✗", Colors.RED)
            visible = f"{info['visible_count']}/{info['tool_count']}"
            print(f"  {ts_key:<22} {enabled:<10} {avail:<10} {visible:<10}")
        print()

    # ── Issues ──────────────────────────────────────────────────────────────
    issues = report.get("issues") or []
    if issues:
        print(f"  Issues ({len(issues)})")
        print(f"  {'─' * 52}")
        for issue in issues:
            sev = issue["severity"]
            sev_color = Colors.RED if sev == "error" else Colors.YELLOW
            sev_icon = "❌" if sev == "error" else "⚠️ "
            msg = issue["message"]
            if issue.get("tool"):
                msg = f"[{issue['tool']}] {msg}"
            if issue.get("fix"):
                msg += f"  → {issue['fix']}"
            print(f"  {sev_icon} {color(sev.upper(), sev_color):<8} {msg}")
        print()
    else:
        print(f"  {color('✓  No issues detected', Colors.GREEN)}")
        print()

    # ── Lazy Deps ───────────────────────────────────────────────────────────
    lazy_deps = report.get("lazy_deps") or {}
    active_or_missing = {
        k: v for k, v in lazy_deps.items()
        if v["is_active"] or not v["installed"]
    }
    if active_or_missing:
        print(f"  Lazy dependencies")
        print(f"  {'─' * 52}")
        for feature, info in sorted(active_or_missing.items()):
            if info["installed"]:
                status_str = color("installed", Colors.GREEN)
                if info["active_in_session"]:
                    status_str += f" {color('(active)', Colors.DIM)}"
            else:
                missing = ", ".join(info["missing_specs"][:2])
                if len(info["missing_specs"]) > 2:
                    missing += f" +{len(info['missing_specs']) - 2} more"
                status_str = color(f"missing: {missing}", Colors.RED)
            print(f"  {feature:<30} {status_str}")
        print()

    print(f"  Run with {color('--json', Colors.DIM)} for machine-readable output.")


def _has_env_var(name: str) -> bool:
    """Check if an environment variable is set (non-empty)."""
    return bool(str(os.environ.get(name) or "").strip())
