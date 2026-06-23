"""Enable the Japan OSINT stack: plugins, toolsets, and e-Gov MCP."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

OSINT_PLUGINS = ("shinka-osint", "worldmonitor-osint")
OSINT_TOOLSETS = frozenset({"shinka_osint", "worldmonitor_osint", "web", "search"})
EGOV_MCP_NAME = "egov-law"


def _egov_mcp_transport() -> dict[str, Any]:
    """Prefer py -3 on Windows; uvx when available."""
    if sys.platform == "win32":
        return {"command": "py", "args": ["-3", "-m", "egov_law_mcp.server"]}
    return {"command": "uvx", "args": ["egov-law-mcp"]}


def _ensure_egov_package() -> dict[str, Any]:
    """Best-effort pip install for egov-law-mcp (Windows / py -3 path)."""
    if sys.platform != "win32":
        return {"skipped": True, "reason": "uvx transport on non-Windows"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "egov-law-mcp>=0.1.0,<1"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return {
            "installed": proc.returncode == 0,
            "returncode": proc.returncode,
            "stderr": (proc.stderr or "")[:500],
        }
    except Exception as exc:  # pragma: no cover
        return {"installed": False, "error": str(exc)}


def enable_osint_stack(
    *,
    platforms: tuple[str, ...] = ("cli", "messaging"),
    install_egov: bool = True,
    install_worldmonitor_mcp: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enable OSINT plugins, toolsets, and register e-Gov Law MCP."""
    from hermes_cli.config import load_config, save_config
    from hermes_cli.mcp_config import _get_mcp_servers, _save_mcp_server
    from hermes_cli.plugins_cmd import _get_enabled_set, _resolve_plugin_key, _save_enabled_set
    from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools

    result: dict[str, Any] = {
        "success": True,
        "dry_run": dry_run,
        "plugins": {},
        "toolsets": {},
        "egov_mcp": {},
        "worldmonitor_mcp": {},
        "next_steps": [],
    }

    enabled = _get_enabled_set()
    for name in OSINT_PLUGINS:
        key = _resolve_plugin_key(name)
        if key is None:
            result["plugins"][name] = "not_found"
            result["success"] = False
            continue
        if dry_run:
            result["plugins"][name] = "would_enable" if key not in enabled else "already_enabled"
        else:
            enabled.add(key)
            result["plugins"][name] = "enabled" if key not in _get_enabled_set() else "already_enabled"
    if not dry_run:
        _save_enabled_set(enabled)

    config = load_config()
    for platform in platforms:
        current = _get_platform_tools(config, platform)
        merged = set(current) | set(OSINT_TOOLSETS)
        if dry_run:
            result["toolsets"][platform] = sorted(merged)
        else:
            _save_platform_tools(config, platform, merged)
            result["toolsets"][platform] = sorted(merged)
        config = load_config()

    servers = _get_mcp_servers()
    if install_egov:
        if EGOV_MCP_NAME in servers:
            result["egov_mcp"] = {"status": "already_configured"}
        elif dry_run:
            result["egov_mcp"] = {"status": "would_install", "transport": _egov_mcp_transport()}
        else:
            pip_result = _ensure_egov_package()
            result["egov_mcp"]["pip"] = pip_result
            saved = _save_mcp_server(EGOV_MCP_NAME, _egov_mcp_transport())
            result["egov_mcp"]["status"] = "installed" if saved else "save_failed"

    if install_worldmonitor_mcp:
        from .auth_setup import _ensure_mcp_oauth

        result["worldmonitor_mcp"] = _ensure_mcp_oauth(dry_run=dry_run)

    result["next_steps"] = [
        "SitDeck (no WM Pro): `hermes sitdeck-osint setup --email <gmail>` + crawl tools.",
        "Free tier (no Pro): `hermes worldmonitor-osint free-crawl` or fusion with `--wm-tier free`.",
        "Local dev: `hermes worldmonitor-osint dev setup` (clone + npm install + npm run dev).",
        "Paid/sidecar: `hermes worldmonitor-osint setup-auth --mode sidecar` or `--mode key`.",
        "OAuth MCP (optional, Pro only): `hermes mcp login worldmonitor` — skipped by default.",
        "Run `hermes shinka-osint setup --root <ShinkaEvolve-OSINT path>` if not configured.",
        "Fusion: `hermes worldmonitor-osint fusion 日本の安全保障 --wm-tier auto --source-mode real --save`.",
        "Agent tool: `worldmonitor_fusion_report` (WM Free/sidecar + Shinka MILSPEC).",
    ]
    return result
