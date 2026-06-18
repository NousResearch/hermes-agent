"""Enable sitdeck-osint and retire World Monitor Pro MCP."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from .credentials import SITDECK_EMAIL_ENV, SITDECK_PASSWORD_ENV, _normalize_email

PLUGIN_NAME = "sitdeck-osint"
TOOLSET_NAME = "sitdeck_osint"
WM_MCP_NAME = "worldmonitor"


def disable_worldmonitor_mcp(*, dry_run: bool = False) -> dict[str, Any]:
    """Set worldmonitor MCP server to disabled in config.yaml."""
    from hermes_cli.mcp_config import _get_mcp_servers, _save_mcp_server

    servers = _get_mcp_servers()
    if WM_MCP_NAME not in servers:
        return {"success": True, "status": "not_configured", "dry_run": dry_run}

    cfg = dict(servers[WM_MCP_NAME])
    if cfg.get("enabled") is False:
        return {"success": True, "status": "already_disabled", "dry_run": dry_run}

    if dry_run:
        return {"success": True, "status": "would_disable", "dry_run": True}

    cfg["enabled"] = False
    saved = _save_mcp_server(WM_MCP_NAME, cfg)
    return {
        "success": saved,
        "status": "disabled" if saved else "save_failed",
        "note": "worldmonitor-osint free-crawl tools remain available without MCP.",
    }


def _upsert_env_var(env_path: Path, key: str, value: str) -> bool:
    if not value:
        return False
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    replaced = False
    out: list[str] = []
    for line in lines:
        if pattern.match(line):
            out.append(f"{key}={value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        if out and out[-1].strip():
            out.append("")
        out.append(f"# SitDeck OSINT (sitdeck-osint plugin)")
        out.append(f"{key}={value}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


def setup_sitdeck_stack(
    *,
    email: str | None = None,
    write_env: bool = True,
    platforms: tuple[str, ...] = ("cli", "messaging"),
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enable sitdeck-osint plugin, toolsets, disable WM MCP, write email to .env."""
    from hermes_cli.plugins_cmd import _get_enabled_set, _resolve_plugin_key, _save_enabled_set
    from hermes_cli.tools_config import _get_platform_tools, _save_platform_tools
    from hermes_cli.config import load_config

    result: dict[str, Any] = {
        "success": True,
        "dry_run": dry_run,
        "plugin": {},
        "toolsets": {},
        "worldmonitor_mcp": {},
        "env": {},
        "next_steps": [],
    }

    key = _resolve_plugin_key(PLUGIN_NAME)
    if key is None:
        result["success"] = False
        result["plugin"] = {"status": "not_found"}
        return result

    enabled = _get_enabled_set()
    if dry_run:
        result["plugin"] = {
            "status": "would_enable" if key not in enabled else "already_enabled"
        }
    else:
        enabled.add(key)
        _save_enabled_set(enabled)
        result["plugin"] = {"status": "enabled"}

    config = load_config()
    for platform in platforms:
        current = _get_platform_tools(config, platform)
        merged = set(current) | {TOOLSET_NAME}
        if dry_run:
            result["toolsets"][platform] = sorted(merged)
        else:
            _save_platform_tools(config, platform, merged)
            result["toolsets"][platform] = sorted(merged)
        config = load_config()

    result["worldmonitor_mcp"] = disable_worldmonitor_mcp(dry_run=dry_run)

    if write_env and email:
        normalized = _normalize_email(email)
        env_path = get_hermes_home() / ".env"
        if dry_run:
            result["env"] = {"would_set": {SITDECK_EMAIL_ENV: normalized}}
        else:
            ok = _upsert_env_var(env_path, SITDECK_EMAIL_ENV, normalized)
            result["env"] = {SITDECK_EMAIL_ENV: "written" if ok else "failed"}

    result["next_steps"] = [
        f"Add {SITDECK_PASSWORD_ENV} to {get_hermes_home() / '.env'} (never commit).",
        'pip install -e ".[sitdeck-osint]" && playwright install chromium',
        "hermes sitdeck-osint status",
        "hermes sitdeck-osint crawl",
        "Agent tools: sitdeck_crawl, sitdeck_osint_digest",
        "World Monitor: use free-crawl / scrapling-feeds — Pro MCP disabled.",
    ]
    if not email:
        result["next_steps"].insert(
            0,
            f"Set {SITDECK_EMAIL_ENV} in .env or: hermes sitdeck-osint setup --email Mine0119",
        )
    return result
