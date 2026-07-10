"""Install notebooklm-mcp-cli and register the MCP server in Hermes config."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from hermes_constants import get_hermes_home

from . import bridge

MCP_SERVER_NAME = "notebooklm-mcp"
STATE_CONFIG = "mcp-stack.json"


def _state_path():
    path = get_hermes_home() / "notebooklm" / STATE_CONFIG
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_mcp_server_config() -> dict[str, Any]:
    """Build a Hermes ``mcp_servers`` entry for notebooklm-mcp."""
    resolved = bridge.resolve_mcp_command()
    if resolved is None:
        return {}
    executable, prefix = resolved
    return {
        "command": executable,
        "args": list(prefix),
        "enabled": True,
    }


def mcp_server_status() -> dict[str, Any]:
    try:
        from hermes_cli.mcp_config import _get_mcp_servers

        servers = _get_mcp_servers()
    except Exception as exc:
        return {"configured": False, "error": str(exc)}
    cfg = servers.get(MCP_SERVER_NAME) or {}
    if not cfg:
        return {"configured": False, "name": MCP_SERVER_NAME}
    return {
        "configured": True,
        "name": MCP_SERVER_NAME,
        "command": cfg.get("command"),
        "args": cfg.get("args"),
        "enabled": cfg.get("enabled", True),
        "tools_include": cfg.get("tools", {}).get("include"),
    }


def ensure_mcp_server(*, dry_run: bool = False) -> dict[str, Any]:
    existing = mcp_server_status()
    if existing.get("configured"):
        return {"status": "already_configured", **existing}
    transport = build_mcp_server_config()
    if not transport:
        return {
            "status": "unavailable",
            "error": "notebooklm-mcp binary not resolvable (need nlm/uvx on PATH).",
        }
    if dry_run:
        return {"status": "would_install", "transport": transport}
    from hermes_cli.mcp_config import _save_mcp_server

    saved = _save_mcp_server(MCP_SERVER_NAME, transport)
    return {
        "status": "installed" if saved else "save_failed",
        "transport": transport,
        "saved": saved,
    }


def install_cli_package(*, dry_run: bool = False) -> dict[str, Any]:
    """Install notebooklm-mcp-cli via ``uv tool install`` when possible."""
    if bridge.cli_available() and shutil.which("nlm"):
        return {"status": "already_available", "command": shutil.which("nlm")}
    spec = bridge._uvx_from_spec()
    if dry_run:
        return {"status": "would_install", "spec": spec}
    if not shutil.which("uv"):
        return {
            "status": "skipped",
            "reason": "uv not on PATH; bridge will use uvx at runtime.",
            "spec": spec,
        }
    cmd = ["uv", "tool", "install", "--force", spec]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "failed", "error": str(exc), "command": cmd}
    ok = result.returncode == 0
    return {
        "status": "installed" if ok else "failed",
        "returncode": result.returncode,
        "command": cmd,
        "stdout": (result.stdout or "").strip()[-2000:],
        "stderr": (result.stderr or "").strip()[-2000:],
        "nlm_path": shutil.which("nlm"),
        "mcp_path": shutil.which("notebooklm-mcp"),
    }


def setup_mcp_stack(
    *,
    install_cli: bool = True,
    register_mcp: bool = True,
    install_skill: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """End-to-end setup for consumer NotebookLM via notebooklm-mcp-cli."""
    result: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "bridge": bridge.bridge_status(),
        "cli_install": {},
        "mcp_server": {},
        "skill": {},
        "auth": {},
        "next_steps": [],
    }

    if install_cli:
        result["cli_install"] = install_cli_package(dry_run=dry_run)
        if result["cli_install"].get("status") == "failed":
            result["ok"] = False

    if register_mcp:
        result["mcp_server"] = ensure_mcp_server(dry_run=dry_run)
        if result["mcp_server"].get("status") in {"save_failed", "unavailable"}:
            result["ok"] = False

    if install_skill and not dry_run and bridge.cli_available():
        result["skill"] = bridge.install_skill_hermes()
        if not result["skill"].get("ok"):
            result["ok"] = False
    elif install_skill and dry_run:
        result["skill"] = {"status": "would_install", "target": "hermes"}

    if not dry_run and bridge.cli_available():
        result["auth"] = bridge.auth_status()

    result["next_steps"] = [
        "Run `hermes notebooklm login` (or `nlm login`) to authenticate with Google NotebookLM.",
        "Optional: set NOTEBOOKLM_MCP_NOTEBOOK_ID in ~/.hermes/.env for default notebook.",
        "Run `hermes notebooklm collect` then `hermes notebooklm sync --consumer`.",
        "Start a new Hermes session so MCP tools from notebooklm-mcp load.",
        "Full MCP surface: `hermes mcp install notebooklm-mcp` (catalog) or `hermes notebooklm setup-mcp`.",
    ]
    if result.get("auth", {}).get("authenticated") is False:
        result["next_steps"].insert(0, "Authentication required: `hermes notebooklm login`")
    return result
