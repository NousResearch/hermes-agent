"""CLI commands for AgentMemory integration management."""

from __future__ import annotations

import json
import os
import shutil
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import yaml

from hermes_constants import get_hermes_home

DEFAULT_URL = "http://localhost:3111"
VIEWER_PORT = "3113"
MCP_SERVER_NAME = "agentmemory"


def _base_url(value: str | None = None) -> str:
    return (value or os.environ.get("AGENTMEMORY_URL") or DEFAULT_URL).rstrip("/")


def _viewer_url(base_url: str | None = None) -> str:
    explicit = os.environ.get("AGENTMEMORY_VIEWER_URL")
    if explicit:
        return explicit
    base = _base_url(base_url)
    parsed = urlparse(base)
    if parsed.hostname:
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{host}:{VIEWER_PORT}"
        return urlunparse((parsed.scheme or "http", netloc, "", "", "", ""))
    return "http://localhost:3113"


def _config_path() -> Path:
    return get_hermes_home() / "config.yaml"


def _read_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))


def _get_json(path: str, base_url: str | None = None) -> dict[str, Any] | None:
    url = f"{_base_url(base_url)}/agentmemory/{path.lstrip('/')}"
    headers = {"Accept": "application/json"}
    secret = os.environ.get("AGENTMEMORY_SECRET", "")
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    req = Request(url, headers=headers, method="GET")
    try:
        with urlopen(req, timeout=2) as resp:
            payload = resp.read().decode()
        data = json.loads(payload or "{}")
        return data if isinstance(data, dict) else None
    except (OSError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def _viewer_reachable(base_url: str | None = None) -> bool:
    req = Request(_viewer_url(base_url), method="GET")
    try:
        with urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def _mcp_configured(config: dict[str, Any]) -> bool:
    server = (config.get("mcp_servers") or {}).get(MCP_SERVER_NAME) or {}
    return server.get("command") == "npx" and "@agentmemory/mcp" in server.get("args", [])


def _provider_enabled(config: dict[str, Any]) -> bool:
    return (config.get("memory") or {}).get("provider") == "agentmemory"


def _configure_mcp() -> None:
    config = _read_config()
    config.setdefault("mcp_servers", {})[MCP_SERVER_NAME] = {
        "command": "npx",
        "args": ["-y", "@agentmemory/mcp"],
    }
    _write_config(config)


def _configure_provider() -> None:
    config = _read_config()
    config.setdefault("memory", {})["provider"] = "agentmemory"
    _write_config(config)


def _disable_provider() -> bool:
    config = _read_config()
    memory = config.setdefault("memory", {})
    if memory.get("provider") != "agentmemory":
        return False
    memory["provider"] = ""
    _write_config(config)
    return True


def cmd_mcp(args) -> None:
    _configure_mcp()
    print("\n  ✓ AgentMemory MCP configured")
    print("  Saved to config.yaml")
    print("  Restart Hermes or start a new session to load MCP tools.\n")


def cmd_provider(args) -> None:
    _configure_provider()
    print("\n  ✓ Memory provider: agentmemory")
    print("  Saved to config.yaml")
    print("  Start the server with: npx @agentmemory/agentmemory\n")


def cmd_enable(args) -> None:
    cmd_provider(args)


def cmd_disable(args) -> None:
    if _disable_provider():
        print("\n  ✓ AgentMemory provider disabled")
        print("  Built-in memory remains active.\n")
    else:
        print("\n  AgentMemory provider was not active.\n")


def cmd_setup(args) -> None:
    _configure_mcp()
    if getattr(args, "provider", False):
        _configure_provider()
    print("\n  ✓ AgentMemory setup saved")
    print("  MCP server: agentmemory -> npx -y @agentmemory/mcp")
    if getattr(args, "provider", False):
        print("  Memory provider: agentmemory")
    else:
        print("  Memory provider: unchanged (use --provider for lifecycle hooks)")
    print("\n  Next:")
    print("    npx @agentmemory/agentmemory")
    print("    hermes agentmemory status")
    print("    Open viewer: http://localhost:3113\n")


def cmd_status(args) -> None:
    base = _base_url(getattr(args, "url", None))
    config = _read_config()
    health = _get_json("health", base)
    flags = _get_json("config/flags", base) or {}
    server_ok = bool(health and health.get("status") in ("healthy", "ok"))
    viewer_ok = _viewer_reachable(base)
    npx_ok = bool(shutil.which("npx"))
    mcp_ok = _mcp_configured(config)
    provider_ok = _provider_enabled(config)

    print("\nAgentMemory")
    print(f"  URL: {base}")
    print(f"  Server... {'OK' if server_ok else 'FAILED'}")
    print(f"  Viewer... {'OK' if viewer_ok else 'FAILED'} ({_viewer_url(base)})")
    print(f"  npx... {'OK' if npx_ok else 'MISSING'}")
    print(f"  MCP config... {'OK' if mcp_ok else 'MISSING'}")
    print(f"  Provider... {'OK' if provider_ok else 'disabled'}")

    flag_items = flags.get("flags") if isinstance(flags, dict) else None
    if flag_items:
        disabled = [f for f in flag_items if isinstance(f, dict) and not f.get("enabled")]
        print(f"  Feature flags... {len(flag_items) - len(disabled)}/{len(flag_items)} enabled")
        for flag in disabled[:5]:
            print(f"    - {flag.get('key')}: off")
    print()


def cmd_doctor(args) -> None:
    cmd_status(args)
    print("Checks and fixes:")
    if not shutil.which("npx"):
        print("  - Install Node.js/npm so npx is available.")
    if not _mcp_configured(_read_config()):
        print("  - Run: hermes agentmemory mcp")
    if not _get_json("health", _base_url(getattr(args, "url", None))):
        print("  - Start server: npx @agentmemory/agentmemory")
    print("  - Run AgentMemory diagnostics: npx @agentmemory/agentmemory doctor\n")


def cmd_viewer(args) -> None:
    url = _viewer_url(getattr(args, "url", None))
    print(url)
    if not getattr(args, "print_only", False):
        try:
            webbrowser.open(url)
        except Exception:
            pass


def agentmemory_command(args) -> None:
    sub = getattr(args, "agentmemory_command", None) or "status"
    if sub == "setup":
        cmd_setup(args)
    elif sub == "status":
        cmd_status(args)
    elif sub == "doctor":
        cmd_doctor(args)
    elif sub == "viewer":
        cmd_viewer(args)
    elif sub == "mcp":
        cmd_mcp(args)
    elif sub == "provider":
        cmd_provider(args)
    elif sub == "enable":
        cmd_enable(args)
    elif sub == "disable":
        cmd_disable(args)
    else:
        print(f"  Unknown agentmemory command: {sub}")


def register_cli(subparser) -> None:
    subparser.description = "Manage AgentMemory persistent cross-agent memory."
    subs = subparser.add_subparsers(dest="agentmemory_command")

    setup = subs.add_parser("setup", help="Configure AgentMemory MCP, optionally as memory provider")
    setup.add_argument("--provider", action="store_true", help="Also enable AgentMemory as the active memory provider")

    status = subs.add_parser("status", help="Show AgentMemory server, MCP, and provider status")
    status.add_argument("--url", help="AgentMemory server URL")

    doctor = subs.add_parser("doctor", help="Diagnose AgentMemory setup")
    doctor.add_argument("--url", help="AgentMemory server URL")

    viewer = subs.add_parser("viewer", help="Open or print the AgentMemory viewer URL")
    viewer.add_argument("--url", help="AgentMemory server URL")
    viewer.add_argument("--print", dest="print_only", action="store_true", help="Print only, do not open browser")

    subs.add_parser("mcp", help="Configure AgentMemory as an MCP server")
    subs.add_parser("provider", help="Enable AgentMemory memory-provider hooks")
    subs.add_parser("enable", help="Alias for provider")
    subs.add_parser("disable", help="Disable AgentMemory as active memory provider")

    subparser.set_defaults(func=agentmemory_command)
