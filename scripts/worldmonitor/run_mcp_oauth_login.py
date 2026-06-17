#!/usr/bin/env python3
"""Run World Monitor MCP OAuth in agent / non-TTY environments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import webbrowser
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tools.mcp_oauth as mcp_oauth

mcp_oauth._is_interactive = lambda: True  # type: ignore[method-assign]

_AUTH_URL_FILE = Path(__file__).resolve().parent / ".last-oauth-url.txt"
_DEFAULT_TIMEOUT = 600.0


async def _patched_redirect_handler(authorization_url: str) -> None:
    url = authorization_url.strip()
    _AUTH_URL_FILE.write_text(url, encoding="utf-8")
    print(f"\n[worldmonitor-oauth] Authorization URL:\n  {url}\n", flush=True)
    try:
        if sys.platform == "win32":
            escaped = url.replace("'", "''")
            subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"Start-Process '{escaped}'",
                ],
                close_fds=True,
            )
        else:
            webbrowser.open(url)
        print("[worldmonitor-oauth] Browser launch requested.", flush=True)
    except Exception as exc:
        print(f"[worldmonitor-oauth] Could not open browser: {exc}", flush=True)


mcp_oauth._redirect_handler = _patched_redirect_handler  # type: ignore[assignment]


def _run_oauth_via_probe(server: str, timeout: float) -> bool:
    """Trigger SDK OAuth via connect probe (works with dynamic client registration)."""
    from hermes_cli.mcp_config import _get_mcp_servers, _oauth_tokens_present, _probe_single_server
    from tools.mcp_oauth_manager import get_manager

    servers = _get_mcp_servers()
    if server not in servers:
        raise ValueError(f"server '{server}' not in mcp_servers config")

    cfg = servers[server]
    get_manager().remove(server)

    _probe_single_server(server, cfg, connect_timeout=timeout)
    return _oauth_tokens_present(server)


def main() -> int:
    parser = argparse.ArgumentParser(description="World Monitor MCP OAuth login")
    parser.add_argument("--server", default="worldmonitor")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from tools.mcp_oauth import HermesTokenStorage

    if args.dry_run:
        storage = HermesTokenStorage(args.server)
        print(
            json.dumps(
                {
                    "success": True,
                    "has_tokens": storage.has_cached_tokens(),
                    "tokens_path": str(storage._tokens_path()),
                },
                indent=2,
            )
        )
        return 0

    _AUTH_URL_FILE.unlink(missing_ok=True)
    print(f"[worldmonitor-oauth] Starting OAuth (timeout={args.timeout}s)...", flush=True)

    try:
        ok = _run_oauth_via_probe(args.server, args.timeout)
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, indent=2))
        return 1

    storage = HermesTokenStorage(args.server)
    has_tokens = storage.has_cached_tokens()
    result = {
        "success": bool(ok and has_tokens),
        "server": args.server,
        "probe_ok": ok,
        "has_tokens": has_tokens,
        "tokens_path": str(storage._tokens_path()),
        "auth_url_file": str(_AUTH_URL_FILE) if _AUTH_URL_FILE.exists() else None,
    }
    print(json.dumps(result, indent=2))

    if result["success"]:
        print("\n[worldmonitor-oauth] Done. Verify: hermes mcp test worldmonitor", flush=True)
        return 0
    print(
        "\n[worldmonitor-oauth] Incomplete — sign in via the URL above "
        "(World Monitor Pro required), then re-run this script.",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
