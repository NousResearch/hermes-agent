#!/usr/bin/env python3
"""Preflight checks for AbletonMCP.

Checks the lazy runtime bits (`uvx`) and the Ableton Remote Script socket. The
script never installs packages and never starts the MCP server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
from pathlib import Path


ABLETON_HOST = "127.0.0.1"
ABLETON_PORT = 9877


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _mac_remote_script_candidates(home: Path) -> list[str]:
    root = home / "Library" / "Preferences" / "Ableton"
    if not root.exists():
        return []
    return [
        str(p / "User Remote Scripts" / "AbletonMCP")
        for p in sorted(root.glob("Live *"))
        if p.is_dir()
    ]


def check(*, which=shutil.which, port_open=_port_open, home=None) -> dict:
    home_path = Path(home or Path.home())
    telemetry_disabled = any(
        os.environ.get(key, "").lower() == "true"
        for key in (
            "ABLETON_MCP_DISABLE_TELEMETRY",
            "DISABLE_TELEMETRY",
            "MCP_DISABLE_TELEMETRY",
        )
    )
    return {
        "uvx": bool(which("uvx")),
        "remote_script_socket": {
            "host": ABLETON_HOST,
            "port": ABLETON_PORT,
            "reachable": port_open(ABLETON_HOST, ABLETON_PORT),
        },
        "telemetry_disabled_in_env": telemetry_disabled,
        "mac_remote_script_candidates": _mac_remote_script_candidates(home_path),
    }


def _summary(s: dict) -> str:
    sock = s["remote_script_socket"]
    lines = [
        "✓ uvx found" if s["uvx"] else "✗ uvx not found — install uv first",
        (
            f"✓ Ableton Remote Script socket: {sock['host']}:{sock['port']} reachable"
            if sock["reachable"]
            else f"✗ Ableton Remote Script socket: {sock['host']}:{sock['port']} not reachable"
        ),
        (
            "✓ telemetry disabled in current env"
            if s["telemetry_disabled_in_env"]
            else "⚠ telemetry not disabled in current env; configure MCP env opt-out"
        ),
    ]
    candidates = s["mac_remote_script_candidates"]
    if candidates:
        lines.append("macOS candidate Remote Script dirs:")
        lines.extend(f"  - {p}" for p in candidates)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check AbletonMCP prerequisites.")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args(argv)
    status = check()
    print(json.dumps(status, indent=2) if args.json else _summary(status))
    return 0 if status["uvx"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
