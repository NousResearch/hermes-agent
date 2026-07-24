#!/usr/bin/env python3
"""Preflight checks for the optional Rive MCP skill.

No installs, no MCP process spawning. Reports whether the official Rive
desktop MCP port is reachable and whether the optional headless RiveMCP path
has the Node/npx bits needed to launch `npx -y rivemcp`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket


RIVE_HOST = "127.0.0.1"
RIVE_PORT = 9791


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check(*, which=shutil.which, port_open=_port_open) -> dict:
    node, npx, rivemcp = which("node"), which("npx"), which("rivemcp")
    return {
        "official_rive": {
            "url": f"http://{RIVE_HOST}:{RIVE_PORT}/mcp",
            "reachable": port_open(RIVE_HOST, RIVE_PORT),
        },
        "rivemcp": {
            "node": bool(node),
            "npx": bool(npx),
            "global_binary": bool(rivemcp),
            "launch": "npx -y rivemcp",
        },
    }


def _summary(s: dict) -> str:
    official, headless = s["official_rive"], s["rivemcp"]
    lines = [
        (
            f"✓ official Rive MCP: {official['url']} reachable"
            if official["reachable"]
            else f"✗ official Rive MCP: {official['url']} not reachable "
            "(open the Rive desktop editor / Early Access app)"
        )
    ]
    if headless["npx"]:
        lines.append("✓ RiveMCP launch path: npx -y rivemcp")
    elif headless["global_binary"]:
        lines.append("✓ RiveMCP global binary found: rivemcp")
    else:
        lines.append("✗ RiveMCP: need npx or a downloaded rivemcp binary")
    if not headless["node"] and not headless["global_binary"]:
        lines.append("⚠ node not found; npm/npx RiveMCP install path unavailable")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check Rive MCP integration paths.")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args(argv)
    status = check()
    print(json.dumps(status, indent=2) if args.json else _summary(status))
    return 0 if status["official_rive"]["reachable"] or status["rivemcp"]["npx"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
