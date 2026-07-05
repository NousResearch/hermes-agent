"""``hermes ollama`` — Ollama Cloud usage and account info.

Usage:
    hermes ollama          Show Ollama Cloud usage (session + weekly quotas)
    hermes ollama --json   Output as JSON
    hermes ollama cookie   Show cookie status (exists/expired/missing)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

COOKIE_FILE = Path.home() / ".hermes" / "ollama_cookie.txt"


def _fetch_usage() -> dict | None:
    """Fetch Ollama Cloud usage via the core account_usage module."""
    try:
        from agent.account_usage import _fetch_ollama_cloud_usage

        snap = _fetch_ollama_cloud_usage()
        if not snap:
            return None
        return {
            "provider": snap.provider,
            "plan": snap.plan,
            "windows": [
                {
                    "label": w.label,
                    "used_percent": w.used_percent,
                    "reset_at": str(w.reset_at) if w.reset_at else None,
                }
                for w in snap.windows
            ],
            "fetched_at": str(snap.fetched_at),
        }
    except Exception:
        return None


def _render_usage(data: dict | None) -> list[str]:
    """Render usage data to text lines."""
    if not data:
        return ["Ollama Cloud usage: unavailable (cookie missing or expired)"]

    lines = []
    plan = data.get("plan") or "Unknown"
    lines.append(f"Ollama Cloud — {plan}")
    for w in data.get("windows", []):
        pct = w.get("used_percent")
        label = w.get("label", "?")
        if pct is not None:
            remaining = max(0, 100 - round(pct))
            reset = w.get("reset_at")
            line = f"  {label}: {remaining}% remaining ({pct:.1f}% used)"
            if reset:
                line += f" • resets {reset}"
            lines.append(line)
        else:
            lines.append(f"  {label}: unavailable")
    return lines


def _cookie_status() -> str:
    """Check cookie file status."""
    if not COOKIE_FILE.exists():
        return "missing"
    try:
        content = COOKIE_FILE.read_text().strip()
        if not content:
            return "empty"
        if not content.startswith("__Secure-session="):
            return "malformed"
        return "present"
    except OSError:
        return "unreadable"


def cmd_ollama(args: argparse.Namespace) -> int:
    """Handle ``hermes ollama``."""
    if args.subcommand == "cookie":
        status = _cookie_status()
        print(f"Cookie: {status}")
        if status == "present":
            size = COOKIE_FILE.stat().st_size
            print(f"  Path: {COOKIE_FILE}")
            print(f"  Size: {size} bytes")
        return 0

    data = _fetch_usage()
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        for line in _render_usage(data):
            print(line)
    return 0 if data else 1


def add_parser(subparsers) -> None:
    """Register ``hermes ollama`` on the given argparse subparsers object."""
    parser = subparsers.add_parser(
        "ollama",
        help="Show Ollama Cloud usage (session + weekly quotas)",
        description=(
            "Show Ollama Cloud usage quotas by scraping the settings page "
            "with a session cookie. Requires ~/.hermes/ollama_cookie.txt "
            "containing '__Secure-session=<value>' from ollama.com/settings."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.set_defaults(func=cmd_ollama)

    subs = parser.add_subparsers(dest="subcommand")
    subs.add_parser("cookie", help="Check cookie file status")
