"""Tool handlers for sitdeck-osint plugin."""

from __future__ import annotations

import json
from typing import Any

from . import browser_crawl
from . import credentials


def check_available() -> bool:
    """Tool gate: credentials required; Playwright optional at import time."""
    creds = credentials.get_credentials()
    return bool(creds["email"] and creds["password"])


STATUS_SCHEMA = {
    "name": "sitdeck_status",
    "description": (
        "SitDeck OSINT plugin status: .env credentials, Playwright, last crawl, "
        "World Monitor MCP replacement path (no Pro required)."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

CRAWL_SCHEMA = {
    "name": "sitdeck_crawl",
    "description": (
        "Browser-login to SitDeck (app.sitdeck.com) and crawl dashboard text plus "
        "JSON API responses. Requires SITDECK_EMAIL and SITDECK_PASSWORD in .env."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "headless": {
                "type": "boolean",
                "description": "Run Chromium headless (default true).",
                "default": True,
            },
            "reuse_session": {
                "type": "boolean",
                "description": "Reuse saved Playwright storage state when present.",
                "default": True,
            },
        },
        "required": [],
    },
}

DIGEST_SCHEMA = {
    "name": "sitdeck_osint_digest",
    "description": (
        "Run SitDeck browser crawl and return a markdown digest for PDB/OSINT briefing "
        "(replaces World Monitor Pro MCP for dashboard intelligence)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "headless": {"type": "boolean", "default": True},
        },
        "required": [],
    },
}


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    last = browser_crawl.load_last_crawl()
    payload = {
        "success": True,
        "credentials": credentials.credential_status(),
        "playwright": browser_crawl._playwright_available(),
        "last_crawl_at": last.get("saved_at") if last else None,
        "last_crawl_ok": bool(last and last.get("success")),
        "replacement_for": "worldmonitor_pro_mcp",
        "notes": [
            "SitDeck Hobbyist is free; no wm_ key or World Monitor OAuth MCP required.",
            "Install: pip install hermes-agent[sitdeck-osint] && playwright install chromium",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def handle_crawl(args: dict[str, Any], **_: Any) -> str:
    result = browser_crawl.crawl_dashboard(
        headless=bool(args.get("headless", True)),
        reuse_session=bool(args.get("reuse_session", True)),
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


def handle_digest(args: dict[str, Any], **_: Any) -> str:
    crawl = browser_crawl.crawl_dashboard(headless=bool(args.get("headless", True)))
    return browser_crawl.build_digest(crawl)


def handle_slash(args: str) -> str:
    """Slash command: /sitdeck-osint [status|crawl|digest]."""
    parts = (args or "").strip().split()
    sub = (parts[0] if parts else "status").lower()
    if sub in {"status", "st"}:
        return handle_status({})
    if sub in {"crawl", "fetch"}:
        return handle_crawl({})
    if sub in {"digest", "brief"}:
        return handle_digest({})
    return (
        "Usage: /sitdeck-osint [status|crawl|digest]\n"
        "SitDeck browser OSINT (no World Monitor Pro MCP)."
    )
