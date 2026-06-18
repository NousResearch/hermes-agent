"""Tool handlers and availability for scrapling-feeds plugin."""

from __future__ import annotations

import json
from typing import Any

from . import feeds_catalog
from . import fetcher
from . import gov_digest
from . import mhlw_designated

STATUS_SCHEMA = {
    "name": "gov_feed_status",
    "description": (
        "Show government RSS feed catalog status (MOD, CISA, NISC, etc.) and "
        "whether Scrapling Fetcher is installed."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

FETCH_SCHEMA = {
    "name": "gov_feed_fetch",
    "description": (
        "Fetch one official government RSS/Atom feed by id (e.g. mod_press, cisa_advisories_all)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "feed_id": {
                "type": "string",
                "description": "Catalog feed id (see gov_feed_status).",
            },
            "hours": {
                "type": "integer",
                "description": "Only entries from the last N hours (default 24).",
                "default": 24,
            },
            "max_items": {
                "type": "integer",
                "description": "Max entries to return (default 15).",
                "default": 15,
            },
        },
        "required": ["feed_id"],
    },
}

DIGEST_SCHEMA = {
    "name": "gov_feed_digest",
    "description": (
        "Digest all enabled government feeds (防衛省, CISA, NISC, デジタル庁) "
        "for the last N hours — PRIMARY sources for PDB/OSINT."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "hours": {
                "type": "integer",
                "description": "Time window in hours (default 24).",
                "default": 24,
            },
            "max_per_feed": {
                "type": "integer",
                "description": "Max entries per feed (default 10).",
                "default": 10,
            },
            "feed_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional subset of feed ids; default = all enabled.",
            },
        },
        "required": [],
    },
}


MHLW_CHECK_SCHEMA = {
    "name": "mhlw_designated_check",
    "description": (
        "Check MHLW 指定薬物部会 meeting notices and 指定薬物 designation/enforcement "
        "announcements (PRIMARY sources). Returns new items since last check."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "record_baseline": {
                "type": "boolean",
                "description": "Mark current items as seen without reporting as new.",
                "default": False,
            },
            "scan_enforcement": {
                "type": "boolean",
                "description": "Run site: search for 指定・施行 announcements.",
                "default": True,
            },
        },
        "required": [],
    },
}


def check_available() -> bool:
    """Plugin is always available (urllib fallback)."""
    return True


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def handle_status(_args: dict[str, Any], **_: Any) -> str:
    feeds = []
    for fid, meta in feeds_catalog.GOV_FEEDS.items():
        feeds.append(
            {
                "id": fid,
                "name": meta.get("name"),
                "url": meta.get("url"),
                "country": meta.get("country"),
                "enabled_by_default": meta.get("enabled_by_default", True),
            }
        )
    return _json(
        {
            "success": True,
            "scrapling_available": fetcher.scrapling_available(),
            "feed_count": len(feeds),
            "enabled_by_default": feeds_catalog.list_feed_ids(enabled_only=True),
            "feeds": feeds,
            "catalog_docs": feeds_catalog.CATALOG_DOCS,
            "install_hint": "pip install 'hermes-agent[scrapling-feeds]' for Scrapling Fetcher",
        }
    )


def handle_fetch(args: dict[str, Any], **_: Any) -> str:
    feed_id = (args.get("feed_id") or "").strip()
    if not feed_id:
        return _json({"success": False, "error": "feed_id is required"})
    hours = int(args.get("hours") or 24)
    max_items = int(args.get("max_items") or 15)
    return _json(
        gov_digest.fetch_single_feed(feed_id, hours=hours, max_items=max_items)
    )


def handle_digest(args: dict[str, Any], **_: Any) -> str:
    hours = int(args.get("hours") or 24)
    max_per_feed = int(args.get("max_per_feed") or 10)
    feed_ids = args.get("feed_ids")
    if feed_ids and not isinstance(feed_ids, list):
        feed_ids = None
    return _json(
        gov_digest.digest_feeds(
            hours=hours,
            max_per_feed=max_per_feed,
            feed_ids=feed_ids,
        )
    )


def handle_mhlw_check(args: dict[str, Any], **_: Any) -> str:
    result = mhlw_designated.check_mhlw_designated(
        record_baseline=bool(args.get("record_baseline")),
        scan_enforcement=bool(args.get("scan_enforcement", True)),
    )
    result["markdown"] = mhlw_designated.build_report_markdown(result)
    return _json(result)


def handle_slash(parts: list[str]) -> str:
    """Slash command: /scrapling-feeds [status|fetch|digest|mhlw] ..."""
    sub = (parts[0] if parts else "status").lower()
    if sub in {"status", "st"}:
        return handle_status({})
    if sub in {"fetch", "f"} and len(parts) >= 2:
        return handle_fetch({"feed_id": parts[1], "hours": 24})
    if sub in {"digest", "d"}:
        return handle_digest({"hours": 24})
    if sub in {"mhlw", "mhlw-check", "薬物"}:
        return handle_mhlw_check({})
    return (
        "Usage: /scrapling-feeds status | fetch <feed_id> | digest | mhlw\n"
        f"Enabled feeds: {', '.join(feeds_catalog.list_feed_ids())}"
    )
