"""Government feed digest — fetch catalog feeds and filter by time window."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from . import feeds_catalog
from . import fetcher
from . import rss_parse


def fetch_single_feed(
    feed_id: str,
    *,
    hours: int = 24,
    max_items: int = 15,
    include_disabled: bool = False,
) -> dict[str, Any]:
    meta = feeds_catalog.get_feed(feed_id)
    if not meta:
        return {"success": False, "error": f"unknown feed_id: {feed_id}"}
    if not include_disabled and not meta.get("enabled_by_default", True):
        return {
            "success": False,
            "error": f"feed {feed_id} is disabled by default",
            "feed_id": feed_id,
        }

    url = meta["url"]
    fetch_result = fetcher.fetch_url(url)
    row: dict[str, Any] = {
        "feed_id": feed_id,
        "name": meta.get("name"),
        "url": url,
        "agency": meta.get("agency"),
        "country": meta.get("country"),
        "category": meta.get("category"),
        "source_tier": meta.get("source_tier", "PRIMARY"),
        "catalog_doc": meta.get("catalog_doc"),
        "fetch": {
            "backend": fetch_result.get("backend"),
            "status": fetch_result.get("status"),
        },
    }
    if not fetch_result.get("success"):
        row["success"] = False
        row["error"] = fetch_result.get("error", "fetch failed")
        return row

    since = datetime.now(timezone.utc) - timedelta(hours=max(1, hours))
    try:
        entries = rss_parse.parse_feed_xml(fetch_result.get("body") or "")
    except Exception as exc:
        row["success"] = False
        row["error"] = f"XML parse failed: {exc}"
        return row
    filtered = rss_parse.filter_entries_since(entries, since=since, max_items=max_items)
    for entry in filtered:
        entry.pop("published_dt", None)
        url_item = (entry.get("url") or "").strip()
        entry["citation"] = (
            f"[出典: {meta.get('agency') or feed_id} — {entry.get('title', '')}] {url_item}"
            if url_item
            else f"[出典: {meta.get('agency') or feed_id}] {url}"
        )

    row.update(
        {
            "success": True,
            "entries": filtered,
            "entry_count": len(filtered),
            "window_hours": hours,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    if meta.get("note"):
        row["note"] = meta["note"]
    return row


def digest_feeds(
    *,
    hours: int = 24,
    feed_ids: list[str] | None = None,
    max_per_feed: int = 10,
    include_disabled: bool = False,
) -> dict[str, Any]:
    """Fetch all enabled (or selected) government feeds."""
    ids = feed_ids or feeds_catalog.list_feed_ids(enabled_only=not include_disabled)
    feeds: list[dict[str, Any]] = []
    ok = 0
    total_entries = 0
    for fid in ids:
        result = fetch_single_feed(
            fid,
            hours=hours,
            max_items=max_per_feed,
            include_disabled=include_disabled,
        )
        feeds.append(result)
        if result.get("success"):
            ok += 1
            total_entries += result.get("entry_count") or 0

    return {
        "success": ok > 0,
        "scrapling_available": fetcher.scrapling_available(),
        "window_hours": hours,
        "feed_count": len(ids),
        "feeds_ok": ok,
        "total_entries": total_entries,
        "feeds": feeds,
        "methodology": (
            "Official government RSS/Atom only; polite HTTP (urllib or Scrapling Fetcher). "
            "No stealth / bot-evasion."
        ),
        "catalog_docs": feeds_catalog.CATALOG_DOCS,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
