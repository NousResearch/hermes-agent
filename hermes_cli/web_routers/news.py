"""News feed route for the dashboard Command Center: ``GET /api/news``.

Aggregates a handful of RSS/Atom feeds into one newest-first list. Feeds come from
``dashboard.news_feeds`` in config.yaml (a list of URLs or ``{name, url}`` mappings) and
fall back to a curated set of AI news sources. Results are cached in-process for
``_CACHE_TTL_S`` so the dashboard's auto-refresh never hammers upstream servers; a single
broken feed is reported per-feed and never fails the whole response.
"""

from __future__ import annotations

import asyncio
import html
import re
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from fastapi import APIRouter, Query

from hermes_cli.web_deps import late
from hermes_cli.web_routers._common import log as _log

router = APIRouter()

load_config = late("load_config", "hermes_cli.config")

DEFAULT_NEWS_FEEDS: List[Dict[str, str]] = [
    {"name": "Hacker News · AI", "url": "https://hnrss.org/newest?q=AI+OR+LLM&points=50"},
    {"name": "Hugging Face", "url": "https://huggingface.co/blog/feed.xml"},
    {"name": "Google AI", "url": "https://blog.google/technology/ai/rss/"},
    {"name": "TechCrunch AI", "url": "https://techcrunch.com/category/artificial-intelligence/feed/"},
    {"name": "The Verge AI", "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"},
]

_CACHE_TTL_S = 15 * 60
_FETCH_TIMEOUT_S = 8.0
_MAX_FEED_BYTES = 2 * 1024 * 1024
_MAX_FEEDS = 12
_MAX_ITEMS_PER_FEED = 25
# In the returned page no single source may fill more than this many slots while others have items.
_MAX_ITEMS_PER_SOURCE_IN_RESULT = 8
# Pause between two fetches to the SAME host (two subreddits, two blogs on one domain): fired together they
# trip burst rate limits (HTTP 429). Different hosts still run in parallel.
_SAME_HOST_GAP_S = 1.5
_SUMMARY_CHARS = 280

_ATOM = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_cache: Dict[Tuple[str, ...], Tuple[float, Dict[str, Any]]] = {}


def _feed_name(url: str) -> str:
    host = urlparse(url).hostname or url
    return host[4:] if host.startswith("www.") else host


def resolve_news_feeds(cfg: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize ``dashboard.news_feeds``; defaults when unset, empty or malformed."""
    dash = (cfg or {}).get("dashboard") if isinstance(cfg, dict) else None
    raw = dash.get("news_feeds") if isinstance(dash, dict) else None
    feeds: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                url, name = entry.strip(), ""
            elif isinstance(entry, dict):
                url = str(entry.get("url") or "").strip()
                name = str(entry.get("name") or "").strip()
            else:
                continue
            if urlparse(url).scheme not in ("http", "https"):
                continue
            feeds.append({"name": name or _feed_name(url), "url": url})
    return feeds[:_MAX_FEEDS] or [dict(f) for f in DEFAULT_NEWS_FEEDS]


def _clean_text(value: Optional[str], limit: Optional[int] = None) -> str:
    text = _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", value or ""))).strip()
    if limit and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _parse_date(value: Optional[str]) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _text(node: Optional[ET.Element], *paths: str) -> str:
    if node is None:
        return ""
    for path in paths:
        found = node.find(path)
        if found is not None and (found.text or "").strip():
            return found.text or ""
    return ""


def parse_feed(payload: bytes, source: str) -> List[Dict[str, Any]]:
    """Parse RSS 2.0 or Atom bytes into news items. Raises ``ValueError`` on bad input."""
    head = payload[:4096].lower()
    # Feeds never need DTDs; refusing them sidesteps entity-expansion and XXE tricks.
    if b"<!doctype" in head or b"<!entity" in payload.lower():
        raise ValueError("feed declares a DTD")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ValueError(f"invalid XML: {exc}") from exc

    items: List[Dict[str, Any]] = []
    if root.tag == f"{_ATOM}feed":
        for entry in root.findall(f"{_ATOM}entry")[:_MAX_ITEMS_PER_FEED]:
            link = ""
            for link_el in entry.findall(f"{_ATOM}link"):
                if link_el.get("rel", "alternate") == "alternate" and link_el.get("href"):
                    link = link_el.get("href", "")
                    break
            items.append({
                "title": _clean_text(_text(entry, f"{_ATOM}title")),
                "link": link,
                "summary": _clean_text(_text(entry, f"{_ATOM}summary", f"{_ATOM}content"), _SUMMARY_CHARS),
                "published": _parse_date(_text(entry, f"{_ATOM}published", f"{_ATOM}updated")),
                "source": source,
            })
    else:
        channel = root.find("channel")
        for item in (channel if channel is not None else root).findall("item")[:_MAX_ITEMS_PER_FEED]:
            items.append({
                "title": _clean_text(_text(item, "title")),
                "link": _text(item, "link").strip(),
                "summary": _clean_text(_text(item, "description"), _SUMMARY_CHARS),
                "published": _parse_date(_text(item, "pubDate", "{http://purl.org/dc/elements/1.1/}date")),
                "source": source,
            })
    return [
        i for i in items
        if i["title"] and urlparse(i["link"]).scheme in ("http", "https")
    ]


async def _fetch_feed(client: httpx.AsyncClient, feed: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta: Dict[str, Any] = {"name": feed["name"], "url": feed["url"], "ok": False, "count": 0, "error": None}
    try:
        async with client.stream("GET", feed["url"]) as resp:
            resp.raise_for_status()
            chunks: List[bytes] = []
            size = 0
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > _MAX_FEED_BYTES:
                    raise ValueError("feed too large")
                chunks.append(chunk)
        items = parse_feed(b"".join(chunks), feed["name"])
    except Exception as exc:  # one bad feed must not sink the rest
        meta["error"] = str(exc) or exc.__class__.__name__
        _log.debug("news feed %s failed: %s", feed["url"], meta["error"])
        return [], meta
    meta.update(ok=True, count=len(items))
    return items, meta


async def _fetch_host_group(client: httpx.AsyncClient, group: List[Tuple[int, Dict[str, str]]]):
    """One host's feeds, one after another with a short gap; returns ``(original_index, result)`` pairs."""
    out = []
    for position, (index, feed) in enumerate(group):
        if position:
            await asyncio.sleep(_SAME_HOST_GAP_S)
        out.append((index, await _fetch_feed(client, feed)))
    return out


async def collect_news(feeds: List[Dict[str, str]]) -> Dict[str, Any]:
    groups: Dict[str, List[Tuple[int, Dict[str, str]]]] = {}
    for index, feed in enumerate(feeds):
        groups.setdefault(urlparse(feed["url"]).hostname or feed["url"], []).append((index, feed))
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT_S,
        follow_redirects=True,
        headers={"User-Agent": "HermesAgent-Dashboard/1.0 (+news)"},
    ) as client:
        gathered = await asyncio.gather(*(_fetch_host_group(client, group) for group in groups.values()))
    # Feeds keep the order they were configured in, whatever order their hosts finished.
    results = [result for _index, result in sorted((pair for batch in gathered for pair in batch), key=lambda p: p[0])]

    seen: set = set()
    items: List[Dict[str, Any]] = []
    for feed_items, _meta in results:
        for item in feed_items:
            if item["link"] in seen:
                continue
            seen.add(item["link"])
            items.append(item)
    items.sort(key=lambda i: i["published"] or 0, reverse=True)
    return {
        "items": items,
        "feeds": [meta for _items, meta in results],
        "fetched_at": time.time(),
    }


def cap_per_source(items: List[Dict[str, Any]], limit: int, per_source: int = _MAX_ITEMS_PER_SOURCE_IN_RESULT
                   ) -> List[Dict[str, Any]]:
    """The newest ``limit`` items with no source over ``per_source`` while others still have some.

    ``items`` is newest-first. Sources under the cap fill the page in date order; any slots left
    are backfilled from the overflow (still newest-first), so a quiet day, or a single source,
    still returns a full page.
    """
    counts: Dict[str, int] = {}
    picked: List[Dict[str, Any]] = []
    overflow: List[Dict[str, Any]] = []
    for item in items:
        source = item.get("source", "")
        if counts.get(source, 0) < per_source:
            counts[source] = counts.get(source, 0) + 1
            picked.append(item)
        else:
            overflow.append(item)
    if len(picked) < limit:
        picked = sorted(picked + overflow[: limit - len(picked)], key=lambda i: i.get("published") or 0, reverse=True)
    return picked[:limit]


@router.get("/api/news")
async def get_news(
    refresh: bool = False,
    limit: int = Query(default=40, ge=1, le=200),
):
    try:
        cfg = await asyncio.to_thread(load_config)
    except Exception:
        cfg = {}
    feeds = resolve_news_feeds(cfg)
    key = tuple(f["url"] for f in feeds)
    cached = _cache.get(key)
    if refresh or cached is None or time.time() - cached[0] > _CACHE_TTL_S:
        data = await collect_news(feeds)
        _cache[key] = (time.time(), data)
    else:
        data = cached[1]
    return {**data, "items": cap_per_source(data["items"], limit), "total": len(data["items"])}
