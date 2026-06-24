#!/usr/bin/env python3
"""ARD navigate client.

Discovers ARD resources from a domain or registry URL by combining:
- RFC8615-style /.well-known/ai-catalog.json discovery
- POST /search with root-level federation='referrals'
- breadth-first traversal of root-level referrals

This intentionally stays small and dependency-light so Hermes can use it as a
script, test fixture, or future /ard navigate backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx

from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

ARD_TYPES = [
    "application/ai-skill",
    "application/mcp-server-card+json",
    "application/a2a-agent-card+json",
    "application/vnd.huggingface.space+json",
]


def _base_origin(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    return urlunparse((parsed.scheme or "https", parsed.netloc, "", "", "", ""))


def normalize_search_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path = parsed.path.rstrip("/")
    if path.endswith("/search") or path == "/search":
        return urlunparse((parsed.scheme or "https", parsed.netloc, path, "", parsed.query, ""))
    if path.endswith("ai-catalog.json"):
        return urljoin(_base_origin(url), "/search")
    return urljoin(_base_origin(url), "/search")


def well_known_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    if parsed.path.rstrip("/").endswith("/.well-known/ai-catalog.json"):
        return urlunparse((parsed.scheme or "https", parsed.netloc, parsed.path, "", parsed.query, ""))
    return urljoin(_base_origin(url), "/.well-known/ai-catalog.json")


def _safe_json_get(url: str, timeout: float) -> dict[str, Any] | None:
    if not is_safe_url(url) or check_website_access(url):
        return None
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if resp.status_code < 200 or resp.status_code >= 300:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _safe_json_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any] | None:
    if not is_safe_url(url) or check_website_access(url):
        return None
    try:
        resp = httpx.post(url, json=payload, timeout=timeout, follow_redirects=True)
    except httpx.HTTPError:
        return None
    if resp.status_code < 200 or resp.status_code >= 300:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _extract_entries(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = data.get(key, [])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _extract_referral_urls(data: dict[str, Any]) -> list[str]:
    raw = data.get("referrals")
    if raw is None and isinstance(data.get("federation"), dict):
        # Legacy compatibility only; caller still emits current root-level shape.
        raw = data["federation"].get("referrals")
    if not isinstance(raw, list):
        return []
    urls: list[str] = []
    for item in raw:
        if isinstance(item, str):
            urls.append(item)
        elif isinstance(item, dict) and isinstance(item.get("url"), str):
            urls.append(item["url"])
    return urls


def _search_payload(query: str, page_size: int, types: list[str] | None = None) -> dict[str, Any]:
    return {
        "query": {
            "text": query,
            "filter": {"type": types or ARD_TYPES},
        },
        "federation": "referrals",
        "pageSize": page_size,
    }


def navigate(
    start: str,
    query: str,
    *,
    page_size: int = 10,
    max_depth: int = 2,
    timeout: float = 20,
    types: list[str] | None = None,
) -> dict[str, Any]:
    entries_by_id: dict[str, dict[str, Any]] = {}
    visited: list[str] = []
    seen_search_urls: set[str] = set()
    queue: deque[tuple[str, int]] = deque()

    catalog = _safe_json_get(well_known_url(start), timeout)
    if catalog:
        for entry in _extract_entries(catalog, "entries"):
            ident = entry.get("identifier")
            if isinstance(ident, str) and ident not in entries_by_id:
                entries_by_id[ident] = entry
        for ref_url in _extract_referral_urls(catalog):
            queue.append((normalize_search_url(ref_url), 1))
        search_endpoint = catalog.get("search_endpoint")
        if isinstance(search_endpoint, dict) and isinstance(search_endpoint.get("url"), str):
            queue.appendleft((urljoin(_base_origin(start), search_endpoint["url"]), 0))
        else:
            queue.appendleft((normalize_search_url(start), 0))
    else:
        queue.append((normalize_search_url(start), 0))

    payload = _search_payload(query, page_size, types)
    errors: list[dict[str, str]] = []
    while queue:
        search_url, depth = queue.popleft()
        if search_url in seen_search_urls or depth > max_depth:
            continue
        seen_search_urls.add(search_url)
        visited.append(search_url)
        data = _safe_json_post(search_url, payload, timeout)
        if data is None:
            errors.append({"url": search_url, "error": "fetch_or_parse_failed"})
            continue
        for entry in _extract_entries(data, "results"):
            ident = entry.get("identifier")
            if isinstance(ident, str) and ident not in entries_by_id:
                entries_by_id[ident] = entry
        for ref_url in _extract_referral_urls(data):
            next_url = normalize_search_url(ref_url)
            if next_url not in seen_search_urls:
                queue.append((next_url, depth + 1))

    return {
        "ok": bool(entries_by_id) or not errors,
        "start": start,
        "query": query,
        "visited": visited,
        "entry_count": len(entries_by_id),
        "entries": list(entries_by_id.values()),
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Navigate ARD catalogs and referrals")
    parser.add_argument("start", help="Domain, ai-catalog URL, or registry/search URL")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--type", action="append", dest="types")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip()
    if not query:
        parser.error("query is required")
    result = navigate(args.start, query, page_size=args.page_size, max_depth=args.max_depth, timeout=args.timeout, types=args.types)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"ARD navigate: {result['entry_count']} entries from {len(result['visited'])} registries")
        for url in result["visited"]:
            print(f"- {url}")
        for entry in result["entries"][:20]:
            print(f"  {entry.get('type','?')} {entry.get('displayName') or entry.get('identifier')}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
