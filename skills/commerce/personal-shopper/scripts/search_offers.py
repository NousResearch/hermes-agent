#!/usr/bin/env python3
"""Search the open web for product offers via a SearXNG instance.

Usage:
  python3 search_offers.py --query 'cafe en grain bio 1kg' \
    [--searxng-url http://127.0.0.1:8888] [--max-results 12] [--json]

Outputs (JSON mode):
  {"hits": [
     {"title": ..., "url": ..., "domain": ..., "snippet": ..., "rank": 1},
     ...
  ]}

Filters out comparator/aggregator/wiki/social-media domains so the agent
sees only direct merchant pages. The downstream LLM (Hermes itself in
production) is responsible for turning these hits into structured
offers via tool-use; this script just does the search and the
domain-level filtering.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.parse import urlparse

import httpx


BLOCKED_DOMAINS: set[str] = {
    "idealo.fr",
    "idealo.com",
    "leguide.com",
    "shopping.google.com",
    "kelkoo.fr",
    "kelkoo.com",
    "twenga.fr",
    "twenga.com",
    "youtube.com",
    "wikipedia.org",
    "fr.wikipedia.org",
    "simple.wikipedia.org",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "pinterest.com",
    "tiktok.com",
    "linkedin.com",
}


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower().lstrip(".")
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def search(
    query: str,
    *,
    searxng_url: str,
    max_results: int = 12,
    language: str = "fr",
) -> list[dict[str, object]]:
    """Run a SearXNG search and return the curated hit list."""
    base = searxng_url.rstrip("/")
    params = {
        "q": f"{query} acheter en ligne",
        "format": "json",
        "language": language,
    }
    with httpx.Client(timeout=httpx.Timeout(20.0)) as client:
        r = client.get(f"{base}/search", params=params)
        r.raise_for_status()
        data = r.json()

    hits: list[dict[str, object]] = []
    for item in data.get("results", []):
        url = (item.get("url") or "").strip()
        if not url:
            continue
        d = _domain(url)
        if not d or d in BLOCKED_DOMAINS:
            continue
        hits.append(
            {
                "rank": len(hits) + 1,
                "title": (item.get("title") or "").strip(),
                "url": url,
                "domain": d,
                "snippet": (item.get("content") or "").strip(),
            }
        )
        if len(hits) >= max_results:
            break
    return hits


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--query", required=True, help="Natural-language product query")
    p.add_argument(
        "--searxng-url",
        default=os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888"),
        help="Base URL of a SearXNG instance (default: env SEARXNG_URL or http://127.0.0.1:8888)",
    )
    p.add_argument("--max-results", type=int, default=12)
    p.add_argument("--language", default="fr")
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = p.parse_args()

    try:
        hits = search(
            args.query,
            searxng_url=args.searxng_url,
            max_results=args.max_results,
            language=args.language,
        )
    except httpx.HTTPError as exc:
        err = {"error": f"searxng request failed: {exc}", "hits": []}
        sys.stdout.write(json.dumps(err, ensure_ascii=False) + "\n")
        return 2

    if args.json:
        sys.stdout.write(json.dumps({"hits": hits}, ensure_ascii=False, indent=2) + "\n")
    else:
        for h in hits:
            sys.stdout.write(f"[{h['rank']:>2}] {h['domain']}\n  {h['title']}\n  {h['url']}\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
