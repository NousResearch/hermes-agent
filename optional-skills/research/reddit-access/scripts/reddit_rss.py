#!/usr/bin/env python3
"""Fetch public Reddit RSS feeds and emit normalized JSON.

This is deliberately RSS-only: it avoids authenticated actions and does not
attempt to bypass Reddit's anti-bot controls.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

DEFAULT_USER_AGENT = "hermes-agent-reddit-access/1.0 (read-only research)"
ATOM = "http://www.w3.org/2005/Atom"
MEDIA = "http://search.yahoo.com/mrss/"
CONTENT = "http://purl.org/rss/1.0/modules/content/"


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    value = html.unescape(value)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</p\s*>", "\n", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _text(element: ET.Element, *names: str) -> str:
    for name in names:
        child = element.find(name)
        if child is not None and child.text:
            return child.text.strip()
    return ""


def _link(element: ET.Element) -> str:
    atom_link = element.find(f"{{{ATOM}}}link")
    if atom_link is not None and atom_link.get("href"):
        return atom_link.get("href", "")
    return _text(element, "link")


def parse_feed(payload: bytes, source_url: str) -> list[dict[str, Any]]:
    root = ET.fromstring(payload)
    entries = root.findall(f"{{{ATOM}}}entry") or root.findall(".//item")
    results: list[dict[str, Any]] = []
    for entry in entries:
        link = _link(entry)
        title = _text(entry, f"{{{ATOM}}}title", "title")
        author = _text(entry, f"{{{ATOM}}}author/{{{ATOM}}}name", "author")
        published = _text(entry, f"{{{ATOM}}}published", f"{{{ATOM}}}updated", "pubDate")
        text = _text(
            entry,
            f"{{{CONTENT}}}encoded",
            f"{{{ATOM}}}content",
            f"{{{MEDIA}}}description",
            "description",
        )
        subreddit = ""
        match = re.search(r"/r/([^/]+)", link)
        if match:
            subreddit = f"r/{match.group(1)}"
        results.append(
            {
                "title": _clean_text(title),
                "url": link,
                "author": author.removeprefix("/u/"),
                "published": published,
                "text": _clean_text(text),
                "subreddit": subreddit,
                "source": source_url,
            }
        )
    return results


def feed_url(subreddit: str | None, query: str | None) -> str:
    if bool(subreddit) == bool(query):
        raise ValueError("provide exactly one of --subreddit or --query")
    if subreddit:
        subreddit = subreddit.strip().removeprefix("r/").strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_+-]+", subreddit):
            raise ValueError("invalid subreddit name")
        return f"https://www.reddit.com/r/{subreddit}/.rss"
    return "https://www.reddit.com/search.rss?" + urllib.parse.urlencode({"q": query})


def fetch(url: str, timeout: float, user_agent: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/atom+xml"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"Reddit RSS returned HTTP {response.status}")
        return parse_feed(response.read(), url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subreddit")
    group.add_argument("--query")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 100:
        parser.error("--limit must be between 1 and 100")
    try:
        url = feed_url(args.subreddit, args.query)
        print(json.dumps(fetch(url, args.timeout, args.user_agent)[: args.limit], ensure_ascii=False, indent=2))
        return 0
    except (OSError, ET.ParseError, RuntimeError, ValueError) as exc:
        print(f"reddit_rss: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
