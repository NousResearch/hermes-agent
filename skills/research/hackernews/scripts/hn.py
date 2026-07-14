#!/usr/bin/env python3
"""Read and search Hacker News from public, keyless APIs."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

FIREBASE_BASE = "https://hacker-news.firebaseio.com/v0"
ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
LIST_ENDPOINTS = {
    "top": "topstories",
    "new": "newstories",
    "best": "beststories",
    "ask": "askstories",
    "show": "showstories",
    "jobs": "jobstories",
}


def _fetch_json(url: str, timeout: float = 15.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "hermes-hackernews-skill/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error from {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from {url}: {exc}") from exc


def _firebase(path: str) -> Any:
    return _fetch_json(f"{FIREBASE_BASE}/{path}.json")


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"<p\s*/?>", "\n\n", value, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _hn_url(item_id: int | str | None) -> str | None:
    if item_id is None:
        return None
    return f"https://news.ycombinator.com/item?id={item_id}"


def _normalize_item(item: dict[str, Any] | None, *, include_kids: bool = False) -> dict[str, Any] | None:
    if not item or item.get("deleted") or item.get("dead"):
        return None

    normalized: dict[str, Any] = {
        "id": item.get("id"),
        "type": item.get("type"),
        "by": item.get("by"),
        "time": item.get("time"),
        "time_iso": _format_time(item.get("time")),
        "title": item.get("title"),
        "url": item.get("url"),
        "hn_url": _hn_url(item.get("id")),
        "score": item.get("score"),
        "comments": item.get("descendants"),
        "parent": item.get("parent"),
        "text": _strip_html(item.get("text")),
    }
    if include_kids:
        normalized["kids"] = item.get("kids", [])
    return {key: value for key, value in normalized.items() if value is not None}


def _format_time(timestamp: Any) -> str | None:
    if not isinstance(timestamp, (int, float)):
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def list_items(kind: str, limit: int) -> list[dict[str, Any]]:
    ids = _firebase(LIST_ENDPOINTS[kind]) or []
    rows: list[dict[str, Any]] = []
    for item_id in ids:
        if len(rows) >= limit:
            break
        row = _normalize_item(_firebase(f"item/{item_id}"))
        if row:
            rows.append(row)
    return rows


def get_item(item_id: int, include_comments: bool = False, comment_limit: int = 20) -> dict[str, Any]:
    raw = _firebase(f"item/{item_id}")
    item = _normalize_item(raw, include_kids=include_comments)
    if item is None:
        raise RuntimeError(f"HN item {item_id} was not found or is deleted/dead")

    if include_comments:
        comments = []
        for child_id in (raw.get("kids") or [])[:comment_limit]:
            comment = _normalize_item(_firebase(f"item/{child_id}"), include_kids=False)
            if comment:
                comments.append(comment)
        item["comments_detail"] = comments
        item["comments_fetched"] = len(comments)
    return item


def get_user(username: str) -> dict[str, Any]:
    user = _firebase(f"user/{urllib.parse.quote(username, safe='')}")
    if not user:
        raise RuntimeError(f"HN user {username!r} was not found")
    return {
        "id": user.get("id"),
        "created": user.get("created"),
        "created_iso": _format_time(user.get("created")),
        "karma": user.get("karma"),
        "about": _strip_html(user.get("about")),
        "submitted_count": len(user.get("submitted") or []),
        "profile_url": f"https://news.ycombinator.com/user?id={urllib.parse.quote(username, safe='')}",
    }


def _normalize_hit(hit: dict[str, Any]) -> dict[str, Any]:
    item_id = hit.get("objectID") or hit.get("id")
    try:
        item_id_int: int | str = int(item_id) if item_id is not None else item_id
    except (TypeError, ValueError):
        item_id_int = item_id

    title = hit.get("title") or hit.get("story_title")
    url = hit.get("url") or hit.get("story_url")
    created = hit.get("created_at_i")
    return {
        key: value
        for key, value in {
            "id": item_id_int,
            "title": title,
            "type": hit.get("_tags", [None])[0] if hit.get("_tags") else None,
            "by": hit.get("author"),
            "time": created,
            "time_iso": _format_time(created),
            "url": url,
            "hn_url": _hn_url(item_id),
            "score": hit.get("points"),
            "comments": hit.get("num_comments"),
            "text": _strip_html(hit.get("comment_text") or hit.get("story_text")),
            "tags": hit.get("_tags"),
        }.items()
        if value is not None
    }


def search(query: str, limit: int, by_date: bool = False, tags: str | None = None) -> list[dict[str, Any]]:
    endpoint = "search_by_date" if by_date else "search"
    params = {"query": query, "hitsPerPage": str(limit)}
    if tags:
        params["tags"] = tags
    url = f"{ALGOLIA_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    payload = _fetch_json(url)
    return [_normalize_hit(hit) for hit in payload.get("hits", [])]


def _emit(payload: Any) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _emit_error(message: str) -> int:
    print(json.dumps({"error": message}, ensure_ascii=False), file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and search Hacker News.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in LIST_ENDPOINTS:
        sub = subparsers.add_parser(command, help=f"Fetch HN {command} items")
        sub.add_argument("-n", "--limit", type=int, default=10, choices=range(1, 101), metavar="1-100")

    item = subparsers.add_parser("item", help="Fetch a story/comment/job/poll by ID")
    item.add_argument("id", type=int)
    item.add_argument("--comments", action="store_true", help="Fetch top-level comments")
    item.add_argument("--comment-limit", type=int, default=20, choices=range(1, 101), metavar="1-100")

    user = subparsers.add_parser("user", help="Fetch a public HN user profile")
    user.add_argument("username")

    search_parser = subparsers.add_parser("search", help="Search HN via Algolia")
    search_parser.add_argument("query")
    search_parser.add_argument("-n", "--limit", type=int, default=10, choices=range(1, 101), metavar="1-100")
    search_parser.add_argument("--by-date", action="store_true", help="Sort newest first")
    search_parser.add_argument("--tags", help="Algolia tags, e.g. story,ask_hn,author_pg")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command in LIST_ENDPOINTS:
            return _emit(list_items(args.command, args.limit))
        if args.command == "item":
            return _emit(get_item(args.id, include_comments=args.comments, comment_limit=args.comment_limit))
        if args.command == "user":
            return _emit(get_user(args.username))
        if args.command == "search":
            return _emit(search(args.query, args.limit, by_date=args.by_date, tags=args.tags))
    except RuntimeError as exc:
        return _emit_error(str(exc))

    return _emit_error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
