#!/usr/bin/env python3
"""Build a compact review brief from a Markdown social/investor watchlist.

The script is intentionally offline: it does not fetch social platforms or
contact anyone. It turns a human-maintained Markdown watchlist into a repeatable
queue a cron job or operator can review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse

URL_RE = re.compile(r"https?://[^\s)>,;]+")
BULLET_RE = re.compile(r"^[-*+]\s+(?P<body>.+?)\s*$")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(?P<title>.+?)\s*$")
PRIORITY_RE = re.compile(r"^\[(?P<priority>high|medium|low)\]\s*", re.IGNORECASE)
GUIDANCE_SECTION_RE = re.compile(
    r"\b(format|instructions?|notes?|safety|permission|template|output)\b", re.IGNORECASE
)

PLATFORM_BY_DOMAIN = {
    "x.com": "x",
    "twitter.com": "x",
    "threads.com": "threads",
    "github.com": "github",
    "substack.com": "substack",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "linkedin.com": "linkedin",
    "medium.com": "medium",
    "mirror.xyz": "mirror",
    "warpcast.com": "farcaster",
    "farcaster.xyz": "farcaster",
}


@dataclass(frozen=True)
class WatchlistEntry:
    """One bullet extracted from a Markdown watchlist."""

    name: str
    section: str
    priority: str
    platforms: list[str]
    urls: list[str]
    raw: str


def _platform_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for domain, platform in PLATFORM_BY_DOMAIN.items():
        if host == domain or host.endswith(f".{domain}"):
            return platform
    return "web"


def _clean_name(body: str, urls: Sequence[str]) -> tuple[str, str]:
    priority = "normal"
    priority_match = PRIORITY_RE.match(body)
    if priority_match:
        priority = priority_match.group("priority").lower()
        body = body[priority_match.end() :].strip()

    # Keep the person/project name before common metadata separators, even when
    # the separator is adjacent to the name (for example "Alice: https://...").
    candidate = re.split(r"\s*(?:—|--|:|\|)\s*", body, maxsplit=1)[0]

    without_urls = body
    for url in urls:
        without_urls = without_urls.replace(url, "")

    candidate = re.sub(r"\s+", " ", candidate).strip(" -—:;|")
    return (candidate or without_urls.strip() or "Untitled entry", priority)


def parse_watchlist(path: str | Path) -> list[WatchlistEntry]:
    """Parse Markdown bullets into watchlist entries.

    The parser is deliberately forgiving so it works with hand-maintained notes:
    any Markdown bullet becomes an entry, the nearest preceding heading becomes
    its section, and URLs are classified by domain.
    """

    watchlist_path = Path(path).expanduser()
    text = watchlist_path.read_text(encoding="utf-8")
    entries: list[WatchlistEntry] = []
    section = "Unsectioned"

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            section = heading.group("title").strip()
            continue

        bullet = BULLET_RE.match(line)
        if not bullet or GUIDANCE_SECTION_RE.search(section):
            continue

        raw = bullet.group("body").strip()
        urls = URL_RE.findall(raw)
        name, priority = _clean_name(raw, urls)
        platforms = sorted({_platform_for_url(url) for url in urls})
        entries.append(
            WatchlistEntry(
                name=name,
                section=section,
                priority=priority,
                platforms=platforms,
                urls=urls,
                raw=raw,
            )
        )

    return entries


def _sort_key(entry: WatchlistEntry) -> tuple[int, str, str]:
    priority_rank = {"high": 0, "medium": 1, "normal": 2, "low": 3}
    return (priority_rank.get(entry.priority, 2), entry.section.lower(), entry.name.lower())


def build_markdown_brief(entries: Iterable[WatchlistEntry], *, limit: int | None = None) -> str:
    """Render a compact Markdown brief for manual review."""

    ordered = sorted(entries, key=_sort_key)
    if limit is not None:
        ordered = ordered[:limit]

    platform_counts: dict[str, int] = {}
    missing_urls: list[WatchlistEntry] = []
    for entry in ordered:
        if not entry.urls:
            missing_urls.append(entry)
        for platform in entry.platforms or ["missing"]:
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

    lines = ["# Watchlist Brief", "", f"Entries: {len(ordered)}", "", "## Review queue"]
    if not ordered:
        lines.append("- No watchlist entries found.")
    for entry in ordered:
        platform_label = ", ".join(entry.platforms) if entry.platforms else "missing URL"
        priority_label = f" [{entry.priority}]" if entry.priority != "normal" else ""
        url_label = f" — {entry.urls[0]}" if entry.urls else ""
        lines.append(f"- {entry.name}{priority_label} ({entry.section}; {platform_label}){url_label}")

    lines.extend(["", "## Source coverage"])
    if platform_counts:
        for platform, count in sorted(platform_counts.items()):
            lines.append(f"- {platform}: {count}")
    else:
        lines.append("- No sources detected.")

    lines.extend(["", "## Missing URLs"])
    if missing_urls:
        for entry in missing_urls:
            lines.append(f"- {entry.name} ({entry.section})")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def build_json_payload(entries: Iterable[WatchlistEntry], *, limit: int | None = None) -> dict[str, object]:
    ordered = sorted(entries, key=_sort_key)
    if limit is not None:
        ordered = ordered[:limit]
    return {"count": len(ordered), "entries": [asdict(entry) for entry in ordered]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("watchlist", help="Markdown watchlist path")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--limit", type=int, default=None, help="Maximum entries to include")
    args = parser.parse_args(argv)

    entries = parse_watchlist(args.watchlist)
    if args.format == "json":
        print(json.dumps(build_json_payload(entries, limit=args.limit), indent=2, sort_keys=True))
    else:
        print(build_markdown_brief(entries, limit=args.limit), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke test
    raise SystemExit(main())
