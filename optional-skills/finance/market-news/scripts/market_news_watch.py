#!/usr/bin/env python3

"""Turn market headlines into trading risk and regime signals."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
WATCHER_STATE_DIR = Path(os.environ.get("WATCHER_STATE_DIR", str(HERMES_HOME / "watcher-state"))).expanduser()
USER_AGENT = "HermesMarketNews/0.1"
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_ITEMS = 5
MAX_SEEN_IDS = 500

PROFILE_FEEDS: dict[str, tuple[str, ...]] = {
    "crypto": (
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ),
    "forex": (
        "https://www.fxstreet.com/rss/news",
        "https://www.forexlive.com/feed/",
    ),
    "stocks": (
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds.marketwatch.com/marketwatch/topstories/",
    ),
    "macro": (
        "https://www.cnbc.com/id/100727362/device/rss/rss.html",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    ),
}

PROFILE_KEYWORDS: dict[str, dict[str, tuple[str, ...]]] = {
    "crypto": {
        "severe": ("hack", "exploit", "bankruptcy", "insolvency", "depeg", "peg break", "halt", "suspension"),
        "high": ("sec", "regulation", "regulatory", "etf", "exchange", "stablecoin", "liquidity", "liquidation"),
        "medium": ("bitcoin", "btc", "ethereum", "eth", "macro", "fed", "fomc", "inflation"),
    },
    "forex": {
        "severe": ("emergency", "intervention", "peg", "capital controls"),
        "high": ("cpi", "nfp", "payrolls", "fomc", "ecb", "boj", "boe", "rate decision", "rate hike", "rate cut"),
        "medium": ("pmi", "gdp", "inflation", "yield", "treasury", "tariff", "trade war"),
    },
    "stocks": {
        "severe": ("trading halt", "bankruptcy", "fraud", "delisting", "chapter 11"),
        "high": ("earnings", "guidance", "sec filing", "antitrust", "merger", "acquisition", "offering"),
        "medium": ("downgrade", "upgrade", "buyback", "layoffs", "product launch", "analyst"),
    },
    "macro": {
        "severe": ("emergency", "default", "bank run", "war", "invasion"),
        "high": ("fomc", "ecb", "boj", "boe", "cpi", "nfp", "payrolls", "rate decision", "inflation"),
        "medium": ("pmi", "gdp", "yields", "tariff", "liquidity", "recession"),
    },
}

ACTION_BY_LEVEL = {
    "severe": "trade_halt",
    "high": "cooldown",
    "medium": "size_down",
    "low": "watch",
}


@dataclass
class NewsItem:
    identity: str
    title: str
    link: str
    summary: str
    published: str | None
    source: str


@dataclass
class ScoredItem:
    title: str
    link: str
    summary: str
    published: str | None
    source: str
    level: str
    action: str
    reason: str
    tags: list[str]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    urls = list(args.url or PROFILE_FEEDS.get(args.profile, ()))
    if not urls:
        raise SystemExit("No feeds configured. Pass --url or use a known --profile.")

    state_path = WATCHER_STATE_DIR / f"{args.name}.json"
    seen_ids = load_seen_ids(state_path)
    items = collect_items(urls, args.timeout)
    current_ids = [item.identity for item in items]
    new_items = [item for item in items if item.identity not in seen_ids]
    save_seen_ids(state_path, current_ids + list(seen_ids))

    if not seen_ids and not args.emit_initial:
        return 0
    if not new_items:
        return 0

    scored = [score_item(args.profile, item) for item in new_items[: args.max_items]]
    overall_action, overall_reason = summarize_action(scored)

    if args.json:
        payload = {
            "profile": args.profile,
            "action": overall_action,
            "reason": overall_reason,
            "count": len(scored),
            "items": [asdict(item) for item in scored],
        }
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0

    lines = [
        f"Profile: {args.profile}",
        f"Action: {overall_action} - {overall_reason}",
    ]
    for item in scored:
        if args.with_summary and item.summary:
            lines.append(
                f"- {item.level} | {compact(item.title, 110)} | {compact(item.summary, 160)}"
            )
        else:
            lines.append(f"- {item.level} | {compact(item.title, 140)}")
    print("\n".join(lines))
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_FEEDS), default="crypto")
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--name", default=None, help="State file key (default: market-news-<profile>)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--emit-initial", action="store_true")
    parser.add_argument("--with-summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.name:
        args.name = f"market-news-{args.profile}"
    return args


def collect_items(urls: Iterable[str], timeout: float) -> list[NewsItem]:
    deduped: dict[str, NewsItem] = {}
    for url in urls:
        xml_text = fetch_url(url, timeout)
        for item in parse_feed(url, xml_text):
            deduped.setdefault(item.identity, item)
    return sorted(deduped.values(), key=sort_key, reverse=True)


def fetch_url(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to fetch feed {url}: {exc}") from exc


def parse_feed(source_url: str, xml_text: str) -> list[NewsItem]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SystemExit(f"Invalid feed XML for {source_url}: {exc}") from exc

    items = root.findall(".//{*}item")
    entries = root.findall(".//{*}entry") if not items else []
    rows = items or entries
    source = urlparse(source_url).netloc or source_url

    parsed: list[NewsItem] = []
    for row in rows:
        title = first_text(row, "title")
        link = first_link(row)
        summary = strip_html(
            first_text(row, "description")
            or first_text(row, "summary")
            or first_text(row, "content")
        )
        guid = first_text(row, "guid") or first_text(row, "id") or link or title
        published = normalize_timestamp(
            first_text(row, "pubDate")
            or first_text(row, "updated")
            or first_text(row, "published")
        )
        if not title or not guid:
            continue
        parsed.append(
            NewsItem(
                identity=guid.strip(),
                title=compact(title, 220),
                link=link.strip(),
                summary=compact(summary, 500),
                published=published,
                source=source,
            )
        )
    return parsed


def first_text(element: ET.Element, local_name: str) -> str:
    child = element.find(f"{{*}}{local_name}")
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def first_link(element: ET.Element) -> str:
    link = element.find("{*}link")
    if link is None:
        return ""
    href = link.attrib.get("href")
    if href:
        return href.strip()
    return (link.text or "").strip()


def normalize_timestamp(value: str) -> str | None:
    if not value:
        return None
    value = value.strip()
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        return datetime.fromisoformat(value).astimezone(timezone.utc).isoformat()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, IndexError):
        return None


def load_seen_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    ids = payload.get("seen_ids")
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids if item]


def save_seen_ids(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
        if len(unique) >= MAX_SEEN_IDS:
            break
    path.write_text(json.dumps({"seen_ids": unique}, indent=2) + "\n", encoding="utf-8")


def score_item(profile: str, item: NewsItem) -> ScoredItem:
    haystack = f"{item.title} {item.summary}".lower()
    profile_rules = PROFILE_KEYWORDS[profile]
    tags: list[str] = []
    level = "low"
    reason = "headline is contextually relevant but not an obvious event shock"

    for severity in ("severe", "high", "medium"):
        matched = [keyword for keyword in profile_rules[severity] if keyword in haystack]
        if not matched:
            continue
        tags = matched[:4]
        level = severity
        if severity == "severe":
            reason = "severe event-risk headline detected"
        elif severity == "high":
            reason = "market-moving event headline detected"
        else:
            reason = "headline may alter volatility, spread, or timing"
        break

    if not tags:
        common_tags = [keyword for keyword in ("fed", "inflation", "liquidity", "earnings") if keyword in haystack]
        tags = common_tags[:4]

    return ScoredItem(
        title=item.title,
        link=item.link,
        summary=item.summary,
        published=item.published,
        source=item.source,
        level=level,
        action=ACTION_BY_LEVEL[level],
        reason=reason,
        tags=tags,
    )


def summarize_action(items: list[ScoredItem]) -> tuple[str, str]:
    if not items:
        return "watch", "no new market-moving headlines"
    priority = {"trade_halt": 3, "cooldown": 2, "size_down": 1, "watch": 0}
    action = max((item.action for item in items), key=lambda value: priority[value])
    top = next(item for item in items if item.action == action)
    tag_text = ", ".join(top.tags[:3]) if top.tags else "headline mix"
    if action == "trade_halt":
        return action, f"severe event-risk news detected ({tag_text})"
    if action == "cooldown":
        return action, f"market-moving news detected ({tag_text})"
    if action == "size_down":
        return action, f"headline mix may raise volatility or slippage ({tag_text})"
    return action, f"watchlist-relevant headlines only ({tag_text})"


def sort_key(item: NewsItem) -> tuple[str, str]:
    return (item.published or "", item.identity)


def strip_html(text: str) -> str:
    return compact(re.sub(r"<[^>]+>", " ", text or ""), 500)


def compact(text: str, limit: int) -> str:
    flattened = " ".join((text or "").split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())