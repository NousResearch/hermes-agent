#!/usr/bin/env python3
"""Build a Joe-style AI/dev/community radar brief from local items or RSS.

This is intentionally small and local-first: feed it manually collected JSON/YAML
items or RSS XML exports, and it renders a concise Traditional Chinese morning
brief with facts, hypotheses, action prompts, and source links.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import html
import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # PyYAML is already a Hermes core dependency.
    import yaml
except Exception:  # pragma: no cover - only hit in stripped environments.
    yaml = None  # type: ignore[assignment]


DEFAULT_THEMES: dict[str, tuple[str, ...]] = {
    "ai agents": ("ai agent", "agentic", "agents", "coding agent", "workflow memory"),
    "developer tooling": ("developer", "github", "pull request", "pr review", "issue triage", "open-source", "open source"),
    "payments": ("payment", "payments", "checkout", "stablecoin", "merchant", "settlement"),
    "fintech": ("neobank", "fintech", "wealth", "yield", "savings", "banking"),
    "funding": ("raises", "raised", "series a", "seed", "funding", "round"),
    "movement bd": ("move", "movement", "blockchain", "onchain", "defi", "wallet"),
    "personal os": ("reminder", "habit", "body composition", "tracking", "personal os"),
}


@dataclass(frozen=True)
class RadarItem:
    """One external signal to rank and summarize."""

    title: str
    url: str = ""
    source: str = ""
    summary: str = ""
    published: str = ""
    score: int = 0
    matched_themes: tuple[str, ...] = field(default_factory=tuple)


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    return " ".join(text.split())


def _child_text(element: ET.Element, names: Sequence[str]) -> str:
    for name in names:
        child = element.find(name)
        if child is not None and child.text:
            return _clean_text(child.text)
    # Namespace-tolerant fallback for common RSS/Atom variants.
    suffixes = tuple(f"}}{name}" for name in names)
    for child in list(element):
        if child.tag in names or child.tag.endswith(suffixes):
            return _clean_text(child.text)
    return ""


def _normalize_date(value: str) -> str:
    value = _clean_text(value)
    if not value:
        return ""
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return value
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc)
    return parsed.date().isoformat()


def parse_rss_xml(xml_text: str, source: str = "") -> list[RadarItem]:
    """Parse RSS/Atom-ish XML into radar items.

    The parser deliberately handles only standard feed fields so the helper stays
    dependency-light and safe for cron usage.
    """

    root = ET.fromstring(xml_text)
    channel_title = _child_text(root.find("channel") or root, ["title"])
    feed_source = source or channel_title
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    parsed: list[RadarItem] = []
    for item in items:
        title = _child_text(item, ["title"])
        url = _child_text(item, ["link"])
        if not url:
            for child in list(item):
                if child.tag.endswith("}link") or child.tag == "link":
                    url = child.attrib.get("href", "")
                    break
        summary = _child_text(item, ["description", "summary", "content"])
        published = _normalize_date(_child_text(item, ["pubDate", "published", "updated"]))
        if title or url or summary:
            parsed.append(
                RadarItem(
                    title=title or url,
                    url=url,
                    source=feed_source,
                    summary=summary,
                    published=published,
                )
            )
    return parsed


def _item_text(item: RadarItem) -> str:
    return f"{item.title} {item.summary} {item.source}".lower()


def score_item(
    item: RadarItem, themes: dict[str, tuple[str, ...]] = DEFAULT_THEMES
) -> RadarItem:
    """Return item annotated with deterministic score and matched themes."""

    text = _item_text(item)
    score = 0
    matched: list[str] = []
    for theme, keywords in themes.items():
        hits = sum(1 for keyword in keywords if keyword.lower() in text)
        if hits:
            matched.append(theme)
            score += 2 + hits
    if item.url:
        score += 1
    if item.published:
        score += 1
    return replace(item, score=score, matched_themes=tuple(matched))


def rank_items(items: Iterable[RadarItem], limit: int = 5) -> list[RadarItem]:
    """Score and rank items for Joe's morning attention."""

    scored = [score_item(item) for item in items]
    return sorted(
        scored,
        key=lambda item: (item.score, item.published, item.title.lower()),
        reverse=True,
    )[:limit]


def _fact_line(item: RadarItem) -> str:
    parts = []
    if item.source:
        parts.append(item.source)
    if item.published:
        parts.append(item.published)
    prefix = " / ".join(parts) if parts else "source provided"
    return f"{prefix} — {item.summary or item.title}"


def _hypothesis_line(item: RadarItem) -> str:
    if item.matched_themes:
        themes = ", ".join(item.matched_themes[:3])
        return f"This may matter because it intersects Joe's {themes} radar."
    return "Potentially low-signal unless it connects to Joe's side-project, BD, or personal-OS goals."


def _action_line(item: RadarItem) -> str:
    themes = set(item.matched_themes)
    if {"payments", "fintech", "funding"} & themes:
        return "Add to top-of-funnel BD watchlist; check whether Movement infra gives them distribution or technical leverage."
    if {"ai agents", "developer tooling"} & themes:
        return "Inspect for a reusable Hermes skill, cron workflow, or small PR idea; avoid rebuilding if the tool is already strong."
    if "personal os" in themes:
        return "Convert only if it reduces recurring friction with a measurable habit/reminder loop."
    return "Skim once; promote only if a concrete next action appears."


def render_brief(
    items: Iterable[RadarItem], limit: int = 5, silent_if_empty: bool = False
) -> str:
    """Render a Traditional Chinese morning radar brief."""

    ranked = rank_items(items, limit=limit)
    if not ranked:
        return "[SILENT]" if silent_if_empty else "## TL;DR\n- 沒有新的 radar item。"

    lines = [
        "## TL;DR",
        f"- 今日整理 {len(ranked)} 個 AI/dev/community radar signals；優先看分數最高、最接近 Joe personal OS / Movement BD / side-project factory 的項目。",
        "- 請把下面的 **Hypothesis** 當成待驗證假設，不是已證明結論。",
        "",
        "## Radar Items",
    ]
    for idx, item in enumerate(ranked, start=1):
        themes = ", ".join(item.matched_themes) if item.matched_themes else "none"
        lines.extend(
            [
                f"### {idx}. {item.title}",
                f"- **Score:** {item.score}；**Themes:** {themes}",
                f"- **Fact / verified:** {_fact_line(item)}",
                f"- **Hypothesis:** {_hypothesis_line(item)}",
                f"- **Action for Joe:** {_action_line(item)}",
            ]
        )
        if item.url:
            lines.append(f"- **Source:** {item.url}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _item_from_mapping(raw: dict[str, Any]) -> RadarItem:
    return RadarItem(
        title=_clean_text(raw.get("title")),
        url=_clean_text(raw.get("url") or raw.get("link")),
        source=_clean_text(raw.get("source")),
        summary=_clean_text(raw.get("summary") or raw.get("description")),
        published=_clean_text(raw.get("published") or raw.get("date") or raw.get("pubDate")),
    )


def load_items_file(path: Path) -> list[RadarItem]:
    """Load radar items from JSON/YAML or RSS/XML file."""

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".xml", ".rss", ".atom"}:
        return parse_rss_xml(text, source=path.stem)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML radar inputs")
        data = yaml.safe_load(text) or []
    else:
        data = json.loads(text or "[]")
    if isinstance(data, dict):
        data = data.get("items", [])
    if not isinstance(data, list):
        raise ValueError(f"Expected list of items in {path}")
    return [_item_from_mapping(item) for item in data if isinstance(item, dict)]


def _read_url_or_path(value: str) -> str:
    if value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=20) as response:  # noqa: S310 - user-supplied CLI utility.
            return response.read().decode("utf-8", errors="replace")
    return Path(value).read_text(encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=[], help="JSON/YAML/XML file with radar items; repeatable")
    parser.add_argument("--rss", action="append", default=[], help="RSS/Atom URL or local XML path; repeatable")
    parser.add_argument("--limit", type=int, default=5, help="Maximum items to render")
    parser.add_argument("--silent-if-empty", action="store_true", help="Print exactly [SILENT] when no items are present")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    items: list[RadarItem] = []
    for path in args.input:
        items.extend(load_items_file(path))
    for rss in args.rss:
        items.extend(parse_rss_xml(_read_url_or_path(rss), source=rss))
    sys.stdout.write(render_brief(items, limit=args.limit, silent_if_empty=args.silent_if_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
