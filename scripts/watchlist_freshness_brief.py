#!/usr/bin/env python3
"""Local-first freshness brief for human-maintained social watchlists.

This script intentionally performs no network requests and mutates no files.
It helps cron jobs decide whether a watchlist needs manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

URL_RE = re.compile(r"https?://[^\s)>,]+")
BULLET_RE = re.compile(r"^\s*[-*]\s+(?:\[[ xX]\]\s*)?(?P<body>.+?)\s*$")
HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True)
class WatchlistEntry:
    label: str
    url: str
    section: str


@dataclass(frozen=True)
class FreshnessItem:
    entry: WatchlistEntry
    last_checked: date
    due_date: date
    cadence_days: int
    priority: str
    notes: str

    def days_overdue_as_of(self, today: date) -> int:
        return max(0, (today - self.due_date).days)

    def days_until_due_as_of(self, today: date) -> int:
        return max(0, (self.due_date - today).days)


@dataclass(frozen=True)
class FreshnessBrief:
    today: date
    overdue: list[FreshnessItem]
    due_soon: list[FreshnessItem]
    missing_state: list[WatchlistEntry]

    @property
    def has_work(self) -> bool:
        return bool(self.overdue or self.due_soon or self.missing_state)


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_watchlist(path: Path) -> list[WatchlistEntry]:
    entries: list[WatchlistEntry] = []
    section = "Uncategorized"

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            section = heading.group("title").strip()
            continue

        bullet = BULLET_RE.match(line)
        if not bullet:
            continue

        body = bullet.group("body").strip()
        url_match = URL_RE.search(body)
        if not url_match:
            continue

        url = url_match.group(0).rstrip(".,;")
        label_part = body[: url_match.start()].strip(" -–—:|")
        label = label_part or url
        entries.append(WatchlistEntry(label=label, url=url, section=section))

    return entries


def load_state(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("state JSON must be an object")

    items = raw.get("items", raw)
    if not isinstance(items, dict):
        raise ValueError("state JSON 'items' must be an object")

    normalized: dict[str, dict[str, Any]] = {}
    for url, metadata in items.items():
        if not isinstance(url, str):
            continue
        if isinstance(metadata, dict):
            normalized[url] = dict(metadata)
        else:
            normalized[url] = {}
    return normalized


def _metadata_for(entry: WatchlistEntry, state: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return state.get(entry.url) or state.get(entry.label)


def _coerce_cadence_days(value: Any, default: int = 14) -> int:
    try:
        cadence = int(value)
    except (TypeError, ValueError):
        return default
    return cadence if cadence > 0 else default


def build_brief(
    entries: list[WatchlistEntry],
    state: dict[str, dict[str, Any]],
    *,
    today: date,
    soon_days: int = 3,
    default_cadence_days: int = 14,
) -> FreshnessBrief:
    overdue: list[FreshnessItem] = []
    due_soon: list[FreshnessItem] = []
    missing_state: list[WatchlistEntry] = []
    soon_cutoff = today + timedelta(days=max(0, soon_days))

    seen_urls: set[str] = set()
    for entry in entries:
        if entry.url in seen_urls:
            continue
        seen_urls.add(entry.url)

        metadata = _metadata_for(entry, state)
        last_checked_raw = metadata.get("last_checked") if metadata else None
        if not isinstance(last_checked_raw, str):
            missing_state.append(entry)
            continue

        try:
            last_checked = parse_date(last_checked_raw)
        except argparse.ArgumentTypeError:
            missing_state.append(entry)
            continue

        cadence_days = _coerce_cadence_days(
            metadata.get("cadence_days") if metadata else None,
            default=default_cadence_days,
        )
        due_date = last_checked + timedelta(days=cadence_days)
        item = FreshnessItem(
            entry=entry,
            last_checked=last_checked,
            due_date=due_date,
            cadence_days=cadence_days,
            priority=str(metadata.get("priority", "normal") if metadata else "normal"),
            notes=str(metadata.get("notes", "") if metadata else ""),
        )

        if due_date < today:
            overdue.append(item)
        elif due_date <= soon_cutoff:
            due_soon.append(item)

    overdue.sort(key=lambda item: (item.due_date, item.entry.section, item.entry.label.lower()))
    due_soon.sort(key=lambda item: (item.due_date, item.entry.section, item.entry.label.lower()))
    missing_state.sort(key=lambda entry: (entry.section, entry.label.lower()))
    return FreshnessBrief(today=today, overdue=overdue, due_soon=due_soon, missing_state=missing_state)


def _item_to_dict(item: FreshnessItem, today: date) -> dict[str, Any]:
    return {
        "label": item.entry.label,
        "url": item.entry.url,
        "section": item.entry.section,
        "last_checked": item.last_checked.isoformat(),
        "due_date": item.due_date.isoformat(),
        "cadence_days": item.cadence_days,
        "priority": item.priority,
        "notes": item.notes,
        "days_overdue": item.days_overdue_as_of(today),
        "days_until_due": item.days_until_due_as_of(today),
    }


def brief_to_dict(brief: FreshnessBrief) -> dict[str, Any]:
    return {
        "today": brief.today.isoformat(),
        "summary": {
            "overdue": len(brief.overdue),
            "due_soon": len(brief.due_soon),
            "missing_state": len(brief.missing_state),
        },
        "overdue": [_item_to_dict(item, brief.today) for item in brief.overdue],
        "due_soon": [_item_to_dict(item, brief.today) for item in brief.due_soon],
        "missing_state": [entry.__dict__ for entry in brief.missing_state],
    }


def render_markdown(brief: FreshnessBrief) -> str:
    if not brief.has_work:
        return "[SILENT]"

    lines = ["# Watchlist Freshness Brief", "", f"Date: {brief.today.isoformat()}", ""]
    lines.append(
        "Summary: "
        f"{len(brief.overdue)} overdue, "
        f"{len(brief.due_soon)} due soon, "
        f"{len(brief.missing_state)} missing tracking state."
    )

    if brief.overdue:
        lines.extend(["", "## Overdue"])
        for item in brief.overdue:
            lines.append(
                f"- **{item.entry.label}** ({item.entry.section}) — "
                f"due {item.due_date.isoformat()} "
                f"({item.days_overdue_as_of(brief.today)}d overdue), "
                f"priority: {item.priority}; {item.entry.url}"
            )
            if item.notes:
                lines.append(f"  - Notes: {item.notes}")

    if brief.due_soon:
        lines.extend(["", "## Due soon"])
        for item in brief.due_soon:
            lines.append(
                f"- **{item.entry.label}** ({item.entry.section}) — "
                f"due {item.due_date.isoformat()} "
                f"(in {item.days_until_due_as_of(brief.today)}d); {item.entry.url}"
            )

    if brief.missing_state:
        lines.extend(["", "## Missing tracking state"])
        for entry in brief.missing_state:
            lines.append(f"- **{entry.label}** ({entry.section}) — {entry.url}")

    lines.extend(
        [
            "",
            "Next action: review overdue items manually, then record `last_checked` in the state file.",
            "Safety: read-only brief; no platform interactions were performed.",
        ]
    )
    return "\n".join(lines)


def default_watchlist_path() -> Path:
    return Path.home() / ".hermes" / "memories" / "INVESTOR_SOCIAL_WATCHLIST.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a local watchlist freshness brief.")
    parser.add_argument("--watchlist", type=Path, default=default_watchlist_path())
    parser.add_argument("--state", type=Path, default=None)
    parser.add_argument("--today", type=parse_date, default=date.today())
    parser.add_argument("--soon-days", type=int, default=3)
    parser.add_argument("--default-cadence-days", type=int, default=14)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entries = parse_watchlist(args.watchlist)
    state = load_state(args.state)
    brief = build_brief(
        entries,
        state,
        today=args.today,
        soon_days=args.soon_days,
        default_cadence_days=args.default_cadence_days,
    )
    if args.json:
        print(json.dumps(brief_to_dict(brief), indent=2, sort_keys=True))
    else:
        print(render_markdown(brief))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
