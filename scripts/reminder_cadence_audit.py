#!/usr/bin/env python3
"""Audit recurring reminders stored as structured Markdown checklist rows.

Spec / plan:
- Input: Markdown files or stdin containing rows like
  `- [ ] Clean fridge | cadence: every 1 month | last: 2026-05-01`.
- Output: deterministic Markdown or JSON grouped into due/upcoming/clear buckets.
- Safety: read-only local helper; never sends messages, mutates files, or calls networks.
"""

from __future__ import annotations

import argparse
import calendar
import dataclasses
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

CHECKLIST_RE = re.compile(r"^\s*[-*]\s+\[(?: |x|X)\]\s+(?P<body>.+?)\s*$")
CADENCE_RE = re.compile(r"^every\s+(?P<count>\d+)\s+(?P<unit>day|days|week|weeks|month|months|year|years)$", re.IGNORECASE)
VALID_UNITS = {"day", "week", "month", "year"}


@dataclasses.dataclass(frozen=True)
class Reminder:
    title: str
    cadence_count: int
    cadence_unit: str
    last_done: dt.date
    source: str
    line: int
    note: str = ""
    owner: str = ""


@dataclasses.dataclass(frozen=True)
class ReminderStatus:
    reminder: Reminder
    due_date: dt.date
    days_delta: int


@dataclasses.dataclass(frozen=True)
class ReminderReport:
    today: dt.date
    lookahead_days: int
    due: list[ReminderStatus]
    upcoming: list[ReminderStatus]
    clear: list[ReminderStatus]


def parse_date(value: str) -> dt.date:
    """Parse an ISO date and reject vague human prose."""
    return dt.date.fromisoformat(value.strip())


def _singular_unit(unit: str) -> str:
    unit = unit.lower().strip()
    return unit[:-1] if unit.endswith("s") else unit


def _parse_cadence(value: str) -> tuple[int, str] | None:
    match = CADENCE_RE.match(value.strip())
    if not match:
        return None
    count = int(match.group("count"))
    unit = _singular_unit(match.group("unit"))
    if count <= 0 or unit not in VALID_UNITS:
        return None
    return count, unit


def _parse_fields(parts: list[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        fields[key.strip().lower()] = value.strip()
    return fields


def parse_markdown(text: str, *, source: str = "<stdin>") -> list[Reminder]:
    """Extract reminder rows from Markdown while ignoring nearby guidance/template text."""
    reminders: list[Reminder] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = CHECKLIST_RE.match(line)
        if not match:
            continue
        parts = [part.strip() for part in match.group("body").split("|")]
        if len(parts) < 3:
            continue
        title = parts[0]
        fields = _parse_fields(parts[1:])
        cadence_raw = fields.get("cadence", "")
        last_raw = fields.get("last", "")
        cadence = _parse_cadence(cadence_raw)
        if cadence is None or not last_raw:
            continue
        try:
            last_done = parse_date(last_raw)
        except ValueError:
            continue
        reminders.append(
            Reminder(
                title=title,
                cadence_count=cadence[0],
                cadence_unit=cadence[1],
                last_done=last_done,
                source=source,
                line=line_number,
                note=fields.get("note", ""),
                owner=fields.get("owner", ""),
            )
        )
    return reminders


def parse_files(paths: Iterable[Path]) -> list[Reminder]:
    reminders: list[Reminder] = []
    for path in paths:
        reminders.extend(parse_markdown(path.read_text(encoding="utf-8"), source=str(path)))
    return reminders


def _add_months(value: dt.date, months: int) -> dt.date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return dt.date(year, month, day)


def next_due_date(reminder: Reminder) -> dt.date:
    count = reminder.cadence_count
    unit = reminder.cadence_unit
    if unit == "day":
        return reminder.last_done + dt.timedelta(days=count)
    if unit == "week":
        return reminder.last_done + dt.timedelta(weeks=count)
    if unit == "month":
        return _add_months(reminder.last_done, count)
    if unit == "year":
        return _add_months(reminder.last_done, count * 12)
    raise ValueError(f"Unsupported cadence unit: {unit}")


def build_report(reminders: Iterable[Reminder], *, today: dt.date | None = None, lookahead_days: int = 14) -> ReminderReport:
    today = today or dt.date.today()
    due: list[ReminderStatus] = []
    upcoming: list[ReminderStatus] = []
    clear: list[ReminderStatus] = []
    for reminder in reminders:
        due_date = next_due_date(reminder)
        days_delta = (today - due_date).days
        status = ReminderStatus(reminder=reminder, due_date=due_date, days_delta=days_delta)
        if days_delta >= 0:
            due.append(status)
        elif abs(days_delta) <= lookahead_days:
            upcoming.append(status)
        else:
            clear.append(status)
    due.sort(key=lambda item: (-item.days_delta, item.due_date, item.reminder.title.lower()))
    upcoming.sort(key=lambda item: (item.due_date, item.reminder.title.lower()))
    clear.sort(key=lambda item: (item.due_date, item.reminder.title.lower()))
    return ReminderReport(today=today, lookahead_days=lookahead_days, due=due, upcoming=upcoming, clear=clear)


def _cadence_label(reminder: Reminder) -> str:
    suffix = "" if reminder.cadence_count == 1 else "s"
    return f"every {reminder.cadence_count} {reminder.cadence_unit}{suffix}"


def _source_label(reminder: Reminder) -> str:
    return f"{reminder.source}:{reminder.line}"


def _render_item(item: ReminderStatus) -> str:
    reminder = item.reminder
    if item.days_delta >= 0:
        age = f"{item.days_delta}d overdue" if item.days_delta else "due today"
    else:
        age = f"in {abs(item.days_delta)}d"
    details = [
        f"due {item.due_date.isoformat()} ({age})",
        f"cadence: {_cadence_label(reminder)}",
        f"last: {reminder.last_done.isoformat()}",
        f"source: {_source_label(reminder)}",
    ]
    if reminder.owner:
        details.append(f"owner: {reminder.owner}")
    if reminder.note:
        details.append(f"note: {reminder.note}")
    return f"- **{reminder.title}** — " + "; ".join(details)


def render_markdown(report: ReminderReport) -> str:
    lines = [
        "# Reminder cadence audit",
        "",
        f"- Today: {report.today.isoformat()}",
        f"- Lookahead: {report.lookahead_days}d",
        f"- Due: {len(report.due)}",
        f"- Upcoming: {len(report.upcoming)}",
        "",
    ]
    if report.due:
        lines.extend(["## Due now", *(_render_item(item) for item in report.due), ""])
    if report.upcoming:
        lines.extend(["## Upcoming", *(_render_item(item) for item in report.upcoming), ""])
    if not report.due and not report.upcoming:
        lines.extend(["## Clear", "No reminders are due or upcoming in the configured lookahead window.", ""])
    return "\n".join(lines).rstrip() + "\n"


def _item_to_dict(item: ReminderStatus) -> dict[str, object]:
    reminder = item.reminder
    payload: dict[str, object] = {
        "title": reminder.title,
        "due_date": item.due_date.isoformat(),
        "last_done": reminder.last_done.isoformat(),
        "cadence": _cadence_label(reminder),
        "source": reminder.source,
        "line": reminder.line,
    }
    if item.days_delta >= 0:
        payload["days_overdue"] = item.days_delta
    else:
        payload["days_until_due"] = abs(item.days_delta)
    if reminder.owner:
        payload["owner"] = reminder.owner
    if reminder.note:
        payload["note"] = reminder.note
    return payload


def render_json(report: ReminderReport) -> str:
    payload = {
        "today": report.today.isoformat(),
        "lookahead_days": report.lookahead_days,
        "due": [_item_to_dict(item) for item in report.due],
        "upcoming": [_item_to_dict(item) for item in report.upcoming],
        "clear": [_item_to_dict(item) for item in report.clear],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Markdown files to scan. Reads stdin when omitted and stdin is piped.")
    parser.add_argument("--today", type=parse_date, default=None, help="Override today's date as YYYY-MM-DD for deterministic runs.")
    parser.add_argument("--lookahead-days", type=int, default=14, help="Include reminders due within this many days. Default: 14.")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown", help="Output format. Default: markdown.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.lookahead_days < 0:
        parser.error("--lookahead-days must be >= 0")

    if args.paths:
        reminders = parse_files(args.paths)
    elif not sys.stdin.isatty():
        reminders = parse_markdown(sys.stdin.read(), source="<stdin>")
    else:
        parser.error("provide at least one Markdown path or pipe Markdown on stdin")

    report = build_report(reminders, today=args.today, lookahead_days=args.lookahead_days)
    output = render_json(report) if args.format == "json" else render_markdown(report)
    sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
