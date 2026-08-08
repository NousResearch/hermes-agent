#!/usr/bin/env python3
"""Render a local-first personal reminder morning brief.

Input is a small YAML/JSON reminder list, for example:

reminders:
  - title: Clean fridge
    due: 2026-05-28
    cadence: monthly
    area: home
    action: Throw expired food and wipe shelves.
    source: Joe Operating Manual

The script intentionally only reads local files and prints markdown. It does
not create reminders, send messages, or modify user data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class Reminder:
    """A single local reminder item."""

    title: str
    due: date
    cadence: str = ""
    area: str = ""
    action: str = ""
    source: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)
    completed: bool = False


@dataclass(frozen=True)
class ReminderGroups:
    """Actionable reminders grouped by urgency."""

    overdue: tuple[Reminder, ...]
    today: tuple[Reminder, ...]
    soon: tuple[Reminder, ...]

    @property
    def has_items(self) -> bool:
        return bool(self.overdue or self.today or self.soon)

    @property
    def count(self) -> int:
        return len(self.overdue) + len(self.today) + len(self.soon)


class ReminderBriefError(ValueError):
    """Raised when reminder input is invalid."""


def _parse_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ReminderBriefError(f"Invalid due date {value!r}; expected YYYY-MM-DD") from exc
    raise ReminderBriefError(f"Invalid due date {value!r}; expected YYYY-MM-DD")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "done", "completed"}
    return bool(value)


def _as_notes(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def _coerce_reminder(raw: Any) -> Reminder:
    if not isinstance(raw, dict):
        raise ReminderBriefError("Each reminder must be an object")
    title = str(raw.get("title") or "").strip()
    if not title:
        raise ReminderBriefError("Reminder is missing required field: title")
    if "due" not in raw:
        raise ReminderBriefError(f"Reminder {title!r} is missing required field: due")
    return Reminder(
        title=title,
        due=_parse_date(raw["due"]),
        cadence=str(raw.get("cadence") or "").strip(),
        area=str(raw.get("area") or "").strip(),
        action=str(raw.get("action") or "").strip(),
        source=str(raw.get("source") or "").strip(),
        notes=_as_notes(raw.get("notes")),
        completed=_as_bool(raw.get("completed", raw.get("done", False))),
    )


def load_reminders(path: str | Path) -> list[Reminder]:
    """Load reminders from a YAML/JSON file.

    Accepts either a top-level list or an object with a ``reminders`` list.
    """

    reminder_path = Path(path)
    text = reminder_path.read_text(encoding="utf-8")
    if reminder_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}

    raw_reminders = data.get("reminders") if isinstance(data, dict) else data
    if raw_reminders is None:
        raw_reminders = []
    if not isinstance(raw_reminders, list):
        raise ReminderBriefError("Input must be a list or contain a reminders list")
    return [_coerce_reminder(item) for item in raw_reminders]


def classify_reminders(
    reminders: Iterable[Reminder], *, today: date | None = None, soon_days: int = 7
) -> ReminderGroups:
    """Group incomplete reminders into overdue, today, and soon buckets."""

    anchor = today or date.today()
    overdue: list[Reminder] = []
    due_today: list[Reminder] = []
    soon: list[Reminder] = []
    for reminder in reminders:
        if reminder.completed:
            continue
        days_until_due = (reminder.due - anchor).days
        if days_until_due < 0:
            overdue.append(reminder)
        elif days_until_due == 0:
            due_today.append(reminder)
        elif days_until_due <= soon_days:
            soon.append(reminder)

    key = lambda item: (item.due, item.title.lower())
    return ReminderGroups(
        overdue=tuple(sorted(overdue, key=key)),
        today=tuple(sorted(due_today, key=key)),
        soon=tuple(sorted(soon, key=key)),
    )


def _format_reminder(reminder: Reminder, anchor: date) -> str:
    day_delta = (reminder.due - anchor).days
    if day_delta < 0:
        timing = f"逾期 {abs(day_delta)} 天"
    elif day_delta == 0:
        timing = "今天到期"
    else:
        timing = f"{day_delta} 天後到期"

    meta = [f"due {reminder.due.isoformat()}", timing]
    if reminder.cadence:
        meta.append(f"cadence: {reminder.cadence}")
    if reminder.area:
        meta.append(f"area: {reminder.area}")
    line = f"- **{reminder.title}** ({'; '.join(meta)})"
    details: list[str] = []
    if reminder.action:
        details.append(f"  - Action for Joe: {reminder.action}")
    if reminder.source:
        details.append(f"  - Source: {reminder.source}")
    for note in reminder.notes:
        details.append(f"  - Note: {note}")
    return "\n".join([line, *details])


def render_brief(
    groups: ReminderGroups, *, today: date | None = None, silent_if_empty: bool = False
) -> str:
    """Render grouped reminders as a Joe-style Traditional Chinese markdown brief."""

    anchor = today or date.today()
    if not groups.has_items:
        if silent_if_empty:
            return "[SILENT]"
        return "## TL;DR\n- 今天沒有到期、逾期或近期提醒。"

    lines = [
        "## TL;DR",
        f"- {anchor.isoformat()} 有 {groups.count} 個需要注意的個人提醒。",
        "- 建議：先處理逾期與今天到期項目；近期項目只做排程，不要過度佔用今天注意力。",
        "",
        "## Fact / verified",
        "- 來源是本機 YAML/JSON 檔；此腳本只讀取檔案並輸出 Markdown，不會建立提醒、不會發訊息。",
    ]

    buckets = [
        ("逾期", groups.overdue),
        ("今天到期", groups.today),
        ("未來 7 天內", groups.soon),
    ]
    for heading, reminders in buckets:
        if not reminders:
            continue
        lines.extend(["", f"### {heading}"])
        lines.extend(_format_reminder(reminder, anchor) for reminder in reminders)

    lines.extend(
        [
            "",
            "## Action for Joe",
            "- 如果提醒仍有效：手動執行或排進今天行程。",
            "- 如果提醒已不重要：在來源檔標記 `completed: true`，下次輸出會自動安靜。",
        ]
    )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a personal reminder morning brief")
    parser.add_argument("--input", required=True, help="YAML/JSON reminder file")
    parser.add_argument("--today", help="Override today's date as YYYY-MM-DD")
    parser.add_argument("--soon-days", type=int, default=7, help="Include reminders due within N days")
    parser.add_argument(
        "--silent-if-empty",
        action="store_true",
        help="Print exact [SILENT] when no actionable reminders exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    anchor = _parse_date(args.today) if args.today else date.today()
    reminders = load_reminders(args.input)
    groups = classify_reminders(reminders, today=anchor, soon_days=args.soon_days)
    print(render_brief(groups, today=anchor, silent_if_empty=args.silent_if_empty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
