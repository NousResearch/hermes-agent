#!/usr/bin/env python3
"""Build a deterministic life-ops reminder brief from local YAML/JSON.

The script is intentionally local-first: it reads one file, prints one report,
and never sends messages or touches external services.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

SILENT = "[SILENT]"


@dataclass(frozen=True)
class Reminder:
    title: str
    due: date
    notes: str = ""
    plan_path: str = ""

    def line(self, today: date) -> str:
        delta = (self.due - today).days
        if delta < 0:
            due_text = f"due {self.due.isoformat()} ({abs(delta)}d overdue)"
        elif delta == 0:
            due_text = "due today"
        else:
            due_text = f"due {self.due.isoformat()} (in {delta}d)"

        parts = [f"{self.title} — {due_text}"]
        if self.notes:
            parts.append(self.notes)
        if self.plan_path:
            parts.append(f"plan: {self.plan_path}")
        return " — ".join(parts)


def parse_date(value: Any, *, field: str) -> date:
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date string: {value!r}") from exc


def load_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("input must be a mapping with an 'items' list")
    return data


def _item_due_date(item: dict[str, Any]) -> date | None:
    if item.get("done") or item.get("completed"):
        return None
    if "due" in item:
        return parse_date(item["due"], field="due")
    if "last_done" in item and "every_days" in item:
        last_done = parse_date(item["last_done"], field="last_done")
        every_days = int(item["every_days"])
        if every_days <= 0:
            raise ValueError("every_days must be positive")
        return last_done + timedelta(days=every_days)
    return None


def evaluate_reminders(payload: dict[str, Any], *, today: str | date | None = None, soon_days: int = 7) -> list[Reminder]:
    today_date = parse_date(today, field="today") if today is not None else date.today()
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("items must be a list")

    reminders: list[Reminder] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each item must be a mapping")
        due = _item_due_date(item)
        if due is None:
            continue
        days_until_due = (due - today_date).days
        if days_until_due > soon_days:
            continue
        title = str(item.get("title") or item.get("name") or "Untitled")
        reminders.append(
            Reminder(
                title=title,
                due=due,
                notes=str(item.get("notes") or ""),
                plan_path=str(item.get("plan_path") or ""),
            )
        )
    return sorted(reminders, key=lambda reminder: (reminder.due, reminder.title.lower()))


def build_brief(payload: dict[str, Any], *, today: str | date | None = None, soon_days: int = 7) -> str:
    today_date = parse_date(today, field="today") if today is not None else date.today()
    reminders = evaluate_reminders(payload, today=today_date, soon_days=soon_days)
    if not reminders:
        return SILENT

    groups = [
        ("Overdue", [reminder for reminder in reminders if reminder.due < today_date]),
        ("Due today", [reminder for reminder in reminders if reminder.due == today_date]),
        ("Soon", [reminder for reminder in reminders if reminder.due > today_date]),
    ]
    lines = ["# Life Ops Reminder Brief"]
    for heading, group in groups:
        if not group:
            continue
        lines.append("")
        lines.append(f"## {heading}")
        lines.extend(f"- {reminder.line(today_date)}" for reminder in group)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local life-ops reminder brief from YAML/JSON.")
    parser.add_argument("input", help="Path to a YAML or JSON file with an items list")
    parser.add_argument("--today", help="Override today's date (YYYY-MM-DD), useful for tests/cron reproducibility")
    parser.add_argument("--soon-days", type=int, default=7, help="Include future tasks due within this many days")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = load_payload(args.input)
    print(build_brief(payload, today=args.today, soon_days=args.soon_days))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
