#!/usr/bin/env python3
"""Build a local-first Joe-style life-ops morning brief.

Input is a JSON/YAML list of items. Example item shapes:

    {"title": "clean fridge", "area": "household", "due": "2026-06-01"}
    {"title": "aircon cleaning", "last_done": "2026-03-10", "every_days": 90}

The script prints exactly ``[SILENT]`` when no item is overdue or due soon,
which lets cron jobs suppress delivery without special wrapper logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

try:  # PyYAML is optional for JSON-only users.
    import yaml
except Exception:  # pragma: no cover - exercised only when PyYAML unavailable.
    yaml = None


@dataclass(frozen=True)
class ActionableItem:
    title: str
    area: str
    due: date
    status: str
    days_delta: int
    notes: str = ""


def _parse_date(value: Any, field_name: str) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be YYYY-MM-DD, got {value!r}")
    return date.fromisoformat(value)


def _today(value: str | date | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _item_due_date(item: dict[str, Any]) -> date | None:
    explicit_due = _parse_date(item.get("due"), "due")
    if explicit_due is not None:
        return explicit_due

    if item.get("last_done") in (None, "") or item.get("every_days") in (None, ""):
        return None

    last_done = _parse_date(item.get("last_done"), "last_done")
    if last_done is None:
        return None
    try:
        every_days = int(item["every_days"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"every_days must be an integer, got {item.get('every_days')!r}") from exc
    if every_days <= 0:
        raise ValueError("every_days must be positive")
    return last_done + timedelta(days=every_days)


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Load life-ops items from JSON or YAML."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        return []

    if source.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("YAML input requires PyYAML; use .json or install pyyaml")
        data = yaml.safe_load(text)

    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("life-ops input must be a list of items")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} must be an object")
    return data


def classify_items(
    items: Iterable[dict[str, Any]], *, today: str | date | None = None, soon_days: int = 7
) -> list[ActionableItem]:
    """Return only overdue or due-soon items, sorted by urgency."""
    anchor = _today(today)
    horizon = anchor + timedelta(days=soon_days)
    actionable: list[ActionableItem] = []

    for item in items:
        if item.get("done") is True or item.get("status") in {"done", "completed", "cancelled"}:
            continue
        due = _item_due_date(item)
        if due is None or due > horizon:
            continue

        days_delta = (due - anchor).days
        status = "overdue" if days_delta < 0 else "soon"
        if days_delta == 0:
            status = "today"
        actionable.append(
            ActionableItem(
                title=str(item.get("title") or item.get("name") or "Untitled"),
                area=str(item.get("area") or "life"),
                due=due,
                status=status,
                days_delta=days_delta,
                notes=str(item.get("notes") or item.get("note") or ""),
            )
        )

    return sorted(actionable, key=lambda row: (row.due, row.area.lower(), row.title.lower()))


def _status_label(item: ActionableItem) -> str:
    if item.status == "overdue":
        return f"逾期 {abs(item.days_delta)} 天"
    if item.status == "today":
        return "今天到期"
    return f"{item.days_delta} 天內到期"


def build_brief(
    items: Iterable[dict[str, Any]], *, today: str | date | None = None, soon_days: int = 7
) -> str:
    """Build a Traditional Chinese morning brief or exact ``[SILENT]``."""
    actionable = classify_items(items, today=today, soon_days=soon_days)
    if not actionable:
        return "[SILENT]"

    overdue_count = sum(1 for item in actionable if item.status == "overdue")
    today_count = sum(1 for item in actionable if item.status == "today")
    soon_count = sum(1 for item in actionable if item.status == "soon")

    lines = [
        f"TL;DR：生活營運有 {len(actionable)} 個項目需要注意（逾期 {overdue_count}、今天 {today_count}、即將到期 {soon_count}）。",
        "",
        "- 事實 / 待處理：",
    ]
    for item in actionable:
        note = f" — {item.notes}" if item.notes else ""
        lines.append(
            f"  - [{item.area}] {item.title}：{_status_label(item)}（due {item.due.isoformat()}）{note}"
        )

    lines.extend(
        [
            "",
            "- 建議：先處理逾期項；若已完成，更新 `last_done` 或 `due`，下次 cron 會自動安靜。",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON/YAML life-ops backlog path")
    parser.add_argument("--today", help="Override today's date as YYYY-MM-DD for deterministic cron/tests")
    parser.add_argument("--soon-days", type=int, default=7, help="Include items due within N days")
    args = parser.parse_args(argv)

    print(build_brief(load_items(args.input), today=args.today, soon_days=args.soon_days))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
