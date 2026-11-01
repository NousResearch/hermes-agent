#!/usr/bin/env python3
"""Build a local-first relationship/contact reminder brief.

Input shape accepts either a list of records or a mapping with a ``contacts``
list. Records can use built-in annual date fields such as ``birthday`` and
``anniversary`` or an ``occasions`` list for custom events.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

ANNUAL_FIELDS = {
    "birthday": "生日",
    "anniversary": "紀念日",
}


@dataclass(frozen=True)
class Occasion:
    name: str
    event: str
    event_label: str
    date: date
    days_until: int
    relationship: str | None = None
    notes: str | None = None
    turning_age: int | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "event": self.event,
            "event_label": self.event_label,
            "date": self.date.isoformat(),
            "days_until": self.days_until,
        }
        if self.relationship:
            data["relationship"] = self.relationship
        if self.notes:
            data["notes"] = self.notes
        if self.turning_age is not None:
            data["turning_age"] = self.turning_age
        return data


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _annual_observed_date(source: date, year: int) -> date:
    if source.month == 2 and source.day == 29 and not _is_leap_year(year):
        return date(year, 2, 28)
    return date(year, source.month, source.day)


def _next_annual_occurrence(source: date, today: date) -> date:
    candidate = _annual_observed_date(source, today.year)
    if candidate < today:
        candidate = _annual_observed_date(source, today.year + 1)
    return candidate


def _age_on_annual_date(source: date, occurrence: date) -> int:
    years = occurrence.year - source.year
    if source.month == 2 and source.day == 29 and occurrence.month == 2 and occurrence.day == 28:
        return years
    if (occurrence.month, occurrence.day) < (source.month, source.day):
        return years - 1
    return years


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load contacts from JSON or YAML without reading any external sources."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on env packaging
            raise RuntimeError("PyYAML is required for YAML input") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("contacts"), list):
        records = payload["contacts"]
    else:
        raise ValueError("Expected a list of contacts or an object with a contacts list")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Every contact record must be an object")
    return records


def _record_occasions(record: dict[str, Any], today: date, window_days: int) -> list[Occasion]:
    name = str(record.get("name", "")).strip()
    if not name:
        return []

    relationship = str(record.get("relationship", "")).strip() or None
    notes = str(record.get("notes", "")).strip() or None
    items: list[Occasion] = []

    for field, label in ANNUAL_FIELDS.items():
        raw_value = record.get(field)
        if not raw_value:
            continue
        source_date = _parse_date(str(raw_value))
        occurrence = _next_annual_occurrence(source_date, today)
        days_until = (occurrence - today).days
        if 0 <= days_until <= window_days:
            items.append(
                Occasion(
                    name=name,
                    event=field,
                    event_label=label,
                    date=occurrence,
                    days_until=days_until,
                    relationship=relationship,
                    notes=notes,
                    turning_age=_age_on_annual_date(source_date, occurrence),
                )
            )

    for occasion in record.get("occasions", []) or []:
        if not isinstance(occasion, dict) or not occasion.get("date"):
            continue
        label = str(occasion.get("label") or occasion.get("event") or "提醒")
        raw_event = str(occasion.get("event") or occasion.get("label") or "occasion")
        source_date = _parse_date(str(occasion["date"]))
        recurring = bool(occasion.get("annual", False))
        occurrence = _next_annual_occurrence(source_date, today) if recurring else source_date
        days_until = (occurrence - today).days
        if 0 <= days_until <= window_days:
            items.append(
                Occasion(
                    name=name,
                    event=raw_event,
                    event_label=label,
                    date=occurrence,
                    days_until=days_until,
                    relationship=relationship,
                    notes=str(occasion.get("notes") or notes or "").strip() or None,
                )
            )

    return items


def build_brief(
    records: list[dict[str, Any]], today: date | None = None, window_days: int = 14
) -> dict[str, Any]:
    """Return a structured brief for occasions due within ``window_days``."""
    today = today or date.today()
    items: list[Occasion] = []
    for record in records:
        items.extend(_record_occasions(record, today, window_days))

    item_dicts = [item.as_dict() for item in sorted(items, key=lambda item: (item.date, item.name))]
    if not item_dicts:
        return {"silent": True, "items": []}
    return {"silent": False, "today": today.isoformat(), "window_days": window_days, "items": item_dicts}


def _relative_zh(days_until: int) -> str:
    if days_until == 0:
        return "今天"
    if days_until == 1:
        return "明天"
    return f"{days_until} 天後"


def render_text(brief: dict[str, Any]) -> str:
    """Render a Joe-style Traditional Chinese brief."""
    if brief.get("silent"):
        return "[SILENT]"

    items = brief.get("items", [])
    lines = [
        f"TL;DR：未來 {brief.get('window_days')} 天有 {len(items)} 個關係提醒需要 Joe 看一眼。",
        "",
        "- 事實 / 已驗證：",
    ]
    for item in items:
        age = f"，{item['turning_age']} 歲" if item.get("turning_age") is not None else ""
        relationship = f"（{item['relationship']}）" if item.get("relationship") else ""
        notes = f"；備註：{item['notes']}" if item.get("notes") else ""
        lines.append(
            f"  - {item['name']}{relationship}：{item['event_label']}在 {item['date']}（{_relative_zh(int(item['days_until']))}{age}）{notes}"
        )

    lines.extend(
        [
            "",
            "- 建議動作：",
            "  - 若要聯絡真人，只起草內容；由 Joe 手動傳送。",
            "  - 今天/明天的項目優先處理，避免最後一刻補救。",
        ]
    )
    return "\n".join(lines)


def _parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to contacts JSON/YAML")
    parser.add_argument("--today", help="Override today as YYYY-MM-DD for deterministic runs")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    if args.window_days < 0:
        parser.error("--window-days must be non-negative")

    brief = build_brief(load_records(args.input), today=_parse_today(args.today), window_days=args.window_days)
    if args.format == "json":
        print(json.dumps(brief, ensure_ascii=False, indent=2))
    else:
        print(render_text(brief))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
