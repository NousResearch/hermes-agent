#!/usr/bin/env python3
"""Build a local-first Markdown prep brief from JSON/YAML meeting notes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

SKIP_STATUSES = {"done", "cancelled", "canceled", "archived"}
PREP_FIELDS = ("goals", "prep", "questions", "materials")


@dataclass(order=True)
class Meeting:
    sort_key: tuple[date, str] = field(init=False, repr=False)
    title: str
    day: date
    time_text: str = ""
    timezone_text: str = ""
    attendees: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    prep: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    link: str = ""
    source: str = ""
    missing_fields: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.sort_key = (self.day, self.time_text or "99:99")

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "date": self.day.isoformat(),
            "time": self.time_text,
            "timezone": self.timezone_text,
            "attendees": self.attendees,
            "goals": self.goals,
            "prep": self.prep,
            "questions": self.questions,
            "materials": self.materials,
            "notes": self.notes,
            "link": self.link,
            "source": self.source,
            "missing_fields": self.missing_fields,
        }


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text and "\n" not in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [line.strip(" -\t") for line in text.splitlines() if line.strip(" -\t")]
    return [str(value).strip()] if str(value).strip() else []


def _parse_date(record: dict[str, Any]) -> tuple[date | None, str]:
    raw_datetime = record.get("datetime")
    if raw_datetime:
        text = str(raw_datetime).strip()
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
            time_text = parsed.strftime("%H:%M") if parsed.time() != time(0, 0) else ""
            return parsed.date(), time_text
        except ValueError:
            return None, ""

    raw_date = record.get("date")
    if not raw_date:
        return None, ""
    text = str(raw_date).strip()
    try:
        parsed_day = date.fromisoformat(text[:10])
    except ValueError:
        return None, ""
    return parsed_day, str(record.get("time") or "").strip()


def _load_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise SystemExit("PyYAML is required for YAML meeting files") from exc
        payload = yaml.safe_load(text) or []
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if isinstance(payload, dict):
        records = payload.get("meetings", [])
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of meetings in {path}")
    return [record for record in records if isinstance(record, dict)]


def _normalize_record(record: dict[str, Any], source_path: Path) -> tuple[Meeting | None, str | None]:
    title = str(record.get("title") or record.get("name") or "Untitled meeting").strip()
    status = str(record.get("status") or "").strip().lower()
    if status in SKIP_STATUSES:
        return None, None

    meeting_day, parsed_time = _parse_date(record)
    if meeting_day is None:
        return None, f"Add an ISO date for `{title}` in {source_path.name}."

    fields = {field: _coerce_list(record.get(field)) for field in PREP_FIELDS}
    missing_fields = [field for field, values in fields.items() if not values]
    meeting = Meeting(
        title=title,
        day=meeting_day,
        time_text=parsed_time,
        timezone_text=str(record.get("timezone") or "").strip(),
        attendees=_coerce_list(record.get("attendees")),
        goals=fields["goals"],
        prep=fields["prep"],
        questions=fields["questions"],
        materials=fields["materials"],
        notes=_coerce_list(record.get("notes")),
        link=str(record.get("link") or "").strip(),
        source=str(record.get("source") or source_path.name).strip(),
        missing_fields=missing_fields,
    )
    return meeting, None


def collect_meetings(paths: list[Path], today: date, days: int) -> tuple[list[Meeting], list[str]]:
    end_day = today + timedelta(days=days)
    meetings: list[Meeting] = []
    quick_actions: list[str] = []

    for path in paths:
        for record in _load_file(path):
            meeting, warning = _normalize_record(record, path)
            if warning:
                quick_actions.append(warning)
            if meeting is None:
                continue
            if today <= meeting.day <= end_day:
                meetings.append(meeting)

    meetings.sort()
    for meeting in meetings:
        if meeting.missing_fields:
            quick_actions.append(
                f"Fill {', '.join(meeting.missing_fields)} for `{meeting.title}` before {meeting.day.isoformat()}."
            )
    return meetings, quick_actions


def build_payload(paths: list[Path | str], today: str | None = None, days: int = 7) -> dict[str, Any]:
    generated_for = date.fromisoformat(today) if today else datetime.now(timezone.utc).date()
    normalized_paths = [Path(path).expanduser() for path in paths]
    meetings, quick_actions = collect_meetings(normalized_paths, generated_for, days)
    return {
        "generated_for": generated_for.isoformat(),
        "window_days": days,
        "meetings": [meeting.to_dict() for meeting in meetings],
        "quick_actions": quick_actions,
    }


def _append_list(lines: list[str], label: str, values: list[str]) -> None:
    if not values:
        return
    lines.append(f"- **{label}:**")
    for value in values:
        lines.append(f"  - {value}")


def format_markdown(payload: dict[str, Any]) -> str:
    meetings = payload["meetings"]
    quick_actions = payload["quick_actions"]
    if not meetings and not quick_actions:
        return "[SILENT]"

    lines = [
        "# Meeting Prep Brief",
        "",
        f"Window: {payload['generated_for']} + {payload['window_days']} days",
        "",
    ]
    if meetings:
        for meeting in meetings:
            when = meeting["date"]
            if meeting.get("time"):
                when += f" {meeting['time']}"
            if meeting.get("timezone"):
                when += f" {meeting['timezone']}"
            lines.extend([f"## {meeting['title']}", "", f"- **When:** {when}"])
            if meeting.get("attendees"):
                lines.append(f"- **Attendees:** {', '.join(meeting['attendees'])}")
            if meeting.get("source"):
                lines.append(f"- **Source:** {meeting['source']}")
            if meeting.get("link"):
                lines.append(f"- **Link:** {meeting['link']}")
            if meeting.get("missing_fields"):
                lines.append(f"- **Warning:** Missing prep fields: {', '.join(meeting['missing_fields'])}")
            _append_list(lines, "Goals", meeting.get("goals", []))
            _append_list(lines, "Prep", meeting.get("prep", []))
            _append_list(lines, "Questions", meeting.get("questions", []))
            _append_list(lines, "Materials", meeting.get("materials", []))
            _append_list(lines, "Notes", meeting.get("notes", []))
            lines.append("")
    else:
        lines.extend(["No dated meetings are due in the selected window.", ""])

    if quick_actions:
        lines.extend(["## Quick Actions", ""])
        for action in quick_actions:
            lines.append(f"- {action}")
    return "\n".join(lines).rstrip()


def build_brief(paths: list[Path | str], today: str | None = None, days: int = 7) -> str:
    return format_markdown(build_payload(paths, today=today, days=days))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="JSON/YAML meeting files")
    parser.add_argument("--today", help="ISO date for deterministic runs, e.g. 2026-06-23")
    parser.add_argument("--days", type=int, default=7, help="Inclusive lookahead window in days")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of Markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(args.paths, today=args.today, days=args.days)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_markdown(payload))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
