#!/usr/bin/env python3
"""Morning brief composer — assembles the four-section daily markdown brief.

Reads three upstream JSON contracts, validates freshness, and renders a
single human-readable brief. Sections degrade independently when a contract
is missing or stale; the script never exits non-zero due to unavailable
contracts alone.

Input paths (env var | CLI flag | default):
  journal_brief.latest.json:
    JOURNAL_BRIEF_PATH | --journal-path
    default: ~/.hermes/contracts/journal_brief.latest.json

  perfcoach_brief.latest.json:
    PERFCOACH_BRIEF_PATH | --perfcoach-path
    default: ~/.hermes/contracts/perfcoach_brief.latest.json

  commander_report.latest.json:
    COMMANDER_REPORT_PATH | --commander-path
    default: ~/.hermes/contracts/commander_report.latest.json

Output (env var | CLI flag | default):
  MORNING_BRIEF_OUTPUT | --output
  default: ~/.hermes/morning_brief.md

Flags:
  --dry-run   Print brief to stdout, exit 0, write no file.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

_BKK = ZoneInfo("Asia/Bangkok")

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_CONTRACTS_DIR = _HERMES_HOME / "contracts"

DEFAULT_JOURNAL_PATH = str(_CONTRACTS_DIR / "journal_brief.latest.json")
DEFAULT_PERFCOACH_PATH = str(_CONTRACTS_DIR / "perfcoach_brief.latest.json")
DEFAULT_COMMANDER_PATH = str(_CONTRACTS_DIR / "commander_report.latest.json")
DEFAULT_OUTPUT_PATH = str(_HERMES_HOME / "morning_brief.md")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def get_today_bangkok() -> str:
    return datetime.now(_BKK).date().isoformat()


def validate_freshness(data: dict, today: str) -> bool:
    return data.get("for_date") == today


# ---------------------------------------------------------------------------
# Contract loader
# ---------------------------------------------------------------------------

def load_contract(path: str | Path) -> tuple[dict | None, str]:
    """Load a JSON contract file and validate freshness.

    Returns (data, reason) where data is None when unavailable and reason
    explains why (non-empty string).
    """
    path = Path(path)
    if not path.exists():
        return None, f"file not found: {path}"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"could not parse {path}: {exc}"

    today = get_today_bangkok()
    if not validate_freshness(data, today):
        stale_date = data.get("for_date", "unknown")
        return None, f"stale: {stale_date}"

    return data, ""


# ---------------------------------------------------------------------------
# Todo helpers
# ---------------------------------------------------------------------------

def normalize_todo_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_todo_text(item: dict) -> str:
    """Return an item's display text.

    The journal contract's canonical field is "content"; "text" is kept as
    a fallback for older/alternate producers.
    """
    return item.get("content") or item.get("text", "")


def filter_todos(todos: list) -> list:
    """Filter, dedup, sort, and annotate todos.

    Steps:
    1. Keep only items with confidence >= 0.6.
    2. Deduplicate by normalised text (case-insensitive, strip punctuation).
    3. Sort by priority descending.
    4. Annotate dev-category items with _approval_route=True.
    """
    kept: list[dict] = []
    seen: set[str] = set()

    for item in todos:
        confidence = item.get("confidence")
        if confidence is None or confidence < 0.6:
            continue
        key = normalize_todo_text(get_todo_text(item))
        if key in seen:
            continue
        seen.add(key)
        annotated = dict(item)
        if annotated.get("category") == "dev":
            annotated["_approval_route"] = True
        kept.append(annotated)

    kept.sort(key=lambda t: t.get("priority", 0), reverse=True)
    return kept


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _unavailable_block(reason: str) -> str:
    return f"> ⚠️ unavailable ({reason})\n"


def render_journal_section(data: dict | None, reason: str) -> str:
    lines = ["## Section 1 — Journal Reflection\n"]
    if data is None:
        lines.append(_unavailable_block(reason))
        return "\n".join(lines)

    reflection = (data.get("reflection") or {}).get("markdown", "")
    lines.append(reflection if reflection else "> (no reflection today)\n")
    return "\n".join(lines)


def render_todo_section(data: dict | None, reason: str) -> str:
    lines = ["## Section 2 — Todo List\n"]
    if data is None:
        lines.append(_unavailable_block(reason))
        return "\n".join(lines)

    todos = filter_todos(data.get("todos") or [])
    if not todos:
        lines.append("> (no todos today)\n")
        return "\n".join(lines)

    for todo in todos:
        text = get_todo_text(todo)
        suffix = " <!-- route: approval -->" if todo.get("_approval_route") else ""
        lines.append(f"- {text}{suffix}")

    return "\n".join(lines)


def render_advisory(advisory) -> str:
    """Render a single perf-coach advisory as a readable bullet.

    Advisories are dicts of the shape {key, severity: "info"|"warn", text}.
    Plain strings are accepted too (legacy producers) and rendered verbatim.
    """
    if isinstance(advisory, dict):
        text = advisory.get("text", "")
        prefix = "⚠️ " if advisory.get("severity") == "warn" else ""
        return f"- {prefix}{text}"
    return f"- {advisory}"


def render_session_value(value) -> str:
    """Render a today/tomorrow/recent_wrap session object as a readable line.

    Sessions are dicts with session_type, intensity, duration_min, notes,
    and a "planned" flag. Null/missing fields are skipped. Plain strings
    (legacy producers) are rendered verbatim.
    """
    if not isinstance(value, dict):
        return str(value)

    if value.get("planned") is False:
        return "Rest / nothing planned"

    parts: list[str] = []
    for field in ("session_type", "intensity", "duration_min", "notes"):
        field_value = value.get(field)
        if field_value is None or field_value == "":
            continue
        parts.append(f"{field.replace('_', ' ')}: {field_value}")

    return ", ".join(parts) if parts else "—"


def render_training_section(data: dict | None, reason: str) -> str:
    lines = ["## Section 3 — Training\n"]
    if data is None:
        lines.append(_unavailable_block(reason))
        return "\n".join(lines)

    advisories = data.get("advisories")
    if advisories:
        for advisory in advisories:
            lines.append(render_advisory(advisory))
        return "\n".join(lines)

    # Fall back to raw fields
    fallback_parts: list[str] = []
    for field in ("today", "tomorrow", "form", "recent_wrap"):
        value = data.get(field)
        if not value:
            continue
        fallback_parts.append(f"**{field}:** {render_session_value(value)}")

    if fallback_parts:
        lines.extend(fallback_parts)
    else:
        lines.append("> (no training data today)\n")

    return "\n".join(lines)


def render_dev_report_section(data: dict | None, reason: str) -> str:
    lines = ["## Section 4 — Overnight Dev Report\n"]
    if data is None:
        lines.append(_unavailable_block(reason))
        return "\n".join(lines)

    completed = data.get("completed") or []
    needs_review = data.get("needs_review") or []
    dead_letter = data.get("dead_letter") or []

    if completed:
        lines.append("**Completed:**")
        for item in completed:
            lines.append(f"- {item}")
    else:
        lines.append("**Completed:** (none)")

    if needs_review:
        lines.append("\n**Needs Review:**")
        for item in needs_review:
            lines.append(f"- {item}")
    else:
        lines.append("\n**Needs Review:** (none)")

    if dead_letter:
        lines.append("\n**Dead Letter:**")
        for item in dead_letter:
            lines.append(f"- {item}")
    else:
        lines.append("\n**Dead Letter:** (none)")

    cost = data.get("cost", "unknown")
    lines.append(f"\nCost: {cost}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def compose_brief(
    journal_data: dict | None,
    journal_reason: str,
    perfcoach_data: dict | None,
    perfcoach_reason: str,
    commander_data: dict | None,
    commander_reason: str,
) -> str:
    sections = [
        "# Morning Brief\n",
        render_journal_section(journal_data, journal_reason),
        "",
        render_todo_section(journal_data, journal_reason),
        "",
        render_training_section(perfcoach_data, perfcoach_reason),
        "",
        render_dev_report_section(commander_data, commander_reason),
    ]
    return "\n".join(sections)


def compose_brief_from_paths(
    journal_path: str | Path,
    perfcoach_path: str | Path,
    commander_path: str | Path,
) -> str:
    """Load the three upstream contracts from disk and compose the brief.

    This is the shared entry point used by both this module's CLI and other
    callers (e.g. the Discord delivery cron script) that just want the
    finished markdown for a given set of contract paths.
    """
    journal_data, journal_reason = load_contract(journal_path)
    perfcoach_data, perfcoach_reason = load_contract(perfcoach_path)
    commander_data, commander_reason = load_contract(commander_path)

    return compose_brief(
        journal_data, journal_reason,
        perfcoach_data, perfcoach_reason,
        commander_data, commander_reason,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose the four-section morning brief from upstream contracts.",
    )
    parser.add_argument(
        "--journal-path",
        default=os.environ.get("JOURNAL_BRIEF_PATH", DEFAULT_JOURNAL_PATH),
        help=f"Path to journal_brief.latest.json (default: {DEFAULT_JOURNAL_PATH})",
    )
    parser.add_argument(
        "--perfcoach-path",
        default=os.environ.get("PERFCOACH_BRIEF_PATH", DEFAULT_PERFCOACH_PATH),
        help=f"Path to perfcoach_brief.latest.json (default: {DEFAULT_PERFCOACH_PATH})",
    )
    parser.add_argument(
        "--commander-path",
        default=os.environ.get("COMMANDER_REPORT_PATH", DEFAULT_COMMANDER_PATH),
        help=f"Path to commander_report.latest.json (default: {DEFAULT_COMMANDER_PATH})",
    )
    parser.add_argument(
        "--output",
        default=os.environ.get("MORNING_BRIEF_OUTPUT", DEFAULT_OUTPUT_PATH),
        help=f"Output file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print brief to stdout; do not write any file.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    brief = compose_brief_from_paths(args.journal_path, args.perfcoach_path, args.commander_path)

    if args.dry_run:
        print(brief)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief, encoding="utf-8")
    logger.info("Morning brief written to %s", output_path)
    print(brief)


if __name__ == "__main__":
    main()
