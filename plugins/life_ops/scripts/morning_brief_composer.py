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
import copy
import json
import logging
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

# Ensure the repo root is on sys.path so plugins.life_ops.{todo_store,away_mode}
# are importable whether this module is invoked directly (``python3
# plugins/life_ops/scripts/morning_brief_composer.py``) or imported as a
# package (tests, the cron delivery script). Mirrors morning_brief_discord.py's
# own bootstrap.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_BKK = ZoneInfo("Asia/Bangkok")

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
_CONTRACTS_DIR = _HERMES_HOME / "contracts"

DEFAULT_BRIEF_RENDER_CONFIG_PATH = str(Path(__file__).resolve().parents[1] / "brief_render.yaml")

_DEFAULT_RENDER_CONFIG = {
    "todo_section": {
        "fields": {"glyph": True, "key": True, "text": True, "recency": True},
        "text_max_chars": 64,
        "header_format": "To-do · {count} open · /done <key>",
    }
}

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
# Render config (config/brief_render.yaml)
# ---------------------------------------------------------------------------

def load_brief_render_config() -> dict:
    """Load ``config/brief_render.yaml`` (path via ``BRIEF_RENDER_CONFIG`` env
    var, else ``<repo_root>/config/brief_render.yaml``).

    Never raises: a missing file, unparsable YAML, or missing PyYAML all fall
    back to :data:`_DEFAULT_RENDER_CONFIG` (deep-copied so callers can't
    mutate the module default). Only recognised keys are honored; anything
    else in the file is ignored.
    """
    cfg = copy.deepcopy(_DEFAULT_RENDER_CONFIG)
    path_str = os.environ.get("BRIEF_RENDER_CONFIG", "").strip() or DEFAULT_BRIEF_RENDER_CONFIG_PATH
    path = Path(path_str)
    if not path.exists():
        return cfg

    try:
        import yaml
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Could not load brief render config %s (%s); using defaults", path, exc)
        return cfg

    if not isinstance(loaded, dict):
        return cfg
    todo_cfg = loaded.get("todo_section")
    if not isinstance(todo_cfg, dict):
        return cfg

    fields = todo_cfg.get("fields")
    if isinstance(fields, dict):
        for field_name in cfg["todo_section"]["fields"]:
            if field_name in fields:
                cfg["todo_section"]["fields"][field_name] = bool(fields[field_name])

    text_max_chars = todo_cfg.get("text_max_chars")
    if isinstance(text_max_chars, int) and text_max_chars > 0:
        cfg["todo_section"]["text_max_chars"] = text_max_chars

    header_format = todo_cfg.get("header_format")
    if isinstance(header_format, str) and header_format.strip():
        cfg["todo_section"]["header_format"] = header_format

    return cfg


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


def _get_open_todos_safe() -> list:
    """``todo_store.get_open_todos()``, defensively.

    Snoozed todos are never included in the store's "open" result set (they
    move to status='snoozed' via close_todo), so there is no separate
    snooze-filtering step here — the store is the single source of truth for
    what counts as "open". Any failure (store unavailable, corrupt DB, import
    error) degrades to an empty list rather than crashing the scheduled brief.
    """
    try:
        from plugins.life_ops import todo_store
    except Exception as exc:
        logger.warning("todo_store unavailable; rendering empty todo section (%s)", exc)
        return []
    try:
        return todo_store.get_open_todos()
    except Exception as exc:
        logger.warning("todo_store.get_open_todos() failed; rendering empty todo section (%s)", exc)
        return []


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars``, appending an ellipsis when cut."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1].rstrip() + "…"


def _format_recency(todo: dict, today: str) -> str:
    """``↻ {N}d`` for recurring todos (days since first_seen), else the
    latest ``source_dates`` entry as ``MM-DD``. Never raises — an
    unparseable/missing date degrades to ``"?"`` rather than crashing.
    """
    try:
        today_date = date.fromisoformat(today)
    except (TypeError, ValueError):
        today_date = None

    if todo.get("recurring"):
        first_seen = todo.get("first_seen")
        if first_seen and today_date is not None:
            try:
                first_seen_date = date.fromisoformat(str(first_seen)[:10])
                days = max(0, (today_date - first_seen_date).days)
                return f"↻ {days}d"
            except ValueError:
                pass
        return "↻ ?"

    source_dates = [d for d in (todo.get("source_dates") or []) if isinstance(d, str)]
    if not source_dates:
        return "?"
    latest = max(source_dates)
    try:
        return date.fromisoformat(latest[:10]).strftime("%m-%d")
    except ValueError:
        return latest[:10]


def _render_todo_row(todo: dict, fields: dict, text_max_chars: int, today: str) -> str:
    """One aligned row: ``  {glyph} {key:<20} {text:<N}  {recency}`` — only
    the fields enabled in ``fields`` are included.
    """
    glyph = "!" if str(todo.get("priority") or "").lower() == "high" else "·"
    key = str(todo.get("key") or "")
    text = _truncate_with_ellipsis(str(todo.get("text") or ""), text_max_chars)
    recency = _format_recency(todo, today)

    parts: list[str] = []
    if fields.get("glyph", True):
        parts.append(glyph)
    if fields.get("key", True):
        parts.append(f"{key:<20}")
    if fields.get("text", True):
        parts.append(f"{text:<{text_max_chars}}")
    if fields.get("recency", True):
        parts.append(recency)
    return "  " + " ".join(parts)


def render_todo_section(data: dict | None, reason: str) -> str:
    """Render the aligned, fixed-width todo block.

    ``data``/``reason`` are the journal contract's load result — used only
    to resolve "today" (``for_date``, falling back to system date) for
    recency computation. The row content itself comes from
    ``todo_store.get_open_todos()``, NOT the raw contract's ``todos`` array:
    the store is the persistent, authoritative record of what is actually
    open across days, while the journal contract is just that day's
    proposal. category/status/id/confidence/origin are intentionally never
    rendered here.
    """
    lines = ["## Section 2 — Todo List\n"]
    render_cfg = load_brief_render_config()["todo_section"]
    today = ((data or {}).get("for_date")) or get_today_bangkok()

    open_todos = _get_open_todos_safe()
    header = render_cfg["header_format"].format(count=len(open_todos))

    if not open_todos:
        block = f"```\n{header}\n\n(no open todos)\n```"
        lines.append(block)
        return "\n".join(lines)

    fields = render_cfg["fields"]
    text_max_chars = render_cfg["text_max_chars"]
    rows = [
        _render_todo_row(todo, fields, text_max_chars, today) for todo in open_todos
    ]
    block = "```\n" + "\n".join([header, ""] + rows) + "\n```"
    lines.append(block)
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


def _render_v3_form_line(form: dict) -> str:
    """Render the v3 fitness-metrics form dict as a single Form line."""
    line = f"**Form:** CTL {form.get('ctl')} · ATL {form.get('atl')} · TSB {form.get('tsb')}"
    acwr = form.get("acwr")
    if acwr is not None:
        line += f" · ACWR {acwr} ({form.get('acwr_state')})"
    line += f" — {form.get('interpretation')}"
    return line


def _render_v3_weight_line(weight: dict) -> str:
    """Render the v3 weight tracking dict as a single Weight line."""
    on_track_str = "on track" if weight.get("on_track") else "off pace"
    return (
        f"**Weight:** {weight.get('current_kg')}kg"
        f" · 7d {weight.get('trend_7d')}"
        f" · target {weight.get('target_kg')} by {weight.get('target_date')}"
        f" ({on_track_str})"
    )


def _render_v3_week_plan_block(week_plan: list) -> str:
    """Render the v3 week_plan list as a heading + one row per day."""
    rows = ["**Week plan:**"]
    for entry in week_plan:
        day = entry.get("day", "")
        if entry.get("planned"):
            rows.append(f"· {day}  {entry.get('session_type')} {entry.get('duration_min')}min")
        else:
            rows.append(f"· {day}  rest")
    return "\n".join(rows)


def render_training_section(data: dict | None, reason: str) -> str:
    lines = ["## Section 3 — Training\n"]
    if data is None:
        lines.append(_unavailable_block(reason))
        return "\n".join(lines)

    # v3 new blocks (AC-1, AC-2, AC-3)
    v3_blocks: list[str] = []

    form = data.get("form")
    if isinstance(form, dict):
        v3_blocks.append(_render_v3_form_line(form))

    weight = data.get("weight")
    if weight is not None:
        v3_blocks.append(_render_v3_weight_line(weight))

    week_plan = data.get("week_plan")
    if isinstance(week_plan, list) and week_plan:
        v3_blocks.append(_render_v3_week_plan_block(week_plan))

    lines.extend(v3_blocks)

    advisories = data.get("advisories")
    if advisories:
        # AC-4: heading only when at least one new block precedes advisories
        if v3_blocks:
            lines.append("**Advisories:**")
        for advisory in advisories:
            lines.append(render_advisory(advisory))
        return "\n".join(lines)

    # AC-5: Legacy fallback when advisories is absent
    v3_form_rendered = isinstance(form, dict)
    fallback_parts: list[str] = []
    for field in ("today", "tomorrow", "form", "recent_wrap"):
        # Skip form when it was already rendered as a v3 fitness-metrics line
        if field == "form" and v3_form_rendered:
            continue
        value = data.get(field)
        if not value:
            continue
        fallback_parts.append(f"**{field}:** {render_session_value(value)}")

    if fallback_parts:
        lines.extend(fallback_parts)
    elif not v3_blocks:
        lines.append("> (no training data today)\n")

    return "\n".join(lines)


_PROJECT_GLYPHS: dict[str, str] = {
    "shipped": "🚀",
    "in_progress": "⏳",
    "blocked": "⛔",
    "waiting_signoff": "📋",
    "idle": "💤",
}


def _render_project_header(project: dict) -> str:
    name = project.get("name", "")
    status = project.get("status", "")
    glyph = _PROJECT_GLYPHS.get(status, "")

    header = f"**{name}** — {glyph} {status}"

    if status == "in_progress":
        ip = project.get("in_progress")
        if isinstance(ip, dict):
            sprint_label = ip.get("sprint_label", "")
            percent = ip.get("percent", "")
            ticket = ip.get("ticket", "")
            header += f" ({sprint_label}, {percent}% — {ticket})"

    # compact counts suffix when any bucket is non-empty
    bucket_counts: list[str] = []
    for bucket_name in ("shipped", "fixed", "stale", "waiting"):
        bucket = project.get(bucket_name) or []
        if bucket:
            bucket_counts.append(f"{bucket_name[0].upper()}{len(bucket)}")
    if bucket_counts:
        header += f" [{' '.join(bucket_counts)}]"

    return header


def _render_shipped_item(item) -> str:
    if not isinstance(item, dict):
        return f"- {item}"
    label = item.get("label", "")
    goal = item.get("goal", "")
    done = item.get("done", "")
    pr_number = item.get("pr_number")
    pr_part = f", PR #{pr_number}" if pr_number is not None else ""
    return f'- Shipped: {label} "{goal}" ({done} done{pr_part})'


def _render_fixed_item(item) -> str:
    if not isinstance(item, dict):
        return f"- {item}"
    issue_number = item.get("issue_number", "")
    title = item.get("title", "")
    return f"- Fixed: #{issue_number} {title}"


def _render_stale_item(item) -> str:
    if not isinstance(item, dict):
        return f"- {item}"
    kind = item.get("kind", "")
    if kind == "blocked":
        issue_number = item.get("issue_number", "")
        age_days = item.get("age_days", "")
        type_ = item.get("type", "")
        title = item.get("title", "")
        return f"- #{issue_number} blocked {age_days}d ({type_}) — {title}"
    if kind == "waiting_signoff":
        label = item.get("label", "")
        age_days = item.get("age_days", "")
        return f"- {label} awaiting sign-off {age_days}d"
    if kind == "backlog":
        label = item.get("label", "")
        age_days = item.get("age_days", "")
        ticket_count = item.get("ticket_count", "")
        return f"- {label} backlog untouched {age_days}d ({ticket_count} tickets)"
    # unknown kind — render as much as possible
    return f"- {item}"


def _render_waiting_item(item) -> str:
    if not isinstance(item, dict):
        return f"- {item}"
    label = item.get("label", "")
    ticket_count = item.get("ticket_count", "")
    estimated_hours = item.get("estimated_hours")
    hours_part = f", ~{estimated_hours}h" if estimated_hours is not None else ""
    return f"- Waiting: {label} sign-off ({ticket_count} tickets{hours_part})"


def _render_project_block(project: dict) -> list[str]:
    lines = [_render_project_header(project)]
    for bucket_name, renderer in (
        ("shipped", _render_shipped_item),
        ("fixed", _render_fixed_item),
        ("stale", _render_stale_item),
        ("waiting", _render_waiting_item),
    ):
        for item in (project.get(bucket_name) or []):
            lines.append(renderer(item))
    return lines


def render_dev_report_section(data: dict | None, reason: str) -> str:
    lines = ["## Section 4 — Overnight Dev Report\n"]
    if data is None:
        lines.append(_unavailable_block(reason))
        return "\n".join(lines)

    projects = data.get("projects")
    if projects:
        for project in projects:
            lines.extend(_render_project_block(project))

        cost = data.get("cost", "unknown")
        lines.append(f"\nCost: {cost}")
        return "\n".join(lines)

    # Legacy flat path — unchanged
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

def _render_away_marker(for_date: str) -> str:
    """One-line away-mode marker, or ``""`` when away mode is not active.

    Never raises: any failure reaching ``away_mode`` (missing module, DB
    error) is treated as "not away" rather than crashing the scheduled brief.
    """
    try:
        from plugins.life_ops import away_mode
    except Exception:
        return ""
    try:
        status = away_mode.away_status(for_date)
    except Exception as exc:
        logger.warning("away_mode.away_status() failed; omitting away marker (%s)", exc)
        return ""
    if not status.get("active"):
        return ""
    until = status.get("until")
    if until:
        return f"> 🌙 Away mode on until {until} — overnight runs and bedtime prompts are paused.\n"
    return "> 🌙 Away mode on — overnight runs and bedtime prompts are paused.\n"


def compose_brief(
    journal_data: dict | None,
    journal_reason: str,
    perfcoach_data: dict | None,
    perfcoach_reason: str,
    commander_data: dict | None,
    commander_reason: str,
) -> str:
    for_date = (journal_data or {}).get("for_date") or get_today_bangkok()
    away_marker = _render_away_marker(for_date)

    sections = [
        "# Morning Brief\n",
    ]
    if away_marker:
        sections.append(away_marker)
    sections.extend(
        [
            render_journal_section(journal_data, journal_reason),
            "",
            render_todo_section(journal_data, journal_reason),
            "",
            render_training_section(perfcoach_data, perfcoach_reason),
            "",
            render_dev_report_section(commander_data, commander_reason),
        ]
    )
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
