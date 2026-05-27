#!/usr/bin/env python3
"""Aggregate-only classifier for Hermes provider stall symptoms.

The script deliberately reports counts and input file paths only. It does not
print raw log lines, prompts, messages, tool payloads, session ids, or user ids.

Usage:
    python scripts/provider_stall_audit.py \
        --logs-dir ~/.hermes/logs \
        --state-db ~/.hermes/state.db \
        --start-date 2026-05-25 --end-date 2026-05-27
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")
LEVEL_RE = re.compile(r"\b(WARNING|ERROR|CRITICAL)\b", re.IGNORECASE)
SESSION_ID_RE = re.compile(r"\[(?P<session>[A-Za-z0-9_:-]{8,})\]")

PROVIDER_STALL_RE = re.compile(
    # Keep this intentionally narrow: broad ``provider.*timeout`` searches are
    # exactly what made prior audits count meta-discussion and follow-on errors
    # as runtime provider stalls. Generic timeouts are counted separately.
    r"non-streaming api call stale for (?P<seconds>\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
TIMEOUT_RE = re.compile(r"\b(?:timeout|timed out|TimeoutError)\b", re.IGNORECASE)
COMPRESSION_FAILURE_RE = re.compile(
    r"(?:compression|compress|summary|summar(?:y|ies)).*"
    r"(?:fail|failed|failure|unavailable|timed out|timeout|no auxiliary|below .*threshold)|"
    r"(?:no auxiliary llm provider for compression)",
    re.IGNORECASE,
)
DISCORD_RESPONSE_RE = re.compile(
    r"response ready:\s*platform=discord\b.*?\btime=(?P<seconds>\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)
DISCORD_SEND_FAILURE_RE = re.compile(
    r"\[Discord\].*(?:failed to send|send failure|rate limited|failed to register|auto-thread creation failed)",
    re.IGNORECASE,
)
SESSION_PROVIDER_MENTION_RE = re.compile(
    r"\b(?:provider(?:_300s| stalls?| stale)?|300s|non-streaming api call stale|stale events?)\b",
    re.IGNORECASE,
)
SESSION_META_RE = re.compile(
    r"\b(?:audit|report|regex|mentions?|meta(?:-discussion)?|classifier|symptom|metric|count|evidence)\b",
    re.IGNORECASE,
)

DEFAULT_LOG_GLOBS = ("*.log", "*.log.[0-9]*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, default=Path.home() / ".hermes" / "logs")
    parser.add_argument(
        "--state-db",
        type=Path,
        action="append",
        default=[],
        help="Hermes state.db to scan for aggregate session-content mentions; repeatable.",
    )
    parser.add_argument("--start-date", help="Inclusive YYYY-MM-DD filter.")
    parser.add_argument("--end-date", help="Inclusive YYYY-MM-DD filter.")
    parser.add_argument(
        "--discord-slow-threshold-seconds",
        type=float,
        default=60.0,
        help="Threshold for Discord response-ready latency symptoms.",
    )
    parser.add_argument(
        "--provider-stale-threshold-seconds",
        type=float,
        default=300.0,
        help="Threshold for direct provider stale-call log lines.",
    )
    parser.add_argument(
        "--include-session-content-mentions",
        action="store_true",
        help="Count session-message symptom mentions separately from direct runtime logs.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def in_date_range(date: str, start: str | None, end: str | None) -> bool:
    if date == "unknown":
        return start is None and end is None
    if start and date < start:
        return False
    if end and date > end:
        return False
    return True


def date_from_log_line(line: str) -> str:
    match = DATE_RE.search(line)
    return match.group(1) if match else "unknown"


def date_from_epoch(value: float | int | None) -> str:
    if value is None:
        return "unknown"
    return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime("%Y-%m-%d")


def source_from_log_path(path: Path) -> str:
    name = path.name
    if name.startswith("gateway"):
        return "gateway_log"
    if name.startswith("agent"):
        return "agent_log"
    if name.startswith("errors"):
        return "errors_log"
    return "other_log"


def nested_counter() -> defaultdict[str, defaultdict[str, defaultdict[str, int]]]:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


def bump(counts: dict[str, Any], category: str, date: str, source: str, amount: int = 1) -> None:
    counts[category][date][source] += amount


def log_event_key(category: str, date: str, source: str, line: str, extra: str = "") -> tuple[str, ...]:
    """Return a privacy-safe in-memory key for duplicate rotated-log events."""
    timestamp = line[:23] if DATE_RE.search(line[:23]) else date
    session_match = SESSION_ID_RE.search(line)
    session_key = session_match.group("session") if session_match else "no-session"
    return (category, date, source, timestamp, session_key, extra)


def bump_once(
    counts: dict[str, Any],
    seen_events: set[tuple[str, ...]],
    category: str,
    date: str,
    source: str,
    line: str,
    extra: str = "",
) -> None:
    key = log_event_key(category, date, source, line, extra)
    if key in seen_events:
        return
    seen_events.add(key)
    bump(counts, category, date, source)


def scan_logs(
    logs_dir: Path,
    start_date: str | None,
    end_date: str | None,
    discord_slow_threshold_seconds: float,
    provider_stale_threshold_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    counts: dict[str, Any] = nested_counter()
    input_files: list[str] = []
    seen_paths: set[Path] = set()
    seen_events: set[tuple[str, ...]] = set()
    for glob in DEFAULT_LOG_GLOBS:
        for path in sorted(logs_dir.glob(glob)):
            if path in seen_paths or not path.is_file():
                continue
            seen_paths.add(path)
            input_files.append(str(path))
            source = source_from_log_path(path)
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    date = date_from_log_line(line)
                    if not in_date_range(date, start_date, end_date):
                        continue

                    provider_match = PROVIDER_STALL_RE.search(line)
                    if provider_match:
                        seconds_text = provider_match.groupdict().get("seconds")
                        if seconds_text is None or float(seconds_text) >= provider_stale_threshold_seconds:
                            bump_once(
                                counts,
                                seen_events,
                                "direct_provider_stalls",
                                date,
                                source,
                                line,
                                extra=seconds_text or "",
                            )

                    if TIMEOUT_RE.search(line):
                        bump_once(counts, seen_events, "timeout_lines", date, source, line)

                    if COMPRESSION_FAILURE_RE.search(line):
                        bump_once(
                            counts,
                            seen_events,
                            "compression_summary_failures",
                            date,
                            source,
                            line,
                        )

                    discord_match = DISCORD_RESPONSE_RE.search(line)
                    if discord_match and float(discord_match.group("seconds")) >= discord_slow_threshold_seconds:
                        bump_once(
                            counts,
                            seen_events,
                            "discord_slow_responses",
                            date,
                            source,
                            line,
                            extra=discord_match.group("seconds"),
                        )
                    if DISCORD_SEND_FAILURE_RE.search(line):
                        bump_once(counts, seen_events, "discord_send_failures", date, source, line)
    return counts, input_files


def has_session_schema(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {row[0] for row in rows}
    return {"sessions", "messages"}.issubset(names)


def scan_session_mentions(
    state_dbs: Iterable[Path], start_date: str | None, end_date: str | None
) -> tuple[dict[str, Any], list[str]]:
    counts: dict[str, Any] = nested_counter()
    input_files: list[str] = []
    for db_path in state_dbs:
        if not db_path.exists() or not db_path.is_file():
            continue
        input_files.append(str(db_path))
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            if not has_session_schema(conn):
                continue
            query = """
                SELECT s.source, m.role, m.content, m.tool_name, m.timestamp
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE m.content IS NOT NULL
                  AND (
                    m.content LIKE '%provider%'
                    OR m.content LIKE '%300s%'
                    OR m.content LIKE '%stale%'
                    OR m.content LIKE '%non-streaming API call stale%'
                  )
            """
            for source, role, content, tool_name, timestamp in conn.execute(query):
                date = date_from_epoch(timestamp)
                if not in_date_range(date, start_date, end_date):
                    continue
                text = content or ""
                if not SESSION_PROVIDER_MENTION_RE.search(text):
                    continue
                source_label = str(source or "unknown")
                if role == "tool" or tool_name:
                    category = "session_tool_log_mentions"
                elif SESSION_META_RE.search(text):
                    category = "session_meta_discussion_mentions"
                else:
                    category = "session_unclassified_provider_mentions"
                bump(counts, category, date, source_label)
        finally:
            conn.close()
    return counts, input_files


def freeze(value: Any) -> Any:
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {key: freeze(value[key]) for key in sorted(value)}
    return value


def merge_counts(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for category, by_date in extra.items():
        for date, by_source in by_date.items():
            for source, count in by_source.items():
                bump(base, category, date, source, count)


def totals_by_category(counts: dict[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for category, by_date in counts.items():
        totals[category] = sum(sum(by_source.values()) for by_source in by_date.values())
    return dict(sorted(totals.items()))


def main() -> int:
    args = parse_args()
    counts, log_files = scan_logs(
        logs_dir=args.logs_dir.expanduser(),
        start_date=args.start_date,
        end_date=args.end_date,
        discord_slow_threshold_seconds=args.discord_slow_threshold_seconds,
        provider_stale_threshold_seconds=args.provider_stale_threshold_seconds,
    )
    session_files: list[str] = []
    if args.include_session_content_mentions:
        session_counts, session_files = scan_session_mentions(
            [path.expanduser() for path in args.state_db], args.start_date, args.end_date
        )
        merge_counts(counts, session_counts)

    frozen_counts = freeze(counts)
    output = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "scope": {
            "start_date": args.start_date,
            "end_date": args.end_date,
            "discord_slow_threshold_seconds": args.discord_slow_threshold_seconds,
            "provider_stale_threshold_seconds": args.provider_stale_threshold_seconds,
        },
        "privacy": "aggregate counts and input file paths only; raw messages/prompts/log lines are never emitted",
        "input_files": {
            "logs": sorted(log_files),
            "state_dbs": sorted(session_files),
        },
        "counts": frozen_counts,
        "totals_by_category": totals_by_category(frozen_counts),
    }
    print(json.dumps(output, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
