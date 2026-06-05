#!/usr/bin/env python3
"""Record structured Hermes task-run telemetry from ``state.db``.

This helper intentionally stores metrics and compact labels only. It does not
copy raw transcript text, tool output, prompts, secrets, or message content into
telemetry files.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from hermes_constants import get_hermes_home

DEFAULT_ROOT = get_hermes_home() / "ops" / "self-improvement-log"
DEFAULT_DB = get_hermes_home() / "state.db"
TASK_RUNS_FILENAME = "task_runs.jsonl"
EVENTS_FILENAME = "events.jsonl"


def latest_session_id(
    con: sqlite3.Connection,
    source: str | None = None,
    *,
    include_open: bool = False,
) -> str:
    """Return the newest matching session id.

    Ended sessions are selected by default so a live review session is not
    accidentally logged as the task that just completed.
    """
    clauses: list[str] = []
    params: list[Any] = []
    if source:
        clauses.append("source = ?")
        params.append(source)
    if not include_open:
        clauses.append("ended_at is not null")
    where = f"where {' and '.join(clauses)}" if clauses else ""
    row = con.execute(
        f"select id from sessions {where} order by started_at desc limit 1",
        params,
    ).fetchone()
    if not row:
        raise SystemExit("No matching Hermes session found")
    return str(row[0])


def compact_tool_names(con: sqlite3.Connection, session_id: str) -> list[str]:
    rows = con.execute(
        """
        select tool_name, count(*) as count
        from messages
        where session_id = ? and tool_name is not null and tool_name != ''
        group by tool_name
        order by count(*) desc, tool_name asc
        """,
        (session_id,),
    ).fetchall()
    return [f"{row['tool_name']}:{int(row['count'] or 0)}" for row in rows]


def role_stats(con: sqlite3.Connection, session_id: str) -> dict[str, dict[str, int]]:
    rows = con.execute(
        """
        select role, count(*) as count, sum(length(coalesce(content, ''))) as chars
        from messages
        where session_id = ?
        group by role
        """,
        (session_id,),
    ).fetchall()
    return {
        str(row["role"]): {
            "count": int(row["count"] or 0),
            "chars": int(row["chars"] or 0),
        }
        for row in rows
    }


def top_context_items(
    con: sqlite3.Connection,
    session_id: str,
    *,
    limit: int = 8,
) -> list[str]:
    """Return compact role/tool/size labels for the largest stored messages."""
    rows = con.execute(
        """
        select role, coalesce(tool_name, '') as tool_name, length(coalesce(content, '')) as chars
        from messages
        where session_id = ?
        order by length(coalesce(content, '')) desc
        limit ?
        """,
        (session_id, limit),
    ).fetchall()
    return [
        f"{row['role']}:{row['tool_name'] or '-'}:{int(row['chars'] or 0)}"
        for row in rows
    ]


def _safe_int(row: sqlite3.Row, key: str) -> int:
    try:
        return int(row[key] or 0)
    except (IndexError, KeyError):
        return 0


def _safe_float(row: sqlite3.Row, key: str) -> float:
    try:
        return float(row[key] or 0.0)
    except (IndexError, KeyError):
        return 0.0


def _safe_str(row: sqlite3.Row, key: str) -> str:
    try:
        return str(row[key] or "")
    except (IndexError, KeyError):
        return ""


def build_task_run(con: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    session = con.execute("select * from sessions where id = ?", (session_id,)).fetchone()
    if session is None:
        raise SystemExit(f"Session not found: {session_id}")

    return {
        "schema_version": 1,
        "kind": "task_run_telemetry",
        "captured_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "session_id": session_id,
        "source": _safe_str(session, "source"),
        "title": _safe_str(session, "title"),
        "started_at": _safe_float(session, "started_at"),
        "ended_at": _safe_float(session, "ended_at"),
        "end_reason": _safe_str(session, "end_reason"),
        "model": _safe_str(session, "model"),
        "message_count": _safe_int(session, "message_count"),
        "tool_call_count": _safe_int(session, "tool_call_count"),
        "api_call_count": _safe_int(session, "api_call_count"),
        "input_tokens": _safe_int(session, "input_tokens"),
        "output_tokens": _safe_int(session, "output_tokens"),
        "reasoning_tokens": _safe_int(session, "reasoning_tokens"),
        "cache_read_tokens": _safe_int(session, "cache_read_tokens"),
        "cache_write_tokens": _safe_int(session, "cache_write_tokens"),
        "estimated_cost_usd": _safe_float(session, "estimated_cost_usd"),
        "actual_cost_usd": _safe_float(session, "actual_cost_usd"),
        "cost_status": _safe_str(session, "cost_status"),
        "cost_source": _safe_str(session, "cost_source"),
        "role_stats": role_stats(con, session_id),
        "tool_stats": compact_tool_names(con, session_id),
        "largest_context_items": top_context_items(con, session_id),
        "attribution_method": "explicit_session_id",
    }


def session_already_logged(path: Path, session_id: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("session_id") == session_id:
                return True
    return False


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def build_compact_event(task_run: dict[str, Any], *, source: str, route: str) -> dict[str, Any]:
    measures = {
        "tool_calls": task_run["tool_call_count"],
        "files_changed": 0,
        "memories_changed": 0,
        "skills_changed": 0,
        "verification": "telemetry_readback",
        "session_id": task_run["session_id"],
        "model": task_run["model"],
        "input_tokens": task_run["input_tokens"],
        "output_tokens": task_run["output_tokens"],
        "reasoning_tokens": task_run["reasoning_tokens"],
        "cache_read_tokens": task_run["cache_read_tokens"],
        "cache_write_tokens": task_run["cache_write_tokens"],
        "api_calls": task_run["api_call_count"],
        "estimated_cost_usd": task_run["estimated_cost_usd"],
        "actual_cost_usd": task_run["actual_cost_usd"],
        "cost_status": task_run["cost_status"],
        "cost_source": task_run["cost_source"],
        "tool_names": task_run["tool_stats"],
        "role_stats": task_run["role_stats"],
        "largest_context_items": task_run["largest_context_items"],
    }
    return {
        "schema_version": 1,
        "ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": source,
        "route": route,
        "intention": "Record structured Hermes task-run telemetry for review.",
        "actions": [
            "captured model/token/cost counters from sessions table",
            "captured compact tool, role, and largest-context labels without raw transcript content",
        ],
        "skills_used": [],
        "skill_performance": [],
        "measures": measures,
        "outcome": "Session telemetry was captured from Hermes state.db without raw transcript content.",
        "end_result": "Task-run metrics are available for later self-improvement review.",
        "improvement_candidate": "Promote task-run telemetry into a Hermes runtime hook/plugin if manual capture remains useful.",
        "follow_up": "Decide whether to capture per-turn deltas automatically instead of whole-session aggregates.",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to Hermes state.db")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Self-improvement log directory")
    parser.add_argument("--session-id", default="", help="Specific session id to record")
    parser.add_argument("--source", default="discord", help="Session source filter when --session-id is omitted")
    parser.add_argument("--include-open", action="store_true", help="Allow selecting an open/current session")
    parser.add_argument("--dry-run", action="store_true", help="Print telemetry without writing JSONL")
    parser.add_argument("--force", action="store_true", help="Append even if this session_id is already logged")
    parser.add_argument("--append-event", action="store_true", help="Also append a compact workflow event to events.jsonl")
    parser.add_argument("--route", default="workflow_improvement", help="Route label for --append-event")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    task_runs_path = root / TASK_RUNS_FILENAME
    events_path = root / EVENTS_FILENAME

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    try:
        session_id = args.session_id or latest_session_id(
            con,
            args.source,
            include_open=args.include_open,
        )
        task_run = build_task_run(con, session_id)
        if not args.session_id:
            task_run["attribution_method"] = "latest_open_allowed" if args.include_open else "latest_ended"

        if args.dry_run:
            print(json.dumps(task_run, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        if not args.force and session_already_logged(task_runs_path, session_id):
            print(json.dumps({"status": "already_logged", "session_id": session_id}, sort_keys=True))
            return 0

        append_jsonl(task_runs_path, task_run)
        if args.append_event:
            append_jsonl(events_path, build_compact_event(task_run, source=args.source, route=args.route))

        print(
            json.dumps(
                {"status": "logged", "task_runs": str(task_runs_path), "session_id": session_id},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
