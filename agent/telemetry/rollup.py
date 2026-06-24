"""Build per-run summary events from the local telemetry tables.

Reads the ``tel_*`` tables and projects each completed run into a summary dict holding
the recorded values: provider, models used, tool names, token totals, duration, and
cost. Powers ``hermes telemetry preview``. No aggregation or bucketing is applied here.
"""

from __future__ import annotations

import platform
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


def _os_family() -> str:
    s = platform.system().lower()
    if s.startswith("lin"):
        return "linux"
    if s == "darwin":
        return "macos"
    if s.startswith("win"):
        return "windows"
    return "other"


def _hermes_version() -> str:
    try:
        from hermes_cli import __version__
        return str(__version__)
    except Exception:
        return "0.0.0"


def _open(db_path: Optional[Path], conn: Optional[sqlite3.Connection]):
    if conn is not None:
        prev = conn.row_factory
        conn.row_factory = sqlite3.Row
        return conn, prev, False
    if db_path is None:
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "state.db"
    c = sqlite3.connect(str(db_path), timeout=5.0)
    c.row_factory = sqlite3.Row
    return c, None, True


def _run_events(c: sqlite3.Connection, since_ns: Optional[int]) -> List[Dict[str, Any]]:
    """Project completed runs into per-run summary dicts."""
    where = " WHERE end_ns IS NOT NULL"
    if since_ns:
        where += f" AND start_ns >= {int(since_ns)}"
    rows = c.execute(
        "SELECT run_id, entrypoint, platform, end_reason, start_ns, end_ns, "
        "model_call_count, tool_call_count, error_count, estimated_cost_usd "
        "FROM tel_runs" + where
    ).fetchall()

    events: List[Dict[str, Any]] = []
    for r in rows:
        # Models actually used in this run (real ids), with token totals.
        models = [
            {"provider": m["provider"], "model": m["model"],
             "calls": m["n"], "input_tokens": int(m["inp"] or 0),
             "output_tokens": int(m["outp"] or 0)}
            for m in c.execute(
                "SELECT provider, model, COUNT(*) n, SUM(input_tokens) inp, "
                "SUM(output_tokens) outp FROM tel_model_calls WHERE run_id = ? "
                "GROUP BY provider, model ORDER BY n DESC",
                (r["run_id"],),
            ).fetchall()
        ]
        tools = [
            row["tool_name"]
            for row in c.execute(
                "SELECT DISTINCT tool_name FROM tel_tool_calls WHERE run_id = ?",
                (r["run_id"],),
            ).fetchall()
            if row["tool_name"]
        ]
        trow = c.execute(
            "SELECT SUM(input_tokens) inp, SUM(output_tokens) outp "
            "FROM tel_model_calls WHERE run_id = ?",
            (r["run_id"],),
        ).fetchone()
        duration_ms = (r["end_ns"] - r["start_ns"]) / 1e6 if r["end_ns"] else None
        events.append({
            "event_name": "workflow_completed",
            "run_id": r["run_id"],
            "entrypoint": r["entrypoint"] or "cli",
            "platform": r["platform"],
            "end_reason": r["end_reason"] or "completed",
            "models_used": models,
            "tools_used": tools,
            "model_call_count": r["model_call_count"] or 0,
            "tool_call_count": r["tool_call_count"] or 0,
            "error_count": r["error_count"] or 0,
            "duration_ms": round(duration_ms, 1) if duration_ms is not None else None,
            "input_tokens": int((trow["inp"] if trow else 0) or 0),
            "output_tokens": int((trow["outp"] if trow else 0) or 0),
            "estimated_cost_usd": r["estimated_cost_usd"],
        })
    return events


def build_aggregate_events(
    *,
    install_id: str,
    db_path: Optional[Path] = None,
    since_ns: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
    include_heartbeat: bool = True,
) -> List[Dict[str, Any]]:
    """Return per-run summary events plus an optional heartbeat."""
    c, prev_factory, owned = _open(db_path, conn)
    try:
        events = _run_events(c, since_ns)
        if include_heartbeat:
            events.append({
                "event_name": "heartbeat",
                "install_id": install_id,
                "hermes_version": _hermes_version(),
                "os_family": _os_family(),
                "entrypoint": "cli",
            })
        return events
    finally:
        if owned:
            c.close()
        elif prev_factory is not None:
            c.row_factory = prev_factory


def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Counts by event_name + field coverage, for status/preview output."""
    by_name: Dict[str, int] = {}
    fields = set()
    for e in events:
        name = e.get("event_name", "?")
        by_name[name] = by_name.get(name, 0) + 1
        fields.update(e.keys())
    return {"total": len(events), "by_event_name": by_name, "fields_present": sorted(fields)}


__all__ = ["build_aggregate_events", "summarize"]
