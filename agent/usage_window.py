"""Read exact usage-event windows with an explicit legacy-coverage boundary.

The event ledger owns sessions that have any recorded event. Their lifetime
aggregates must never be charged again to a shorter reporting window. Older
sessions without events retain the historical session-start-window fallback.
All helpers are read-only; no analytics request creates or migrates a database.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import sqlite3
from typing import Any

from hermes_state_common import USAGE_EVENTS_COVERAGE_START_KEY

_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "api_call_count")
_COSTS = ("estimated_cost_usd", "actual_cost_usd")
_ROUTES = ("model", "billing_provider", "billing_base_url", "billing_mode", "task")


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _dict_rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    keys = [column[0] for column in cursor.description or ()]
    return [dict(zip(keys, row)) for row in cursor.fetchall()]


def get_window_usage_rows(conn: sqlite3.Connection, cutoff: float, source: str | None = None) -> list[dict[str, Any]]:
    """Return exact in-window events plus nonoverlapping legacy route totals."""
    has_events = _has_table(conn, "session_model_usage_events")
    source_sql = " AND s.source=?" if source else ""
    params = [cutoff, source] if source else [cutoff]
    result: list[dict[str, Any]] = []
    if has_events:
        result = _dict_rows(conn.execute(
            "SELECT e.*, s.source AS source, 'event' AS usage_origin, e.recorded_at AS last_seen "
            "FROM session_model_usage_events e JOIN sessions s ON s.id=e.session_id "
            "WHERE e.recorded_at>=?" + source_sql, params))
    no_events = " AND NOT EXISTS (SELECT 1 FROM session_model_usage_events e WHERE e.session_id=s.id)" if has_events else ""
    sessions = _dict_rows(conn.execute("SELECT s.* FROM sessions s WHERE s.started_at>=?" + no_events + source_sql, params))
    usage_by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if _has_table(conn, "session_model_usage"):
        for row in _dict_rows(conn.execute(
            "SELECT u.*, s.source AS source, s.started_at AS recorded_at FROM session_model_usage u JOIN sessions s ON s.id=u.session_id "
            "WHERE s.started_at>=?" + no_events + source_sql, params)):
            row.setdefault("task", "")
            row["usage_origin"] = "legacy"
            usage_by_session[row["session_id"]].append(row)
    for session in sessions:
        rows = usage_by_session[session["id"]]
        result.extend(rows)
        # Auxiliary rows never contribute to the session's main-loop counters.
        main = [row for row in rows if not row.get("task")]
        residual = {key: max(0, (session.get(key) or 0) - sum(row.get(key) or 0 for row in main)) for key in (*_COUNTERS, *_COSTS)}
        if any(residual.values()) or not rows:
            result.append({**{key: session.get(key) or "" for key in _ROUTES}, **residual,
                           "session_id": session["id"], "source": session.get("source"), "task": "", "recorded_at": session["started_at"],
                           "last_seen": session["started_at"], "usage_origin": "legacy",
                           "cost_status": session.get("cost_status"), "cost_source": session.get("cost_source")})
    return result


def get_usage_window_coverage(conn: sqlite3.Connection, cutoff: float, *, usage_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Describe exact-ledger availability without fabricating historical events."""
    available = _has_table(conn, "session_model_usage_events")
    start = None
    if _has_table(conn, "state_meta"):
        row = conn.execute("SELECT value FROM state_meta WHERE key=?", (USAGE_EVENTS_COVERAGE_START_KEY,)).fetchone()
        if row is not None:
            try:
                start = float(row[0])
            except (TypeError, ValueError):
                pass
    rows = usage_rows if usage_rows is not None else get_window_usage_rows(conn, cutoff)
    legacy = any(row.get("usage_origin") == "legacy" for row in rows)
    return {"coverage_start": start, "exact_events_available": available,
            "legacy_fallback_used": legacy, "window_complete": bool(available and start is not None and cutoff >= start and not legacy)}


def _aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) or ("unknown" if field == "model" else "") for field in keys)
        target = grouped.setdefault(key, {**dict(zip(keys, key)), **dict.fromkeys((*_COUNTERS, *_COSTS), 0), "_sessions": set(), "last_used_at": 0})
        target["_sessions"].add(row["session_id"])
        for field in (*_COUNTERS, *_COSTS):
            target[field] += row.get(field) or 0
        target["last_used_at"] = max(target["last_used_at"], row.get("recorded_at") or row.get("last_seen") or 0)
    result = []
    for target in grouped.values():
        target["sessions"] = len(target.pop("_sessions"))
        target["estimated_cost"] = target.pop("estimated_cost_usd")
        target["actual_cost"] = target.pop("actual_cost_usd")
        target["api_calls"] = target.pop("api_call_count")
        result.append(target)
    return sorted(result, key=lambda row: row["input_tokens"] + row["output_tokens"], reverse=True)


def aggregate_window_usage(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Aggregate dashboard day/model/totals with unique session counts."""
    dated = [{**row, "day": datetime.fromtimestamp(float(row.get("recorded_at") or row.get("last_seen") or 0), timezone.utc).date().isoformat()} for row in rows]
    daily = sorted(_aggregate(dated, ("day",)), key=lambda row: row["day"])
    by_model = _aggregate(rows, ("model",))
    totals = {"total_input": sum(row.get("input_tokens") or 0 for row in rows),
              "total_output": sum(row.get("output_tokens") or 0 for row in rows),
              "total_cache_read": sum(row.get("cache_read_tokens") or 0 for row in rows),
              "total_cache_write": sum(row.get("cache_write_tokens") or 0 for row in rows),
              "total_reasoning": sum(row.get("reasoning_tokens") or 0 for row in rows),
              "total_estimated_cost": sum(row.get("estimated_cost_usd") or 0 for row in rows),
              "total_actual_cost": sum(row.get("actual_cost_usd") or 0 for row in rows),
              "total_api_calls": sum(row.get("api_call_count") or 0 for row in rows),
              "total_sessions": len({row["session_id"] for row in rows})}
    return daily, by_model, totals


def aggregate_aux_usage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Auxiliary calls only, grouped by model, task, and provider."""
    return _aggregate([row for row in rows if row.get("task")], ("model", "task", "billing_provider"))


def aggregate_model_usage(rows: list[dict[str, Any]], tool_rows: list[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    """Aggregate model cards without collapsing auxiliary task routes."""
    # ``task`` is part of the historical Models API grouping contract: an
    # auxiliary call on the same model/provider remains a distinct row.
    result = _aggregate(rows, ("model", "billing_provider", "task"))
    by_key = {(row["model"], row["billing_provider"], row["task"]): row for row in result}
    for row in result:
        row["tool_calls"] = 0
        row["avg_tokens_per_session"] = (
            (row["input_tokens"] + row["output_tokens"]) / max(row["sessions"], 1)
            if not row["task"] else 0
        )
    # Tool counts are session-start metrics and have no auxiliary task route;
    # attach them only to the main (empty-task) model row.
    for tool_row in tool_rows:
        row = by_key.get((tool_row.get("model") or "unknown", tool_row.get("billing_provider") or "", ""))
        if row is not None:
            row["tool_calls"] += tool_row.get("tool_calls") or 0
    return result
