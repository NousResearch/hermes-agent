"""Derive metric rollups from the local telemetry tables.

Reads the ``tel_*`` tables in state.db and returns aggregates for /usage, /insights,
and local dashboards. Metrics are computed by querying the event log rather than being
emitted on the hot path.

Each function accepts either an open caller-owned ``conn`` (reused, not closed) or a
``db_path`` (opened and closed internally). InsightsEngine passes its existing
connection; a standalone dashboard passes a path.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@contextmanager
def _cursor(
    conn: Optional[sqlite3.Connection], db_path: Optional[Path]
) -> Iterator[sqlite3.Connection]:
    """Yield a Row-factory connection. Closes it only if we opened it."""
    if conn is not None:
        prev_factory = conn.row_factory
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.row_factory = prev_factory
        return
    if db_path is None:
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "state.db"
    c = sqlite3.connect(str(db_path), timeout=5.0)
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def _since_clause(since_ns: Optional[int], col: str = "start_ns") -> str:
    return f" WHERE {col} >= {int(since_ns)}" if since_ns else ""


def workflow_summary(
    db_path: Optional[Path] = None,
    since_ns: Optional[int] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Run-level counters + duration percentiles (local plane, exact)."""
    with _cursor(conn, db_path) as c:
        where = _since_clause(since_ns)
        total = c.execute(f"SELECT COUNT(*) n FROM tel_runs{where}").fetchone()["n"]
        by_reason = {
            r["end_reason"] or "unknown": r["n"]
            for r in c.execute(
                f"SELECT end_reason, COUNT(*) n FROM tel_runs{where} GROUP BY end_reason"
            ).fetchall()
        }
        by_entry = {
            r["entrypoint"] or "unknown": r["n"]
            for r in c.execute(
                f"SELECT entrypoint, COUNT(*) n FROM tel_runs{where} GROUP BY entrypoint"
            ).fetchall()
        }
        dur_where = (where + " AND end_ns IS NOT NULL") if where else " WHERE end_ns IS NOT NULL"
        durations = [
            (r["end_ns"] - r["start_ns"]) / 1e6
            for r in c.execute(
                f"SELECT start_ns, end_ns FROM tel_runs{dur_where}"
            ).fetchall()
        ]
        return {
            "total_runs": total,
            "by_end_reason": by_reason,
            "by_entrypoint": by_entry,
            "duration_ms_p50": _pct(durations, 50),
            "duration_ms_p95": _pct(durations, 95),
            "success_rate": round(by_reason.get("completed", 0) / total, 4) if total else 0.0,
        }


def model_call_summary(
    db_path: Optional[Path] = None,
    since_ns: Optional[int] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    with _cursor(conn, db_path) as c:
        rows = c.execute(
            "SELECT provider, model, COUNT(*) n, "
            "SUM(input_tokens) inp, SUM(output_tokens) outp, "
            "SUM(cache_read_tokens) cache, AVG(latency_ms) avg_latency "
            "FROM tel_model_calls GROUP BY provider, model"
        ).fetchall()
        by_provider: Dict[str, int] = {}
        by_model: Dict[str, int] = {}
        tokens = {"input": 0, "output": 0, "cache_read": 0}
        breakdown: List[Dict[str, Any]] = []
        for r in rows:
            prov = r["provider"] or "unknown"
            mdl = r["model"] or "unknown"
            by_provider[prov] = by_provider.get(prov, 0) + r["n"]
            by_model[mdl] = by_model.get(mdl, 0) + r["n"]
            tokens["input"] += r["inp"] or 0
            tokens["output"] += r["outp"] or 0
            tokens["cache_read"] += r["cache"] or 0
            breakdown.append({
                "provider": r["provider"],
                "model": r["model"],
                "calls": r["n"],
                "avg_latency_ms": round(r["avg_latency"] or 0, 1),
            })
        cache_total = tokens["cache_read"] + tokens["input"]
        return {
            "by_provider": by_provider,
            "by_model": by_model,
            "tokens": tokens,
            "cache_hit_rate": round(tokens["cache_read"] / cache_total, 4) if cache_total else 0.0,
            "breakdown": breakdown,
        }


def tool_call_summary(
    db_path: Optional[Path] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    with _cursor(conn, db_path) as c:
        by_tool = {
            r["tool_name"] or "unknown": r["n"]
            for r in c.execute(
                "SELECT tool_name, COUNT(*) n FROM tel_tool_calls GROUP BY tool_name"
            ).fetchall()
        }
        fails = {
            r["tool_name"] or "unknown": r["n"]
            for r in c.execute(
                "SELECT tool_name, COUNT(*) n FROM tel_tool_calls "
                "WHERE result_class IN ('error','timeout','blocked') GROUP BY tool_name"
            ).fetchall()
        }
        total = sum(by_tool.values())
        total_fail = sum(fails.values())
        return {
            "by_tool": by_tool,
            "failures_by_tool": fails,
            "total": total,
            "failure_rate": round(total_fail / total, 4) if total else 0.0,
        }


def error_summary(
    db_path: Optional[Path] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    with _cursor(conn, db_path) as c:
        return {
            "by_class": {
                r["error_class"] or "unknown": r["n"]
                for r in c.execute(
                    "SELECT error_class, COUNT(*) n FROM tel_error_events GROUP BY error_class"
                ).fetchall()
            },
        }


def _pct(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 2)


def overview(
    db_path: Optional[Path] = None,
    since_ns: Optional[int] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """One call for a dashboard: all the rollups."""
    return {
        "workflows": workflow_summary(db_path, since_ns, conn=conn),
        "model_calls": model_call_summary(db_path, since_ns, conn=conn),
        "tool_calls": tool_call_summary(db_path, conn=conn),
        "errors": error_summary(db_path, conn=conn),
    }


def has_data(
    db_path: Optional[Path] = None,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """True when any telemetry runs exist (cheap guard for /insights rendering)."""
    try:
        with _cursor(conn, db_path) as c:
            return c.execute("SELECT 1 FROM tel_runs LIMIT 1").fetchone() is not None
    except Exception:
        return False


__all__ = [
    "workflow_summary",
    "model_call_summary",
    "tool_call_summary",
    "error_summary",
    "overview",
    "has_data",
]
