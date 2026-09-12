"""Run-level usage snapshots and per-task aggregation for Kanban."""

from __future__ import annotations

import time
from typing import Any, Optional


_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "api_call_count",
    "turns",
)
def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _nonnegative_float(value: Any, *, nullable: bool = False) -> Optional[float]:
    if value is None and nullable:
        return None
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError, OverflowError):
        return None if nullable else 0.0


def normalized_run_usage(usage: Optional[dict]) -> dict[str, Any]:
    """Return a bounded, column-ready snapshot; absent usage becomes zeroes."""
    raw = usage if isinstance(usage, dict) else {}
    recorded_at = raw.get("usage_recorded_at")
    return {
        **{field: _nonnegative_int(raw.get(field)) for field in _TOKEN_FIELDS},
        "estimated_cost_usd": _nonnegative_float(raw.get("estimated_cost_usd")),
        "auxiliary_estimated_cost_usd": _nonnegative_float(raw.get("auxiliary_estimated_cost_usd")),
        "actual_cost_usd": _nonnegative_float(raw.get("actual_cost_usd"), nullable=True),
        "session_id": str(raw.get("session_id") or "").strip() or None,
        "model": str(raw.get("model") or "").strip() or None,
        "provider": str(raw.get("provider") or "").strip() or None,
        "usage_recorded_at": (
            _nonnegative_int(recorded_at)
            if recorded_at is not None
            else (int(time.time()) if raw else None)
        ),
    }


def task_usage(conn, task_id: str) -> dict[str, Any]:
    """Aggregate immutable closed-run snapshots, including profile segments."""
    sum_columns = ", ".join(
        f"COALESCE(SUM({field}), 0) AS {field}"
        for field in (*_TOKEN_FIELDS, "estimated_cost_usd", "auxiliary_estimated_cost_usd")
    )
    preferred_cost = (
        "COALESCE(SUM(CASE WHEN actual_cost_usd IS NOT NULL "
        "THEN actual_cost_usd + auxiliary_estimated_cost_usd "
        "ELSE estimated_cost_usd END), 0) AS cost_usd"
    )

    def _summary(where: str, params: tuple[Any, ...]) -> dict[str, Any]:
        row = conn.execute(
            f"SELECT COUNT(*) AS runs, {sum_columns}, {preferred_cost} FROM task_runs WHERE {where}",
            params,
        ).fetchone()
        return {
            "runs": int(row["runs"] or 0),
            **{field: int(row[field] or 0) for field in _TOKEN_FIELDS},
            "estimated_cost_usd": float(row["estimated_cost_usd"] or 0.0),
            "auxiliary_estimated_cost_usd": float(row["auxiliary_estimated_cost_usd"] or 0.0),
            "cost_usd": float(row["cost_usd"] or 0.0),
        }

    total = _summary("task_id = ? AND ended_at IS NOT NULL", (task_id,))
    profiles = []
    rows = conn.execute(
        "SELECT DISTINCT COALESCE(profile, '') AS profile FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL ORDER BY profile",
        (task_id,),
    ).fetchall()
    for row in rows:
        profile = row["profile"] or None
        segment = _summary(
            "task_id = ? AND ended_at IS NOT NULL AND COALESCE(profile, '') = ?",
            (task_id, row["profile"]),
        )
        profiles.append({"profile": profile, **segment})
    return {**total, "profiles": profiles}
