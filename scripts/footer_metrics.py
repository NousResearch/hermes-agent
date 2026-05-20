from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_state_db_path(payload: dict[str, Any]) -> Path:
    hermes_home = (
        payload.get("hermes_home")
        or os.environ.get("HERMES_HOME")
        or os.path.expanduser("~/.hermes")
    )
    return Path(os.path.expanduser(str(hermes_home))).resolve() / "state.db"


def compute_footer_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        return {}

    db_path = _resolve_state_db_path(payload)
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_write_tokens,
                reasoning_tokens,
                estimated_cost_usd,
                actual_cost_usd,
                cost_status,
                api_call_count
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {}

    prompt_tokens = _coerce_int(row["input_tokens"])
    completion_tokens = _coerce_int(row["output_tokens"])
    cache_read_tokens = _coerce_int(row["cache_read_tokens"])
    cache_write_tokens = _coerce_int(row["cache_write_tokens"])
    reasoning_tokens = _coerce_int(row["reasoning_tokens"])
    api_calls = _coerce_int(row["api_call_count"])

    actual_cost = _coerce_float(row["actual_cost_usd"])
    estimated_cost = _coerce_float(row["estimated_cost_usd"])
    chosen_cost = actual_cost if actual_cost is not None else estimated_cost

    cost_status = str(row["cost_status"] or "").strip()
    if actual_cost is not None and not cost_status:
        cost_status = "actual"

    metrics: dict[str, Any] = {
        "total_tokens": prompt_tokens + completion_tokens,
        "api_calls": api_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "reasoning_tokens": reasoning_tokens,
    }

    if chosen_cost is not None:
        metrics["estimated_cost_usd"] = cast(float, chosen_cost)
    if cost_status:
        metrics["cost_status"] = cost_status

    return metrics


def main() -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print("{}")
        return 0

    try:
        payload = json.loads(raw)
    except Exception:
        print("{}")
        return 0

    if not isinstance(payload, dict):
        print("{}")
        return 0

    try:
        result = compute_footer_metrics(payload)
    except Exception:
        result = {}

    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
