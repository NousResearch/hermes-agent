"""Month-to-date usage per billing provider, measured against ``usage.budgets``.

A token plan is budgeted in tokens and pay-as-you-go in dollars. Most sessions on a token plan carry no
price at all (``cost_status = 'unknown'``), so tokens are always counted, money only where it is known,
and the number of unpriced sessions is reported instead of being shown as $0.
"""
from __future__ import annotations

import calendar
import sqlite3
from datetime import datetime, timedelta

# "Tokens used" is CanonicalUsage.total_tokens(): input excludes cache reads/writes, so nothing doubles.
_MAIN_SQL = """
    SELECT COALESCE(NULLIF(billing_provider, ''), 'unknown') AS provider,
           SUM(COALESCE(input_tokens, 0) + COALESCE(cache_read_tokens, 0)
               + COALESCE(cache_write_tokens, 0) + COALESCE(output_tokens, 0)) AS tokens,
           COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost,
           COALESCE(SUM(actual_cost_usd), 0) AS actual_cost,
           COUNT(*) AS sessions,
           SUM(CASE WHEN (cost_status IS NULL OR cost_status = 'unknown')
                         AND COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0) > 0
                    THEN 1 ELSE 0 END) AS unpriced_sessions
    FROM sessions WHERE started_at >= ? GROUP BY 1
"""
# Auxiliary calls (compression, vision, ...) never touch the sessions counters: add-only, no double count.
_AUX_SQL = """
    SELECT COALESCE(NULLIF(u.billing_provider, ''), 'unknown') AS provider,
           SUM(u.input_tokens + u.cache_read_tokens + u.cache_write_tokens + u.output_tokens) AS tokens,
           COALESCE(SUM(u.estimated_cost_usd), 0) AS estimated_cost
    FROM session_model_usage u JOIN sessions s ON s.id = u.session_id
    WHERE s.started_at >= ? AND u.task != '' GROUP BY 1
"""


def _empty_row(provider: str) -> dict:
    return {"provider": provider, "tokens": 0, "estimated_cost": 0.0, "actual_cost": 0.0,
            "sessions": 0, "unpriced_sessions": 0}


def month_window(now: datetime) -> tuple[datetime, int, float]:
    """(start of *now*'s calendar month in *now*'s timezone, days in that month, days elapsed)."""
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    days = calendar.monthrange(now.year, now.month)[1]
    return start, days, (now - start).total_seconds() / 86400


def month_usage_rows(conn: sqlite3.Connection, start_ts: float) -> list[dict]:
    """Per billing provider since *start_ts*: tokens, estimated/actual cost, sessions, unpriced sessions."""
    rows = {r["provider"]: dict(r) for r in conn.execute(_MAIN_SQL, (start_ts,))}
    try:
        aux = conn.execute(_AUX_SQL, (start_ts,)).fetchall()
    except sqlite3.OperationalError:  # a store older than session_model_usage.task
        aux = []
    for r in aux:
        row = rows.setdefault(r["provider"], _empty_row(r["provider"]))
        row["tokens"] += r["tokens"] or 0
        row["estimated_cost"] += r["estimated_cost"] or 0
    return list(rows.values())


def _evaluate(kind: str, limit: float, row: dict, start: datetime, days: int, elapsed: float) -> dict:
    used = row["tokens"] if kind == "tokens" else row["estimated_cost"]
    rate = used / max(elapsed, 1 / 24)  # never extrapolate from the first minutes of a month
    runs_out = start + timedelta(days=limit / rate) if rate and rate * days > limit else None
    return {"kind": kind, "limit": limit, "used": used, "used_ratio": used / limit,
            "projected_ratio": rate * days / limit,
            "runs_out_on": runs_out.date().isoformat() if runs_out else None}


def summarize_month(rows: list[dict], budgets: dict[str, tuple[str, float]], now: datetime) -> dict:
    """Month-to-date rows with each budgeted provider's used/projected share and overrun date.

    A budgeted provider with no usage yet still appears. Budgeted providers sort first, nearest their
    limit first; then the rest by tokens."""
    start, days, elapsed = month_window(now)
    by_provider = {r["provider"]: dict(r) for r in rows}
    for provider in budgets.keys() - by_provider.keys():
        by_provider[provider] = _empty_row(provider)
    providers = []
    for row in by_provider.values():
        spec = budgets.get(row["provider"])
        row["budget"] = _evaluate(*spec, row, start, days, elapsed) if spec else None
        providers.append(row)
    providers.sort(key=lambda r: (r["budget"] is None, -(r["budget"] or {}).get("used_ratio", 0), -r["tokens"]))
    return {"month": start.strftime("%Y-%m"), "days_in_month": days, "days_elapsed": round(elapsed, 2),
            "providers": providers}


def read_budgets(config: dict) -> dict[str, tuple[str, float]]:
    """``usage.budgets`` as ``{provider: ("tokens" | "usd", limit)}``; null or non-positive limits are no budget."""
    raw = ((config.get("usage") or {}).get("budgets") or {}) if isinstance(config, dict) else {}
    budgets: dict[str, tuple[str, float]] = {}
    for provider, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        for key, kind in (("monthly_tokens", "tokens"), ("monthly_usd", "usd")):
            value = spec.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                budgets[str(provider)] = (kind, float(value))
                break
    return budgets
