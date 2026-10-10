"""Month-to-date usage per billing provider, measured against ``usage.budgets``.

A token plan is budgeted in tokens and pay-as-you-go in dollars. Most sessions on a token plan carry no
price at all (``cost_status = 'unknown'``), so tokens are always counted, money only where it is known,
and the number of unpriced sessions is reported instead of being shown as $0.
"""
from __future__ import annotations

import calendar
import sqlite3
from datetime import datetime, timedelta

# Read from usage_hourly: every per-call delta (main loop and auxiliary tasks) in the hour it was spent,
# under the provider that served it. A session started last month counts its calls from this month, and
# a mid-session /model switch splits by provider. "Tokens used" is CanonicalUsage.total_tokens(): input
# excludes cache reads/writes, so nothing doubles.
_MONTH_SQL = """
    SELECT COALESCE(NULLIF(billing_provider, ''), 'unknown') AS provider,
           SUM(input_tokens + cache_read_tokens + cache_write_tokens + output_tokens) AS tokens,
           COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost,
           COALESCE(SUM(actual_cost_usd), 0) AS actual_cost,
           COUNT(DISTINCT session_id) AS sessions,
           COUNT(DISTINCT CASE WHEN unpriced_calls > 0 THEN session_id END) AS unpriced_sessions
    FROM usage_hourly WHERE hour >= ? GROUP BY 1
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
    """Per billing provider, usage spent since *start_ts*: tokens, estimated/actual cost, sessions, and
    sessions with at least one call of unknown price."""
    try:
        return [dict(r) for r in conn.execute(_MONTH_SQL, (start_ts,))]
    except sqlite3.OperationalError:  # a store no writer has opened since usage_hourly arrived (schema v32)
        return []


def ledger_started_at(conn: sqlite3.Connection) -> float | None:
    """Start of the earliest hour usage_hourly holds: nothing before it was recorded by time."""
    try:
        return conn.execute("SELECT MIN(hour) FROM usage_hourly").fetchone()[0]
    except sqlite3.OperationalError:
        return None


def _evaluate(kind: str, limit: float, row: dict, counted_from: datetime, span: float, counted: float) -> dict:
    """*span*: days from *counted_from* to month end; *counted*: how many of them have passed."""
    used = row["tokens"] if kind == "tokens" else row["estimated_cost"]
    rate = used / max(counted, 1 / 24)  # never extrapolate from the first minutes of a month
    runs_out = counted_from + timedelta(days=limit / rate) if rate and rate * span > limit else None
    return {"kind": kind, "limit": limit, "used": used, "used_ratio": used / limit,
            "projected_ratio": rate * span / limit,
            "runs_out_on": runs_out.date().isoformat() if runs_out else None}


def summarize_month(rows: list[dict], budgets: dict[str, tuple[str, float]], now: datetime, *,
                    ledger_start: datetime | None = None) -> dict:
    """Month-to-date rows with each budgeted provider's used/projected share and overrun date.

    A budgeted provider with no usage yet still appears. Budgeted providers sort first, nearest their
    limit first; then the rest by tokens. When the usage ledger began after this month did
    (*ledger_start*: the upgrade that added it), the days before were never recorded by time: the pace
    covers the counted part only and counted_since says where it starts."""
    start, days, elapsed = month_window(now)
    counted_from = ledger_start if ledger_start and ledger_start > start else start
    offset = (counted_from - start).total_seconds() / 86400
    by_provider = {r["provider"]: dict(r) for r in rows}
    for provider in budgets.keys() - by_provider.keys():
        by_provider[provider] = _empty_row(provider)
    providers = []
    for row in by_provider.values():
        spec = budgets.get(row["provider"])
        row["budget"] = _evaluate(*spec, row, counted_from, days - offset, elapsed - offset) if spec else None
        providers.append(row)
    providers.sort(key=lambda r: (r["budget"] is None, -(r["budget"] or {}).get("used_ratio", 0), -r["tokens"]))
    return {"month": start.strftime("%Y-%m"), "days_in_month": days, "days_elapsed": round(elapsed, 2),
            "counted_since": counted_from.isoformat() if counted_from > start else None, "providers": providers}


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
