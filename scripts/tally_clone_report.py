#!/usr/bin/env python3
"""Build a Tally-style billing context for Hermes.

Combines:
1. Legacy Production Hub billing endpoints (localhost:5053) for parity with old Tally
2. Hermes native session DB (~/.hermes/state.db) for direct vs aggregator split tracking

Prints JSON to stdout for use by Hermes cron `script=` context injection.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError, HTTPError

TZ = timezone(timedelta(hours=7))
NOW = datetime.now(TZ)
TODAY = NOW.date()
YESTERDAY = TODAY - timedelta(days=1)
MONTH_START = TODAY.replace(day=1)
PRODUCTION_HUB = os.environ.get("TALLY_BILLING_BASE", "http://localhost:5053")
STATE_DB = Path(os.environ.get("HERMES_STATE_DB", str(Path.home() / ".hermes" / "state.db")))

ENDPOINTS = {
    "budget_status": "/api/billing/budget-status",
    "subscriptions": "/api/billing/subscriptions",
    "anthropic": "/api/billing/anthropic-usage",
    "codex": "/api/billing/codex-usage",
    "openai": "/api/billing/openai-usage",
    "google": "/api/billing/google-usage",
    "openrouter": "/api/billing/openrouter-usage",
    "zai": "/api/billing/zai-usage",
    "xai": "/api/billing/xai-usage",
    "groq": "/api/billing/groq-usage",
    "fireworks": "/api/billing/fireworks-usage",
    "brave": "/api/billing/brave-usage",
    "apify": "/api/billing/apify",
}


@dataclass
class RouteSummary:
    route: str
    sessions: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    models: list[str] | None = None


def _json_get(path: str) -> dict[str, Any]:
    url = f"{PRODUCTION_HUB}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Hermes-Tally-Clone/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_json_get(path: str) -> dict[str, Any]:
    try:
        data = _json_get(path)
        return {"ok": True, "status": 200, "data": data}
    except HTTPError as e:
        return {"ok": False, "status": e.code, "error": str(e)}
    except URLError as e:
        return {"ok": False, "status": None, "error": str(e)}
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": False, "status": None, "error": str(e)}


def _parse_dt(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _freshness_hours(updated_at: Any) -> float | None:
    dt = _parse_dt(updated_at)
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 3600.0, 2)


def _endpoint_health() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in ENDPOINTS.items():
        res = _safe_json_get(path)
        entry: dict[str, Any] = {
            "ok": res["ok"],
            "status": res.get("status"),
        }
        if res["ok"]:
            data = res["data"]
            updated_at = data.get("updatedAt") if isinstance(data, dict) else None
            hours = _freshness_hours(updated_at)
            entry.update(
                {
                    "updatedAt": updated_at,
                    "freshness_hours": hours,
                    "fresh": (hours is not None and hours <= 24.0),
                }
            )
        else:
            entry["error"] = res.get("error")
        out[name] = entry
    return out


def _session_rows() -> list[sqlite3.Row]:
    if not STATE_DB.exists():
        return []
    conn = sqlite3.connect(str(STATE_DB))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT id, started_at, model, billing_provider, billing_base_url,
                   input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
                   reasoning_tokens, estimated_cost_usd, actual_cost_usd
            FROM sessions
            WHERE billing_provider IS NOT NULL OR billing_base_url IS NOT NULL
            """
        ).fetchall()
    finally:
        conn.close()


def _route_key(provider: str | None, base_url: str | None, model: str | None) -> str | None:
    p = (provider or "").lower()
    b = (base_url or "").lower()
    m = (model or "").lower()
    if "openrouter" in p or "openrouter.ai" in b:
        return "openrouter_kimi" if "kimi" in m else "openrouter_other"
    if "kimi" in p or "moonshot" in p or "moonshot.ai" in b or "api.kimi.com" in b:
        return "moonshot_direct_kimi" if "kimi" in m else "moonshot_direct_other"
    return None


def _empty_summary(route: str) -> RouteSummary:
    return RouteSummary(route=route, models=[])


def _summaries_from_rows(rows: list[sqlite3.Row], start_date: datetime.date, end_date: datetime.date) -> dict[str, RouteSummary]:
    buckets: dict[str, RouteSummary] = {
        "moonshot_direct_kimi": _empty_summary("moonshot_direct_kimi"),
        "openrouter_kimi": _empty_summary("openrouter_kimi"),
        "moonshot_direct_other": _empty_summary("moonshot_direct_other"),
        "openrouter_other": _empty_summary("openrouter_other"),
    }
    for row in rows:
        started = datetime.fromtimestamp(float(row["started_at"]), TZ).date()
        if started < start_date or started > end_date:
            continue
        route = _route_key(row["billing_provider"], row["billing_base_url"], row["model"])
        if not route:
            continue
        bucket = buckets.setdefault(route, _empty_summary(route))
        bucket.sessions += 1
        bucket.input_tokens += int(row["input_tokens"] or 0)
        bucket.output_tokens += int(row["output_tokens"] or 0)
        bucket.cache_tokens += int(row["cache_read_tokens"] or 0) + int(row["cache_write_tokens"] or 0)
        bucket.reasoning_tokens += int(row["reasoning_tokens"] or 0)
        bucket.estimated_cost_usd += float(row["estimated_cost_usd"] or 0.0)
        bucket.actual_cost_usd += float(row["actual_cost_usd"] or 0.0)
        model = row["model"] or "unknown"
        if bucket.models is not None and model not in bucket.models:
            bucket.models.append(model)
    for bucket in buckets.values():
        bucket.estimated_cost_usd = round(bucket.estimated_cost_usd, 6)
        bucket.actual_cost_usd = round(bucket.actual_cost_usd, 6)
        if bucket.models is not None:
            bucket.models.sort()
    return buckets


def _route_comparison(rows: list[sqlite3.Row]) -> dict[str, Any]:
    yesterday = _summaries_from_rows(rows, YESTERDAY, YESTERDAY)
    mtd = _summaries_from_rows(rows, MONTH_START, TODAY)

    def _cost(summary: RouteSummary) -> float:
        return summary.actual_cost_usd or summary.estimated_cost_usd

    moon_y = _cost(yesterday["moonshot_direct_kimi"])
    or_y = _cost(yesterday["openrouter_kimi"])
    moon_m = _cost(mtd["moonshot_direct_kimi"])
    or_m = _cost(mtd["openrouter_kimi"])

    notes: list[str] = []
    if moon_y == 0 and or_y == 0 and moon_m == 0 and or_m == 0:
        notes.append("No Hermes-native Kimi spend recorded yet in ~/.hermes/state.db.")
    else:
        if moon_m and or_m:
            cheaper = "moonshot_direct_kimi" if moon_m < or_m else "openrouter_kimi" if or_m < moon_m else "tie"
            notes.append(f"MTD cheaper route so far: {cheaper}.")
        elif moon_m and not or_m:
            notes.append("Only Moonshot direct Kimi spend recorded so far.")
        elif or_m and not moon_m:
            notes.append("Only OpenRouter Kimi spend recorded so far.")

    return {
        "yesterday": {k: asdict(v) for k, v in yesterday.items()},
        "mtd": {k: asdict(v) for k, v in mtd.items()},
        "notes": notes,
    }


def main() -> int:
    budget = _safe_json_get(ENDPOINTS["budget_status"])
    subscriptions = _safe_json_get(ENDPOINTS["subscriptions"])
    health = _endpoint_health()
    rows = _session_rows()
    report = {
        "generated_at": NOW.isoformat(),
        "timezone": "Asia/Saigon",
        "legacy_budget_status": budget.get("data") if budget.get("ok") else None,
        "legacy_subscriptions": subscriptions.get("data") if subscriptions.get("ok") else None,
        "legacy_errors": {
            "budget_status": None if budget.get("ok") else budget,
            "subscriptions": None if subscriptions.get("ok") else subscriptions,
        },
        "endpoint_health": health,
        "hermes_native_split": _route_comparison(rows),
        "comparison_notes": [
            "Legacy Production Hub endpoints preserve apples-to-apples Tally parity.",
            "Hermes native split uses ~/.hermes/state.db session costs grouped by provider/base URL.",
            "Kimi via Moonshot direct and Kimi via OpenRouter are tracked separately when Hermes records billing metadata.",
        ],
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
