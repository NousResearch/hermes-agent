"""Pure analytics computations over decoded token time-series.

DB-free so the math is unit-testable in isolation. Inputs come from
:meth:`hermes_state.SessionDB.get_message_token_timeseries` (1-minute
buckets) plus provider quota dicts from :mod:`agent.provider_quotas`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _pct(value: int, limit: Optional[int]) -> Optional[float]:
    if not limit or limit <= 0:
        return None
    return round(100.0 * value / limit, 1)


def _stat(values: List[int]) -> Dict[str, float]:
    """min/max/mean/median/p95 over a list (0s for empty)."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    return {"min": s[0], "max": s[-1], "mean": round(mean, 2), "median": round(median, 2), "p95": p95}


def compute_usage_rates(
    minute_buckets: List[Dict[str, int]],
    daily_totals: Dict[str, int],
    provider_quotas: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """RPM/TPM peaks + RPD/TPD totals, compared to provider limits.

    ``minute_buckets``: 1-minute time-series rows (requests/input/output/…).
    ``daily_totals``: aggregate over the last 24h ({requests,input,output,…}).
    ``provider_quotas``: quota dicts for providers active in the window.
    """
    req = [b.get("requests", 0) for b in minute_buckets]
    tpm = [b.get("input", 0) + b.get("output", 0) for b in minute_buckets]
    tpm_in = [b.get("input", 0) for b in minute_buckets]
    tpm_out = [b.get("output", 0) for b in minute_buckets]

    peak_rpm = max(req, default=0)
    peak_tpm = max(tpm, default=0)
    current_rpm = req[-1] if req else 0
    current_tpm = tpm[-1] if tpm else 0

    rpd = int(daily_totals.get("requests", 0))
    tpd = int(daily_totals.get("input", 0)) + int(daily_totals.get("output", 0))

    providers_view: List[Dict[str, Any]] = []
    for q in provider_quotas:
        # Peak-vs-limit %, conservative (treats the global peak as if it all
        # went to this provider — exact for the single-provider common case).
        providers_view.append({
            "provider": q.get("provider"),
            "display": q.get("display"),
            "tier": q.get("tier"),
            "limits": {
                "rpm": q.get("rpm"), "rpd": q.get("rpd"),
                "tpm_input": q.get("tpm_input"), "tpm_output": q.get("tpm_output"),
                "tpd": q.get("tpd"),
            },
            "pct_of_limit": {
                "rpm": _pct(peak_rpm, q.get("rpm")),
                "rpd": _pct(rpd, q.get("rpd")),
                "tpm_input": _pct(max(tpm_in, default=0), q.get("tpm_input")),
                "tpm_output": _pct(max(tpm_out, default=0), q.get("tpm_output")),
                "tpd": _pct(tpd, q.get("tpd")),
            },
            "source_url": q.get("source_url"),
            "as_of": q.get("as_of"),
        })

    return {
        "rpm": {"current": current_rpm, "peak": peak_rpm},
        "tpm": {
            "current": current_tpm, "peak": peak_tpm,
            "peak_input": max(tpm_in, default=0), "peak_output": max(tpm_out, default=0),
        },
        "rpd": rpd,
        "tpd": tpd,
        "window_totals": {
            "requests": sum(req),
            "input": sum(tpm_in),
            "output": sum(tpm_out),
        },
        "providers": providers_view,
    }


def compute_token_trends(buckets: List[Dict[str, int]]) -> Dict[str, Any]:
    """Per-bucket series + per-request averages + cache-hit rate.

    ``buckets``: time-series rows at the caller's chosen granularity.
    """
    series: List[Dict[str, Any]] = []
    per_call_input: List[int] = []
    per_call_output: List[int] = []
    total_input = total_output = total_cache = total_reasoning = total_req = 0

    for b in buckets:
        reqs = int(b.get("requests", 0))
        inp = int(b.get("input", 0))
        out = int(b.get("output", 0))
        cache = int(b.get("cache_read", 0))
        reason = int(b.get("reasoning", 0))
        total_input += inp
        total_output += out
        total_cache += cache
        total_reasoning += reason
        total_req += reqs
        cache_hit = round(100.0 * cache / inp, 1) if inp else None
        series.append({
            "bucket_start": int(b.get("bucket_start", 0)),
            "requests": reqs,
            "input": inp, "output": out, "cache_read": cache, "reasoning": reason,
            "cache_hit_rate": cache_hit,
            "avg_input_per_request": round(inp / reqs, 1) if reqs else 0,
            "avg_output_per_request": round(out / reqs, 1) if reqs else 0,
        })
        if reqs:
            per_call_input.append(round(inp / reqs))
            per_call_output.append(round(out / reqs))

    overall_cache_hit = round(100.0 * total_cache / total_input, 1) if total_input else None
    return {
        "series": series,
        "totals": {
            "requests": total_req, "input": total_input, "output": total_output,
            "cache_read": total_cache, "reasoning": total_reasoning,
        },
        "averages_per_request": {
            "input": round(total_input / total_req, 1) if total_req else 0,
            "output": round(total_output / total_req, 1) if total_req else 0,
            "reasoning": round(total_reasoning / total_req, 1) if total_req else 0,
            "cache_read": round(total_cache / total_req, 1) if total_req else 0,
            "input_distribution": _stat(per_call_input),
            "output_distribution": _stat(per_call_output),
        },
        "cache_hit_rate": overall_cache_hit,
    }


def _price(per_million: Any, tokens: int) -> Optional[float]:
    """Cost in USD for ``tokens`` at ``per_million`` USD/1M, or None if unpriced."""
    if per_million is None:
        return None
    return round(float(per_million) * tokens / 1_000_000.0, 6)


def compute_cost_estimate(
    groups: List[Dict[str, Any]],
    window_seconds: int,
    price_lookup,
) -> Dict[str, Any]:
    """Per-model cost broken down by price tier, with daily/monthly projection.

    ``groups``: rows from
    :meth:`hermes_state.SessionDB.get_session_cost_aggregates`.
    ``price_lookup(model, provider, base_url)``: returns a pricing entry with
    ``input_cost_per_million`` / ``output_cost_per_million`` /
    ``cache_read_cost_per_million`` / ``cache_write_cost_per_million`` (or
    None). When a model has no known pricing the stored ``estimated_cost_usd``
    is used as a fallback and flagged.
    """
    models: List[Dict[str, Any]] = []
    total = 0.0
    total_input_cost = total_output_cost = total_cache_cost = 0.0
    any_unpriced = False

    for g in groups:
        entry = None
        try:
            entry = price_lookup(g.get("model"), g.get("billing_provider"), g.get("billing_base_url"))
        except Exception:
            entry = None

        in_cost = out_cost = cr_cost = cw_cost = None
        if entry is not None:
            in_cost = _price(getattr(entry, "input_cost_per_million", None), g["input_tokens"])
            out_cost = _price(getattr(entry, "output_cost_per_million", None), g["output_tokens"])
            cr_cost = _price(getattr(entry, "cache_read_cost_per_million", None), g["cache_read_tokens"])
            cw_cost = _price(getattr(entry, "cache_write_cost_per_million", None), g["cache_write_tokens"])

        priced = any(c is not None for c in (in_cost, out_cost, cr_cost, cw_cost))
        if priced:
            grp_total = round(sum(c or 0.0 for c in (in_cost, out_cost, cr_cost, cw_cost)), 6)
            source = "pricing"
        else:
            grp_total = round(float(g.get("estimated_cost_usd") or 0.0), 6)
            source = "stored_estimate"
            any_unpriced = True

        total += grp_total
        total_input_cost += in_cost or 0.0
        total_output_cost += out_cost or 0.0
        total_cache_cost += (cr_cost or 0.0) + (cw_cost or 0.0)

        models.append({
            "model": g.get("model"),
            "provider": g.get("billing_provider"),
            "sessions": g.get("sessions", 0),
            "tokens": {
                "input": g["input_tokens"], "output": g["output_tokens"],
                "cache_read": g["cache_read_tokens"], "cache_write": g["cache_write_tokens"],
                "reasoning": g["reasoning_tokens"],
            },
            "cost_breakdown": {
                "input": in_cost, "output": out_cost,
                "cache_read": cr_cost, "cache_write": cw_cost,
            },
            "cost_usd": grp_total,
            "cost_source": source,
        })

    models.sort(key=lambda m: m["cost_usd"], reverse=True)
    total = round(total, 6)
    days = max(window_seconds / 86400.0, 1e-9)
    daily = round(total / days, 6)
    return {
        "total_cost_usd": total,
        "cost_by_tier": {
            "input": round(total_input_cost, 6),
            "output": round(total_output_cost, 6),
            "cache": round(total_cache_cost, 6),
        },
        "projection": {"daily_usd": daily, "monthly_usd": round(daily * 30, 6)},
        "has_unpriced_models": any_unpriced,
        "models": models,
    }
