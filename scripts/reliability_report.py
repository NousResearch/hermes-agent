#!/usr/bin/env python3
"""Reliability report builder with dream-scope daily premium savings projection.

Reads a routing telemetry JSONL store and emits a markdown report containing:
 - Per-model table with request count, Wilson 95% CI for success rate,
   latency stats, premium units consumed and dominant failure type.
 - Worst / optimal model identification (delegated to summarize()).
 - Dream scope: projected daily and 30-day savings versus an all-Opus baseline,
   both conservative and optimistic.

Usage:
    python scripts/reliability_report.py [--store PATH] [--out PATH]
                                         [--since 7d] [--synthetic]

When --synthetic is set, a SYNTHETIC header is prepended to the report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running from worktree root without installing as package
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from agent.routing_telemetry import (  # noqa: E402
    DEFAULT_STORE,
    PREMIUM_MULTIPLIERS,
    load_events,
    summarize,
)


_SINCE_RE = re.compile(r"^(\d+)([hdw])$")


def parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    m = _SINCE_RE.match(value.strip())
    if not m:
        raise ValueError(f"invalid --since value: {value!r} (expected e.g. 24h, 7d, 2w)")
    n, unit = int(m.group(1)), m.group(2)
    delta = {"h": timedelta(hours=n), "d": timedelta(days=n), "w": timedelta(weeks=n)}[unit]
    return datetime.now(timezone.utc) - delta


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n <= 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    halfw = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    lo = max(0.0, center - halfw)
    hi = min(1.0, center + halfw)
    return (lo, hi)


def _event_tokens(event) -> int:
    return int(event.tokens_in or 0) + int(event.tokens_out or 0)


def _days_span(events) -> float:
    if not events:
        return 1.0
    ts = []
    for e in events:
        try:
            ts.append(datetime.fromisoformat(e.timestamp))
        except Exception:
            continue
    if len(ts) < 2:
        return 1.0
    span = (max(ts) - min(ts)).total_seconds() / 86400.0
    return max(span, 1.0)


def build_dream_scope(events) -> dict:
    """Projected daily premium-unit savings vs all-Opus baseline."""
    if not events:
        return {
            "events_per_day": 0.0,
            "actual_premium_per_day": 0.0,
            "baseline_all_opus_per_day": 0.0,
            "savings_per_day": 0.0,
            "savings_pct": 0.0,
            "projection_30d_conservative": 0.0,
            "projection_30d_optimistic": 0.0,
            "days_span": 0.0,
        }
    days = _days_span(events)
    total_tokens = sum(_event_tokens(e) for e in events)
    actual_units = sum(float(e.premium_units or 0) for e in events)
    opus_mult = PREMIUM_MULTIPLIERS.get("claude-opus-4.6", 5.0)
    baseline_units = total_tokens * opus_mult / 1000.0

    actual_per_day = actual_units / days
    baseline_per_day = baseline_units / days
    savings_per_day = max(0.0, baseline_per_day - actual_per_day)
    pct = savings_per_day / baseline_per_day if baseline_per_day > 0 else 0.0

    return {
        "events_per_day": len(events) / days,
        "actual_premium_per_day": actual_per_day,
        "baseline_all_opus_per_day": baseline_per_day,
        "savings_per_day": savings_per_day,
        "savings_pct": pct,
        "projection_30d_conservative": savings_per_day * 30,
        "projection_30d_optimistic": savings_per_day * 30 * 1.2,
        "days_span": days,
    }


def _fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def build_report_markdown(events, summary: dict, dream: dict, synthetic: bool) -> str:
    lines: list[str] = []
    if synthetic:
        lines.append("> **DATOS SINTÉTICOS — reemplazar tras 100+ turnos reales.**\n")

    lines.append("# Routing Reliability Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    if events:
        first = min(e.timestamp for e in events)
        last = max(e.timestamp for e in events)
        lines.append(f"Event span: {first} → {last} ({_fmt_num(dream['days_span'])} days)")
    lines.append(f"Total events: {len(events)}")
    lines.append("")

    lines.append("## Model table")
    lines.append("")
    lines.append("| model | req | success% (95% CI) | avg_ms | p95_ms | premium_units | top_fail |")
    lines.append("|-------|-----|-------------------|--------|--------|---------------|----------|")
    by_model = summary.get("by_model", {})
    for model, stats in sorted(by_model.items()):
        n = stats["requests"]
        k = stats["successes"]
        lo, hi = wilson_interval(k, n)
        sr_str = f"{_fmt_num(stats['success_rate']*100, 1)}% [{_fmt_num(lo*100,1)}, {_fmt_num(hi*100,1)}]"
        avg_ms = _fmt_num(stats["avg_latency_ms"] or 0, 1)
        p95_ms = _fmt_num(stats["p95_latency_ms"] or 0, 1)
        units = _fmt_num(stats["total_premium_units"], 2)
        fails = stats.get("fail_types") or {}
        top_fail = max(fails.items(), key=lambda kv: kv[1])[0] if fails else "-"
        lines.append(f"| {model} | {n} | {sr_str} | {avg_ms} | {p95_ms} | {units} | {top_fail} |")
    lines.append("")

    lines.append("## Worst / Optimal")
    lines.append("")
    lines.append(f"- **worst_model**: `{summary.get('worst_model')}`")
    lines.append(f"- **optimal_model**: `{summary.get('optimal_model')}`")
    lines.append("")

    lines.append("## Dream scope (ahorro proyectado)")
    lines.append("")
    lines.append(f"- turnos/día promedio: **{_fmt_num(dream['events_per_day'],1)}**")
    lines.append(f"- unidades premium/día actual: **{_fmt_num(dream['actual_premium_per_day'],2)}**")
    lines.append(f"- baseline all-Opus/día: **{_fmt_num(dream['baseline_all_opus_per_day'],2)}**")
    lines.append(f"- **AHORRO/día: {_fmt_num(dream['savings_per_day'],2)} unidades ({_fmt_num(dream['savings_pct']*100,1)}%)**")
    lines.append(f"- Proyección 30d conservadora: {_fmt_num(dream['projection_30d_conservative'],1)} unidades")
    lines.append(f"- Proyección 30d optimista: {_fmt_num(dream['projection_30d_optimistic'],1)} unidades")
    lines.append("")

    lines.append("## Savings vs baseline (lifetime of store)")
    lines.append("")
    lines.append(f"- total_premium_units: {_fmt_num(summary.get('total_premium_units',0),2)}")
    lines.append(f"- baseline_all_opus_units: {_fmt_num(summary.get('baseline_all_opus_units',0),2)}")
    lines.append(f"- savings_vs_baseline: **{_fmt_num(summary.get('savings_vs_baseline',0)*100,1)}%**")
    lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--since", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args(argv)

    since = parse_since(args.since)
    if not Path(args.store).exists():
        print(f"no events: store not found at {args.store}", file=sys.stderr)
        return 0

    events = load_events(store=args.store, since=since)
    if not events:
        print("no events in store", file=sys.stderr)
        return 0

    summary = summarize(events)
    dream = build_dream_scope(events)
    md = build_report_markdown(events, summary, dream, args.synthetic)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
