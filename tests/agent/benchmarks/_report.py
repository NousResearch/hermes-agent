"""Aggregate benchmark results into a markdown table.

Each test calls ``record(benchmark_id, metric_name, baseline, candidate, unit)``
and the session-end hook writes
``docs/research/2026-05-02-qwen-aware-compaction-benchmark-report.md``.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

_RESULTS: dict[str, list[dict]] = defaultdict(list)


def record(bid: str, metric: str, baseline: float, candidate: float,
           unit: str = "", note: str = "") -> None:
    _RESULTS[bid].append({
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "delta_pct": (candidate - baseline) / baseline * 100 if baseline else 0.0,
        "unit": unit,
        "note": note,
    })


def emit_report(path: Path) -> None:
    lines = ["# Compaction Benchmark Report", "",
             "| Benchmark | Metric | Baseline | Candidate | Δ% | Unit | Note |",
             "|---|---|---|---|---|---|---|"]
    for bid in sorted(_RESULTS):
        for r in _RESULTS[bid]:
            lines.append(
                f"| {bid} | {r['metric']} | {r['baseline']:.2f} "
                f"| {r['candidate']:.2f} | {r['delta_pct']:+.1f}% "
                f"| {r['unit']} | {r['note']} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    # Also dump raw JSON for downstream tooling.
    path.with_suffix(".json").write_text(
        json.dumps(_RESULTS, indent=2, default=str)
    )
