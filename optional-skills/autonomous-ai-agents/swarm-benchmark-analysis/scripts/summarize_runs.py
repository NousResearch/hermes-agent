#!/usr/bin/env python3
"""Summarize measured agent benchmark runs using only the Python standard library."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


class BenchmarkDataError(ValueError):
    """Raised when a benchmark record is incomplete or inconsistent."""


def read_runs(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BenchmarkDataError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise BenchmarkDataError(f"line {line_number}: each record must be a JSON object")
        row["_line"] = line_number
        records.append(row)
    return records


def _validate(records: list[dict[str, Any]]) -> str:
    if not records:
        raise BenchmarkDataError("no benchmark records found")
    benchmark_ids: set[str] = set()
    observed: set[tuple[str, str]] = set()
    required = ("benchmark_id", "case_id", "variant", "accepted", "tokens", "latency_ms")
    for row in records:
        line = row.get("_line", "?")
        for field in required:
            if field not in row:
                raise BenchmarkDataError(f"line {line}: missing {field}")
        if not all(isinstance(row[field], str) and row[field] for field in required[:3]):
            raise BenchmarkDataError(f"line {line}: benchmark_id, case_id, and variant must be non-empty strings")
        benchmark_ids.add(row["benchmark_id"])
        if type(row["accepted"]) is not bool:
            raise BenchmarkDataError(f"line {line}: accepted must be a boolean")
        if type(row["tokens"]) is not int or row["tokens"] < 0:
            raise BenchmarkDataError(f"line {line}: tokens must be a non-negative integer")
        latency = row["latency_ms"]
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise BenchmarkDataError(f"line {line}: latency_ms must be a finite non-negative number")
        key = (row["case_id"], row["variant"])
        if key in observed:
            raise BenchmarkDataError(f"line {line}: duplicate case/variant pair {key!r}")
        observed.add(key)
        if "cost_microusd" in row:
            cost = row["cost_microusd"]
            if type(cost) is not int or cost < 0:
                raise BenchmarkDataError(f"line {line}: cost_microusd must be a non-negative integer")
            if not isinstance(row.get("cost_source"), str) or not row["cost_source"].strip():
                raise BenchmarkDataError(f"line {line}: measured cost requires a non-empty cost_source")
    if len(benchmark_ids) != 1:
        raise BenchmarkDataError("all records must use the same benchmark_id")
    return next(iter(benchmark_ids))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _wilson(successes: int, total: int) -> dict[str, float]:
    if total == 0:
        return {"lower": 0.0, "upper": 0.0}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def summarize(records: list[dict[str, Any]], baseline: str | None = None) -> dict[str, Any]:
    benchmark_id = _validate(records)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["variant"]].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for variant, rows in sorted(grouped.items()):
        accepted = sum(row["accepted"] for row in rows)
        cost_rows = [row for row in rows if "cost_microusd" in row]
        all_costs_known = len(cost_rows) == len(rows)
        total_cost = sum(row["cost_microusd"] for row in cost_rows) if all_costs_known else None
        summaries[variant] = {
            "runs": len(rows),
            "accepted": accepted,
            "acceptance_rate": accepted / len(rows),
            "acceptance_wilson_95": _wilson(accepted, len(rows)),
            "latency_ms_p50": _percentile([float(row["latency_ms"]) for row in rows], 0.50),
            "latency_ms_p95": _percentile([float(row["latency_ms"]) for row in rows], 0.95),
            "tokens_total": sum(row["tokens"] for row in rows),
            "tokens_per_run_mean": sum(row["tokens"] for row in rows) / len(rows),
            "cost_known_runs": len(cost_rows),
            "cost_microusd_total": total_cost,
            "cost_microusd_per_accepted": (total_cost / accepted if total_cost is not None and accepted else None),
        }

    paired: dict[str, Any] = {}
    if baseline is not None:
        if baseline not in grouped:
            raise BenchmarkDataError(f"baseline variant not found: {baseline}")
        baseline_rows = {row["case_id"]: row for row in grouped[baseline]}
        for variant, rows in sorted(grouped.items()):
            if variant == baseline:
                continue
            variant_rows = {row["case_id"]: row for row in rows}
            cases = sorted(set(baseline_rows) & set(variant_rows))
            if not cases:
                paired[variant] = {"baseline": baseline, "paired_cases": 0,
                                   "acceptance_delta": None, "latency_ms_median_delta": None}
                continue
            acceptance_deltas = [int(variant_rows[case]["accepted"]) - int(baseline_rows[case]["accepted"])
                                 for case in cases]
            latency_deltas = [float(variant_rows[case]["latency_ms"]) - float(baseline_rows[case]["latency_ms"])
                              for case in cases]
            paired[variant] = {
                "baseline": baseline,
                "paired_cases": len(cases),
                "acceptance_delta": sum(acceptance_deltas) / len(cases),
                "latency_ms_median_delta": _percentile(latency_deltas, 0.50),
            }

    return {"benchmark_id": benchmark_id, "variants": summaries, "paired_comparisons": paired}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", help="one measured run per JSONL line")
    parser.add_argument("--baseline", help="variant name for paired comparison")
    args = parser.parse_args(argv)
    try:
        report = summarize(read_runs(args.jsonl), baseline=args.baseline)
    except (OSError, BenchmarkDataError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
