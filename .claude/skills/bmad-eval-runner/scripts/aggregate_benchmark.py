#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# ///
"""Variance benchmark: summarize a metric across N runs, and compare two configs.

A single skill run is noisy. Running the same case N times and summarizing the
spread tells you whether a difference between two versions is real or just noise.
This script computes, per numeric metric, the mean, the sample standard deviation
(n-1, the unbiased estimator for a sample), the min, and the max across N runs.
Given two such config summaries it reports the delta on each shared metric so a
"did the change help" question gets a number instead of a guess.

Input shapes accepted for a single config:
  - a list of run records, each a flat dict of metric -> number
      [{"elapsed_s": 12.1, "total_tokens": 800}, {"elapsed_s": 11.4, ...}]
  - {"runs": [ ...records... ]}
  - a directory of run folders, each holding timing.json files written by
    run_evals.py (the script reads every timing.json under the directory and
    treats each as one run record)

Usage:
  Summarize one config across its runs:
    python3 aggregate_benchmark.py --runs CONFIG_A.json
    python3 aggregate_benchmark.py --runs RUN_DIR/        (reads timing.json files)

  Compare two configs (each summarized, then delta = B - A):
    python3 aggregate_benchmark.py --baseline A.json --variant B.json

  Self-test on a known fixture (no external input needed):
    python3 aggregate_benchmark.py --self-test

Output is one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


NUMERIC = (int, float)


# --- statistics -------------------------------------------------------------

def sample_stddev(values: list[float]) -> float:
    """Sample standard deviation using n-1 (Bessel's correction).

    Returns 0.0 for fewer than two values, where the sample variance is
    undefined and reporting zero spread is the least surprising choice.
    """
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def summarize_metric(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": (sum(values) / len(values)) if values else 0.0,
        "stddev": sample_stddev(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def collect_numeric_metrics(records: list[dict]) -> dict[str, list[float]]:
    """Group every numeric field across records by metric name."""
    by_metric: dict[str, list[float]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        for key, val in rec.items():
            if isinstance(val, bool):
                continue  # bools are ints in Python; not a metric
            if isinstance(val, NUMERIC):
                by_metric.setdefault(key, []).append(float(val))
    return by_metric


def summarize_config(records: list[dict]) -> dict:
    by_metric = collect_numeric_metrics(records)
    return {
        "runs": len(records),
        "metrics": {name: summarize_metric(vals)
                    for name, vals in sorted(by_metric.items())},
    }


def delta_configs(baseline: dict, variant: dict) -> dict:
    """Per shared metric, delta = variant.mean - baseline.mean, plus context."""
    b_metrics = baseline.get("metrics", {})
    v_metrics = variant.get("metrics", {})
    shared = sorted(set(b_metrics) & set(v_metrics))
    out: dict[str, dict] = {}
    for name in shared:
        b = b_metrics[name]
        v = v_metrics[name]
        diff = v["mean"] - b["mean"]
        pct = (diff / b["mean"] * 100.0) if b["mean"] != 0 else None
        out[name] = {
            "baseline_mean": b["mean"],
            "variant_mean": v["mean"],
            "delta": diff,
            "delta_pct": pct,
            "baseline_stddev": b["stddev"],
            "variant_stddev": v["stddev"],
        }
    return out


# --- input loading ----------------------------------------------------------

def load_records(path: Path) -> list[dict]:
    """Load run records from a JSON file, a {'runs': [...]} file, or a dir of
    timing.json files."""
    if path.is_dir():
        records: list[dict] = []
        for f in sorted(path.rglob("timing.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                records.append(data)
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "runs" in data:
        data = data["runs"]
    if not isinstance(data, list):
        raise ValueError(f"expected a list of run records in {path}")
    return [r for r in data if isinstance(r, dict)]


# --- self-test --------------------------------------------------------------

def run_self_test() -> int:
    """Verify mean/stddev/min/max/delta on a known fixture."""
    config_a = [
        {"elapsed_s": 10.0, "total_tokens": 100},
        {"elapsed_s": 12.0, "total_tokens": 200},
        {"elapsed_s": 14.0, "total_tokens": 300},
    ]
    summary_a = summarize_config(config_a)
    el = summary_a["metrics"]["elapsed_s"]
    # mean of 10,12,14 = 12; n-1 stddev = sqrt(((-2)^2+0+2^2)/2)=sqrt(4)=2
    assert el["n"] == 3, el
    assert abs(el["mean"] - 12.0) < 1e-9, el
    assert abs(el["stddev"] - 2.0) < 1e-9, el
    assert el["min"] == 10.0 and el["max"] == 14.0, el
    tok = summary_a["metrics"]["total_tokens"]
    # mean of 100,200,300 = 200; n-1 stddev = sqrt((10000+0+10000)/2)=100
    assert abs(tok["mean"] - 200.0) < 1e-9, tok
    assert abs(tok["stddev"] - 100.0) < 1e-9, tok

    # single value -> stddev 0
    one = summarize_config([{"x": 5}])
    assert one["metrics"]["x"]["stddev"] == 0.0, one

    # bools are not treated as metrics
    with_bool = summarize_config([{"ok": True, "x": 1}, {"ok": False, "x": 3}])
    assert "ok" not in with_bool["metrics"], with_bool
    assert abs(with_bool["metrics"]["x"]["mean"] - 2.0) < 1e-9, with_bool

    # delta: variant slower by 3s on mean, faster question answered by sign
    config_b = [
        {"elapsed_s": 13.0, "total_tokens": 90},
        {"elapsed_s": 15.0, "total_tokens": 110},
        {"elapsed_s": 17.0, "total_tokens": 100},
    ]
    summary_b = summarize_config(config_b)
    d = delta_configs(summary_a, summary_b)
    # elapsed mean: A=12, B=15 -> delta +3, pct +25%
    assert abs(d["elapsed_s"]["delta"] - 3.0) < 1e-9, d
    assert abs(d["elapsed_s"]["delta_pct"] - 25.0) < 1e-9, d
    # tokens mean: A=200, B=100 -> delta -100, pct -50%
    assert abs(d["total_tokens"]["delta"] + 100.0) < 1e-9, d
    assert abs(d["total_tokens"]["delta_pct"] + 50.0) < 1e-9, d

    print(json.dumps({"self_test": "passed",
                      "checked": ["mean", "stddev_n_minus_1", "min", "max",
                                  "single_value_stddev", "bool_excluded",
                                  "delta", "delta_pct"]}))
    return 0


# --- main -------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--runs", type=Path,
                   help="summarize one config (JSON file or dir of timing.json)")
    p.add_argument("--baseline", type=Path,
                   help="baseline config for a two-config comparison")
    p.add_argument("--variant", type=Path,
                   help="variant config for a two-config comparison")
    p.add_argument("--self-test", action="store_true",
                   help="run the built-in fixture self-test and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if args.baseline and args.variant:
        b = summarize_config(load_records(args.baseline))
        v = summarize_config(load_records(args.variant))
        out = {
            "baseline": b,
            "variant": v,
            "delta": delta_configs(b, v),
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.runs:
        out = summarize_config(load_records(args.runs))
        print(json.dumps(out, indent=2))
        return 0

    p.error("provide --runs, or both --baseline and --variant, or --self-test")
    return 2


if __name__ == "__main__":
    sys.exit(main())
