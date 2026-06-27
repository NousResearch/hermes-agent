#!/usr/bin/env python3
"""skew-distribution — read-only reporter over COMPACTION_SKEW telemetry (T3 / AC-6).

Parses the dedicated ``~/.hermes/state/skew-samples.log`` sink (primary; can't
rotate out before N accrues) and optionally the rotating gateway logs, and prints
the skew distribution: count + min + p1/p5/p50 of the real/rough ratio, grouped by
model, FILTERED to ``task=main`` (the distribution the compaction trigger uses).

This is the artifact the v0.2 ``compression.skew_floor`` tune reads:
    "p1 of skew over N≥<floor> main-turn samples = X → set skew_floor below X."

READ-ONLY (INV-5): never writes, rotates, or deletes any log.

Usage:
    python scripts/skew-distribution.py                 # the dedicated sink
    python scripts/skew-distribution.py --include-logs   # also scan gateway/agent logs
    python scripts/skew-distribution.py --task all       # don't filter to main
"""
from __future__ import annotations

import argparse
import glob
import os
import re

_SKEW_RE = re.compile(
    r"COMPACTION_SKEW\s+rough=(?P<rough>\d+)\s+real=(?P<real>\d+)\s+"
    r"ratio=(?P<ratio>[0-9.]+)\s+task=(?P<task>\S+)\s+model=(?P<model>\S+)"
)


def _hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or os.path.join(
        os.path.expanduser("~"), ".hermes"
    )


def _percentile(sorted_vals, pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _iter_lines(paths):
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    yield line
        except OSError:
            continue


def collect(paths, task_filter: str | None):
    """Return {model: [ratios]} parsed from COMPACTION_SKEW lines. Malformed lines
    are skipped, never raised on (INV: a bad line must not crash the reader)."""
    by_model: dict[str, list[float]] = {}
    total = 0
    for line in _iter_lines(paths):
        m = _SKEW_RE.search(line)
        if not m:
            continue
        if task_filter and m.group("task") != task_filter:
            continue
        try:
            ratio = float(m.group("ratio"))
        except ValueError:
            continue
        by_model.setdefault(m.group("model"), []).append(ratio)
        total += 1
    return by_model, total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-logs", action="store_true",
                    help="also scan rotating gateway/agent logs (not just the dedicated sink)")
    ap.add_argument("--task", default="main",
                    help="task to filter to (default: main; 'all' = no filter)")
    args = ap.parse_args()

    home = _hermes_home()
    paths = [os.path.join(home, "state", "skew-samples.log")]
    if args.include_logs:
        paths += glob.glob(os.path.join(home, "logs", "*.log*"))
        paths += glob.glob(os.path.join(home, "profiles", "*", "logs", "*.log*"))
    paths = [p for p in paths if os.path.exists(p)]

    task_filter = None if args.task == "all" else args.task
    by_model, total = collect(paths, task_filter)

    print(f"skew distribution  (task={args.task}, sources={len(paths)})")
    print(f"total samples: {total}")
    if total == 0:
        print("  (no COMPACTION_SKEW samples yet — telemetry needs live turns to accrue)")
        return 0

    for model in sorted(by_model):
        vals = sorted(by_model[model])
        n = len(vals)
        print(f"\n  model={model}  n={n}")
        print(f"    min   = {vals[0]:.3f}")
        print(f"    p1    = {_percentile(vals, 1):.3f}")
        print(f"    p5    = {_percentile(vals, 5):.3f}")
        print(f"    p50   = {_percentile(vals, 50):.3f}")
        print(f"    max   = {vals[-1]:.3f}")
        # floor-tune hint
        p1 = _percentile(vals, 1)
        print(f"    → skew_floor candidate (below p1 with margin): {max(0.5, p1 - 0.05):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
