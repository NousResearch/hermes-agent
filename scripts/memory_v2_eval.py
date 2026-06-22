#!/usr/bin/env python3
"""Run deterministic Memory v2 eval fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from plugins.memory.memory_v2.evals.baselines import MemoryV2Baseline, NoMemoryBaseline, RawFTSBaseline
from plugins.memory.memory_v2.evals.datasets import load_eval_dataset
from plugins.memory.memory_v2.evals.reports import write_json_report
from plugins.memory.memory_v2.evals.runners import run_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Memory v2 evaluation fixtures.")
    parser.add_argument("--dataset", action="append", required=True, help="YAML dataset fixture path. Repeatable.")
    parser.add_argument("--baseline", action="append", choices=["no_memory", "raw_fts", "memory_v2"], default=[])
    parser.add_argument("--workdir", default="", help="Directory for temporary baseline stores.")
    parser.add_argument("--output", default="", help="Write JSON report to this path instead of stdout.")
    args = parser.parse_args(argv)

    baseline_names = args.baseline or ["no_memory", "raw_fts", "memory_v2"]
    with tempfile.TemporaryDirectory(prefix="memory-v2-eval-") as temp_dir:
        workdir = Path(args.workdir).expanduser().resolve() if args.workdir else Path(temp_dir)
        workdir.mkdir(parents=True, exist_ok=True)
        reports = []
        for dataset_path in args.dataset:
            try:
                dataset = load_eval_dataset(dataset_path)
            except FileNotFoundError:
                print(f"error: dataset not found: {dataset_path}", file=sys.stderr)
                return 2
            except Exception as exc:
                print(f"error: failed to load dataset {dataset_path}: {exc}", file=sys.stderr)
                return 2
            report = run_eval(dataset, baselines=_build_baselines(baseline_names, workdir / dataset.name))
            reports.append(report.to_dict())

    payload = reports[0] if len(reports) == 1 else {"reports": reports}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        write_json_report(payload, args.output)
    else:
        print(rendered)
    return 0


def _build_baselines(names: list[str], workdir: Path):
    baselines = []
    for index, name in enumerate(names):
        baseline_dir = workdir / f"{index}_{name}"
        if name == "no_memory":
            baselines.append(NoMemoryBaseline())
        elif name == "raw_fts":
            baselines.append(RawFTSBaseline(baseline_dir / "raw.sqlite"))
        elif name == "memory_v2":
            baselines.append(MemoryV2Baseline(baseline_dir / "memory_v2"))
        else:  # argparse choices should prevent this.
            raise ValueError(f"unknown baseline: {name}")
    return baselines


if __name__ == "__main__":
    raise SystemExit(main())
