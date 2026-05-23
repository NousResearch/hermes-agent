from __future__ import annotations

import argparse

from benchmarks.hermes_memory_bench.core import run_benchmark, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Hermes Memory Bench.")
    parser.add_argument("--suite", default="smoke", choices=("smoke",), help="Benchmark suite to run.")
    parser.add_argument("--output", help="Optional JSON report path. Defaults to stdout.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_benchmark(suite=args.suite)
    write_report(report, args.output)


if __name__ == "__main__":
    main()
