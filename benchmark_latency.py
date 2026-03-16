#!/usr/bin/env python3
"""Latency benchmark for Langfuse observability feature.

Runs 10 tasks, 2 times each (20 runs total) with and without observability enabled,
measuring total latency per request.

Usage:
    source .venv/bin/activate
    python benchmark_latency.py

Requirements:
    - Hermes CLI installed and configured
    - Valid API keys in ~/.hermes/.env
"""

import subprocess
import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict
import os
import sys

# 10 different simple tasks to benchmark
BENCHMARK_TASKS = [
    "What is 2 + 2?",
    "Write a one-sentence hello world in Python.",
    "Name three primary colors.",
    "What is the capital of France?",
    "List two benefits of exercise.",
    "What does CPU stand for?",
    "Give me a synonym for 'happy'.",
    "What is the opposite of hot?",
    "How many days are in a week?",
    "Name a fruit that is yellow.",
]

REPETITIONS = 2


@dataclass
class BenchmarkResult:
    task: str
    run_number: int
    latency_seconds: float
    success: bool
    error_message: str = ""


def run_single_query(prompt: str) -> tuple[float, bool, str]:
    """Run a single hermes query and measure latency.

    Returns:
        tuple: (latency_seconds, success, error_or_output)
    """
    start_time = time.time()
    try:
        result = subprocess.run(
            ["hermes", "chat", "-q", prompt],
            capture_output=True,
            text=True,
            timeout=600.0,  # 10 minute timeout per query
        )
        end_time = time.time()
        latency = end_time - start_time

        if result.returncode == 0:
            return latency, True, result.stdout
        else:
            return latency, False, result.stderr
    except subprocess.TimeoutExpired:
        return 600.0, False, "Timeout (600s)"
    except Exception as e:
        end_time = time.time()
        latency = end_time - start_time
        return latency, False, str(e)


def run_benchmark(observability_enabled: bool) -> List[BenchmarkResult]:
    """Run all benchmark tasks with the specified observability setting."""
    # Set observability via environment variable
    os.environ["HERMES_LANGFUSE_ENABLED"] = "true" if observability_enabled else "false"

    print(f"\n{'=' * 60}")
    print(f"Running benchmark with observability={'ON' if observability_enabled else 'OFF'}")
    print(f"{'=' * 60}")

    results: List[BenchmarkResult] = []

    for run_num in range(1, REPETITIONS + 1):
        print(f"\n--- Run {run_num}/{REPETITIONS} ---")

        for i, task in enumerate(BENCHMARK_TASKS, 1):
            print(f"  Task {i}/{len(BENCHMARK_TASKS)}: {task[:50]}...", end=" ", flush=True)

            latency, success, output = run_single_query(task)
            results.append(BenchmarkResult(
                task=task,
                run_number=run_num,
                latency_seconds=latency,
                success=success,
                error_message=output if not success else ""
            ))

            status = "✓" if success else "✗"
            print(f"[{status}] {latency:.2f}s")

    return results


def analyze_results(results: List[BenchmarkResult], label: str) -> Dict:
    """Analyze benchmark results and return statistics."""
    latencies = [r.latency_seconds for r in results if r.success]
    failed_count = sum(1 for r in results if not r.success)

    if not latencies:
        return {
            "label": label,
            "total_runs": len(results),
            "successful_runs": 0,
            "failed_runs": failed_count,
            "mean_latency": None,
            "median_latency": None,
            "std_dev": None,
            "min_latency": None,
            "max_latency": None,
        }

    return {
        "label": label,
        "total_runs": len(results),
        "successful_runs": len(latencies),
        "failed_runs": failed_count,
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_latency": min(latencies),
        "max_latency": max(latencies),
    }


def print_summary(results_with: List[BenchmarkResult], results_without: List[BenchmarkResult]):
    """Print a formatted summary of benchmark results."""
    stats_with = analyze_results(results_with, "With Observability")
    stats_without = analyze_results(results_without, "Without Observability")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for stats in [stats_without, stats_with]:
        print(f"\n{stats['label']}:")
        print(f"  Total runs: {stats['total_runs']}")
        print(f"  Successful: {stats['successful_runs']}")
        print(f"  Failed: {stats['failed_runs']}")

        if stats['mean_latency'] is not None:
            print(f"  Mean latency: {stats['mean_latency']:.3f}s")
            print(f"  Median latency: {stats['median_latency']:.3f}s")
            print(f"  Std dev: {stats['std_dev']:.3f}s")
            print(f"  Min latency: {stats['min_latency']:.3f}s")
            print(f"  Max latency: {stats['max_latency']:.3f}s")

    # Comparison
    if stats_with['mean_latency'] is not None and stats_without['mean_latency'] is not None:
        diff = stats_with['mean_latency'] - stats_without['mean_latency']
        pct_diff = (diff / stats_without['mean_latency']) * 100

        print(f"\n{'=' * 70}")
        print("COMPARISON (Observability ON vs OFF)")
        print("=" * 70)
        print(f"  Absolute difference: {diff:+.3f}s ({pct_diff:+.1f}%)")

        if pct_diff > 10:
            print(f"  Impact: HIGH (>10% overhead)")
        elif pct_diff > 5:
            print(f"  Impact: MODERATE (5-10% overhead)")
        elif pct_diff > 1:
            print(f"  Impact: LOW (1-5% overhead)")
        else:
            print(f"  Impact: MINIMAL (<1% overhead)")


def save_results(results_with: List[BenchmarkResult], results_without: List[BenchmarkResult]):
    """Save results to a JSON file for further analysis."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "tasks": BENCHMARK_TASKS,
        "repetitions": REPETITIONS,
        "with_observability": [asdict(r) for r in results_with],
        "without_observability": [asdict(r) for r in results_without],
        "summary": {
            "with": analyze_results(results_with, "With Observability"),
            "without": analyze_results(results_without, "Without Observability"),
        }
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    print("Hermes Agent Observability Latency Benchmark")
    print("=" * 70)
    print(f"Tasks: {len(BENCHMARK_TASKS)}")
    print(f"Repetitions per task: {REPETITIONS}")
    print(f"Total runs per configuration: {len(BENCHMARK_TASKS) * REPETITIONS}")

    # Check hermes is available
    try:
        result = subprocess.run(["hermes", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("\nError: 'hermes' command not found or not working.")
            print("Make sure Hermes is installed and in your PATH.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError checking hermes: {e}")
        sys.exit(1)

    print("\nHermes CLI detected. Starting benchmark...")

    # Run benchmarks
    # Order: without observability first to establish baseline
    results_without = run_benchmark(observability_enabled=False)
    results_with = run_benchmark(observability_enabled=True)

    # Print and save results
    print_summary(results_with, results_without)
    save_results(results_with, results_without)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
