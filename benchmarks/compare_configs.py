#!/usr/bin/env python3
"""
Compare multiple memory backend configurations side-by-side.

Usage:
    # Compare profiles
    python -m benchmarks.compare_configs --profiles balanced,developer,researcher

    # Compare embedding models
    python -m benchmarks.compare_configs --embeddings tfidf,auto

    # Ablation: test features individually
    python -m benchmarks.compare_configs --ablation

    # Custom parameter sweep
    python -m benchmarks.compare_configs --sweep w_importance=0.1,0.2,0.3,0.4

    # Full comparison table
    python -m benchmarks.compare_configs --profiles balanced --embeddings tfidf,auto --runs 3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.interface import BenchmarkConfig, AggregateResult
from benchmarks.runner import (
    run_benchmark, print_results, register_backend,
    BACKENDS,
)

# Plugin backends are auto-discovered by runner.py from benchmarks/backends/


def run_config(name: str, params: dict, runs: int = 3,
               judge_model: str = "heuristic") -> tuple:
    """Run a single configuration and return (AggregateResult, runs)."""
    config = BenchmarkConfig(
        backend_name="baseline-flat",
        profile=params.get("profile", "balanced"),
        embedding_model=params.get("embedding_model", "tfidf"),
        num_runs=runs,
        judge_model=judge_model,
        parameters=params,
    )
    print(f"\n--- {name} ---")
    agg, run_results = run_benchmark(config)
    return agg, run_results, config


def print_comparison_table(results: Dict[str, tuple]):
    """Print a side-by-side comparison table."""
    names = list(results.keys())
    if not names:
        return

    # Collect all categories
    all_cats = set()
    for name, (agg, runs, config) in results.items():
        all_cats.update(agg.per_category_mean.keys())
    cats = sorted(all_cats)

    # Header
    col_width = max(len(n) for n in names) + 2
    col_width = max(col_width, 12)
    print(f"\n{'='*80}")
    print(f"  CONFIGURATION COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Category':<25}", end="")
    for name in names:
        print(f" {name:>{col_width}}", end="")
    print()
    print(f"  {'─'*25}", end="")
    for _ in names:
        print(f" {'─'*col_width}", end="")
    print()

    # Category rows
    for cat in cats:
        print(f"  {cat:<25}", end="")
        for name in names:
            agg = results[name][0]
            score = agg.per_category_mean.get(cat, 0)
            print(f" {score:>{col_width}.1%}", end="")
        print()

    # Overall
    print(f"  {'─'*25}", end="")
    for _ in names:
        print(f" {'─'*col_width}", end="")
    print()
    print(f"  {'OVERALL':<25}", end="")
    for name in names:
        agg = results[name][0]
        print(f" {agg.mean_score:>{col_width}.1%}", end="")
    print()

    # Token cost
    print(f"  {'tokens/query':<25}", end="")
    for name in names:
        _, runs, _ = results[name]
        if runs:
            avg_tok = sum(
                r.token_usage.get("avg_recall_tokens_per_query", 0) for r in runs
            ) // len(runs)
            print(f" {'~' + str(avg_tok):>{col_width}}", end="")
        else:
            print(f" {'?':>{col_width}}", end="")
    print()

    # Timing
    print(f"  {'wall_time/run':<25}", end="")
    for name in names:
        _, runs, _ = results[name]
        if runs:
            avg_time = sum(r.wall_time_seconds for r in runs) / len(runs)
            print(f" {f'{avg_time:.1f}s':>{col_width}}", end="")
        else:
            print(f" {'?':>{col_width}}", end="")
    print()
    print(f"{'='*80}\n")


def run_profile_comparison(profiles: List[str], embedding: str, runs: int,
                           judge: str) -> Dict[str, tuple]:
    """Compare different config profiles."""
    results = {}

    # Always include baseline
    baseline_config = BenchmarkConfig(
        backend_name="baseline-flat",
        num_runs=runs,
        judge_model=judge,
    )
    print("\n--- baseline-flat ---")
    agg, run_results = run_benchmark(baseline_config)
    results["baseline"] = (agg, run_results, baseline_config)

    for profile in profiles:
        name = f"cog/{profile}"
        params = {"profile": profile, "embedding_model": embedding}
        agg, run_results, config = run_config(name, params, runs, judge)
        results[name] = (agg, run_results, config)

    return results


def run_embedding_comparison(embeddings: List[str], profile: str, runs: int,
                             judge: str) -> Dict[str, tuple]:
    """Compare different embedding models."""
    results = {}

    baseline_config = BenchmarkConfig(
        backend_name="baseline-flat",
        num_runs=runs,
        judge_model=judge,
    )
    print("\n--- baseline-flat ---")
    agg, run_results = run_benchmark(baseline_config)
    results["baseline"] = (agg, run_results, baseline_config)

    for emb in embeddings:
        name = f"cog/{emb}"
        params = {"profile": profile, "embedding_model": emb}
        agg, run_results, config = run_config(name, params, runs, judge)
        results[name] = (agg, run_results, config)

    return results


def run_parameter_sweep(param_name: str, values: List[str], profile: str,
                        embedding: str, runs: int, judge: str) -> Dict[str, tuple]:
    """Sweep a single parameter across values."""
    results = {}

    for val in values:
        name = f"{param_name}={val}"
        params = {
            "profile": profile,
            "embedding_model": embedding,
            param_name: float(val) if '.' in val else int(val),
        }
        agg, run_results, config = run_config(name, params, runs, judge)
        results[name] = (agg, run_results, config)

    return results


def run_ablation(embedding: str, runs: int, judge: str) -> Dict[str, tuple]:
    """Run ablation: test feature contributions by disabling them."""
    configs = {
        "baseline": {"backend": "baseline-flat"},
        "full": {"profile": "balanced", "embedding_model": embedding},
        "no_contradiction": {
            "profile": "balanced", "embedding_model": embedding,
            "contradiction_threshold": 1.0,  # effectively disable
        },
        "low_importance": {
            "profile": "balanced", "embedding_model": embedding,
            "w_importance": 0.01,  # nearly disable importance
        },
        "no_hebbian": {
            "profile": "balanced", "embedding_model": embedding,
            "hebbian_learning_rate": 0.0,  # disable Hebbian links
        },
        "high_decay": {
            "profile": "balanced", "embedding_model": embedding,
            "d": 0.8,  # aggressive decay
        },
        "low_decay": {
            "profile": "balanced", "embedding_model": embedding,
            "d": 0.3,  # slow decay
        },
    }

    results = {}
    for name, params in configs.items():
        if params.get("backend") == "baseline-flat":
            config = BenchmarkConfig(
                backend_name="baseline-flat",
                num_runs=runs,
                judge_model=judge,
            )
            print(f"\n--- {name} ---")
            agg, run_results = run_benchmark(config)
            results[name] = (agg, run_results, config)
        else:
            agg, run_results, config = run_config(name, params, runs, judge)
            results[name] = (agg, run_results, config)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare cognitive memory configurations"
    )
    parser.add_argument("--profiles", default=None,
                        help="Comma-separated profiles to compare (e.g., balanced,developer)")
    parser.add_argument("--embeddings", default=None,
                        help="Comma-separated embedding models (e.g., tfidf,auto)")
    parser.add_argument("--sweep", default=None,
                        help="Parameter sweep: param=val1,val2,val3")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study")
    parser.add_argument("--profile", default="balanced",
                        help="Base profile for embedding/sweep comparisons")
    parser.add_argument("--embedding", default="tfidf",
                        help="Base embedding for profile/sweep comparisons")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per configuration")
    parser.add_argument("--judge-model", default="heuristic",
                        help="Judge model")
    parser.add_argument("--output", default=None,
                        help="Save results JSON to this path")

    args = parser.parse_args()

    if args.ablation:
        results = run_ablation(args.embedding, args.runs, args.judge_model)
    elif args.profiles:
        profiles = [p.strip() for p in args.profiles.split(",")]
        results = run_profile_comparison(profiles, args.embedding, args.runs, args.judge_model)
    elif args.embeddings:
        embeddings = [e.strip() for e in args.embeddings.split(",")]
        results = run_embedding_comparison(embeddings, args.profile, args.runs, args.judge_model)
    elif args.sweep:
        param, vals = args.sweep.split("=")
        values = [v.strip() for v in vals.split(",")]
        results = run_parameter_sweep(param, values, args.profile, args.embedding,
                                      args.runs, args.judge_model)
    else:
        # Default: compare baseline vs cognitive balanced
        results = run_profile_comparison(["balanced"], args.embedding, args.runs, args.judge_model)

    print_comparison_table(results)

    if args.output:
        output_data = {}
        for name, (agg, runs, config) in results.items():
            output_data[name] = {
                "overall": agg.mean_score,
                "std": agg.std_score,
                "categories": agg.per_category_mean,
                "num_runs": agg.num_runs,
            }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
