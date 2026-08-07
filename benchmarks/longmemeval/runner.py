"""
LongMemEval Benchmark Runner for Hermes Cognitive Memory.

Usage:
    python -m benchmarks.longmemeval.runner --sample 5          # Quick test
    python -m benchmarks.longmemeval.runner                     # Full 500 questions
    python -m benchmarks.longmemeval.runner --judge-model claude-haiku-4-5
    python -m benchmarks.longmemeval.runner --json
    python -m benchmarks.longmemeval.runner --local /path/to/longmemeval_oracle.json

The runner downloads the dataset from HuggingFace on first run (streaming,
no local copy needed). Results are saved to benchmarks/results/longmemeval.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.judge import HeuristicJudge, MemoryJudge
from benchmarks.longmemeval.adapter import (
    LongMemSummary,
    load_longmemeval_dataset,
    load_longmemeval_local,
    run_longmemeval,
)


def print_summary(summary: LongMemSummary, elapsed: float, stratified: bool = False) -> None:
    """Print a formatted results table."""
    print(f"\n{'='*60}")
    print(f"  LONGMEMEVAL RESULTS")
    print(f"{'='*60}")
    print(f"  Total:   {summary.total}")
    print(f"  Correct: {summary.correct}")
    print(f"  Score:   {summary.score:.3f} ({summary.score*100:.1f}%)")
    print(f"  Time:    {elapsed:.1f}s")
    print(f"{'─'*60}")
    print(f"  {'Question Type':<30} {'N':>5} {'Score':>8}")
    print(f"  {'─'*30} {'─'*5} {'─'*8}")
    for qtype, stats in sorted(summary.by_type.items(), key=lambda x: -x[1]["score"]):
        print(
            f"  {qtype:<30} {stats['total']:>5} {stats['score']:>8.3f}"
        )

    # Print overall retrieval metrics
    if summary.mean_metrics:
        m = summary.mean_metrics
        print(f"{'─'*60}")
        print(f"  Retrieval Metrics:")
        print(f"    Recall@1:   {m.get('recall_at_1', 0.0):.3f}")
        print(f"    Recall@5:   {m.get('recall_at_5', 0.0):.3f}")
        print(f"    MRR:        {m.get('mrr', 0.0):.3f}")
        print(f"    Token F1:   {m.get('token_f1', 0.0):.3f}")

    # Print per-type metrics breakdown if --stratified
    if stratified and summary.by_type:
        print(f"{'─'*60}")
        print(f"  Per-Type Retrieval Metrics:")
        print(f"  {'Question Type':<30} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'TF1':>6}")
        print(f"  {'─'*30} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
        for qtype, stats in sorted(summary.by_type.items(), key=lambda x: -x[1]["score"]):
            tm = stats.get("mean_metrics", {})
            r1 = tm.get("recall_at_1", 0.0)
            r5 = tm.get("recall_at_5", 0.0)
            m_mrr = tm.get("mrr", 0.0)
            tf1 = tm.get("token_f1", 0.0)
            print(f"  {qtype:<30} {r1:>6.3f} {r5:>6.3f} {m_mrr:>6.3f} {tf1:>6.3f}")

    print(f"{'='*60}\n")


def save_results(summary: LongMemSummary, output_path: Path, elapsed: float) -> None:
    """Save results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _METRIC_SUBSET = {"recall_at_1", "recall_at_5", "mrr", "token_f1", "exact_match"}

    data = {
        "benchmark": "longmemeval",
        "dataset": "xiaowu0162/longmemeval-cleaned (longmemeval_oracle)",
        "total": summary.total,
        "correct": summary.correct,
        "score": summary.score,
        "wall_time_seconds": elapsed,
        "mean_metrics": summary.mean_metrics,
        "by_type": summary.by_type,
        "results": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "correct": r.correct,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "recalled": r.recalled[:200] if r.recalled else "",
                "metrics": {k: v for k, v in r.metrics.items() if k in _METRIC_SUBSET},
            }
            for r in summary.results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LongMemEval benchmark against Hermes cognitive memory"
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only run first N questions (default: all 500)",
    )
    parser.add_argument(
        "--judge-model", default="heuristic",
        help="Judge model: 'heuristic' (default) or 'claude-haiku-4-5' for LLM judge",
    )
    parser.add_argument(
        "--embedding", default="auto",
        help="Embedding model: 'auto', 'sentence-transformers', 'tfidf'",
    )
    parser.add_argument(
        "--profile", default="balanced",
        help="Cognitive memory profile",
    )
    parser.add_argument(
        "--question-type", default=None,
        help="Only run questions of this type",
    )
    parser.add_argument(
        "--local", default=None,
        help="Load from local JSON file instead of HuggingFace",
    )
    parser.add_argument(
        "--hf-cache", default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output", default="benchmarks/results/longmemeval.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print results as JSON to stdout",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-question results",
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="Print per-type retrieval metrics breakdown in summary",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of memories to recall per question (default: 10)",
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Use explore() (PPR graph walk) instead of recall() for retrieval",
    )
    parser.add_argument(
        "--ingest", default="raw", choices=["raw", "chunk", "summarize"],
        help="Ingestion strategy: raw (default), chunk (semantic grouping), summarize (LLM facts)",
    )
    args = parser.parse_args()

    # Load questions
    if args.local:
        questions = load_longmemeval_local(
            args.local,
            sample=args.sample,
            question_type_filter=args.question_type,
        )
    else:
        questions = load_longmemeval_dataset(
            hf_cache=args.hf_cache,
            sample=args.sample,
            question_type_filter=args.question_type,
        )

    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)

    # Set up judge
    if args.judge_model == "heuristic":
        judge = HeuristicJudge(model="heuristic")
    else:
        judge = MemoryJudge(model=args.judge_model)

    # Set up backend kwargs
    backend_kwargs = {
        "profile": args.profile,
        "embedding_model": args.embedding,
    }

    print(f"\nRunning LongMemEval ({len(questions)} questions)...")
    print(f"  Backend: cognitive memory ({args.profile}, {args.embedding})")
    print(f"  Judge:   {args.judge_model}")
    print(f"  Mode:    {'explore (PPR graph walk)' if args.explore else 'recall (default)'}")
    print(f"  Ingest:  {args.ingest}")
    print()

    start = time.time()
    summary = run_longmemeval(
        questions=questions,
        judge=judge,
        backend_kwargs=backend_kwargs,
        top_k=args.top_k,
        verbose=args.verbose,
        explore=args.explore,
        ingest_strategy=args.ingest,
    )
    elapsed = time.time() - start

    if args.json:
        result_dict = {
            "benchmark": "longmemeval",
            "total": summary.total,
            "correct": summary.correct,
            "score": summary.score,
            "mean_metrics": summary.mean_metrics,
            "by_type": summary.by_type,
        }
        print(json.dumps(result_dict, indent=2))
    else:
        print_summary(summary, elapsed, stratified=args.stratified)

    output_path = Path(args.output)
    save_results(summary, output_path, elapsed)


if __name__ == "__main__":
    main()
