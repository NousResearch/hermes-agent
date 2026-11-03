"""
LoCoMo Benchmark Runner for Hermes Cognitive Memory.

Usage:
    python -m benchmarks.locomo.runner --sample 5          # Quick smoke test
    python -m benchmarks.locomo.runner                     # Full dataset
    python -m benchmarks.locomo.runner --judge-model claude-haiku-4-5
    python -m benchmarks.locomo.runner --json
    python -m benchmarks.locomo.runner --local /path/to/locomo.json

The runner downloads the dataset from HuggingFace on first run (streaming,
no local copy needed).  Results are saved to benchmarks/results/locomo.json.

Reference:
    Maharana et al. (2024). "Evaluating Very Long-Term Conversational Memory
    of LLM Agents." arXiv:2402.17753. https://arxiv.org/abs/2402.17753
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path so imports work when run as __main__
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.judge import HeuristicJudge, MemoryJudge
from benchmarks.locomo.adapter import (
    LoCoMoSummary,
    load_locomo_dataset,
    load_locomo_local,
    run_locomo,
)


# ── Output formatting ──


def print_summary(summary: LoCoMoSummary, elapsed: float) -> None:
    """Print a formatted results table to stdout."""
    print(f"\n{'='*65}")
    print(f"  LOCOMO RESULTS")
    print(f"{'='*65}")
    print(f"  Total:   {summary.total}")
    print(f"  Correct: {summary.correct}")
    print(f"  Score:   {summary.score:.3f} ({summary.score*100:.1f}%)")
    print(f"  Time:    {elapsed:.1f}s")

    # Per-type breakdown
    print(f"{'─'*65}")
    print(f"  {'Question Type':<20} {'N':>5} {'Score':>8}")
    print(f"  {'─'*20} {'─'*5} {'─'*8}")
    for qtype, stats in sorted(summary.by_type.items(), key=lambda x: -x[1]["score"]):
        print(
            f"  {qtype:<20} {stats['total']:>5} {stats['score']:>8.3f}"
        )

    # Retrieval metrics summary (if present)
    if summary.mean_metrics:
        print(f"{'─'*65}")
        print(f"  RETRIEVAL METRICS (mean across all questions)")
        print(f"{'─'*65}")
        metric_pairs = [
            ("recall_at_1",       "Recall@1"),
            ("recall_at_5",       "Recall@5"),
            ("recall_at_10",      "Recall@10"),
            ("mrr",               "MRR"),
            ("ndcg_at_5",         "NDCG@5"),
            ("ndcg_at_10",        "NDCG@10"),
            ("average_precision", "MAP"),
            ("token_f1",          "Token F1"),
            ("exact_match",       "Exact Match"),
        ]
        for key, label in metric_pairs:
            val = summary.mean_metrics.get(key)
            if val is not None:
                print(f"  {label:<22} {val:.3f}")

    print(f"{'='*65}\n")


def save_results(summary: LoCoMoSummary, output_path: Path, elapsed: float) -> None:
    """Save results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark": "locomo",
        "dataset": "snap-research/locomo",
        "reference": "Maharana et al. (2024) arXiv:2402.17753",
        "total": summary.total,
        "correct": summary.correct,
        "score": summary.score,
        "wall_time_seconds": elapsed,
        "by_type": summary.by_type,
        "mean_metrics": summary.mean_metrics,
        "results": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "correct": r.correct,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "recalled": r.recalled[:200] if r.recalled else "",
                "recall_count": r.recall_count,
                "metrics": r.metrics,
            }
            for r in summary.results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Results saved to {output_path}")


# ── CLI ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo benchmark against Hermes cognitive memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only run first N questions (default: all ~2000)",
    )
    parser.add_argument(
        "--judge-model", default="heuristic",
        help=(
            "Judge model: 'heuristic' (default, no API calls) "
            "or a Claude model name like 'claude-haiku-4-5'"
        ),
    )
    parser.add_argument(
        "--embedding", default="auto",
        help="Embedding model for cognitive store: 'auto', 'sentence-transformers', 'tfidf'",
    )
    parser.add_argument(
        "--profile", default="balanced",
        help="Cognitive memory profile (default: balanced)",
    )
    parser.add_argument(
        "--question-type", default=None,
        help="Only run questions of this type (single_hop|multi_hop|temporal|open_domain)",
    )
    parser.add_argument(
        "--local", default=None,
        help="Load from a local JSON file instead of HuggingFace",
    )
    parser.add_argument(
    "--hf-cache", default=None,
    help="HuggingFace cache directory (default: HF_DATASETS_CACHE or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--output", default="benchmarks/results/locomo.json",
        help="Output JSON path (default: benchmarks/results/locomo.json)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print results as JSON to stdout instead of the formatted table",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-question results during evaluation",
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

    # ── Load questions ──
    if args.local:
        questions = load_locomo_local(
            args.local,
            sample=args.sample,
            question_type_filter=args.question_type,
        )
    else:
        try:
            questions = load_locomo_dataset(
                hf_cache=args.hf_cache,
                sample=args.sample,
                question_type_filter=args.question_type,
            )
        except RuntimeError as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            sys.exit(1)

    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)

    # ── Set up judge ──
    if args.judge_model == "heuristic":
        judge = HeuristicJudge(model="heuristic")
    else:
        judge = MemoryJudge(model=args.judge_model)

    # ── Backend configuration ──
    backend_kwargs = {
        "profile": args.profile,
        "embedding_model": args.embedding,
    }

    print(f"\nRunning LoCoMo ({len(questions)} questions)...")
    print(f"  Backend: cognitive memory ({args.profile}, {args.embedding})")
    print(f"  Judge:   {args.judge_model}")
    print(f"  top_k:   {args.top_k}")
    print(f"  Mode:    {'explore (PPR graph walk)' if args.explore else 'recall (default)'}")
    print(f"  Ingest:  {args.ingest}")
    print()

    start = time.time()
    summary = run_locomo(
        questions=questions,
        judge=judge,
        backend_kwargs=backend_kwargs,
        top_k=args.top_k,
        verbose=args.verbose,
        explore=args.explore,
        ingest_strategy=args.ingest,
    )
    elapsed = time.time() - start

    # ── Output ──
    if args.json:
        result_dict = {
            "benchmark": "locomo",
            "total": summary.total,
            "correct": summary.correct,
            "score": summary.score,
            "by_type": summary.by_type,
            "mean_metrics": summary.mean_metrics,
        }
        print(json.dumps(result_dict, indent=2))
    else:
        print_summary(summary, elapsed)

    output_path = Path(args.output)
    save_results(summary, output_path, elapsed)


if __name__ == "__main__":
    main()
