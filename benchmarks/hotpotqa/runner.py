"""
HotpotQA Benchmark Runner for Hermes Cognitive Memory.

Usage:
    python -m benchmarks.hotpotqa.runner --sample 5           # Quick smoke test
    python -m benchmarks.hotpotqa.runner                      # Full 500 questions
    python -m benchmarks.hotpotqa.runner --judge-model claude-haiku-4-5
    python -m benchmarks.hotpotqa.runner --question-type bridge
    python -m benchmarks.hotpotqa.runner --difficulty easy
    python -m benchmarks.hotpotqa.runner --json
    python -m benchmarks.hotpotqa.runner --local /path/to/hotpot_dev_distractor_v1.json

Downloads the dataset from HuggingFace on first run (streaming, no full local
copy needed by default).  Results are saved to benchmarks/results/hotpotqa.json.

Reference:
    Yang et al. (2018). HotpotQA: A Dataset for Diverse, Explainable
    Multi-hop Question Answering. EMNLP 2018. https://arxiv.org/abs/1809.09600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path so we can run as a module from any CWD
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.judge import HeuristicJudge, MemoryJudge
from benchmarks.hotpotqa.adapter import (
    HotpotSummary,
    load_hotpotqa_dataset,
    load_hotpotqa_local,
    run_hotpotqa,
)


# ── Formatting ──


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_row(label: str, n: int, score: float, sf_recall: float, f1: float, em: float) -> str:
    return (
        f"  {label:<28} {n:>5}  {_pct(score):>8}  "
        f"{_pct(sf_recall):>12}  {f1:>7.3f}  {_pct(em):>7}"
    )


def print_summary(summary: HotpotSummary, elapsed: float) -> None:
    """Print a formatted results table to stdout."""
    sep = "=" * 80
    thin = "-" * 80

    print(f"\n{sep}")
    print(f"  HOTPOTQA BENCHMARK RESULTS")
    print(sep)
    print(f"  Total questions : {summary.total}")
    print(f"  Correct         : {summary.correct}")
    print(f"  Accuracy        : {_pct(summary.score)}")
    print(f"  SF Recall       : {_pct(summary.avg_supporting_facts_recall)}  "
          f"(fraction of gold paragraphs retrieved)")
    print(f"  Token F1        : {summary.avg_token_f1:.3f}")
    print(f"  Exact Match     : {_pct(summary.avg_exact_match)}")
    print(f"  Wall time       : {elapsed:.1f}s")
    print(thin)

    header = (
        f"  {'Category':<28} {'N':>5}  {'Accuracy':>8}  "
        f"{'SF Recall':>12}  {'Token F1':>8}  {'EM':>7}"
    )
    print(header)
    print(f"  {'─'*28} {'─'*5}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*7}")

    print(f"  {'-- By Question Type --':<28}")
    for qtype in sorted(summary.by_type):
        s = summary.by_type[qtype]
        print(_fmt_row(
            f"  {qtype}",
            s["total"], s["score"],
            s["avg_supporting_facts_recall"],
            s["avg_token_f1"],
            s["avg_exact_match"],
        ))

    print(f"  {'-- By Difficulty --':<28}")
    for diff in ["easy", "medium", "hard"]:
        if diff not in summary.by_difficulty:
            continue
        s = summary.by_difficulty[diff]
        print(_fmt_row(
            f"  {diff}",
            s["total"], s["score"],
            s["avg_supporting_facts_recall"],
            s["avg_token_f1"],
            s["avg_exact_match"],
        ))

    print(thin)
    print(_fmt_row(
        "  OVERALL",
        summary.total, summary.score,
        summary.avg_supporting_facts_recall,
        summary.avg_token_f1,
        summary.avg_exact_match,
    ))
    print(f"{sep}\n")


def save_results(summary: HotpotSummary, output_path: Path, elapsed: float) -> None:
    """Save full results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark": "hotpotqa",
        "dataset": f"{DATASET_REF} (distractor / validation)",
        "total": summary.total,
        "correct": summary.correct,
        "score": summary.score,
        "avg_supporting_facts_recall": summary.avg_supporting_facts_recall,
        "avg_token_f1": summary.avg_token_f1,
        "avg_exact_match": summary.avg_exact_match,
        "wall_time_seconds": elapsed,
        "by_type": summary.by_type,
        "by_difficulty": summary.by_difficulty,
        "results": [
            {
                "question_id": r.question_id,
                "question_type": r.question_type,
                "difficulty": r.difficulty,
                "correct": r.correct,
                "supporting_facts_recall": r.supporting_facts_recall,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "predicted_answer": r.predicted_answer[:300] if r.predicted_answer else "",
                "metrics": r.metrics,
            }
            for r in summary.results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Results saved to {output_path}")


DATASET_REF = "hotpotqa/hotpot_qa"


# ── CLI entry point ──


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HotpotQA multi-hop QA benchmark against Hermes cognitive memory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample", type=int, default=500,
        help="Number of questions (stratified sample across type × difficulty)",
    )
    parser.add_argument(
        "--judge-model", default="heuristic",
        help="Judge model: 'heuristic' or e.g. 'claude-haiku-4-5' for LLM judge",
    )
    parser.add_argument(
        "--embedding", default="auto",
        help="Embedding model: 'auto', 'sentence-transformers', 'tfidf'",
    )
    parser.add_argument(
        "--profile", default="balanced",
        help="Cognitive memory profile passed to the backend",
    )
    parser.add_argument(
        "--difficulty", default=None, choices=["easy", "medium", "hard"],
        help="Only run questions of this difficulty level",
    )
    parser.add_argument(
        "--question-type", default=None, choices=["bridge", "comparison"],
        help="Only run questions of this type",
    )
    parser.add_argument(
        "--local", default=None,
        help="Load from a local JSON file instead of HuggingFace",
    )
    parser.add_argument(
        "--hf-cache", default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--output", default="benchmarks/results/hotpotqa.json",
        help="Path to write the JSON results file",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print summary as JSON to stdout (in addition to saving the file)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-question results as they are evaluated",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of memories to recall per question",
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Use explore() (PPR graph walk) instead of recall() for retrieval",
    )
    args = parser.parse_args()

    # ── Load questions ──
    if args.local:
        questions = load_hotpotqa_local(
            args.local,
            sample=args.sample,
            difficulty_filter=args.difficulty,
            question_type_filter=args.question_type,
        )
    else:
        try:
            questions = load_hotpotqa_dataset(
                sample=args.sample,
                hf_cache=args.hf_cache,
                difficulty_filter=args.difficulty,
                question_type_filter=args.question_type,
            )
        except Exception as exc:
            print(f"  HuggingFace load failed: {exc}", file=sys.stderr)
            print("  Tip: use --local to load from a local JSON file.", file=sys.stderr)
            sys.exit(1)

    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)

    # ── Set up judge ──
    if args.judge_model == "heuristic":
        judge = HeuristicJudge(model="heuristic")
    else:
        judge = MemoryJudge(model=args.judge_model)

    # ── Set up backend kwargs ──
    backend_kwargs = {
        "profile": args.profile,
        "embedding_model": args.embedding,
    }

    # ── Run ──
    print(f"\nRunning HotpotQA ({len(questions)} questions)...")
    print(f"  Backend : cognitive memory (profile={args.profile}, embedding={args.embedding})")
    print(f"  Judge   : {args.judge_model}")
    print(f"  Top-K   : {args.top_k}")
    print(f"  Mode    : {'explore (PPR graph walk)' if args.explore else 'recall (default)'}")
    print()

    start = time.time()
    summary = run_hotpotqa(
        questions=questions,
        judge=judge,
        backend_kwargs=backend_kwargs,
        top_k=args.top_k,
        verbose=args.verbose,
        explore=args.explore,
    )
    elapsed = time.time() - start

    # ── Output ──
    if args.json:
        result_dict = {
            "benchmark": "hotpotqa",
            "total": summary.total,
            "correct": summary.correct,
            "score": summary.score,
            "avg_supporting_facts_recall": summary.avg_supporting_facts_recall,
            "avg_token_f1": summary.avg_token_f1,
            "avg_exact_match": summary.avg_exact_match,
            "by_type": summary.by_type,
            "by_difficulty": summary.by_difficulty,
        }
        print(json.dumps(result_dict, indent=2))
    else:
        print_summary(summary, elapsed)

    output_path = Path(args.output)
    save_results(summary, output_path, elapsed)


if __name__ == "__main__":
    main()
