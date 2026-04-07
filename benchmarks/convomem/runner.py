"""
ConvoMem Benchmark Runner for Hermes Memory Backends.

Usage:
    # Quick smoke test
    python -m benchmarks.convomem.runner --sample 5

    # Default run (all n_evidence levels) against a specific backend
    python -m benchmarks.convomem.runner --backend hindsight

    # Filter to one evidence level
    python -m benchmarks.convomem.runner --backend honcho --n-evidence 3

    # Full dataset
    python -m benchmarks.convomem.runner --backend baseline-flat --full

    # Local JSON instead of HuggingFace
    python -m benchmarks.convomem.runner --local /path/to/convomem.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.judge import HeuristicJudge, MemoryJudge
from benchmarks.convomem.adapter import (
    N_EVIDENCE_LEVELS,
    ConvoMemSummary,
    load_convomem_dataset,
    load_convomem_local,
    run_convomem,
)


def _resolve_backend_class(name: str):
    """Resolve a backend class from the runner.py BACKENDS registry."""
    import benchmarks.runner as runner_mod
    backends = getattr(runner_mod, "BACKENDS", None)
    if not backends:
        raise RuntimeError(
            "benchmarks.runner.BACKENDS registry not found. "
            "Has the registry export name changed?"
        )
    if name not in backends:
        available = sorted(backends.keys())
        raise SystemExit(f"Unknown backend: {name!r}. Available: {available}")
    return backends[name]


def print_summary(summary: ConvoMemSummary, elapsed: float, backend: str) -> None:
    print(f"\n{'='*65}")
    print(f"  CONVOMEM RESULTS  (backend={backend})")
    print(f"{'='*65}")
    print(f"  Total:   {summary.total}")
    print(f"  Correct: {summary.correct}")
    print(f"  Score:   {summary.score:.3f} ({summary.score*100:.1f}%)")
    print(f"  Time:    {elapsed:.1f}s")
    if summary.by_type:
        print(f"{'─'*65}")
        print(f"  {'Evidence Type':<32} {'N':>5} {'Score':>8}")
        print(f"  {'─'*32} {'─'*5} {'─'*8}")
        for etype, stats in sorted(summary.by_type.items(), key=lambda x: -x[1]["score"]):
            print(f"  {etype:<32} {stats['total']:>5} {stats['score']:>8.3f}")
    if summary.by_n_evidence:
        print(f"{'─'*65}")
        print(f"  {'n_evidence':<32} {'N':>5} {'Score':>8}")
        print(f"  {'─'*32} {'─'*5} {'─'*8}")
        for level, stats in sorted(summary.by_n_evidence.items()):
            print(f"  {level:<32} {stats['total']:>5} {stats['score']:>8.3f}")
    if summary.mean_metrics:
        m = summary.mean_metrics
        print(f"{'─'*65}")
        print(f"  Retrieval Metrics:")
        print(f"    Recall@1:  {m.get('recall_at_1', 0.0):.3f}")
        print(f"    Recall@5:  {m.get('recall_at_5', 0.0):.3f}")
        print(f"    MRR:       {m.get('mrr', 0.0):.3f}")
        print(f"    Token F1:  {m.get('token_f1', 0.0):.3f}")
    print(f"{'='*65}\n")


def save_results(
    summary: ConvoMemSummary,
    output_path: Path,
    elapsed: float,
    backend: str,
    n_evidence: int | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _METRIC_SUBSET = {"recall_at_1", "recall_at_5", "mrr", "token_f1", "exact_match"}
    data = {
        "benchmark": "convomem",
        "dataset": "Salesforce/ConvoMem",
        "backend": backend,
        "n_evidence_filter": n_evidence,
        "total": summary.total,
        "correct": summary.correct,
        "score": summary.score,
        "wall_time_seconds": elapsed,
        "mean_metrics": summary.mean_metrics,
        "by_type": summary.by_type,
        "by_n_evidence": {str(k): v for k, v in summary.by_n_evidence.items()},
        "results": [
            {
                "question_id": r.question_id,
                "evidence_type": r.evidence_type,
                "n_evidence": r.n_evidence,
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
        description="Run the ConvoMem benchmark against a Hermes memory backend"
    )
    parser.add_argument("--backend", default="baseline-flat",
        help="Memory backend name (resolved via benchmarks.runner.BACKENDS). "
             "Default: baseline-flat")
    parser.add_argument("--sample", type=int, default=None,
        help="Limit to first N questions after filtering")
    parser.add_argument("--full", action="store_true",
        help="Disable n_evidence filtering (use all levels)")
    parser.add_argument("--n-evidence", type=int, default=None,
        choices=N_EVIDENCE_LEVELS,
        help=f"Filter to one n_evidence level (default: all levels). "
             f"Valid: {N_EVIDENCE_LEVELS}")
    parser.add_argument("--category", default=None,
        help="Filter to one evidence_type")
    parser.add_argument("--judge-model", default="heuristic",
        help="Judge: 'heuristic' (default) or an LLM model name")
    parser.add_argument("--local", default=None,
        help="Load from local JSON file instead of HuggingFace")
    parser.add_argument("--hf-cache", default=None,
        help="HuggingFace cache directory")
    parser.add_argument("--output", default=None,
        help="Output JSON path (default: benchmarks/results/convomem_<backend>.json)")
    parser.add_argument("--json", action="store_true",
        help="Print results as JSON to stdout")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Print per-question results")
    parser.add_argument("--top-k", type=int, default=10,
        help="Memories to recall per question (default: 10)")
    args = parser.parse_args()

    backend_cls = _resolve_backend_class(args.backend)

    # n_evidence filtering: None means all levels; --full also means all levels
    n_evidence_filter = None if (args.full or args.n_evidence is None) else args.n_evidence

    if args.local:
        questions = load_convomem_local(
            args.local,
            sample=args.sample,
            n_evidence=n_evidence_filter,
            category_filter=args.category,
        )
    else:
        questions = load_convomem_dataset(
            hf_cache=args.hf_cache,
            sample=args.sample,
            n_evidence=n_evidence_filter,
            category_filter=args.category,
        )

    if not questions:
        print("No questions loaded. Exiting.")
        sys.exit(1)

    judge = HeuristicJudge(model="heuristic") if args.judge_model == "heuristic" \
        else MemoryJudge(model=args.judge_model)

    # Most plugin backends accept no kwargs from the caller — they read their
    # own config. FlatMemoryStore also takes no kwargs. Keep it empty to avoid
    # TypeError on unexpected keyword arguments.
    backend_kwargs: dict = {}

    print(f"\nRunning ConvoMem ({len(questions)} questions)...")
    print(f"  Backend:     {args.backend}")
    print(f"  n_evidence:  {'all' if n_evidence_filter is None else n_evidence_filter}")
    print(f"  Category:    {args.category or 'all'}")
    print(f"  Judge:       {args.judge_model}")
    print()

    start = time.time()
    summary = run_convomem(
        questions=questions,
        judge=judge,
        backend_cls=backend_cls,
        backend_kwargs=backend_kwargs,
        top_k=args.top_k,
        verbose=args.verbose,
    )
    elapsed = time.time() - start

    if args.json:
        print(json.dumps({
            "benchmark": "convomem",
            "backend": args.backend,
            "total": summary.total,
            "correct": summary.correct,
            "score": summary.score,
            "mean_metrics": summary.mean_metrics,
            "by_type": summary.by_type,
            "by_n_evidence": {str(k): v for k, v in summary.by_n_evidence.items()},
        }, indent=2))
    else:
        print_summary(summary, elapsed, backend=args.backend)

    output_path = Path(args.output or f"benchmarks/results/convomem_{args.backend}.json")
    save_results(summary, output_path, elapsed, backend=args.backend,
                 n_evidence=n_evidence_filter)


if __name__ == "__main__":
    main()
