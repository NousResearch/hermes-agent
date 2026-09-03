#!/usr/bin/env python3
"""Evaluate strict FTS5 versus opt-in NL search on privacy-safe corpora.

Cases are synthetic and each is materialized in a separate temporary SQLite DB.
The runner selects only cases whose ``packs`` are available in the checked-out
branch, so a stacked core/Latin/Slavic PR remains self-validating.

Usage:
  PYTHONPATH=. python scripts/nl_search_eval.py
  PYTHONPATH=. python scripts/nl_search_eval.py --packs default
  PYTHONPATH=. python scripts/nl_search_eval.py --packs default,es,fr,de,pt,it
  PYTHONPATH=. python scripts/nl_search_eval.py --packs all --json-out /tmp/nl-eval.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from hermes_state import SessionDB
from hermes_state_nl_expansion import _NL_LANG_PACKS

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "hermes_state" / "fixtures" / "nl_search_eval_v1.json"
_GENERIC_DISTRACTORS = (
    "generic scheduling record",
    "unrelated visual dashboard",
    "temporary document archive",
    "network inventory note",
    "ordinary release calendar",
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def resolve_packs(raw: str) -> set[str]:
    """Validate a requested pack set against packs in this checkout."""
    available = set(_NL_LANG_PACKS)
    if raw == "all":
        return available
    selected = {item.strip() for item in raw.split(",") if item.strip()}
    unknown = selected - available
    if unknown:
        raise ValueError(f"requested packs unavailable in this branch: {sorted(unknown)}")
    return selected


def select_cases(corpus: dict[str, Any], packs: set[str]) -> list[dict[str, Any]]:
    cases = corpus["cases"] + corpus.get("adversarial_cases", [])
    return [case for case in cases if set(case["packs"]).issubset(packs)]


def case_relevant(case: dict[str, Any]) -> list[str]:
    if "relevant" in case:
        return case["relevant"]
    return [case["target"]]


def case_distractors(case: dict[str, Any]) -> list[str]:
    return case.get("distractors", list(_GENERIC_DISTRACTORS))


def evaluate_case(case: dict[str, Any], natural_language: bool) -> dict[str, Any]:
    """Materialize one isolated relevance corpus and score the top five rows."""
    relevant = case_relevant(case)
    with tempfile.TemporaryDirectory(prefix="hermes-nl-eval-") as tmp:
        db = SessionDB(db_path=Path(tmp) / "state.db")
        relevant_sessions: set[str] = set()
        for index, text in enumerate(relevant):
            session_id = f"relevant-{case['id']}-{index}"
            relevant_sessions.add(session_id)
            db.create_session(session_id, source="eval")
            db.append_message(session_id, "assistant", text)
        for index, text in enumerate(case_distractors(case)):
            session_id = f"distractor-{case['id']}-{index}"
            db.create_session(session_id, source="eval")
            db.append_message(session_id, "assistant", text)
        started = time.perf_counter()
        rows = db.search_messages(case["query"], limit=5, natural_language=natural_language)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        db.close()

    ids = [row["session_id"] for row in rows]
    ranks = [index + 1 for index, session_id in enumerate(ids) if session_id in relevant_sessions]
    relevant_returned = len(ranks)
    if relevant:
        hit1 = bool(ranks and ranks[0] == 1)
        recall5 = relevant_returned / len(relevant)
        precision5 = relevant_returned / len(rows) if rows else 0.0
        rr = 1.0 / ranks[0] if ranks else 0.0
        absent_correct = None
    else:
        hit1 = rows == []
        recall5 = 1.0 if not rows else 0.0
        precision5 = 1.0 if not rows else 0.0
        rr = 1.0 if not rows else 0.0
        absent_correct = not rows
    return {
        "id": case["id"], "lang": case["lang"], "scenario": case["scenario"],
        "latency_ms": elapsed_ms, "returned": len(rows), "relevant": len(relevant),
        "relevant_returned": relevant_returned, "hit1": hit1, "recall5": recall5,
        "precision5": precision5, "rr": rr, "absent_correct": absent_correct,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(results)
    absent = [row for row in results if row["absent_correct"] is not None]
    return {
        "cases": count,
        "hit_at_1": sum(row["hit1"] for row in results) / count,
        "recall_at_5": sum(row["recall5"] for row in results) / count,
        "precision_at_5": sum(row["precision5"] for row in results) / count,
        "mrr": sum(row["rr"] for row in results) / count,
        "absent_query_accuracy": (
            sum(row["absent_correct"] for row in absent) / len(absent) if absent else None
        ),
        "latency_p50_ms": percentile([row["latency_ms"] for row in results], 0.50),
        "latency_p95_ms": percentile([row["latency_ms"] for row in results], 0.95),
        "latency_mean_ms": statistics.fmean(row["latency_ms"] for row in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--packs", default="all", help="comma-separated installed packs or 'all'")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--min-nl-recall", type=float)
    parser.add_argument("--min-nl-precision", type=float)
    parser.add_argument("--min-absent-accuracy", type=float)
    args = parser.parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    packs = resolve_packs(args.packs)
    cases = select_cases(corpus, packs)
    if not cases:
        raise ValueError("pack selection chose zero evaluation cases")
    modes = {
        "strict_fts5": [evaluate_case(case, False) for case in cases],
        "nl_opt_in": [evaluate_case(case, True) for case in cases],
    }
    report = {
        "corpus_version": corpus["version"], "selected_packs": sorted(packs),
        "corpus_cases": len(cases), "summary": {mode: summarize(rows) for mode, rows in modes.items()},
        "by_case": modes,
    }
    for mode, summary in report["summary"].items():
        absent = summary["absent_query_accuracy"]
        absent_text = "n/a" if absent is None else f"{absent:.3f}"
        print(
            f"{mode}: cases={summary['cases']} hit@1={summary['hit_at_1']:.3f} "
            f"recall@5={summary['recall_at_5']:.3f} precision@5={summary['precision_at_5']:.3f} "
            f"MRR={summary['mrr']:.3f} absent={absent_text} "
            f"p50={summary['latency_p50_ms']:.1f}ms p95={summary['latency_p95_ms']:.1f}ms"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    nl_summary = report["summary"]["nl_opt_in"]
    thresholds = {
        "recall@5": (args.min_nl_recall, nl_summary["recall_at_5"]),
        "precision@5": (args.min_nl_precision, nl_summary["precision_at_5"]),
        "absent accuracy": (args.min_absent_accuracy, nl_summary["absent_query_accuracy"]),
    }
    failed = [
        f"{name}={actual:.3f} < {minimum:.3f}"
        for name, (minimum, actual) in thresholds.items()
        if minimum is not None and (actual is None or actual < minimum)
    ]
    if failed:
        print("NL evaluation threshold failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
