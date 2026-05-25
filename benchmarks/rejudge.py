"""Re-score a saved benchmark result JSON with a different judge.

Standard / academic benchmark runners (locomo, longmemeval, hotpotqa,
convomem) save per-question records that include the question text,
gold answer, and recalled context. That's enough to re-run the judge
without re-running the benchmark itself — useful for swapping the
heuristic judge for an LLM judge after the fact.

Usage:
    python -m benchmarks.rejudge \\
        --judge-model claude-haiku-4-5 \\
        --input  runs/standard/locomo.mnemoria.json \\
        --output runs/standard/locomo.mnemoria.haiku.json

The output JSON has the same shape as the input but with `correct`
flags re-set, `judge_model` recorded, and the top-level aggregate
score re-computed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Make the package importable when run as __main__.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from benchmarks.judge import HeuristicJudge, MemoryJudge


def _select_judge(model: str):
    if model == "heuristic":
        return HeuristicJudge(model="heuristic")
    return MemoryJudge(model=model)


def _per_record_score(rec: dict, judge) -> bool:
    question = rec.get("question") or rec.get("query") or ""
    gold = rec.get("gold_answer") or rec.get("answer") or ""
    context = rec.get("context") or rec.get("recalled") or ""
    if not (question and gold):
        return bool(rec.get("correct", False))
    jr = judge.judge_answer(question, gold, context)
    return bool(jr.correct)


def rejudge_file(input_path: Path, output_path: Path, model: str) -> dict:
    data = json.loads(input_path.read_text())
    judge = _select_judge(model)

    records = data.get("results") or data.get("records") or []
    if not records:
        raise SystemExit(
            f"No per-question records in {input_path} — re-running the judge "
            "needs a saved per-question breakdown."
        )

    correct = 0
    by_type: dict[str, list[bool]] = defaultdict(list)

    start = time.time()
    for rec in records:
        ok = _per_record_score(rec, judge)
        rec["correct"] = ok
        if ok:
            correct += 1
        qtype = rec.get("question_type") or rec.get("type") or "default"
        by_type[qtype].append(ok)
    elapsed = time.time() - start

    total = len(records)
    data["correct"] = correct
    data["total"] = total
    data["score"] = correct / total if total else 0.0
    data["judge_model"] = model
    data["rejudge_wall_time_seconds"] = elapsed
    data["by_type"] = {
        t: {
            "total": len(rs),
            "correct": sum(rs),
            "score": sum(rs) / len(rs),
        }
        for t, rs in by_type.items()
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    return {
        "input": str(input_path),
        "output": str(output_path),
        "judge": model,
        "score": data["score"],
        "total": total,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing result JSON")
    parser.add_argument("--output", required=True, help="Where to write the re-scored JSON")
    parser.add_argument("--judge-model", required=True,
                        help="Judge model: 'heuristic' or a Claude model name")
    args = parser.parse_args()

    summary = rejudge_file(Path(args.input), Path(args.output), args.judge_model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
