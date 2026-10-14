#!/usr/bin/env python3
"""Validate blinded ratings and produce a descriptive summary, never a winner."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUBRIC = json.loads((ROOT / "rubric.json").read_text(encoding="utf-8"))
WEIGHTS = {name: row["weight"] for name, row in RUBRIC["blind_clinical_dimensions"].items()}
DIMS = tuple(WEIGHTS)
GATES = tuple(RUBRIC["gates"])


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate(scores: list[dict], key: dict) -> dict:
    if not scores:
        raise ValueError("no clinical ratings")
    grouped = defaultdict(list)
    seen = set()
    for row in scores:
        response_id, rater = row.get("response_id"), row.get("rater_id")
        if not isinstance(response_id, str) or response_id not in key:
            raise ValueError("unknown response_id")
        if not isinstance(rater, str) or not rater.strip() or (response_id, rater) in seen:
            raise ValueError("missing or duplicate rater_id for a response")
        seen.add((response_id, rater))
        clinical, gates = row.get("clinical", {}), row.get("gates", {})
        if not isinstance(clinical, dict) or not all(
            isinstance(clinical.get(dim), (int, float)) and not isinstance(clinical[dim], bool)
            and math.isfinite(clinical[dim]) and 0 <= clinical[dim] <= 4 for dim in DIMS
        ):
            raise ValueError("incomplete or invalid clinical scores")
        if not isinstance(gates, dict) or not all(type(gates.get(gate)) is bool for gate in GATES):
            raise ValueError("all clinical gates must be explicitly rated")
        values = dict(clinical)
        if gates["fabricated_source"]:
            values["evidence_retrieved"] = values["traceability"] = 0
        grouped[response_id].append({"weighted": sum(values[d] * WEIGHTS[d] for d in DIMS), "gates": gates})

    per_provider = defaultdict(list)
    for response_id, ratings in grouped.items():
        identity = key[response_id]
        per_provider[identity["provider_id"]].append({
            "case_id": identity["case_id"], "raters": len(ratings),
            "weighted": statistics.mean(row["weighted"] for row in ratings),
            "gates": {gate: any(row["gates"][gate] for row in ratings) for gate in GATES},
        })
    report = {"status": "descriptive_only", "comparison_complete": False,
              "warning": "Requires two independent raters, adjudication, paired coverage and uncertainty analysis.",
              "missing_response_ids": sorted(set(key) - set(grouped)),
              "minimum_two_raters": all(len(rows) >= 2 for rows in grouped.values()), "providers": {}}
    for provider, responses in sorted(per_provider.items()):
        by_case = defaultdict(list)
        for response in responses:
            by_case[response["case_id"]].append(response["weighted"])
        report["providers"][provider] = {
            "rated_rows": sum(row["raters"] for row in responses), "rated_responses": len(responses),
            "rated_cases": len(by_case),
            "mean_weighted_0_to_4": round(statistics.mean(statistics.mean(v) for v in by_case.values()), 3),
            "critical_failure_count": sum(row["gates"]["critical_safety_failure"] for row in responses),
            "unsupported_number_count": sum(row["gates"]["unsupported_clinical_number"] for row in responses),
            "fabricated_source_count": sum(row["gates"]["fabricated_source"] for row in responses),
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--scores", type=Path)
    args = parser.parse_args(argv)
    key = json.loads((args.run_dir / "identity_key.json").read_text(encoding="utf-8"))
    try:
        report = aggregate(rows(args.scores or args.run_dir / "clinical_scores.jsonl"), key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    (args.run_dir / "aggregate.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
