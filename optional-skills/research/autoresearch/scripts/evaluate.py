#!/usr/bin/env python3
"""Score autoresearch experiments and append decision records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _util import exclusive_lock, now_iso

DECISIONS = ("MERGE", "REVERT")


class EvaluationError(ValueError):
    """Raised when an evaluation command violates its contract."""


def clean_log_field(value: str) -> str:
    """Keep user/research text inside one unambiguous log field."""
    return " ".join(value.replace("---", "—").splitlines()).strip()


def rubric_score(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 5:
        raise argparse.ArgumentTypeError("must be an integer from 1 to 5")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def score_knowledge(
    evidence: int,
    accuracy: int,
    depth: int,
    relevance: int,
    net_improvement: int,
) -> dict[str, Any]:
    values = (evidence, accuracy, depth, relevance, net_improvement)
    if any(value < 1 or value > 5 for value in values):
        raise EvaluationError("knowledge scores must be integers from 1 to 5")
    total = sum(values)
    scores = {
        "evidence": evidence,
        "accuracy": accuracy,
        "depth": depth,
        "relevance": relevance,
        "net_improvement": net_improvement,
        "total": total,
        "max": 25,
    }
    gates = {
        "evidence": evidence,
        "relevance": relevance,
        "net_improvement": net_improvement,
    }
    failed = [name for name, value in gates.items() if value < 3]
    if failed:
        return {
            "scores": scores,
            "decision": "REVERT",
            "reason": f"Failed gates: {', '.join(failed)}",
        }
    if total >= 13:
        return {
            "scores": scores,
            "decision": "MERGE",
            "reason": f"Accepted (total={total}/25)",
        }
    return {
        "scores": scores,
        "decision": "REVERT",
        "reason": f"Below threshold (total={total}/25)",
    }


def score_ml(
    metric_value: float, prev_best: float, lower_is_better: bool
) -> dict[str, Any]:
    improved = metric_value < prev_best if lower_is_better else metric_value > prev_best
    delta = prev_best - metric_value if lower_is_better else metric_value - prev_best
    decision = "MERGE" if improved else "REVERT"
    return {
        "decision": decision,
        "metric_value": metric_value,
        "prev_best": prev_best,
        "delta": delta,
        "reason": f"Metric {'improved' if improved else 'not improved'} by {delta:.6f}",
    }


def results_path(run_dir: str) -> Path:
    path = Path(run_dir) / "results.log"
    if not path.is_file():
        raise EvaluationError(f"run is not initialized; missing {path}")
    return path


def append_entry(run_dir: str, lines: list[str]) -> None:
    with exclusive_lock(Path(run_dir) / ".autoresearch.lock"):
        path = results_path(run_dir)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n" + "\n".join(lines) + "\n---\n")


def log_result(
    run_dir: str,
    experiment_id: int,
    description: str,
    experiment_type: str,
    target: str,
    decision: str,
    reason: str,
    scores: str | None = None,
) -> dict[str, Any]:
    lines = [
        f"## Experiment {experiment_id}: {clean_log_field(description)}",
        f"Time: {now_iso()}",
        f"Type: {clean_log_field(experiment_type)}",
        f"Target: {clean_log_field(target)}",
    ]
    if scores:
        lines.append(f"Scores: {clean_log_field(scores)}")
    lines.extend([f"Decision: {decision}", f"Reason: {clean_log_field(reason)}"])
    append_entry(run_dir, lines)
    return {"status": "logged", "experiment_id": experiment_id, "decision": decision}


def log_result_ml(
    run_dir: str,
    experiment_id: int,
    description: str,
    metric_name: str,
    metric_value: float,
    prev_best: float,
    decision: str,
    reason: str,
) -> dict[str, Any]:
    append_entry(
        run_dir,
        [
            f"## Experiment {experiment_id}: {clean_log_field(description)}",
            f"Time: {now_iso()}",
            f"Metric: {clean_log_field(metric_name)}={metric_value} (previous best: {prev_best})",
            f"Decision: {decision}",
            f"Reason: {clean_log_field(reason)}",
        ],
    )
    return {"status": "logged", "experiment_id": experiment_id, "decision": decision}


def result_entries(run_dir: str) -> list[str]:
    content = results_path(run_dir).read_text(encoding="utf-8")
    return [entry.strip() for entry in content.split("---\n") if entry.strip()]


def read_results(run_dir: str, last_n: int | None = None) -> str:
    entries = result_entries(run_dir)
    selected = entries[-last_n:] if last_n else entries
    return "\n---\n".join(selected)


def stats(run_dir: str) -> dict[str, Any]:
    entries = result_entries(run_dir)
    merged = sum("Decision: MERGE" in entry.splitlines() for entry in entries)
    reverted = sum("Decision: REVERT" in entry.splitlines() for entry in entries)
    by_type: dict[str, int] = {}
    for entry in entries:
        for line in entry.splitlines():
            if line.startswith("Type: "):
                experiment_type = line.removeprefix("Type: ").strip()
                by_type[experiment_type] = by_type.get(experiment_type, 0) + 1
    return {
        "total": len(entries),
        "merged": merged,
        "reverted": reverted,
        "merge_rate": f"{merged / len(entries) * 100:.0f}%" if entries else "0%",
        "by_type": by_type,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    score_parser = commands.add_parser("score")
    for name in ("evidence", "accuracy", "depth", "relevance", "net_improvement"):
        score_parser.add_argument(name, type=rubric_score)

    ml_parser = commands.add_parser("score-ml")
    ml_parser.add_argument("metric_value", type=float)
    ml_parser.add_argument("prev_best", type=float)
    direction = ml_parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--lower-is-better", action="store_true")
    direction.add_argument("--higher-is-better", action="store_true")

    log_parser = commands.add_parser("log-result")
    log_parser.add_argument("run_dir")
    log_parser.add_argument("experiment_id", type=positive_int)
    log_parser.add_argument("description")
    log_parser.add_argument("experiment_type")
    log_parser.add_argument("target")
    log_parser.add_argument("decision", choices=DECISIONS)
    log_parser.add_argument("reason")
    log_parser.add_argument("--scores")

    log_ml_parser = commands.add_parser("log-result-ml")
    log_ml_parser.add_argument("run_dir")
    log_ml_parser.add_argument("experiment_id", type=positive_int)
    log_ml_parser.add_argument("description")
    log_ml_parser.add_argument("metric_name")
    log_ml_parser.add_argument("metric_value", type=float)
    log_ml_parser.add_argument("prev_best", type=float)
    log_ml_parser.add_argument("decision", choices=DECISIONS)
    log_ml_parser.add_argument("reason")

    read_parser = commands.add_parser("read-results")
    read_parser.add_argument("run_dir")
    read_parser.add_argument("--last", type=positive_int)

    stats_parser = commands.add_parser("stats")
    stats_parser.add_argument("run_dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            result: Any = score_knowledge(
                args.evidence,
                args.accuracy,
                args.depth,
                args.relevance,
                args.net_improvement,
            )
        elif args.command == "score-ml":
            result = score_ml(args.metric_value, args.prev_best, args.lower_is_better)
        elif args.command == "log-result":
            result = log_result(
                args.run_dir,
                args.experiment_id,
                args.description,
                args.experiment_type,
                args.target,
                args.decision,
                args.reason,
                args.scores,
            )
        elif args.command == "log-result-ml":
            result = log_result_ml(
                args.run_dir,
                args.experiment_id,
                args.description,
                args.metric_name,
                args.metric_value,
                args.prev_best,
                args.decision,
                args.reason,
            )
        elif args.command == "read-results":
            print(read_results(args.run_dir, args.last))
            return 0
        else:
            result = stats(args.run_dir)
        print(json.dumps(result, indent=2))
    except EvaluationError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
