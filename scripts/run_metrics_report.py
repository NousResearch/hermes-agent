#!/usr/bin/env python3
"""Aggregate opt-in Hermes run-metrics snapshots without reading transcripts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(percentile * len(ordered)) - 1)], 3)


def _diagnosis(run: dict[str, Any]) -> dict[str, str | None]:
    """Exact stop evidence is separate from a measured time-dominance hint."""
    reason = str(run.get("turn_exit_reason") or "")
    attempts = run.get("attempts") or []
    summary = run.get("summary") or {}
    status = str(run.get("status") or "unknown")
    # Only the terminal event can identify a stop cause. Historical retries
    # explain cost, not why this run finally stopped.
    if status in {"completed", "interrupted", "running", "output_capped"}:
        stop = status
    elif reason in {"text_response(finish_reason=length)", "truncated"} and attempts and attempts[-1].get("status") == "output_capped":
        stop = "output_capped"
    elif "timeout" in reason:
        stop = "timeout"
    elif "context" in reason:
        stop = "context_exhausted"
    elif reason in {"api_error", "provider_error", ""} and attempts and attempts[-1].get("status") == "error":
        last = attempts[-1]
        error = (str(last.get("error_reason") or "") + " " + str(last.get("error_type") or "")).lower()
        stop = "timeout" if "timeout" in error else "context_exhausted" if "context" in error else "failed"
    elif reason in {"max_iterations", "max_iterations_reached", "iteration_budget", "truncated"}:
        stop = reason
    else:
        stop = status
    model = summary.get("model_time_s") or 0
    tool = summary.get("tool_time_s") or 0
    wall = summary.get("wall_time_s") or 0
    dominant = "model" if model > tool and model >= wall * 0.5 else "tool" if tool > model and tool >= wall * 0.5 else None
    return {"stop_cause": stop, "dominant_time": dominant}


def aggregate(directory: Path) -> dict[str, Any]:
    runs = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != 1 or not data.get("run_id"):
            continue
        runs.append(data)
    roots = [run for run in runs if not run.get("parent_run_id")]
    status_counts = Counter(str(run.get("status") or "unknown") for run in roots)
    cause_counts = Counter(_diagnosis(run)["stop_cause"] for run in roots)
    walls = [float(run["summary"]["wall_time_s"]) for run in roots if run.get("status") != "running" and run.get("summary", {}).get("wall_time_s") is not None]
    decode_samples = []
    for run in roots:
        for attempt in run.get("attempts", []):
            delivered = next((nested for nested in reversed(attempt.get("transport_attempts", []))
                              if nested.get("token_source") == "provider_reported"), None)
            timing = delivered or (attempt if not attempt.get("transport_attempts") else None)
            if (timing and attempt.get("output_tokens") is not None
                    and timing.get("duration_s") is not None
                    and timing.get("time_to_first_delta_s") is not None
                    and timing["duration_s"] > timing["time_to_first_delta_s"]):
                decode_samples.append((attempt["output_tokens"],
                                       timing["duration_s"] - timing["time_to_first_delta_s"]))
    return {
        "schema_version": 1,
        "root_runs": len(roots),
        "child_runs": len(runs) - len(roots),
        "statuses": dict(status_counts),
        "stop_causes": dict(cause_counts),
        "wall_time_s": {"p50": _percentile(walls, 0.5), "p95": _percentile(walls, 0.95)},
        "provider_attempts": sum(run.get("summary", {}).get("provider_attempts") or 0 for run in roots),
        "runs_with_incomplete_attempt_coverage": sum(
            run.get("summary", {}).get("provider_attempts_complete") is False for run in roots
        ),
        "logical_model_calls": sum(run.get("summary", {}).get("logical_model_calls") or 0 for run in roots),
        "provider_reported_input_tokens": sum(run.get("summary", {}).get("provider_reported_input_tokens") or 0 for run in roots),
        "provider_reported_output_tokens": sum(run.get("summary", {}).get("provider_reported_output_tokens") or 0 for run in roots),
        "measured_decode_tokens_per_s": round(sum(tokens for tokens, _ in decode_samples) / sum(seconds for _, seconds in decode_samples), 2) if decode_samples else None,
        "runs_with_missing_provider_usage": sum(
            bool(run.get("summary", {}).get("attempts_with_missing_usage")) for run in roots
        ),
        "runs": [
            {
                "run_id": run["run_id"], "task_id": run.get("task_id"),
                "status": run.get("status"), "wall_time_s": run.get("summary", {}).get("wall_time_s"),
                "diagnosis": _diagnosis(run),
                "model_time_s": run.get("summary", {}).get("model_time_s"),
                "tool_time_s": run.get("summary", {}).get("tool_time_s"),
                "measured_decode_tokens_per_s": run.get("summary", {}).get("measured_decode_tokens_per_s"),
            }
            for run in roots
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing Hermes run-metrics JSON files")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
