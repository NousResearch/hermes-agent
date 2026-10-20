#!/usr/bin/env python3
"""Benchmark the pure dynamic-orchestration capacity replan contract.

The workload is fixed, secret-free, local, and deterministic. It performs no
network calls, runtime dispatch, credential access, or persistent writes.
Wall-clock results are evidence for local profiling, not CI pass/fail gates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

orchestration = importlib.import_module("agent.dynamic_orchestration")

CARDINALITIES = (1, 64, 256)
DEFAULT_WARMUPS = 2
DEFAULT_SAMPLES = 7
MAX_WARMUPS = 100
MAX_SAMPLES = 100


def _route_values() -> dict[str, object]:
    return {
        "provider": "OpenAI",
        "product": "ChatGPT",
        "surface": "API",
        "account_id": "benchmark",
        "billing_pool_id": "failed-billing",
        "quota_pool_id": "failed-quota",
        "model": "failed-model",
        "endpoint": "https://api.example.com/v1",
        "region": "US",
    }


def _task_payload() -> dict[str, object]:
    return {
        "schema_version": "task-envelope/v1",
        "task_id": "benchmark-task",
        "objective": "benchmark deterministic capacity replanning",
        "deliverables": ["decision"],
        "capabilities_required": ["filesystem.write"],
        "tools_allowed": ["patch"],
        "permissions_required": ["repository.write"],
        "context": {
            "classification": "internal",
            "max_tokens": 900,
            "token_count": 12,
            "allowed_sources": ["repository"],
        },
        "privacy": {
            "classification": "internal",
            "outbound_allowed": False,
            "retention": "ephemeral",
        },
        "risk": {
            "level": "medium",
            "reversibility": "reversible",
            "impact": "repository-only",
        },
        "effort": "E2",
        "budget": {
            "currency": "USD",
            "paid_allowed": False,
            "soft_cap": 0,
            "hard_cap": 0,
        },
        "verification": {
            "minimum": "V0",
            "independent_required": False,
            "human_gate_required": False,
        },
        "policy_version": "policy/v1",
    }


def _build_replan_inputs(cardinality: int) -> dict[str, object]:
    failed = orchestration.RouteV1.from_mapping(_route_values())
    task = orchestration.TaskEnvelope.from_mapping(_task_payload())
    classification = orchestration.RuntimeErrorClassificationV1(
        kind=orchestration.ErrorKind.CAPACITY_EXHAUSTED,
        source="typed-runtime-error",
        attempted_route_id=failed.route_id,
        quota_pool_id=failed.quota_pool_id,
        classified_at="2026-07-27T12:00:00Z",
    )
    candidates = []
    for index in range(cardinality):
        candidate_values = _route_values()
        candidate_values.update(
            {
                "billing_pool_id": f"billing-{index}",
                "quota_pool_id": f"quota-{index}",
                "model": f"model-{index}",
            }
        )
        candidate_route = orchestration.RouteV1.from_mapping(candidate_values)
        candidates.append(
            orchestration.CandidateEvaluation(
                candidate_route,
                True,
                score=float(cardinality - index),
                score_factors=("quality",),
            )
        )
    return {
        "trusted_task": task,
        "task_id": task.task_id,
        "attempt_id": "benchmark-attempt",
        "failed_route": failed,
        "classification": classification,
        "candidates": tuple(candidates),
        "decision_id": "benchmark-decision",
        "parent_decision_id": "benchmark-parent",
        "created_at": "2026-07-27T12:00:01Z",
        "policy_version": task.policy_version,
        "router_version": "router/pure-v1",
        "capacity_view_id": "capacity-view:benchmark",
        "effort": task.effort,
        "verification": "V0",
    }


def _decision_fingerprint(decision: object) -> str:
    payload = json.dumps(
        asdict(decision),  # type: ignore[arg-type]
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _p95_nearest_rank(samples: list[float]) -> float:
    rank = max(1, math.ceil(0.95 * len(samples)))
    return sorted(samples)[rank - 1]


def _benchmark_cardinality(
    cardinality: int,
    *,
    warmups: int,
    sample_count: int,
) -> dict[str, object]:
    inputs = _build_replan_inputs(cardinality)
    decision = None
    for _ in range(warmups):
        decision = orchestration.replan_after_capacity_exhaustion(**inputs)

    samples_ms: list[float] = []
    for _ in range(sample_count):
        started_ns = time.perf_counter_ns()
        decision = orchestration.replan_after_capacity_exhaustion(**inputs)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        samples_ms.append(round(elapsed_ms, 6))

    if decision is None:
        raise RuntimeError("benchmark did not produce a route decision")
    return {
        "cardinality": cardinality,
        "warmups": warmups,
        "sample_count": sample_count,
        "samples_ms": samples_ms,
        "median_ms": round(statistics.median(samples_ms), 6),
        "p95_nearest_rank_ms": round(_p95_nearest_rank(samples_ms), 6),
        "max_ms": round(max(samples_ms), 6),
        "selected_route_id": decision.selected_route_id,
        "decision_sha256": _decision_fingerprint(decision),
    }


def _environment_metadata() -> dict[str, object]:
    source_root = REPOSITORY_ROOT / "agent" / "dynamic_orchestration"
    source_bytes = b"".join(
        path.name.encode() + b"\0" + path.read_bytes()
        for path in sorted(source_root.glob("*.py"))
    )
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "randomized"),
        "gc_enabled": gc.isenabled(),
        "timer": "time.perf_counter_ns",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def _positive_sample_count(value: str) -> int:
    parsed = int(value)
    if parsed < 3 or parsed > MAX_SAMPLES:
        raise argparse.ArgumentTypeError(
            f"samples must be between 3 and {MAX_SAMPLES}"
        )
    return parsed


def _non_negative_warmups(value: str) -> int:
    parsed = int(value)
    if parsed < 0 or parsed > MAX_WARMUPS:
        raise argparse.ArgumentTypeError(
            f"warmups must be between 0 and {MAX_WARMUPS}"
        )
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=_positive_sample_count,
        default=DEFAULT_SAMPLES,
        help=f"timed samples per cardinality (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--warmups",
        type=_non_negative_warmups,
        default=DEFAULT_WARMUPS,
        help=f"warmup calls per cardinality (default: {DEFAULT_WARMUPS})",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="indent the otherwise compact machine-readable JSON",
    )
    args = parser.parse_args(argv)

    result: dict[str, Any] = {
        "benchmark": "dynamic_orchestration.replan_after_capacity_exhaustion",
        "schema_version": "dynamic-orchestration-benchmark/v1",
        "environment": _environment_metadata(),
        "results": [
            _benchmark_cardinality(
                cardinality,
                warmups=args.warmups,
                sample_count=args.samples,
            )
            for cardinality in CARDINALITIES
        ],
    }
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
