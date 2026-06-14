from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost_from_static_pricing


@dataclass(frozen=True)
class EvalAttempt:
    """One offline candidate/run for a fixture-backed eval task."""

    passed: bool
    cost_usd: float | None = 0.0
    cost_status: str = "actual"
    cost_source: str = "fixture"
    latency_ms: int = 0
    api_calls: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VeraFinding:
    """A normalized Vera issue finding used for precision/recall scoring."""

    id: str
    severity: str = ""
    category: str = ""
    title: str = ""


@dataclass(frozen=True)
class EvalTask:
    """A fixture task for Bob/Steve/Vera eval scoring."""

    task_id: str
    agent: str
    attempts: Sequence[EvalAttempt]
    expected_findings: Sequence[VeraFinding] = field(default_factory=tuple)
    reported_findings: Sequence[VeraFinding] = field(default_factory=tuple)
    source: str = "fixture"


@dataclass(frozen=True)
class CostTelemetry:
    amount_usd: float | None
    status: str
    source: str = "none"


# UNKNOWN(MAT-524): MiniSWERunner currently emits `completed`, `api_calls`, and
# `metadata.model`, but not durable latency/cost/token telemetry. This harness
# accepts fixture-provided `usage`/`telemetry` blocks now so real telemetry can be
# wired in without changing the scorer API once the runner starts emitting it.
_USAGE_KEYS = ("usage", "telemetry", "usage_telemetry", "metrics")


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round_float(value: float, places: int = 6) -> float:
    quant = Decimal("1") if places <= 0 else Decimal("1." + ("0" * places))
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def _cost_sum(attempts: Iterable[EvalAttempt]) -> float:
    return _round_float(sum(float(a.cost_usd or 0.0) for a in attempts), 6)


def _cost_summary(attempts: Sequence[EvalAttempt], known_total: float | None = None) -> dict[str, float | int]:
    known_statuses = {"actual", "estimated", "included"}
    known_total_usd = _cost_sum(attempts) if known_total is None else known_total
    return {
        "usd": known_total_usd,
        "known_usd": known_total_usd,
        "known_attempts": sum(1 for attempt in attempts if attempt.cost_status in known_statuses),
        "unknown_attempts": sum(1 for attempt in attempts if attempt.cost_status not in known_statuses),
        "actual_attempts": sum(1 for attempt in attempts if attempt.cost_status == "actual"),
        "estimated_attempts": sum(1 for attempt in attempts if attempt.cost_status == "estimated"),
        "included_attempts": sum(1 for attempt in attempts if attempt.cost_status == "included"),
    }


def _usage_block(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in _USAGE_KEYS:
        block = record.get(key)
        if isinstance(block, Mapping):
            return block
    return {}


def _extract_text_field(record: Mapping[str, Any], *keys: str) -> str:
    usage = _usage_block(record)
    metadata = record.get("metadata")
    sources: tuple[Mapping[str, Any], ...]
    if isinstance(metadata, Mapping):
        sources = (usage, metadata, record)
    else:
        sources = (usage, record)
    for source in sources:
        for key in keys:
            value = source.get(key)
            if value:
                return str(value).strip()
    return ""


def _extract_provider(record: Mapping[str, Any], model: str = "") -> str:
    provider = _extract_text_field(record, "provider", "provider_name")
    return provider.strip().lower()


def _extract_base_url(record: Mapping[str, Any]) -> str:
    return _extract_text_field(record, "base_url", "api_base", "endpoint", "endpoint_url")


def _extract_cost(record: Mapping[str, Any]) -> CostTelemetry:
    usage = _usage_block(record)
    for source in (usage, record):
        for key in ("cost_usd", "actual_cost_usd", "estimated_cost_usd"):
            if key in source:
                return CostTelemetry(
                    amount_usd=_as_float(source.get(key)),
                    status="actual",
                    source="fixture",
                )

    model = _extract_model(record)
    provider = _extract_provider(record, model)
    canonical_usage = CanonicalUsage(
        input_tokens=_extract_token_count(record, "input_tokens", "prompt_tokens"),
        output_tokens=_extract_token_count(record, "output_tokens", "completion_tokens"),
        cache_read_tokens=_extract_token_count(record, "cache_read_tokens", "cached_tokens", "cache_read_input_tokens"),
        cache_write_tokens=_extract_token_count(record, "cache_write_tokens", "cache_creation_input_tokens"),
        request_count=max(0, _extract_token_count(record, "api_calls", "request_count")),
    )
    if not model:
        return CostTelemetry(amount_usd=None, status="unknown", source="none")
    if canonical_usage.total_tokens <= 0:
        return CostTelemetry(amount_usd=None, status="no_usage", source="none")
    if not provider and not _extract_base_url(record):
        return CostTelemetry(amount_usd=None, status="unknown", source="none")

    result = estimate_usage_cost_from_static_pricing(
        model,
        canonical_usage,
        provider=provider or None,
        base_url=_extract_base_url(record) or None,
    )
    if result.amount_usd is None:
        return CostTelemetry(
            amount_usd=None,
            status=getattr(result, "status", "unknown"),
            source=getattr(result, "source", "none"),
        )
    return CostTelemetry(
        amount_usd=_round_float(float(result.amount_usd), 6),
        status=getattr(result, "status", "estimated"),
        source=getattr(result, "source", "static_pricing"),
    )


def _extract_latency_ms(record: Mapping[str, Any]) -> int:
    usage = _usage_block(record)
    for source in (usage, record):
        for key in ("latency_ms", "duration_ms", "elapsed_ms"):
            if key in source:
                return _as_int(source.get(key))
        for key in ("latency_seconds", "duration_seconds", "elapsed_seconds"):
            if key in source:
                return int(_as_float(source.get(key)) * 1000)
    return 0


def _extract_token_count(record: Mapping[str, Any], *keys: str) -> int:
    usage = _usage_block(record)
    for source in (usage, record):
        for key in keys:
            if key in source:
                return _as_int(source.get(key))
    return 0


def _extract_model(record: Mapping[str, Any]) -> str:
    return _extract_text_field(record, "model", "model_name")


def _finding_id(raw: Mapping[str, Any] | str) -> str:
    if isinstance(raw, str):
        return raw.strip()
    for key in ("id", "finding_id", "rule_id", "slug", "title"):
        value = raw.get(key)
        if value:
            return str(value).strip()
    return ""


def _normalize_finding(raw: Mapping[str, Any] | str) -> VeraFinding:
    if isinstance(raw, str):
        return VeraFinding(id=raw.strip())
    return VeraFinding(
        id=_finding_id(raw),
        severity=str(raw.get("severity") or ""),
        category=str(raw.get("category") or raw.get("type") or ""),
        title=str(raw.get("title") or raw.get("message") or ""),
    )


def _normalize_findings(raw: Any) -> tuple[VeraFinding, ...]:
    if not raw:
        return ()
    if isinstance(raw, Mapping):
        raw_items = raw.values()
    else:
        raw_items = raw
    findings = []
    for item in raw_items:
        if isinstance(item, (Mapping, str)):
            finding = _normalize_finding(item)
            if finding.id:
                findings.append(finding)
    return tuple(findings)


def _attempt_from_record(record: Mapping[str, Any]) -> EvalAttempt:
    passed = bool(record.get("passed", record.get("completed", record.get("success", False))))
    cost = _extract_cost(record)
    return EvalAttempt(
        passed=passed,
        cost_usd=cost.amount_usd,
        cost_status=cost.status,
        cost_source=cost.source,
        latency_ms=_extract_latency_ms(record),
        api_calls=_as_int(record.get("api_calls", record.get("request_count", 0))),
        model=_extract_model(record),
        input_tokens=_extract_token_count(record, "input_tokens", "prompt_tokens"),
        output_tokens=_extract_token_count(record, "output_tokens", "completion_tokens"),
        raw=dict(record),
    )


def _task_from_record(record: Mapping[str, Any], index: int) -> EvalTask:
    attempts_raw = record.get("attempts", record.get("runs", record.get("candidates")))
    if attempts_raw is None:
        attempts_raw = [record]
    attempts = tuple(
        _attempt_from_record(item)
        for item in attempts_raw
        if isinstance(item, Mapping)
    )
    return EvalTask(
        task_id=str(record.get("task_id") or record.get("id") or f"fixture-{index}"),
        agent=str(record.get("agent") or record.get("profile") or record.get("role") or "unknown").lower(),
        attempts=attempts,
        expected_findings=_normalize_findings(
            record.get("expected_findings") or record.get("gold_findings") or record.get("labels")
        ),
        reported_findings=_normalize_findings(
            record.get("reported_findings") or record.get("vera_findings") or record.get("findings")
        ),
        source=str(record.get("source") or "fixture"),
    )


def load_fixture(path: str | Path) -> list[EvalTask]:
    """Load JSON/JSONL eval fixtures into normalized tasks.

    Accepted shapes:
    - {"tasks": [{...}]}
    - [{...}, {...}]
    - JSONL, one task object per line

    Each task may use `attempts`, `runs`, or `candidates`. A single
    MiniSWERunner-shaped object (`completed`, `api_calls`, `metadata`) is also
    treated as one attempt.
    """

    fixture_path = Path(path)
    text = fixture_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    records: list[Mapping[str, Any]] = []
    if fixture_path.suffix.lower() == ".jsonl":
        for line in stripped.splitlines():
            if line.strip():
                item = json.loads(line)
                if isinstance(item, Mapping):
                    records.append(item)
    else:
        payload = json.loads(stripped)
        if isinstance(payload, Mapping) and isinstance(payload.get("tasks"), list):
            records = [item for item in payload["tasks"] if isinstance(item, Mapping)]
        elif isinstance(payload, list):
            records = [item for item in payload if isinstance(item, Mapping)]
        elif isinstance(payload, Mapping):
            records = [payload]
        else:
            raise ValueError(f"Unsupported fixture root in {fixture_path}")

    return [_task_from_record(record, index) for index, record in enumerate(records, start=1)]


def _pass_at_1(tasks: Sequence[EvalTask]) -> float:
    if not tasks:
        return 0.0
    solved = sum(1 for task in tasks if task.attempts and task.attempts[0].passed)
    return solved / len(tasks)


def _pass_power_k(tasks: Sequence[EvalTask], k: int) -> float:
    if not tasks:
        return 0.0
    limit = max(1, int(k))
    solved = sum(1 for task in tasks if any(attempt.passed for attempt in task.attempts[:limit]))
    return solved / len(tasks)


def _latency_summary(attempts: Sequence[EvalAttempt]) -> dict[str, float | int]:
    latencies = [attempt.latency_ms for attempt in attempts if attempt.latency_ms >= 0]
    if not latencies:
        return {"mean": 0, "p95": 0, "total": 0}
    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "mean": _round_float(statistics.mean(ordered), 3),
        "p95": ordered[p95_index],
        "total": sum(ordered),
    }


def _vera_counts(tasks: Sequence[EvalTask]) -> dict[str, float | int]:
    tp = fp = fn = 0
    for task in tasks:
        expected = {finding.id for finding in task.expected_findings if finding.id}
        reported = {finding.id for finding in task.reported_findings if finding.id}
        tp += len(expected & reported)
        fp += len(reported - expected)
        fn += len(expected - reported)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _round_float(precision, 6),
        "recall": _round_float(recall, 6),
    }


def _score_group(tasks: Sequence[EvalTask], *, k: int) -> dict[str, Any]:
    attempts = [attempt for task in tasks for attempt in task.attempts]
    known_cost_usd = _cost_sum(attempts)
    result: dict[str, Any] = {
        "task_count": len(tasks),
        "attempt_count": len(attempts),
        "pass@1": _round_float(_pass_at_1(tasks), 6),
        f"pass^{k}": _round_float(_pass_power_k(tasks, k), 6),
        "cost_usd": known_cost_usd,
        "cost": _cost_summary(attempts, known_cost_usd),
        "latency_ms": _latency_summary(attempts),
        "api_calls": sum(attempt.api_calls for attempt in attempts),
        "input_tokens": sum(attempt.input_tokens for attempt in attempts),
        "output_tokens": sum(attempt.output_tokens for attempt in attempts),
    }
    if any(task.expected_findings or task.reported_findings for task in tasks):
        result["vera"] = _vera_counts(tasks)
    return result


def score_eval_suite(tasks: Sequence[EvalTask], *, k: int = 1) -> dict[str, Any]:
    """Score Bob/Steve/Vera fixture tasks.

    Metrics:
    - pass@1: first attempt success rate per task.
    - pass^k: deterministic fixture success rate if any of the first k attempts
      passed for each task.
    - cost_usd: sum of fixture cost telemetry across attempts.
    - latency_ms: mean/p95/total latency across attempts.
    - Vera precision/recall: exact finding-id matching for expected vs reported.
    """

    grouped: dict[str, list[EvalTask]] = {}
    for task in tasks:
        grouped.setdefault(task.agent or "unknown", []).append(task)

    return {
        "k": k,
        "overall": _score_group(tasks, k=k),
        "agents": {agent: _score_group(agent_tasks, k=k) for agent, agent_tasks in sorted(grouped.items())},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score fixture-backed Hermes Bob/Steve/Vera evals.")
    parser.add_argument("--fixture", required=True, help="Path to JSON or JSONL fixture file.")
    parser.add_argument("--k", type=int, default=1, help="Number of attempts for pass^k scoring.")
    parser.add_argument("--output", help="Optional path to write the JSON summary.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tasks = load_fixture(args.fixture)
    summary = score_eval_suite(tasks, k=args.k)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
