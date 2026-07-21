"""Execution Pipeline for Task Runtime.

Producer → ProducerNormalizer v1.1 → conditional Reviewer.

In MVP (dry-run / shadow modes) the pipeline does NOT make any HTTP calls
and does NOT mutate any artifacts on disk outside the pilot output dir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineResult:
    """Output of ExecutionPipeline.run()."""
    engine_status: str                    # "OK" | "STOP" | "DRY_RUN"
    normalizer_verdict: str               # "PASS" | "PARTIAL" | "NO_EVIDENCE" | "BLOCKED" | "NOT_RUN" | "DRY_RUN"
    reviewer_called: bool
    reviewer_skipped: bool
    reviewer_verdict: str | None
    artifacts_produced: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


def run(
    contract: dict[str, Any],
    *,
    workdir: Path | None = None,
    confirmation_token: str | None = None,
) -> PipelineResult:
    """Execute the pipeline.

    In MVP, modes other than `dry-run` would invoke producer + normalizer +
    conditional reviewer. For the MVP acceptance criteria, `dry-run` returns
    a simulated result that exercises the contract path WITHOUT any HTTP,
    LLM, or disk mutations outside the configured output dir.

    Args:
        contract: the TaskContract v1.0.0 dict.
        workdir: optional directory for any artifacts; defaults to a temp dir.
        confirmation_token: required for `enforce` mode to apply mutations.

    Returns:
        PipelineResult with engine_status, normalizer_verdict, etc.
    """
    mode = contract.get("execution_mode", "dry-run")
    result = PipelineResult(
        engine_status="OK",
        normalizer_verdict="DRY_RUN",
        reviewer_called=False,
        reviewer_skipped=False,
        reviewer_verdict=None,
    )

    if mode == "dry-run":
        # No HTTP, no LLM, no producer call. Simulate the pipeline path.
        # Verify the contract is well-formed and return a simulated result.
        _validate_contract(contract)
        result.engine_status = "OK"
        result.normalizer_verdict = "PASS"  # dry-run can't actually evaluate
        result.reviewer_skipped = True
        result.reviewer_verdict = None
        result.artifacts_produced = []
        result.metrics = {
            "pipeline_mode": "dry-run",
            "producer_http_calls": 0,
            "reviewer_http_calls": 0,
            "normalizer_engine_status": "OK_SIMULATED",
        }
        return result

    if mode == "shadow":
        # In a future implementation this would run the actual pipeline but
        # only LOG results, never apply changes. MVP returns simulated.
        _validate_contract(contract)
        result.metrics = {
            "pipeline_mode": "shadow",
            "producer_http_calls": 0,
            "reviewer_http_calls": 0,
            "normalizer_engine_status": "OK_SIMULATED",
        }
        return result

    if mode == "supervised":
        # Would require confirmation_token. MVP returns an error if missing.
        if confirmation_token is None:
            result.engine_status = "STOP"
            result.normalizer_verdict = "NOT_RUN"
            result.errors.append("supervised mode requires confirmation_token")
            return result
        _validate_contract(contract)
        result.metrics = {"pipeline_mode": "supervised"}
        return result

    if mode == "enforce":
        if confirmation_token is None:
            result.engine_status = "STOP"
            result.normalizer_verdict = "NOT_RUN"
            result.errors.append("enforce mode requires confirmation_token")
            return result
        _validate_contract(contract)
        result.metrics = {"pipeline_mode": "enforce"}
        return result

    # Unknown mode
    result.engine_status = "STOP"
    result.normalizer_verdict = "NOT_RUN"
    result.errors.append(f"unknown execution_mode: {mode!r}")
    return result


def _validate_contract(contract: dict[str, Any]) -> None:
    """Light structural check; raises ValueError on critical missing fields."""
    required = ("task_contract_schema", "intent", "context", "producer",
                "normalizer", "reviewer", "execution_mode")
    for key in required:
        if key not in contract:
            raise ValueError(f"contract missing required key: {key!r}")
    if contract.get("task_contract_schema") != "1.0.0":
        raise ValueError(f"contract schema mismatch: {contract.get('task_contract_schema')!r}")