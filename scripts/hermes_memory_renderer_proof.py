#!/usr/bin/env python3
"""Write Phase 155 memory answer renderer proof artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.memory_answer_renderer import contains_lifetime_certainty, render_memory_answer  # noqa: E402


DEFAULT_PHASE_DIR = (
    Path("/home/lauratom/Asztal/ai/atado/Brainstack-phase50")
    / ".planning/phases/155-memory-answer-renderer-gate"
)

CONTRACT = {
    "allowed_tool_profile": "conversation_direct",
    "external_capability_required_for_memory_answer": False,
}


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _run_case(
    *,
    name: str,
    answerability: Mapping[str, Any],
    answer_evidence: list[Mapping[str, Any]],
    expected_used: bool,
    expected_text: str | None = None,
    turn_contract: Mapping[str, Any] | None = None,
    capability_health: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    start = perf_counter()
    answer = render_memory_answer(
        turn_contract=turn_contract or CONTRACT,
        answerability=answerability,
        answer_evidence=answer_evidence,
        capability_health=capability_health,
    )
    latency_ms = int((perf_counter() - start) * 1000)
    data = answer.to_dict()
    pass_conditions = {
        "expected_renderer_usage": answer.used_renderer is expected_used,
        "expected_text": expected_text is None or answer.text == expected_text,
        "no_tool_call": answer.no_tool_call is True,
        "no_memory_mutation": answer.no_memory_mutation is True,
        "bounded_event_humble": answer.renderer_claim_style != "bounded_event"
        or not contains_lifetime_certainty(answer.text),
        "sub_second": latency_ms < 1000,
    }
    return {
        "name": name,
        "latency_ms": latency_ms,
        "result": data,
        "pass_conditions": pass_conditions,
        "passed": all(pass_conditions.values()),
    }


def build_proof() -> dict[str, Any]:
    scenarios = [
        _run_case(
            name="explicit_fact",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "explicit_user_fact",
            },
            answer_evidence=[{"id": "profile:debug_marker", "value": "1231231X"}],
            expected_used=True,
            expected_text="Recorded value: 1231231X.",
        ),
        _run_case(
            name="exact_literal",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "exact_literal",
            },
            answer_evidence=[{"id": "profile:debug_marker", "literal": "1231231X"}],
            expected_used=True,
            expected_text="Recorded identifier: 1231231X.",
        ),
        _run_case(
            name="unsupported_abstain",
            answerability={
                "state": "unanswerable",
                "max_claim_strength": "none",
                "answer_type": "none",
                "answer_evidence_count": 0,
                "reason_code": "NO_SUPPORTED_MEMORY_TRUTH",
            },
            answer_evidence=[],
            expected_used=True,
            expected_text="No supported memory evidence for this request.",
        ),
        _run_case(
            name="typed_assignment_absence",
            answerability={
                "state": "unanswerable",
                "max_claim_strength": "none",
                "answer_type": "current_assignment_absence",
                "reason_code": "NO_TYPED_CURRENT_ASSIGNMENT_EVIDENCE",
            },
            answer_evidence=[],
            expected_used=True,
            expected_text="No typed current-assignment evidence is recorded.",
        ),
        _run_case(
            name="bounded_prior_event",
            answerability={
                "state": "answerable",
                "max_claim_strength": "bounded_event",
                "answer_type": "conversation_event",
            },
            answer_evidence=[{"id": "event:turn-7", "preview": "User asked about marker 1231231X"}],
            expected_used=True,
            expected_text="Recorded event in searched scope: User asked about marker 1231231X.",
        ),
        _run_case(
            name="pulse_background_assignment_forbidden",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "current_assignment",
                "authority": "runtime_state_supporting",
            },
            answer_evidence=[{"id": "operating:pulse", "value": "Pulse running", "source_role": "runtime"}],
            expected_used=False,
        ),
        _run_case(
            name="conflict_forbidden",
            answerability={
                "state": "conflicted",
                "max_claim_strength": "none",
                "answer_type": "conflict_report",
            },
            answer_evidence=[{"id": "profile:old", "value": "1231231Y"}],
            expected_used=False,
        ),
        _run_case(
            name="degraded_backend_forbidden",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "explicit_user_fact",
                "required_backends": ["chroma"],
            },
            answer_evidence=[{"id": "profile:debug_marker", "value": "1231231X"}],
            capability_health={"chroma": "unavailable"},
            expected_used=False,
        ),
        _run_case(
            name="heavy_external_forbidden",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "explicit_user_fact",
            },
            answer_evidence=[{"id": "profile:repo", "value": "https://example.test"}],
            turn_contract={
                "allowed_tool_profile": "heavy_web",
                "external_capability_required_for_memory_answer": False,
            },
            expected_used=False,
        ),
        _run_case(
            name="redaction_private_path",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "explicit_user_fact",
            },
            answer_evidence=[{"id": "profile:path", "value": "/home/lauratom/private/project"}],
            expected_used=True,
            expected_text="Recorded value: [REDACTED_PRIVATE_PATH]",
        ),
    ]
    passed = all(case["passed"] for case in scenarios)
    return {
        "schema": "hermes.phase155.memory_renderer_proof.v1",
        "phase": 155,
        "verdict": "pass" if passed else "fail",
        "no_provider_call_required": True,
        "no_tool_call": True,
        "no_memory_mutation": True,
        "canary_type": "deterministic_renderer_gate",
        "scenarios": scenarios,
        "summary": {
            "scenario_count": len(scenarios),
            "passed_count": sum(1 for case in scenarios if case["passed"]),
            "max_latency_ms": max(case["latency_ms"] for case in scenarios),
            "sub_second_all": all(case["pass_conditions"]["sub_second"] for case in scenarios),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    proof = build_proof()
    _json_dump(args.output_dir / "155-RENDERER-PROOF.json", proof)
    _json_dump(args.output_dir / "155-RENDERER-CANARY.json", proof)
    print(json.dumps({"verdict": proof["verdict"], "summary": proof["summary"]}, sort_keys=True))
    return 0 if proof["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
