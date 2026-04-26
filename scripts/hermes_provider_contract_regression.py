#!/usr/bin/env python3
"""Write Phase 156 proof-carrying provider request regression artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.proof_carrying_request import (  # noqa: E402
    FAILURE_TAXONOMY,
    build_external_action_audit,
    build_provider_request_contract,
    build_provider_response_contract,
    build_regression_result,
)
from gateway.provider_contract import (  # noqa: E402
    build_idempotency_decision,
    build_idempotency_key,
    build_stale_response_decision,
)
from gateway.tool_profile_snapshot import (  # noqa: E402
    build_tool_loader_contract,
    build_tool_registry,
    evaluate_tool_load_request,
)
from gateway.turn_contract import build_hidden_tool_prompt_audit  # noqa: E402


DEFAULT_PHASE_DIR = (
    Path("/home/lauratom/Asztal/ai/atado/Brainstack-phase50")
    / ".planning/phases/156-proof-carrying-provider-request-and-regression-pack"
)


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _turn(
    *,
    tool_profile: str = "conversation_direct",
    model_profile: str = "conversation_renderer",
    forbidden_claims: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "turn_contract_id": f"turn:{tool_profile}",
        "allowed_tool_profile": tool_profile,
        "allowed_model_profile": model_profile,
        "latency_slo": "conversation_warm" if not tool_profile.startswith("heavy") else "heavy_visible_progress",
        "forbidden_claims": forbidden_claims or [],
    }


def _profile(profile_name: str, tool_names: list[str]) -> dict[str, Any]:
    return {
        "profile_name": profile_name,
        "profile_version": "v1",
        "tool_names": tool_names,
        "tool_schema_hash": f"{profile_name}:schema",
        "static_prefix_hash": f"{profile_name}:prefix",
    }


def _contract_case(
    *,
    name: str,
    family: str,
    answerability: Mapping[str, Any],
    answer_evidence: list[Mapping[str, Any]],
    turn_contract: Mapping[str, Any] | None = None,
    profile_snapshot: Mapping[str, Any] | None = None,
    claims_made: list[str] | None = None,
    tool_calls_made: list[str] | None = None,
    latency_slo_satisfied: bool = True,
    expected_satisfied: bool = True,
) -> dict[str, Any]:
    request = build_provider_request_contract(
        turn_contract=turn_contract or _turn(),
        profile_snapshot=profile_snapshot or _profile("conversation_direct", []),
        context_budget={"context_budget_id": "budget:phase156"},
        answerability=answerability,
        answer_evidence=answer_evidence,
        prompt_snapshot=f"phase156:{name}",
        brainstack_packet_id=f"brainstack:{name}",
        regression_family=family,
    )
    response = build_provider_response_contract(
        request,
        claims_made=claims_made or [],
        answer_evidence_used=[str(item.get("id")) for item in answer_evidence if item.get("id")],
        tool_calls_made=tool_calls_made or [],
        latency_slo_satisfied=latency_slo_satisfied,
        degradation_policy_used="none",
    )
    passed = response.contract_satisfied is expected_satisfied
    return build_regression_result(
        name,
        passed=passed,
        family=family,
        details={
            "request": request.to_dict(),
            "response": response.to_dict(),
            "expected_contract_satisfied": expected_satisfied,
        },
    )


def _build_provider_regressions() -> list[dict[str, Any]]:
    cases = [
        _contract_case(
            name="memory_answerable_fact_literal",
            family="memory_answerable",
            answerability={
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "explicit_user_fact",
            },
            answer_evidence=[{"id": "profile:debug_marker", "value": "1231231X"}],
        ),
        _contract_case(
            name="unsupported_abstain",
            family="unsupported_abstain",
            answerability={"state": "unanswerable", "max_claim_strength": "none", "answer_type": "none"},
            answer_evidence=[],
            turn_contract=_turn(forbidden_claims=["memory_truth"]),
        ),
        _contract_case(
            name="prior_event_bounded",
            family="prior_event",
            answerability={
                "state": "answerable",
                "max_claim_strength": "bounded_event",
                "answer_type": "conversation_event",
            },
            answer_evidence=[{"id": "event:turn-7", "preview": "User asked marker question"}],
        ),
        _contract_case(
            name="current_assignment_vs_pulse_background",
            family="current_assignment_authority",
            answerability={
                "state": "unanswerable",
                "max_claim_strength": "none",
                "answer_type": "current_assignment_absence",
            },
            answer_evidence=[],
            turn_contract=_turn(forbidden_claims=["current_assignment"]),
        ),
        _contract_case(
            name="correction_supersession_conflict",
            family="correction_supersession_conflict",
            answerability={"state": "conflicted", "max_claim_strength": "none", "answer_type": "conflict_report"},
            answer_evidence=[{"id": "profile:prior", "value": "1231231Y"}, {"id": "profile:current", "value": "1231231X"}],
        ),
        _contract_case(
            name="heavy_explicit_route",
            family="heavy_explicit_route",
            answerability={"state": "unanswerable", "max_claim_strength": "none", "answer_type": "none"},
            answer_evidence=[],
            turn_contract=_turn(tool_profile="heavy_web", model_profile="reasoning_heavy"),
            profile_snapshot=_profile("heavy_web", ["web_search"]),
            tool_calls_made=["web_search"],
        ),
        _contract_case(
            name="negative_heavy_false_positive_url_memory_mention",
            family="negative_heavy_false_positive",
            answerability={"state": "answerable", "max_claim_strength": "memory_truth", "answer_type": "explicit_user_fact"},
            answer_evidence=[{"id": "profile:repo_url", "value": "https://example.test/repo"}],
            turn_contract=_turn(tool_profile="conversation_direct"),
            profile_snapshot=_profile("conversation_direct", []),
        ),
    ]
    stale = build_stale_response_decision(
        platform="discord",
        channel_id="c1",
        user_id="u1",
        turn_id="turn-a",
        causal_index=1,
        latest_completed_index=2,
        superseded_by="turn-b",
    )
    cases.append(
        build_regression_result(
            "stale_response_suppression",
            passed=stale.stale_response_suppressed is True,
            family="stale_response",
            details=stale.to_dict(),
        )
    )
    key = build_idempotency_key(
        platform="discord",
        guild_id="g1",
        channel_id="c1",
        message_id="m1",
        author_id="u1",
    )
    duplicate = build_idempotency_decision(idempotency_key_hash=key, seen_keys={key})
    cases.append(
        build_regression_result(
            "duplicate_event_idempotency",
            passed=duplicate.duplicate_suppressed is True and duplicate.duplicate_provider_call_suppressed is True,
            family="duplicate_event",
            details=duplicate.to_dict(),
        )
    )
    audit = build_hidden_tool_prompt_audit(tool_profile="no-tools", tool_names=[], system_prompt="Answer normally.")
    cases.append(
        build_regression_result(
            "no_tools_hidden_prompt_audit",
            passed=audit.hidden_tool_prompt_detected is False and audit.tool_count == 0,
            family="hidden_tool_prompt",
            details=audit.to_dict(),
        )
    )
    clean_audit = build_external_action_audit()
    cases.append(
        build_regression_result(
            "external_action_audit_read_only_clean",
            passed=clean_audit.read_only_clean is True and clean_audit.unexpected_side_effects is False,
            family="external_action_audit",
            details=clean_audit.to_dict(),
        )
    )
    return cases


def _build_tool_loader_regressions() -> list[dict[str, Any]]:
    schemas = [
        {"function": {"name": "brainstack_recall", "description": "Recall memory"}},
        {"function": {"name": "terminal", "description": "Run shell command"}},
    ]
    registry = build_tool_registry(schemas)
    conv_loader = build_tool_loader_contract("conversation_tools", ["brainstack_recall"], registry)
    heavy_loader = build_tool_loader_contract("heavy_code", ["terminal"], registry)
    terminal = registry["terminal"]

    allowed, allowed_reason = evaluate_tool_load_request(conv_loader, registry, "brainstack_recall")
    cross_ok, cross_reason = evaluate_tool_load_request(conv_loader, registry, "terminal")
    disabled_registry = {**registry, "terminal": replace(terminal, config_disabled=True)}
    gated_registry = {**registry, "terminal": replace(terminal, gated=True)}
    disabled_ok, disabled_reason = evaluate_tool_load_request(heavy_loader, disabled_registry, "terminal")
    gated_ok, gated_reason = evaluate_tool_load_request(heavy_loader, gated_registry, "terminal")
    side_effect_ok, side_effect_reason = evaluate_tool_load_request(heavy_loader, registry, "terminal")
    pinned_ok, pinned_reason = evaluate_tool_load_request(heavy_loader, registry, "terminal", lifetime="pinned", approval_granted=True)

    return [
        build_regression_result(
            "tool_loader_constrained_enum_success",
            passed=allowed is True and allowed_reason == "ALLOWED",
            family="tool_loader",
            details={"allowed": allowed, "reason": allowed_reason, "contract": conv_loader.to_dict()},
        ),
        build_regression_result(
            "tool_loader_cross_profile_denial",
            passed=cross_ok is False and cross_reason == "TOOL_NOT_IN_ALLOWED_ENUM",
            family="tool_loader",
            details={"allowed": cross_ok, "reason": cross_reason},
        ),
        build_regression_result(
            "tool_loader_config_disabled_denial",
            passed=disabled_ok is False and disabled_reason == "TOOL_CONFIG_DISABLED",
            family="tool_loader",
            details={"allowed": disabled_ok, "reason": disabled_reason},
        ),
        build_regression_result(
            "tool_loader_gated_denial",
            passed=gated_ok is False and gated_reason == "TOOL_GATED_UNAVAILABLE",
            family="tool_loader",
            details={"allowed": gated_ok, "reason": gated_reason},
        ),
        build_regression_result(
            "tool_loader_side_effect_approval_preserved",
            passed=side_effect_ok is False and side_effect_reason == "TOOL_SIDE_EFFECT_APPROVAL_REQUIRED",
            family="tool_loader",
            details={"allowed": side_effect_ok, "reason": side_effect_reason},
        ),
        build_regression_result(
            "tool_loader_ephemeral_and_pinned_cleanup",
            passed=(
                heavy_loader.turn_end_cleanup_required
                and heavy_loader.session_end_cleanup_required
                and heavy_loader.profile_change_cleanup_required
                and pinned_ok
                and pinned_reason == "ALLOWED"
            ),
            family="tool_loader",
            details={
                "pinned_allowed": pinned_ok,
                "pinned_reason": pinned_reason,
                "contract": heavy_loader.to_dict(),
            },
        ),
    ]


def build_regression_pack() -> dict[str, Any]:
    provider_cases = _build_provider_regressions()
    loader_cases = _build_tool_loader_regressions()
    all_cases = provider_cases + loader_cases
    passed = all(case["passed"] for case in all_cases)
    return {
        "schema": "hermes.phase156.provider_contract_regression.v1",
        "phase": 156,
        "verdict": "pass" if passed else "fail",
        "failure_taxonomy": list(FAILURE_TAXONOMY),
        "provider_contract_regressions": provider_cases,
        "tool_loader_regressions": loader_cases,
        "summary": {
            "scenario_count": len(all_cases),
            "passed_count": sum(1 for case in all_cases if case["passed"]),
            "provider_contract_count": len(provider_cases),
            "tool_loader_count": len(loader_cases),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHASE_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack = build_regression_pack()
    _json_dump(args.output_dir / "156-PROVIDER-CONTRACT-REGRESSION.json", pack)
    _json_dump(
        args.output_dir / "156-TOOL-LOADER-REGRESSION.json",
        {
            "schema": "hermes.phase156.tool_loader_regression.v1",
            "phase": 156,
            "verdict": "pass" if all(case["passed"] for case in pack["tool_loader_regressions"]) else "fail",
            "tool_loader_regressions": pack["tool_loader_regressions"],
        },
    )
    print(json.dumps({"verdict": pack["verdict"], "summary": pack["summary"]}, sort_keys=True))
    return 0 if pack["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
