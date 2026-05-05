#!/usr/bin/env python3
"""Write Phase 153 provider/model/degradation proof artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gateway.provider_contract import (  # noqa: E402
    SCHEMA_VERSION,
    build_aux_barrier,
    build_degradation_decision,
    build_discord_interaction_trace,
    build_idempotency_decision,
    build_idempotency_key,
    build_model_contract,
    build_provider_timing_trace,
    build_rate_limit_trace,
    build_readiness_report,
    build_stale_response_decision,
    evaluate_fast_model_parity,
)


def _json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def build_matrix() -> dict[str, Any]:
    parity_ok, parity_reason = evaluate_fast_model_parity(
        same_answerability=True,
        same_answer_evidence=True,
        forbidden_claims=[],
        unsupported_hallucination=False,
        assignment_promotion=False,
        exact_literal_missing=False,
    )
    simple_no_parity = build_model_contract(
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        turn_class="conversation.memory_fact",
        parity_proven=False,
    )
    simple_with_parity = build_model_contract(
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        turn_class="conversation.memory_fact",
        parity_proven=parity_ok,
    )
    conflict = build_model_contract(
        answerability={
            "state": "conflicted",
            "max_claim_strength": "none",
            "answer_type": "conflict_report",
        },
        turn_class="conversation.memory_fact",
        conflict=True,
        parity_proven=True,
    )
    heavy = build_model_contract(
        answerability={"state": "unanswerable", "max_claim_strength": "none", "answer_type": "none"},
        turn_class="heavy_work",
        external_tool_required=True,
        parity_proven=True,
    )

    slow_timing = build_provider_timing_trace(final_ms=50_000, first_token_ms=None)
    slow_degradation = build_degradation_decision(
        timing=slow_timing,
        model_contract=simple_no_parity,
        first_user_visible_commitment_ms=None,
        turn_profile="conversation_direct",
    )
    heavy_timing = build_provider_timing_trace(final_ms=35_000, first_token_ms=4_000, output_tokens=500)
    heavy_degradation = build_degradation_decision(
        timing=heavy_timing,
        model_contract=heavy,
        first_user_visible_commitment_ms=8_000,
        turn_profile="heavy_work",
    )
    stale = build_stale_response_decision(
        platform="discord",
        channel_id="phase153",
        user_id="canary",
        turn_id="turn-A",
        causal_index=1,
        latest_completed_index=2,
        superseded_by="turn-B",
    )
    idem_key = build_idempotency_key(
        platform="discord",
        guild_id="guild",
        channel_id="phase153",
        message_id="msg-1",
        author_id="canary",
    )
    duplicate = build_idempotency_decision(idempotency_key_hash=idem_key, seen_keys={idem_key})
    aux = build_aux_barrier(
        raw_turn_committed_before_response=True,
        explicit_write_status="pending",
        trace_id_committed=True,
        post_response_tasks_pending=("tier2_consolidation", "title_generation"),
        aux_blocking_ms=0,
    )
    interaction = build_discord_interaction_trace(
        is_interaction=True,
        interaction_ack_ms=900,
        ack_type="deferred_response",
        followup_used=True,
    )
    rate_limit = build_rate_limit_trace(messages_created=1, messages_edited=1)
    return {
        "schema": SCHEMA_VERSION,
        "fast_model_parity": {"passed": parity_ok, "reason": parity_reason},
        "model_contracts": {
            "simple_memory_without_parity": simple_no_parity.to_dict(),
            "simple_memory_with_parity": simple_with_parity.to_dict(),
            "conflict_reasoning": conflict.to_dict(),
            "heavy_external_workflow": heavy.to_dict(),
        },
        "provider_timing": {
            "slow_unobservable": slow_timing.to_dict(),
            "heavy_observable": heavy_timing.to_dict(),
        },
        "degradation": {
            "slow_conversation": slow_degradation.to_dict(),
            "heavy_progress": heavy_degradation.to_dict(),
        },
        "stale_response": stale.to_dict(),
        "idempotency": duplicate.to_dict(),
        "aux_barrier": aux.to_dict(),
        "discord_interaction": interaction.to_dict(),
        "rate_limit": rate_limit.to_dict(),
        "verdict": {
            "passed": (
                not simple_no_parity.fast_model_allowed
                and simple_with_parity.fast_model_allowed
                and not conflict.fast_model_allowed
                and slow_degradation.activated
                and stale.stale_response_suppressed
                and duplicate.duplicate_provider_call_suppressed
                and interaction.ack_slo_satisfied is True
            )
        },
    }


def build_readiness() -> dict[str, Any]:
    pulse_degraded = build_readiness_report(
        chat_provider_health="ok",
        pulse_provider_health="reauth_failed",
        memory_health="ok",
        corpus_semantic_health="ok",
        heavy_tool_health="ok",
        cold_backend_live_download=False,
    )
    cold_backend = build_readiness_report(
        chat_provider_health="ok",
        pulse_provider_health="ok",
        memory_health="ok",
        corpus_semantic_health="ok",
        heavy_tool_health="ok",
        cold_backend_live_download=True,
    )
    return {
        "schema": SCHEMA_VERSION,
        "reports": {
            "pulse_degraded_chat_ok": pulse_degraded.to_dict(),
            "cold_backend_blocks_ready": cold_backend.to_dict(),
        },
        "verdict": {
            "passed": (
                pulse_degraded.chat_ready
                and pulse_degraded.provider_health_isolated
                and not cold_backend.chat_ready
                and not cold_backend.memory_ready
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    readiness = build_readiness()
    _json_dump(output_dir / "153-PROVIDER-MODEL-MATRIX.json", matrix)
    _json_dump(output_dir / "153-READINESS-REPORT.json", readiness)
    passed = matrix["verdict"]["passed"] and readiness["verdict"]["passed"]
    print(json.dumps({"schema": SCHEMA_VERSION, "passed": passed}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
