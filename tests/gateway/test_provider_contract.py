from __future__ import annotations

from gateway.provider_contract import (
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


def test_provider_timing_splits_queue_and_generation_when_observable() -> None:
    queue = build_provider_timing_trace(final_ms=22_000, first_token_ms=18_000, output_tokens=100)
    generation = build_provider_timing_trace(final_ms=25_000, first_token_ms=2_000, output_tokens=100)
    unobservable = build_provider_timing_trace(final_ms=7_000, first_token_ms=None)

    assert queue.latency_owner == "provider_queue_or_prompt_processing"
    assert generation.latency_owner == "provider_generation"
    assert unobservable.queue_generation_observable is False


def test_fast_model_needs_parity_even_for_simple_memory() -> None:
    contract = build_model_contract(
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        turn_class="conversation.memory_fact",
        parity_proven=False,
    )
    parity_ok, reason = evaluate_fast_model_parity(
        same_answerability=True,
        same_answer_evidence=True,
        forbidden_claims=[],
        unsupported_hallucination=False,
        assignment_promotion=False,
        exact_literal_missing=False,
    )
    allowed = build_model_contract(
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        turn_class="conversation.memory_fact",
        parity_proven=parity_ok,
    )

    assert contract.fast_model_allowed is False
    assert parity_ok is True
    assert reason == "PARITY_OK"
    assert allowed.fast_model_allowed is True


def test_current_assignment_absence_is_simple_memory_contract() -> None:
    contract = build_model_contract(
        answerability={
            "state": "unanswerable",
            "max_claim_strength": "none",
            "answer_type": "current_assignment_absence",
        },
        turn_class="conversation.current_assignment_absence",
        parity_proven=False,
    )

    assert contract.required_reasoning == "faithful_rendering"
    assert contract.model_profile == "conversation_renderer"
    assert contract.reason_code == "SIMPLE_MEMORY_RENDERING_PARITY_REQUIRED"
    assert contract.fast_model_allowed is False


def test_fast_model_forbidden_for_conflict_without_quality_model() -> None:
    contract = build_model_contract(
        answerability={
            "state": "conflicted",
            "max_claim_strength": "none",
            "answer_type": "conflict_report",
        },
        turn_class="conversation.memory_fact",
        conflict=True,
        parity_proven=True,
    )

    assert contract.fast_model_allowed is False
    assert contract.model_profile == "reasoning_heavy"
    assert contract.reason_code == "CONFLICT_REASONING_REQUIRED"


def test_degradation_detects_silent_wait_violation() -> None:
    timing = build_provider_timing_trace(final_ms=50_000, first_token_ms=None)
    contract = build_model_contract(
        answerability={"state": "answerable", "max_claim_strength": "memory_truth", "answer_type": "explicit_user_fact"},
        turn_class="conversation.memory_fact",
    )
    decision = build_degradation_decision(
        timing=timing,
        model_contract=contract,
        first_user_visible_commitment_ms=None,
        turn_profile="conversation_direct",
    )

    assert decision.activated is True
    assert decision.no_silent_wait_satisfied is False
    assert decision.reason_code == "FIRST_VISIBLE_COMMITMENT_SLO_MISSED"


def test_pulse_provider_failure_isolated_from_chat_readiness() -> None:
    report = build_readiness_report(
        chat_provider_health="ok",
        pulse_provider_health="reauth_failed",
        memory_health="ok",
        corpus_semantic_health="ok",
        heavy_tool_health="ok",
        cold_backend_live_download=False,
    )

    assert report.chat_ready is True
    assert report.provider_health_isolated is True
    assert "PULSE_PROVIDER_DEGRADED_NONBLOCKING_FOR_CHAT" in report.reason_codes


def test_cold_backend_live_download_blocks_readiness() -> None:
    report = build_readiness_report(
        chat_provider_health="ok",
        pulse_provider_health="ok",
        memory_health="ok",
        corpus_semantic_health="ok",
        heavy_tool_health="ok",
        cold_backend_live_download=True,
    )

    assert report.chat_ready is False
    assert report.memory_ready is False
    assert "COLD_BACKEND_LIVE_DOWNLOAD_BLOCKS_READINESS" in report.reason_codes


def test_stale_response_suppressed_after_newer_turn_completes() -> None:
    decision = build_stale_response_decision(
        platform="discord",
        channel_id="chan",
        user_id="user",
        turn_id="A",
        causal_index=1,
        latest_completed_index=2,
        superseded_by="B",
    )

    assert decision.stale_response_suppressed is True
    assert decision.superseded_by == "B"


def test_duplicate_event_idempotency_suppresses_side_effects() -> None:
    key = build_idempotency_key(
        platform="discord",
        guild_id="guild",
        channel_id="chan",
        message_id="msg",
        author_id="author",
    )
    first = build_idempotency_decision(idempotency_key_hash=key, seen_keys=set())
    duplicate = build_idempotency_decision(idempotency_key_hash=key, seen_keys={key})

    assert first.duplicate_event is False
    assert duplicate.duplicate_provider_call_suppressed is True
    assert duplicate.duplicate_memory_write_suppressed is True


def test_aux_barrier_keeps_pending_write_visible_to_next_turn() -> None:
    barrier = build_aux_barrier(
        raw_turn_committed_before_response=True,
        explicit_write_status="pending",
        trace_id_committed=True,
        post_response_tasks_pending=["tier2_consolidation", "title_generation"],
        aux_blocking_ms=0,
    )

    assert barrier.next_turn_barrier_policy == "next_turn_wait_300ms_or_degrade_pending_write_barrier"
    assert barrier.post_response_tasks_pending == ("tier2_consolidation", "title_generation")


def test_discord_ack_and_rate_limit_progress_policy() -> None:
    interaction = build_discord_interaction_trace(
        is_interaction=True,
        interaction_ack_ms=800,
        ack_type="deferred_response",
        followup_used=True,
    )
    rate_limit = build_rate_limit_trace(messages_created=1, messages_edited=1)

    assert interaction.ack_slo_satisfied is True
    assert rate_limit.progress_policy == "edit_single_progress_message_no_heartbeat_spam"
