from __future__ import annotations

from gateway.memory_answer_renderer import (
    contains_lifetime_certainty,
    evaluate_renderer_eligibility,
    render_memory_answer,
)


CONTRACT = {
    "allowed_tool_profile": "conversation_direct",
    "external_capability_required_for_memory_answer": False,
}


def test_explicit_fact_renders_single_authoritative_evidence() -> None:
    answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        answer_evidence=[{"id": "profile:debug_marker", "value": "1231231X"}],
    )

    assert answer.used_renderer is True
    assert answer.text == "Recorded value: 1231231X."
    assert answer.evidence_ids == ("profile:debug_marker",)
    assert answer.no_tool_call is True
    assert answer.no_memory_mutation is True


def test_unsupported_abstain_renders_without_evidence() -> None:
    answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "unanswerable",
            "max_claim_strength": "none",
            "answer_type": "none",
            "answer_evidence_count": 0,
            "reason_code": "NO_SUPPORTED_MEMORY_TRUTH",
        },
        answer_evidence=[],
    )

    assert answer.used_renderer is True
    assert answer.text == "No supported memory evidence for this request."
    assert answer.evidence_ids == ()


def test_forbidden_cases_fall_back_to_provider() -> None:
    cases = [
        (
            {"state": "conflicted", "max_claim_strength": "none", "answer_type": "conflict_report"},
            {},
        ),
        (
            {"state": "degraded", "max_claim_strength": "memory_truth", "answer_type": "explicit_user_fact"},
            {},
        ),
        (
            {
                "state": "answerable",
                "max_claim_strength": "memory_truth",
                "answer_type": "explicit_user_fact",
                "required_backends": ["chroma"],
            },
            {"chroma": "unavailable"},
        ),
    ]

    for answerability, health in cases:
        answer = render_memory_answer(
            turn_contract=CONTRACT,
            answerability=answerability,
            answer_evidence=[{"id": "profile:x", "value": "x"}],
            capability_health=health,
        )
        assert answer.used_renderer is False
        assert answer.fallback_reason


def test_heavy_or_external_turn_falls_back() -> None:
    eligibility = evaluate_renderer_eligibility(
        turn_contract={
            "allowed_tool_profile": "heavy_web",
            "external_capability_required_for_memory_answer": False,
        },
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        answer_evidence=[{"id": "profile:repo", "value": "https://example.test"}],
    )

    assert eligibility.eligible is False
    assert eligibility.reason_code == "HEAVY_OR_EXTERNAL_TASK"


def test_private_path_and_secret_shaped_values_are_redacted() -> None:
    path_answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "explicit_user_fact",
        },
        answer_evidence=[{"id": "profile:path", "value": "/home/lauratom/private/project"}],
    )
    secret_answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "exact_literal",
        },
        answer_evidence=[{"id": "profile:secret", "value": "sk-abc123456789xyz"}],
    )

    assert "[REDACTED_PRIVATE_PATH]" in path_answer.text
    assert path_answer.redacted is True
    assert "[REDACTED_SECRET_SHAPED]" in secret_answer.text
    assert secret_answer.redacted is True


def test_supporting_context_never_becomes_answer_truth() -> None:
    answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "supporting_context",
            "answer_type": "explicit_user_fact",
        },
        answer_evidence=[{"id": "continuity:bg", "value": "maybe 1231231Y"}],
    )

    assert answer.used_renderer is False
    assert answer.reason_code == "ONLY_SUPPORTING_CONTEXT"


def test_pulse_background_never_assignment_authority() -> None:
    answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "current_assignment",
            "authority": "runtime_state_supporting",
        },
        answer_evidence=[
            {
                "id": "operating:pulse",
                "value": "Brainstack Proactive Pulse running",
                "source_role": "runtime",
            }
        ],
    )

    assert answer.used_renderer is False
    assert answer.reason_code == "ASSIGNMENT_REQUIRES_TYPED_AUTHORITY"


def test_typed_assignment_and_absence_have_humble_claims() -> None:
    presence = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "memory_truth",
            "answer_type": "current_assignment",
            "authority": "typed_current_assignment",
        },
        answer_evidence=[{"id": "task:current", "value": "Phase 155", "current_assignment_authority": True}],
    )
    absence = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "unanswerable",
            "max_claim_strength": "none",
            "answer_type": "current_assignment_absence",
            "reason_code": "NO_TYPED_CURRENT_ASSIGNMENT_EVIDENCE",
        },
        answer_evidence=[],
    )

    assert presence.used_renderer is True
    assert presence.text == "Typed current assignment: Phase 155."
    assert absence.used_renderer is True
    assert absence.text == (
        "No typed current-assignment evidence is recorded. "
        "Background runtime/Pulse evidence alone is not current assignment."
    )


def test_bounded_event_never_claims_lifetime_certainty() -> None:
    answer = render_memory_answer(
        turn_contract=CONTRACT,
        answerability={
            "state": "answerable",
            "max_claim_strength": "bounded_event",
            "answer_type": "conversation_event",
        },
        answer_evidence=[{"id": "event:turn-7", "preview": "User asked about marker 1231231X"}],
    )

    assert answer.used_renderer is True
    assert answer.renderer_claim_style == "bounded_event"
    assert contains_lifetime_certainty(answer.text) is False
    assert answer.text == "Recorded event in searched scope: User asked about marker 1231231X."
