from agent.memory_evidence_repair_recovery_execution_preview import (
    build_evidence_repair_recovery_execution_preview,
)
from agent.memory_evidence_repair_recovery_human_approval_token import (
    RECOVERY_APPROVAL_TOKEN_SCOPE,
    RECOVERY_APPROVAL_TOKEN_TYPE,
    RECOVERY_TOKEN_STATUS_BLOCKED,
    RECOVERY_TOKEN_STATUS_DRAFT_READY,
    RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED,
    build_evidence_repair_recovery_human_approval_token,
    empty_evidence_repair_recovery_human_approval_token,
)
from tests.agent.test_memory_evidence_repair_recovery_execution_preview import (
    _manual_rollback_decision,
    _preparedness_decision,
)


def _manual_rollback_execution():
    return build_evidence_repair_recovery_execution_preview(
        recovery_decision=_manual_rollback_decision()
    ).to_dict()


def _preparedness_execution():
    return build_evidence_repair_recovery_execution_preview(
        recovery_decision=_preparedness_decision()
    ).to_dict()


def test_manual_rollback_execution_creates_recovery_token_draft():
    payload = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_manual_rollback_execution(),
        approver="han",
        approval_reason="manual recovery approval",
        expires_in_minutes=20,
    ).to_dict()

    token = payload["tokens"][0]
    assert payload["status"] == RECOVERY_TOKEN_STATUS_DRAFT_READY
    assert payload["summary"]["token_count"] == 1
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert token["id"].startswith("recovery-approval-token-")
    assert token["token_type"] == RECOVERY_APPROVAL_TOKEN_TYPE
    assert token["scope"] == RECOVERY_APPROVAL_TOKEN_SCOPE
    assert token["approver"] == "han"
    assert token["approval_reason"] == "manual recovery approval"
    assert token["expires_in_minutes"] == 20
    assert token["operation_ids"] == ["dryrun-op-123"]
    assert token["future_mutation_step_ids"] == [
        "recovery-execution-step-5-apply_manual_rollback_restore"
    ]
    assert len(token["token_digest"]) == 64
    assert token["one_time_constraints"]["one_time_use"] is True
    assert token["required_confirmation_text"].startswith("APPROVE ")


def test_preparedness_execution_does_not_need_recovery_token():
    payload = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_preparedness_execution()
    ).to_dict()

    assert payload["status"] == RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED
    assert payload["tokens"] == []
    assert payload["summary"]["no_action_count"] == 1


def test_blocked_execution_preview_does_not_create_token():
    payload = build_evidence_repair_recovery_human_approval_token(
        recovery_execution={
            "status": "blocked",
            "blocking_reasons": ["recovery decision missing"],
            "required_actions": ["produce_recovery_decision_gate"],
        }
    ).to_dict()

    assert payload["status"] == RECOVERY_TOKEN_STATUS_BLOCKED
    assert payload["tokens"] == []
    assert payload["blocking_reasons"] == ["recovery decision missing"]
    assert payload["required_actions"] == ["produce_recovery_decision_gate"]


def test_recovery_token_id_and_digest_are_stable():
    first = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_manual_rollback_execution(),
        approver="han",
        approval_reason="manual recovery approval",
    ).to_dict()
    second = build_evidence_repair_recovery_human_approval_token(
        recovery_execution=_manual_rollback_execution(),
        approver="han",
        approval_reason="manual recovery approval",
    ).to_dict()

    assert first["tokens"][0]["id"] == second["tokens"][0]["id"]
    assert first["tokens"][0]["token_digest"] == second["tokens"][0]["token_digest"]


def test_empty_recovery_token_report_is_read_only():
    payload = empty_evidence_repair_recovery_human_approval_token().to_dict()

    assert payload["status"] == RECOVERY_TOKEN_STATUS_NO_ACTION_NEEDED
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["would_mutate_memory"] is False
    assert payload["tokens"] == []
