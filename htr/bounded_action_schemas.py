"""Closed enums and constants for Task 28 Phase 28A."""

from __future__ import annotations

from enum import Enum

PROTOCOL_VERSION = "htr.bounded_action.phase28a.v1"
SCHEMA_VERSION = "1"

BOUNDED_ACTION_MAX_PROPOSALS_PER_SUCCESSOR = 16
BOUNDED_ACTION_MAX_ESCALATIONS_PER_SUCCESSOR = 8
BOUNDED_ACTION_MAX_TERMINALS_PER_PROPOSAL = 1

RECORD_TYPE_PROPOSAL = "bounded_action_proposal"
RECORD_TYPE_REVIEW = "bounded_action_review_decision"
RECORD_TYPE_ESCALATION = "bounded_action_escalation"

NAMESPACE_INTEGRITY_SCHEMA = "htr.bounded_action.namespace_integrity.v1"
TARGET_AGGREGATE_SCHEMA = "htr.bounded_action.target_aggregate.v1"

AUTHORITY_BOOLEAN_FIELDS = (
    "execution_authority_granted",
    "successor_mutation_allowed",
    "retry_allowed",
    "repair_allowed",
    "marker_disposition_allowed",
    "automatic_continuation_allowed",
    "external_side_effect_allowed",
    "source_run_mutation_allowed",
    "outcome_rewrite_allowed",
)


class ProposalSubject(str, Enum):
    bounded_action_architecture_candidate = "bounded_action_architecture_candidate"
    future_forward_repair_candidate = "future_forward_repair_candidate"
    future_retry_candidate = "future_retry_candidate"
    reconciliation_escalation_candidate = "reconciliation_escalation_candidate"
    unsupported_action_observed = "unsupported_action_observed"


class RiskClass(str, Enum):
    advisory_read_only_low = "advisory_read_only_low"
    advisory_integrity_review_high = "advisory_integrity_review_high"
    prohibited_execution_or_external_effect = "prohibited_execution_or_external_effect"


class ConfidenceClass(str, Enum):
    proven = "proven"
    high = "high"
    insufficient = "insufficient"
    indeterminate = "indeterminate"


class DecisionClass(str, Enum):
    accepted_for_future_architecture_review = "accepted_for_future_architecture_review"
    rejected = "rejected"
    deferred = "deferred"


class EscalationClass(str, Enum):
    integrity_anomaly = "integrity_anomaly"
    marker_residue = "marker_residue"
    source_evidence_drift = "source_evidence_drift"
    successor_evidence_drift = "successor_evidence_drift"
    task27_lineage_drift = "task27_lineage_drift"
    publication_budget_exhausted = "publication_budget_exhausted"
    unsupported_action = "unsupported_action"
    human_review_required = "human_review_required"


class EvidenceDriftClassification(str, Enum):
    no_drift = "no_drift"
    raw_byte_drift = "raw_byte_drift"
    semantic_drift = "semantic_drift"
    path_identity_drift = "path_identity_drift"
    marker_state_drift = "marker_state_drift"
    task27_lineage_drift = "task27_lineage_drift"
    aggregate_state_drift = "aggregate_state_drift"
    indeterminate = "indeterminate"


PROPOSAL_REASON_CODES = frozenset(
    {
        "architecture_review_requested",
        "integrity_findings_present",
        "reconciliation_signal_present",
        "successor_state_ambiguous",
        "unsupported_action_detected",
        "human_checkpoint_recommended",
        "marker_residue_observed",
    }
)

REVIEW_REASON_CODES = frozenset(
    {
        "evidence_sufficient_for_advisory_acceptance",
        "evidence_insufficient",
        "integrity_blocking",
        "proposal_subject_not_actionable",
        "defer_pending_human_review",
    }
)

ESCALATION_REASON_CODES = frozenset(
    {
        "fresh_evidence_drift_detected",
        "marker_present_unexpected",
        "aggregate_cap_exhausted",
        "case_state_indeterminate",
        "malformed_predecessor",
        "duplicate_identity_detected",
    }
)

SUBJECT_MATRIX: dict[str, dict[str, frozenset[str] | bool]] = {
    ProposalSubject.bounded_action_architecture_candidate.value: {
        "marker_required_absent": True,
        "risk": frozenset({RiskClass.advisory_read_only_low.value, RiskClass.advisory_integrity_review_high.value}),
        "confidence": frozenset({ConfidenceClass.proven.value, ConfidenceClass.high.value}),
        "required_reasons": frozenset({"architecture_review_requested"}),
        "forbidden_reasons": frozenset({"marker_residue_observed"}),
    },
    ProposalSubject.future_forward_repair_candidate.value: {
        "marker_required_absent": True,
        "risk": frozenset({RiskClass.advisory_integrity_review_high.value}),
        "confidence": frozenset(
            {ConfidenceClass.proven.value, ConfidenceClass.high.value, ConfidenceClass.insufficient.value}
        ),
        "required_reasons": frozenset({"architecture_review_requested"}),
        "forbidden_reasons": frozenset({"marker_residue_observed"}),
    },
    ProposalSubject.future_retry_candidate.value: {
        "marker_required_absent": True,
        "risk": frozenset({RiskClass.advisory_integrity_review_high.value}),
        "confidence": frozenset({ConfidenceClass.proven.value, ConfidenceClass.high.value}),
        "required_reasons": frozenset({"architecture_review_requested"}),
        "forbidden_reasons": frozenset({"marker_residue_observed"}),
    },
    ProposalSubject.reconciliation_escalation_candidate.value: {
        "marker_required_absent": False,
        "risk": frozenset(
            {
                RiskClass.advisory_integrity_review_high.value,
                RiskClass.prohibited_execution_or_external_effect.value,
            }
        ),
        "confidence": frozenset(
            {ConfidenceClass.proven.value, ConfidenceClass.high.value, ConfidenceClass.insufficient.value}
        ),
        "required_reasons": frozenset({"marker_residue_observed", "reconciliation_signal_present"}),
        "forbidden_reasons": frozenset({"unsupported_action_detected"}),
    },
    ProposalSubject.unsupported_action_observed.value: {
        "marker_required_absent": False,
        "risk": frozenset({RiskClass.prohibited_execution_or_external_effect.value}),
        "confidence": frozenset(
            {
                ConfidenceClass.proven.value,
                ConfidenceClass.high.value,
                ConfidenceClass.insufficient.value,
                ConfidenceClass.indeterminate.value,
            }
        ),
        "required_reasons": frozenset({"unsupported_action_detected"}),
        "forbidden_reasons": frozenset({"marker_residue_observed"}),
    },
}


def authority_booleans_false() -> dict[str, bool]:
    return {field: False for field in AUTHORITY_BOOLEAN_FIELDS}
