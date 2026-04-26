"""Deterministic renderer for simple Gateway memory answerability packets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from gateway.context_budget import redact_model_facing_text


SCHEMA_VERSION = "hermes.memory_answer_renderer.v1"

SUPPORTED_FACT_TYPES = {"explicit_user_fact", "exact_literal"}
SUPPORTED_EVENT_TYPES = {"conversation_event", "prior_conversation_event", "bounded_event"}
SUPPORTED_ASSIGNMENT_TYPES = {"current_assignment", "current_assignment_absence"}
BAD_BACKEND_STATES = {"degraded", "unavailable", "down", "error", "missing"}
HEAVY_OR_EXTERNAL_PROFILES = {"heavy", "heavy_work", "heavy_web", "heavy_file", "heavy_code", "heavy_full_debug"}
TYPED_ASSIGNMENT_AUTHORITIES = {"typed_current_assignment", "current_assignment_authority"}
LIFETIME_CERTAINTY_WORDS = ("always", "never", "forever", "lifetime")


@dataclass(frozen=True)
class RendererEligibility:
    schema: str
    eligible: bool
    answer_type: str
    reason_code: str
    fallback_reason: str | None
    evidence_ids: tuple[str, ...]
    renderer_claim_style: str
    no_tool_call: bool = True
    no_memory_mutation: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_ids"] = list(self.evidence_ids)
        return data


@dataclass(frozen=True)
class RenderedMemoryAnswer:
    schema: str
    used_renderer: bool
    text: str
    answer_type: str
    reason_code: str
    fallback_reason: str | None
    evidence_ids: tuple[str, ...]
    renderer_claim_style: str
    redacted: bool
    no_tool_call: bool = True
    no_memory_mutation: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence_ids"] = list(self.evidence_ids)
        return data


def evaluate_renderer_eligibility(
    *,
    turn_contract: Mapping[str, Any] | Any,
    answerability: Mapping[str, Any],
    answer_evidence: Sequence[Mapping[str, Any]] = (),
    capability_health: Mapping[str, str] | None = None,
) -> RendererEligibility:
    """Return fail-closed eligibility for deterministic memory rendering."""

    contract = _as_mapping(turn_contract)
    answer_type = str(answerability.get("answer_type", "none"))
    state = str(answerability.get("state", "unanswerable"))
    max_claim_strength = str(answerability.get("max_claim_strength", "none"))
    evidence_ids = _evidence_ids(answerability, answer_evidence)

    if _is_heavy_or_external(contract):
        return _ineligible(answer_type, "HEAVY_OR_EXTERNAL_TASK", "provider_required_for_heavy_or_external_task")

    if state in {"conflicted", "conflict"}:
        return _ineligible(answer_type, "CONFLICT_REQUIRES_MODEL", "conflict_requires_model")

    if state == "degraded":
        return _ineligible(answer_type, "DEGRADED_STATE_REQUIRES_MODEL", "degraded_state_requires_model")

    backend_reason = _degraded_required_backend(answerability, capability_health or {})
    if backend_reason:
        return _ineligible(answer_type, "REQUIRED_BACKEND_DEGRADED", backend_reason)

    if max_claim_strength == "supporting_context":
        return _ineligible(answer_type, "ONLY_SUPPORTING_CONTEXT", "supporting_context_is_not_answer_truth")

    if _is_unsupported(answer_type, state, max_claim_strength, answerability, answer_evidence):
        return _eligible(
            answer_type="none",
            reason_code="UNSUPPORTED_ABSTAIN",
            evidence_ids=(),
            claim_style="unsupported",
        )

    if answer_type in SUPPORTED_FACT_TYPES:
        if state != "answerable" or max_claim_strength != "memory_truth":
            return _ineligible(answer_type, "FACT_NOT_MEMORY_TRUTH", "fact_requires_memory_truth_answerability")
        if len(answer_evidence) != 1 or not evidence_ids:
            return _ineligible(answer_type, "FACT_REQUIRES_SINGLE_EVIDENCE", "fact_requires_single_answer_evidence")
        return _eligible(answer_type, "AUTHORITATIVE_SIMPLE_FACT", evidence_ids, "explicit_fact")

    if answer_type in SUPPORTED_EVENT_TYPES:
        if state != "answerable" or max_claim_strength != "bounded_event":
            return _ineligible(answer_type, "EVENT_NOT_BOUNDED", "event_requires_bounded_event_strength")
        if not evidence_ids:
            return _ineligible(answer_type, "EVENT_REQUIRES_EVIDENCE", "event_requires_evidence_id")
        return _eligible(answer_type, "BOUNDED_EVENT", evidence_ids, "bounded_event")

    if answer_type in SUPPORTED_ASSIGNMENT_TYPES:
        return _assignment_eligibility(answer_type, answerability, answer_evidence, evidence_ids)

    return _ineligible(answer_type, "UNSUPPORTED_ANSWER_TYPE", f"unsupported_answer_type:{answer_type}")


def render_memory_answer(
    *,
    turn_contract: Mapping[str, Any] | Any,
    answerability: Mapping[str, Any],
    answer_evidence: Sequence[Mapping[str, Any]] = (),
    capability_health: Mapping[str, str] | None = None,
) -> RenderedMemoryAnswer:
    """Render simple memory answer or return explicit provider fallback trace."""

    eligibility = evaluate_renderer_eligibility(
        turn_contract=turn_contract,
        answerability=answerability,
        answer_evidence=answer_evidence,
        capability_health=capability_health,
    )
    if not eligibility.eligible:
        return RenderedMemoryAnswer(
            schema=SCHEMA_VERSION,
            used_renderer=False,
            text="",
            answer_type=eligibility.answer_type,
            reason_code=eligibility.reason_code,
            fallback_reason=eligibility.fallback_reason,
            evidence_ids=eligibility.evidence_ids,
            renderer_claim_style=eligibility.renderer_claim_style,
            redacted=False,
        )

    raw = _render_text(eligibility.answer_type, eligibility.renderer_claim_style, answer_evidence)
    text = redact_model_facing_text(raw)
    return RenderedMemoryAnswer(
        schema=SCHEMA_VERSION,
        used_renderer=True,
        text=text,
        answer_type=eligibility.answer_type,
        reason_code=eligibility.reason_code,
        fallback_reason=None,
        evidence_ids=eligibility.evidence_ids,
        renderer_claim_style=eligibility.renderer_claim_style,
        redacted=text != raw,
    )


def _as_mapping(value: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict"):
        mapped = value.to_dict()
        if isinstance(mapped, Mapping):
            return mapped
    return {}


def _is_heavy_or_external(contract: Mapping[str, Any]) -> bool:
    profile = str(contract.get("allowed_tool_profile", ""))
    external = bool(contract.get("external_capability_required_for_memory_answer", False))
    return external or profile in HEAVY_OR_EXTERNAL_PROFILES or profile.startswith("heavy_")


def _degraded_required_backend(answerability: Mapping[str, Any], capability_health: Mapping[str, str]) -> str | None:
    if bool(answerability.get("required_backend_degraded", False)):
        return "answerability_marked_required_backend_degraded"
    required = answerability.get("required_backends") or answerability.get("required_backend_keys") or ()
    if isinstance(required, str):
        required = (required,)
    for backend in required:
        status = str(capability_health.get(str(backend), "unknown")).lower()
        if status in BAD_BACKEND_STATES:
            return f"required_backend_{backend}_{status}"
    return None


def _is_unsupported(
    answer_type: str,
    state: str,
    max_claim_strength: str,
    answerability: Mapping[str, Any],
    answer_evidence: Sequence[Mapping[str, Any]],
) -> bool:
    no_answer_type = answer_type in {"none", "unsupported", "no_evidence"}
    no_evidence = not answer_evidence and int(answerability.get("answer_evidence_count", 0) or 0) == 0
    return state == "unanswerable" and max_claim_strength == "none" and no_answer_type and no_evidence


def _assignment_eligibility(
    answer_type: str,
    answerability: Mapping[str, Any],
    answer_evidence: Sequence[Mapping[str, Any]],
    evidence_ids: tuple[str, ...],
) -> RendererEligibility:
    state = str(answerability.get("state", "unanswerable"))
    reason_code = str(answerability.get("reason_code", "UNKNOWN"))
    if answer_type == "current_assignment_absence" or reason_code == "NO_TYPED_CURRENT_ASSIGNMENT_EVIDENCE":
        return _eligible(
            answer_type="current_assignment_absence",
            reason_code="NO_TYPED_CURRENT_ASSIGNMENT_EVIDENCE",
            evidence_ids=evidence_ids,
            claim_style="current_assignment_absence",
        )

    if state != "answerable":
        return _ineligible(answer_type, "ASSIGNMENT_NOT_ANSWERABLE", "assignment_presence_requires_answerable_state")

    authority = str(answerability.get("authority", ""))
    if authority not in TYPED_ASSIGNMENT_AUTHORITIES and not any(
        _has_typed_assignment_authority(evidence) for evidence in answer_evidence
    ):
        return _ineligible(answer_type, "ASSIGNMENT_REQUIRES_TYPED_AUTHORITY", "no_typed_assignment_authority")

    if not evidence_ids:
        return _ineligible(answer_type, "ASSIGNMENT_REQUIRES_EVIDENCE", "assignment_requires_evidence_id")

    return _eligible(answer_type, "TYPED_CURRENT_ASSIGNMENT", evidence_ids, "current_assignment_presence")


def _has_typed_assignment_authority(evidence: Mapping[str, Any]) -> bool:
    if bool(evidence.get("current_assignment_authority", False)):
        return True
    schema = str(evidence.get("current_assignment_authority_schema", ""))
    authority = str(evidence.get("authority", ""))
    return bool(schema) or authority in TYPED_ASSIGNMENT_AUTHORITIES


def _evidence_ids(
    answerability: Mapping[str, Any],
    answer_evidence: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    ids: list[str] = []
    raw = answerability.get("answer_evidence_ids") or answerability.get("evidence_ids") or ()
    if isinstance(raw, str):
        raw = (raw,)
    for value in raw:
        _append_unique(ids, str(value))
    for evidence in answer_evidence:
        for key in ("evidence_id", "id", "candidate_id", "stable_key"):
            value = evidence.get(key)
            if value:
                _append_unique(ids, str(value))
                break
    return tuple(ids)


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _render_text(answer_type: str, claim_style: str, answer_evidence: Sequence[Mapping[str, Any]]) -> str:
    if claim_style == "unsupported":
        return "No supported memory evidence for this request."
    if claim_style == "current_assignment_absence":
        return "No typed current-assignment evidence is recorded. Background runtime/Pulse evidence alone is not current assignment."

    value = _answer_value(answer_evidence)
    if claim_style == "bounded_event":
        return f"Recorded event in searched scope: {value}."
    if claim_style == "current_assignment_presence":
        return f"Typed current assignment: {value}."
    if answer_type == "exact_literal":
        return f"Recorded identifier: {value}."
    return f"Recorded value: {value}."


def _answer_value(answer_evidence: Sequence[Mapping[str, Any]]) -> str:
    if not answer_evidence:
        return "none"
    evidence = answer_evidence[0]
    for key in ("value", "literal", "recorded_value", "answer", "summary", "preview", "text", "content"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            nested = _answer_value((value,))
            if nested != "recorded evidence":
                return nested
    return "recorded evidence"


def _eligible(
    answer_type: str,
    reason_code: str,
    evidence_ids: Sequence[str],
    claim_style: str,
) -> RendererEligibility:
    return RendererEligibility(
        schema=SCHEMA_VERSION,
        eligible=True,
        answer_type=answer_type,
        reason_code=reason_code,
        fallback_reason=None,
        evidence_ids=tuple(evidence_ids),
        renderer_claim_style=claim_style,
    )


def _ineligible(answer_type: str, reason_code: str, fallback_reason: str) -> RendererEligibility:
    return RendererEligibility(
        schema=SCHEMA_VERSION,
        eligible=False,
        answer_type=answer_type,
        reason_code=reason_code,
        fallback_reason=fallback_reason,
        evidence_ids=(),
        renderer_claim_style="provider_fallback",
    )


def contains_lifetime_certainty(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in LIFETIME_CERTAINTY_WORDS)
