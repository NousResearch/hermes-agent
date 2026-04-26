"""Provider/model/degradation contracts for Hermes gateway turns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "hermes.provider_contract.v1"

INTERACTIVE_FIRST_VISIBLE_LIMIT_MS = 15_000
HEAVY_PROGRESS_VISIBLE_LIMIT_MS = 10_000


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_parts(parts: Sequence[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderTimingTrace:
    schema: str
    request_build_ms: int | None
    http_connect_ms: int | None
    time_to_first_byte_ms: int | None
    first_token_ms: int | None
    generation_after_first_ms: int | None
    final_ms: int
    output_tokens: int | None
    tokens_per_second_after_first: float | None
    queue_generation_observable: bool
    latency_owner: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelContract:
    schema: str
    required_reasoning: str
    risk_class: str
    evidence_strength: str
    latency_class_required: str
    model_profile: str
    fast_model_allowed: bool
    fast_model_requires_parity: bool
    reason_code: str
    forbidden_without_parity: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["forbidden_without_parity"] = list(self.forbidden_without_parity)
        return data


@dataclass(frozen=True)
class DegradationDecision:
    schema: str
    activated: bool
    reason_code: str
    policy: str
    first_user_visible_commitment_ms: int | None
    commitment_type: str
    no_silent_wait_satisfied: bool
    progress_visible_required: bool
    fallback_selected: str
    why_not_fast_model: str | None
    why_not_direct_renderer: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessReport:
    schema: str
    chat_ready: bool
    memory_ready: bool
    heavy_ready: bool
    corpus_semantic_ready: bool
    chat_provider_health: str
    pulse_provider_health: str
    cold_backend_live_download_blocked: bool
    provider_health_isolated: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reason_codes"] = list(self.reason_codes)
        return data


@dataclass(frozen=True)
class StaleResponseDecision:
    schema: str
    stream_key_hash: str
    turn_id: str
    causal_index: int
    latest_completed_index: int
    superseded_by: str | None
    stale_response_suppressed: bool
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IdempotencyDecision:
    schema: str
    idempotency_key_hash: str
    duplicate_event: bool
    duplicate_suppressed: bool
    duplicate_provider_call_suppressed: bool
    duplicate_reply_suppressed: bool
    duplicate_memory_write_suppressed: bool
    duplicate_event_index_suppressed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuxBarrier:
    schema: str
    raw_turn_committed_before_response: bool
    explicit_write_status: str
    trace_id_committed: bool
    post_response_tasks_pending: tuple[str, ...]
    aux_blocking_ms: int
    aux_blocking_reason: str
    next_turn_barrier_policy: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["post_response_tasks_pending"] = list(self.post_response_tasks_pending)
        return data


@dataclass(frozen=True)
class DiscordInteractionTrace:
    schema: str
    is_interaction: bool
    interaction_ack_ms: int | None
    ack_type: str | None
    followup_used: bool
    ack_slo_satisfied: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RateLimitTrace:
    schema: str
    messages_created: int
    messages_edited: int
    retry_after_ms: int
    rate_limit_wait_ms: int
    progress_policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_provider_timing_trace(
    *,
    final_ms: int,
    request_build_ms: int | None = None,
    http_connect_ms: int | None = None,
    first_token_ms: int | None = None,
    output_tokens: int | None = None,
) -> ProviderTimingTrace:
    generation_after_first = None
    tokens_per_second = None
    if first_token_ms is not None:
        generation_after_first = max(0, final_ms - first_token_ms)
        if output_tokens and generation_after_first > 0:
            tokens_per_second = round(output_tokens / (generation_after_first / 1000.0), 3)

    if first_token_ms is None:
        owner = "unobservable_provider_or_gateway"
    elif first_token_ms > INTERACTIVE_FIRST_VISIBLE_LIMIT_MS:
        owner = "provider_queue_or_prompt_processing"
    elif generation_after_first and generation_after_first > INTERACTIVE_FIRST_VISIBLE_LIMIT_MS:
        owner = "provider_generation"
    else:
        owner = "within_interactive_budget"

    return ProviderTimingTrace(
        schema=SCHEMA_VERSION,
        request_build_ms=request_build_ms,
        http_connect_ms=http_connect_ms,
        time_to_first_byte_ms=first_token_ms,
        first_token_ms=first_token_ms,
        generation_after_first_ms=generation_after_first,
        final_ms=final_ms,
        output_tokens=output_tokens,
        tokens_per_second_after_first=tokens_per_second,
        queue_generation_observable=first_token_ms is not None,
        latency_owner=owner,
    )


def build_model_contract(
    *,
    answerability: Mapping[str, Any],
    turn_class: str,
    conflict: bool = False,
    external_tool_required: bool = False,
    parity_proven: bool = False,
) -> ModelContract:
    state = str(answerability.get("state", "unanswerable"))
    strength = str(answerability.get("max_claim_strength", "none"))
    answer_type = str(answerability.get("answer_type", "none"))

    simple_memory = (
        not conflict
        and not external_tool_required
        and state in {"answerable", "unanswerable"}
        and strength in {"memory_truth", "bounded_event", "none"}
        and answer_type
        in {
            "explicit_user_fact",
            "conversation_event",
            "current_assignment",
            "current_assignment_absence",
            "none",
        }
        and turn_class.startswith("conversation")
    )
    if conflict:
        return ModelContract(
            schema=SCHEMA_VERSION,
            required_reasoning="multi_evidence_conflict_resolution",
            risk_class="medium_high",
            evidence_strength="conflicting",
            latency_class_required="quality_first",
            model_profile="reasoning_heavy",
            fast_model_allowed=False,
            fast_model_requires_parity=True,
            reason_code="CONFLICT_REASONING_REQUIRED",
            forbidden_without_parity=("fast_model_default", "memory_direct_renderer"),
        )
    if external_tool_required:
        return ModelContract(
            schema=SCHEMA_VERSION,
            required_reasoning="heavy_tool_workflow",
            risk_class="high",
            evidence_strength=strength,
            latency_class_required="heavy_visible_progress",
            model_profile="reasoning_heavy",
            fast_model_allowed=False,
            fast_model_requires_parity=True,
            reason_code="EXTERNAL_TOOL_WORKFLOW_REQUIRED",
            forbidden_without_parity=("fast_model_default", "silent_heavy_work"),
        )
    if simple_memory:
        return ModelContract(
            schema=SCHEMA_VERSION,
            required_reasoning="faithful_rendering",
            risk_class="low",
            evidence_strength=strength,
            latency_class_required="interactive",
            model_profile="conversation_renderer",
            fast_model_allowed=parity_proven,
            fast_model_requires_parity=True,
            reason_code="SIMPLE_MEMORY_RENDERING_PARITY_REQUIRED",
            forbidden_without_parity=("fast_model_default",),
        )
    return ModelContract(
        schema=SCHEMA_VERSION,
        required_reasoning="multi_evidence_synthesis",
        risk_class="medium",
        evidence_strength=strength,
        latency_class_required="quality_first",
        model_profile="reasoning_heavy",
        fast_model_allowed=False,
        fast_model_requires_parity=True,
        reason_code="SYNTHESIS_OR_AMBIGUITY_REQUIRED",
        forbidden_without_parity=("fast_model_default", "memory_direct_renderer"),
    )


def evaluate_fast_model_parity(
    *,
    same_answerability: bool,
    same_answer_evidence: bool,
    forbidden_claims: Sequence[str],
    unsupported_hallucination: bool,
    assignment_promotion: bool,
    exact_literal_missing: bool,
) -> tuple[bool, str]:
    if not same_answerability:
        return False, "ANSWERABILITY_MISMATCH"
    if not same_answer_evidence:
        return False, "ANSWER_EVIDENCE_MISMATCH"
    if forbidden_claims:
        return False, "FORBIDDEN_CLAIM"
    if unsupported_hallucination:
        return False, "UNSUPPORTED_HALLUCINATION"
    if assignment_promotion:
        return False, "ASSIGNMENT_PROMOTION"
    if exact_literal_missing:
        return False, "EXACT_LITERAL_MISSING"
    return True, "PARITY_OK"


def build_degradation_decision(
    *,
    timing: ProviderTimingTrace,
    model_contract: ModelContract,
    first_user_visible_commitment_ms: int | None,
    turn_profile: str,
) -> DegradationDecision:
    limit = HEAVY_PROGRESS_VISIBLE_LIMIT_MS if turn_profile.startswith("heavy") else INTERACTIVE_FIRST_VISIBLE_LIMIT_MS
    no_silent_wait = (
        first_user_visible_commitment_ms is not None and first_user_visible_commitment_ms <= limit
    )
    activated = not no_silent_wait or timing.final_ms > limit
    if not no_silent_wait:
        reason = "FIRST_VISIBLE_COMMITMENT_SLO_MISSED"
        policy = "send_visible_status_or_degraded_response"
        commitment_type = "missing"
    elif timing.final_ms > limit and turn_profile.startswith("heavy"):
        reason = "HEAVY_PROGRESS_REQUIRED"
        policy = "visible_progress_then_continue"
        commitment_type = "progress_notice"
    elif timing.final_ms > limit:
        reason = "PROVIDER_LATENCY_SLO_RISK"
        policy = "fail_visible_or_use_parity_proven_fallback"
        commitment_type = "final_or_status"
    else:
        reason = "NO_DEGRADATION"
        policy = "normal_final_response"
        commitment_type = "final_answer"
    return DegradationDecision(
        schema=SCHEMA_VERSION,
        activated=activated,
        reason_code=reason,
        policy=policy,
        first_user_visible_commitment_ms=first_user_visible_commitment_ms,
        commitment_type=commitment_type,
        no_silent_wait_satisfied=no_silent_wait,
        progress_visible_required=turn_profile.startswith("heavy") and timing.final_ms > HEAVY_PROGRESS_VISIBLE_LIMIT_MS,
        fallback_selected="none",
        why_not_fast_model=None if model_contract.fast_model_allowed else model_contract.reason_code,
        why_not_direct_renderer=(
            None
            if model_contract.required_reasoning == "faithful_rendering"
            else model_contract.reason_code
        ),
    )


def build_readiness_report(
    *,
    chat_provider_health: str,
    pulse_provider_health: str,
    memory_health: str,
    corpus_semantic_health: str,
    heavy_tool_health: str,
    cold_backend_live_download: bool,
) -> ReadinessReport:
    reason_codes: list[str] = []
    chat_ready = chat_provider_health == "ok" and not cold_backend_live_download
    memory_ready = memory_health == "ok" and not cold_backend_live_download
    corpus_ready = corpus_semantic_health == "ok" and not cold_backend_live_download
    heavy_ready = chat_ready and heavy_tool_health == "ok"
    if pulse_provider_health != "ok":
        reason_codes.append("PULSE_PROVIDER_DEGRADED_NONBLOCKING_FOR_CHAT")
    if cold_backend_live_download:
        reason_codes.append("COLD_BACKEND_LIVE_DOWNLOAD_BLOCKS_READINESS")
    if chat_provider_health != "ok":
        reason_codes.append("CHAT_PROVIDER_NOT_READY")
    return ReadinessReport(
        schema=SCHEMA_VERSION,
        chat_ready=chat_ready,
        memory_ready=memory_ready,
        heavy_ready=heavy_ready,
        corpus_semantic_ready=corpus_ready,
        chat_provider_health=chat_provider_health,
        pulse_provider_health=pulse_provider_health,
        cold_backend_live_download_blocked=cold_backend_live_download,
        provider_health_isolated=True,
        reason_codes=tuple(reason_codes),
    )


def build_stream_key_hash(*, platform: str, channel_id: str, user_id: str) -> str:
    return _hash_parts([platform, channel_id, user_id])


def build_stale_response_decision(
    *,
    platform: str,
    channel_id: str,
    user_id: str,
    turn_id: str,
    causal_index: int,
    latest_completed_index: int,
    superseded_by: str | None = None,
) -> StaleResponseDecision:
    stale = causal_index < latest_completed_index
    return StaleResponseDecision(
        schema=SCHEMA_VERSION,
        stream_key_hash=build_stream_key_hash(platform=platform, channel_id=channel_id, user_id=user_id),
        turn_id=turn_id,
        causal_index=causal_index,
        latest_completed_index=latest_completed_index,
        superseded_by=superseded_by if stale else None,
        stale_response_suppressed=stale,
        policy="suppress_late_final_response" if stale else "deliver_if_contract_current",
    )


def build_idempotency_key(
    *,
    platform: str,
    guild_id: str | None,
    channel_id: str,
    message_id: str | None,
    author_id: str,
    interaction_id: str | None = None,
) -> str:
    return _hash_parts(
        [
            platform,
            guild_id or "",
            channel_id,
            message_id or "",
            author_id,
            interaction_id or "",
        ]
    )


def build_idempotency_decision(
    *,
    idempotency_key_hash: str,
    seen_keys: set[str],
) -> IdempotencyDecision:
    duplicate = idempotency_key_hash in seen_keys
    return IdempotencyDecision(
        schema=SCHEMA_VERSION,
        idempotency_key_hash=idempotency_key_hash,
        duplicate_event=duplicate,
        duplicate_suppressed=duplicate,
        duplicate_provider_call_suppressed=duplicate,
        duplicate_reply_suppressed=duplicate,
        duplicate_memory_write_suppressed=duplicate,
        duplicate_event_index_suppressed=duplicate,
    )


def build_aux_barrier(
    *,
    raw_turn_committed_before_response: bool,
    explicit_write_status: str,
    trace_id_committed: bool,
    post_response_tasks_pending: Sequence[str],
    aux_blocking_ms: int,
) -> AuxBarrier:
    answer_critical_ok = raw_turn_committed_before_response and trace_id_committed
    if explicit_write_status in {"pending", "diverged"}:
        policy = "next_turn_wait_300ms_or_degrade_pending_write_barrier"
    elif answer_critical_ok:
        policy = "post_response_aux_nonblocking"
    else:
        policy = "answer_critical_commit_missing_fail_visible"
    return AuxBarrier(
        schema=SCHEMA_VERSION,
        raw_turn_committed_before_response=raw_turn_committed_before_response,
        explicit_write_status=explicit_write_status,
        trace_id_committed=trace_id_committed,
        post_response_tasks_pending=tuple(sorted(post_response_tasks_pending)),
        aux_blocking_ms=aux_blocking_ms,
        aux_blocking_reason="ANSWER_CRITICAL" if aux_blocking_ms and not answer_critical_ok else "NONE",
        next_turn_barrier_policy=policy,
    )


def build_discord_interaction_trace(
    *,
    is_interaction: bool,
    interaction_ack_ms: int | None = None,
    ack_type: str | None = None,
    followup_used: bool = False,
) -> DiscordInteractionTrace:
    return DiscordInteractionTrace(
        schema=SCHEMA_VERSION,
        is_interaction=is_interaction,
        interaction_ack_ms=interaction_ack_ms,
        ack_type=ack_type,
        followup_used=followup_used,
        ack_slo_satisfied=(interaction_ack_ms <= 3000) if is_interaction and interaction_ack_ms is not None else (None if is_interaction else True),
    )


def build_rate_limit_trace(
    *,
    messages_created: int,
    messages_edited: int,
    retry_after_ms: int = 0,
    rate_limit_wait_ms: int = 0,
) -> RateLimitTrace:
    return RateLimitTrace(
        schema=SCHEMA_VERSION,
        messages_created=messages_created,
        messages_edited=messages_edited,
        retry_after_ms=retry_after_ms,
        rate_limit_wait_ms=rate_limit_wait_ms,
        progress_policy="edit_single_progress_message_no_heartbeat_spam",
    )
