"""Observe-only turn contract tracing for gateway runtime decisions.

Phase 150 intentionally does not route tools, models, or execution. It records
the contract the runtime would need before later phases are allowed to change
Discord defaults.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping, Sequence


class TurnReasonCode(StrEnum):
    MEMORY_SUFFICIENT_NO_EXTERNAL_TOOLS = "MEMORY_SUFFICIENT_NO_EXTERNAL_TOOLS"
    OBSERVE_NO_TOOLS = "OBSERVE_NO_TOOLS"
    OBSERVE_COMPACT_TOOLS = "OBSERVE_COMPACT_TOOLS"
    OBSERVE_FULL_TOOLS = "OBSERVE_FULL_TOOLS"
    STRUCTURAL_EXTERNAL_CANDIDATE = "STRUCTURAL_EXTERNAL_CANDIDATE"
    HEAVY_COMMAND = "HEAVY_COMMAND"
    NO_SUPPORTED_MEMORY_TRUTH = "NO_SUPPORTED_MEMORY_TRUTH"
    UNKNOWN_OBSERVE_ONLY = "UNKNOWN_OBSERVE_ONLY"


class ClaimStrength(StrEnum):
    MEMORY_TRUTH = "memory_truth"
    BOUNDED_EVENT = "bounded_event"
    SOURCE_SAYS = "source_says"
    SUPPORTING_CONTEXT = "supporting_context"
    NONE = "none"


@dataclass(frozen=True)
class BrainstackSufficiency:
    state: str = "unknown"
    max_claim_strength: str = ClaimStrength.NONE.value
    answer_type: str = "none"
    answer_evidence_count: int = 0
    supporting_context_count: int = 0
    requires_external_tools_for_memory_answer: bool = False
    reason_code: str = "UNKNOWN"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "BrainstackSufficiency":
        if not value:
            return cls()
        return cls(
            state=str(value.get("state") or "unknown"),
            max_claim_strength=str(value.get("max_claim_strength") or ClaimStrength.NONE.value),
            answer_type=str(value.get("answer_type") or "none"),
            answer_evidence_count=_safe_int(value.get("answer_evidence_count")),
            supporting_context_count=_safe_int(value.get("supporting_context_count")),
            requires_external_tools_for_memory_answer=bool(
                value.get("requires_external_tools_for_memory_answer", False)
            ),
            reason_code=str(value.get("reason_code") or "UNKNOWN"),
        )


@dataclass(frozen=True)
class StructuralSignals:
    slash_command: str | None = None
    attachment_count: int = 0
    url_count: int = 0
    pending_approval: bool = False
    active_workflow: bool = False
    explicit_heavy: bool = False

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        attachment_count: int = 0,
        pending_approval: bool = False,
        active_workflow: bool = False,
    ) -> "StructuralSignals":
        stripped = (prompt or "").strip()
        slash_command = None
        if stripped.startswith("/"):
            slash_command = stripped.split(maxsplit=1)[0].casefold()
        url_count = len(re.findall(r"https?://\S+", prompt or ""))
        explicit_heavy = bool(slash_command and slash_command.startswith("/heavy"))
        return cls(
            slash_command=slash_command,
            attachment_count=max(0, int(attachment_count or 0)),
            url_count=url_count,
            pending_approval=bool(pending_approval),
            active_workflow=bool(active_workflow),
            explicit_heavy=explicit_heavy,
        )


@dataclass(frozen=True)
class TurnContract:
    schema: str
    turn_contract_id: str
    observe_only: bool
    platform: str
    turn_class: str
    evidence_sufficiency: str
    max_claim_strength: str
    external_capability_required_for_memory_answer: bool
    allowed_tool_profile: str
    allowed_model_profile: str
    latency_slo: str
    context_budget_profile: str
    degradation_policy: str
    forbidden_claims: tuple[str, ...]
    reason_code: str
    structural_signals: dict[str, Any]
    brainstack_sufficiency: dict[str, Any]
    idempotency_key_hash: str
    causal_index: int | None = None
    superseded_by: str | None = None
    stale_response_suppressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimingTrace:
    request_build_ms: float | None = None
    http_connect_ms: float | None = None
    provider_queue_or_time_to_first_byte_ms: float | None = None
    first_token_ms: float | None = None
    generation_after_first_token_ms: float | None = None
    provider_final_ms: float | None = None
    final_latency_ms: float | None = None
    first_user_visible_commitment_ms: float | None = None
    output_tokens_est: int | None = None
    tokens_per_second_after_first: float | None = None
    observable: bool = False
    unobservable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObserverEffect:
    extra_provider_calls: int = 0
    extra_brainstack_recalls: int = 0
    extra_memory_writes: int = 0
    trace_write_ms: float = 0.0
    total_observer_overhead_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HiddenToolPromptAudit:
    tools_array_empty: bool
    tool_count: int
    tool_names: tuple[str, ...] = field(default_factory=tuple)
    tool_use_system_prompt_present: bool = False
    tool_catalog_prose_tokens_est: int = 0
    hidden_tool_prompt_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_turn_contract(
    *,
    platform: str,
    prompt: str,
    tool_profile: str,
    model_profile: str | None = None,
    brainstack_sufficiency: Mapping[str, Any] | None = None,
    structural_signals: StructuralSignals | None = None,
    idempotency_parts: Sequence[str] = (),
    causal_index: int | None = None,
) -> TurnContract:
    """Build observe-only contract. Does not change runtime behavior."""
    suff = BrainstackSufficiency.from_mapping(brainstack_sufficiency)
    signals = structural_signals or StructuralSignals.from_prompt(prompt)
    reason_code = _reason_code(tool_profile, signals, suff)
    turn_class = _turn_class(signals, suff)
    allowed_tool_profile = _allowed_tool_profile(tool_profile)
    forbidden_claims = _forbidden_claims(suff)
    id_key = _hash_idempotency_parts(idempotency_parts or (platform, prompt[:160]))
    contract_id = _stable_contract_id(platform, prompt, tool_profile, id_key)
    return TurnContract(
        schema="hermes.turn_contract.v1",
        turn_contract_id=contract_id,
        observe_only=True,
        platform=platform,
        turn_class=turn_class,
        evidence_sufficiency=suff.state,
        max_claim_strength=suff.max_claim_strength,
        external_capability_required_for_memory_answer=(
            suff.requires_external_tools_for_memory_answer
        ),
        allowed_tool_profile=allowed_tool_profile,
        allowed_model_profile=model_profile or "current_config",
        latency_slo="observe_only",
        context_budget_profile="observe_only",
        degradation_policy="observe_only_no_behavior_change",
        forbidden_claims=forbidden_claims,
        reason_code=reason_code.value,
        structural_signals=asdict(signals),
        brainstack_sufficiency=asdict(suff),
        idempotency_key_hash=id_key,
        causal_index=causal_index,
    )


def build_hidden_tool_prompt_audit(
    *,
    tool_profile: str,
    tool_names: Sequence[str] | None,
    system_prompt: str = "",
) -> HiddenToolPromptAudit:
    names = tuple(sorted({str(name) for name in (tool_names or []) if str(name)}))
    prompt_norm = " ".join((system_prompt or "").casefold().split())
    tool_catalog_tokens = 0
    tool_prompt_present = False
    if prompt_norm:
        tool_prompt_present = any(
            marker in prompt_norm
            for marker in ("you can use tools", "available tools", "tool calls")
        )
        tool_catalog_tokens = len(prompt_norm.split()) if tool_prompt_present else 0
    hidden = tool_profile == "no-tools" and (bool(names) or tool_prompt_present)
    return HiddenToolPromptAudit(
        tools_array_empty=not names,
        tool_count=len(names),
        tool_names=names,
        tool_use_system_prompt_present=tool_prompt_present,
        tool_catalog_prose_tokens_est=tool_catalog_tokens,
        hidden_tool_prompt_detected=hidden,
    )


def build_timing_trace(
    *,
    start_monotonic: float,
    final_monotonic: float | None = None,
    first_token_monotonic: float | None = None,
    output_tokens_est: int | None = None,
) -> TimingTrace:
    final = final_monotonic if final_monotonic is not None else time.monotonic()
    final_ms = max(0.0, (final - start_monotonic) * 1000.0)
    first_ms = None
    generation_ms = None
    tokens_per_second = None
    if first_token_monotonic is not None:
        first_ms = max(0.0, (first_token_monotonic - start_monotonic) * 1000.0)
        generation_ms = max(0.0, final_ms - first_ms)
        if output_tokens_est and generation_ms > 0:
            tokens_per_second = output_tokens_est / (generation_ms / 1000.0)
    return TimingTrace(
        provider_queue_or_time_to_first_byte_ms=first_ms,
        first_token_ms=first_ms,
        generation_after_first_token_ms=generation_ms,
        provider_final_ms=final_ms,
        final_latency_ms=final_ms,
        first_user_visible_commitment_ms=final_ms,
        output_tokens_est=output_tokens_est,
        tokens_per_second_after_first=tokens_per_second,
        observable=first_token_monotonic is not None,
        unobservable_reason=None
        if first_token_monotonic is not None
        else "gateway_dispatch_only_no_provider_ttfb_hook",
    )


def build_observer_effect(
    *,
    started_monotonic: float,
    finished_monotonic: float | None = None,
    extra_provider_calls: int = 0,
    extra_brainstack_recalls: int = 0,
    extra_memory_writes: int = 0,
) -> ObserverEffect:
    finished = finished_monotonic if finished_monotonic is not None else time.monotonic()
    overhead_ms = max(0.0, (finished - started_monotonic) * 1000.0)
    return ObserverEffect(
        extra_provider_calls=max(0, int(extra_provider_calls)),
        extra_brainstack_recalls=max(0, int(extra_brainstack_recalls)),
        extra_memory_writes=max(0, int(extra_memory_writes)),
        trace_write_ms=0.0,
        total_observer_overhead_ms=overhead_ms,
    )


def sample_size_latency_summary(latencies_seconds: Sequence[float]) -> dict[str, Any]:
    values = sorted(float(v) for v in latencies_seconds)
    count = len(values)
    if not values:
        return {
            "sample_size": 0,
            "max_latency_seconds": 0.0,
            "p95_seconds": None,
            "p95_claim_allowed": False,
            "small_sample_rule": "p95 requires at least 30 warm turns",
        }
    return {
        "sample_size": count,
        "max_latency_seconds": values[-1],
        "p50_seconds": _percentile(values, 50),
        "p90_seconds": _percentile(values, 90),
        "p95_seconds": _percentile(values, 95) if count >= 30 else None,
        "p95_claim_allowed": count >= 30,
        "small_sample_rule": None
        if count >= 30
        else "p95 requires at least 30 warm turns",
    }


def _reason_code(
    tool_profile: str,
    signals: StructuralSignals,
    suff: BrainstackSufficiency,
) -> TurnReasonCode:
    if signals.explicit_heavy:
        return TurnReasonCode.HEAVY_COMMAND
    if signals.attachment_count or signals.url_count or signals.pending_approval or signals.active_workflow:
        return TurnReasonCode.STRUCTURAL_EXTERNAL_CANDIDATE
    if suff.state == "answerable" and not suff.requires_external_tools_for_memory_answer:
        return TurnReasonCode.MEMORY_SUFFICIENT_NO_EXTERNAL_TOOLS
    if suff.state in {"unanswerable", "none"}:
        return TurnReasonCode.NO_SUPPORTED_MEMORY_TRUTH
    if tool_profile == "no-tools":
        return TurnReasonCode.OBSERVE_NO_TOOLS
    if tool_profile in {"memory-only", "compact"}:
        return TurnReasonCode.OBSERVE_COMPACT_TOOLS
    if tool_profile == "full":
        return TurnReasonCode.OBSERVE_FULL_TOOLS
    return TurnReasonCode.UNKNOWN_OBSERVE_ONLY


def _turn_class(signals: StructuralSignals, suff: BrainstackSufficiency) -> str:
    if signals.explicit_heavy:
        return "external.heavy_requested"
    if signals.attachment_count or signals.url_count:
        return "external.capability_candidate"
    if suff.answer_type and suff.answer_type != "none":
        return f"conversation.{suff.answer_type}"
    return "conversation.observe_only"


def _allowed_tool_profile(tool_profile: str) -> str:
    if tool_profile == "no-tools":
        return "none"
    if tool_profile in {"memory-only", "compact"}:
        return "conversation_tools"
    if tool_profile == "full":
        return "heavy_work_observed"
    return f"observed:{tool_profile}"


def _forbidden_claims(suff: BrainstackSufficiency) -> tuple[str, ...]:
    claims = ["external_verification_without_tool"]
    if suff.max_claim_strength != ClaimStrength.MEMORY_TRUTH.value:
        claims.append("memory_truth")
    if suff.answer_type != "current_assignment":
        claims.append("current_assignment")
    return tuple(claims)


def _hash_idempotency_parts(parts: Sequence[str]) -> str:
    payload = "\\x1f".join(str(part) for part in parts)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_contract_id(platform: str, prompt: str, tool_profile: str, id_key: str) -> str:
    payload = "\\x1f".join((platform, prompt[:240], tool_profile, id_key))
    return "tc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _percentile(values: Sequence[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight
