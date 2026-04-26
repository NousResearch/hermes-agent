"""Observe-only context budget compiler for Hermes gateway prompts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "hermes.context_budget.v1"

SECRET_SHAPED_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{12,}|api[_-]?key|token|secret|password|bearer\s+[a-z0-9._-]{12,})"
)
PRIVATE_PATH_RE = re.compile(r"(?<![\w.-])/(?:home|Users)/[^\s'\"`]+")

PROTECTED_CATEGORIES = {
    "answerability",
    "answer_evidence",
    "forbidden_claims",
    "conflict_degraded_status",
    "current_assignment_authority_status",
    "current_user_message",
}

CATEGORY_PRIORITY = {
    "answerability": 0,
    "answer_evidence": 1,
    "forbidden_claims": 2,
    "conflict_degraded_status": 3,
    "current_assignment_authority_status": 4,
    "current_user_message": 5,
    "tool_schema": 6,
    "memory_packet": 7,
    "supporting_context": 8,
    "recent_history": 9,
    "diagnostics": 10,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def estimate_text_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_json_tokens(value: Any) -> int:
    return estimate_text_tokens(_canonical_json(value))


def redact_model_facing_text(text: str) -> str:
    text = SECRET_SHAPED_RE.sub("[REDACTED_SECRET_SHAPED]", text)
    return PRIVATE_PATH_RE.sub("[REDACTED_PRIVATE_PATH]", text)


@dataclass(frozen=True)
class ContextBudget:
    total_input_budget_tokens: int
    static_system_budget_tokens: int
    tool_schema_budget_tokens: int
    memory_packet_budget_tokens: int
    recent_history_budget_tokens: int
    safety_margin_tokens: int
    overflow_policy: tuple[str, ...] = (
        "drop_low_authority_supporting_context",
        "replace_verbose_diagnostics_with_inspect_ref",
        "compress_recent_history",
        "minimum_viable_context",
        "fail_or_degrade_visible_if_required_evidence_cannot_fit",
    )

    @property
    def dynamic_budget_tokens(self) -> int:
        reserved = self.static_system_budget_tokens + self.tool_schema_budget_tokens + self.safety_margin_tokens
        return max(0, self.total_input_budget_tokens - reserved)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["overflow_policy"] = list(self.overflow_policy)
        data["dynamic_budget_tokens"] = self.dynamic_budget_tokens
        return data


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    category: str
    text: str
    tokens_est: int
    protected: bool
    priority: int
    source: str = "runtime"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResponseEnvelope:
    tool_name: str
    model_facing: dict[str, Any]
    inspect_ref: str
    truncated: bool
    continuation_token: str | None
    cap_tokens: int
    tokens_est: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromptAssembly:
    schema: str
    profile_name: str
    static_prefix_hash: str
    dynamic_suffix_hash: str
    total_tokens_est: int
    static_tokens_est: int
    dynamic_tokens_est: int
    selected_items: tuple[ContextItem, ...]
    dropped_items: tuple[ContextItem, ...]
    minimum_viable_context_used: bool
    overflow: bool
    overflow_reason: str | None
    provider_cache_observable: bool
    provider_reported_cached_tokens: int | None
    provider_cache_hit: bool | None
    cache_candidate: bool
    cache_break_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["selected_items"] = [item.to_dict() for item in self.selected_items]
        data["dropped_items"] = [item.to_dict() for item in self.dropped_items]
        return data


def make_context_item(
    item_id: str,
    category: str,
    text: str,
    *,
    source: str = "runtime",
    tokens_est: int | None = None,
) -> ContextItem:
    redacted = redact_model_facing_text(text)
    protected = category in PROTECTED_CATEGORIES
    return ContextItem(
        item_id=item_id,
        category=category,
        text=redacted,
        tokens_est=tokens_est if tokens_est is not None else estimate_text_tokens(redacted),
        protected=protected,
        priority=CATEGORY_PRIORITY.get(category, 50),
        source=source,
    )


def build_policy_static_prefix(
    *,
    profile_name: str,
    tool_profile: str,
    max_claim_strength: str,
    latency_mode: str,
    forbidden_claims: Sequence[str],
) -> str:
    policy = {
        "schema": "hermes.policy_variables.v1",
        "profile_name": profile_name,
        "tool_profile": tool_profile,
        "max_claim_strength": max_claim_strength,
        "latency_mode": latency_mode,
        "forbidden_claims": sorted(set(forbidden_claims)),
        "rule": "Use memory answerability and answer evidence as upper bound for memory claims.",
    }
    return "Hermes gateway base prompt\n" + _canonical_json(policy)


def compact_tool_response(
    tool_name: str,
    payload: Mapping[str, Any],
    *,
    cap_tokens: int,
    inspect_ref: str,
    continuation_token: str | None = None,
) -> ToolResponseEnvelope:
    model_facing: dict[str, Any] = {
        "tool_name": tool_name,
        "answerability": payload.get("answerability") or payload.get("memory_answerability"),
        "answer_evidence": payload.get("answer_evidence", []),
        "supporting_context": payload.get("supporting_context", []),
        "forbidden_claims": payload.get("forbidden_claims", []),
        "inspect_ref": inspect_ref,
    }
    raw_tokens = estimate_json_tokens(model_facing)
    truncated = raw_tokens > cap_tokens
    if truncated:
        model_facing["supporting_context"] = []
        model_facing["truncated"] = True
        model_facing["continuation_token"] = continuation_token
    else:
        model_facing["truncated"] = False
        model_facing["continuation_token"] = continuation_token
    model_facing = json.loads(redact_model_facing_text(_canonical_json(model_facing)))
    return ToolResponseEnvelope(
        tool_name=tool_name,
        model_facing=model_facing,
        inspect_ref=inspect_ref,
        truncated=truncated,
        continuation_token=continuation_token,
        cap_tokens=cap_tokens,
        tokens_est=min(raw_tokens, estimate_json_tokens(model_facing)),
    )


def compile_context_budget(
    *,
    profile_name: str,
    budget: ContextBudget,
    static_prefix: str,
    tool_schema_tokens_est: int,
    items: Sequence[ContextItem],
    provider_cache_observable: bool = False,
    provider_reported_cached_tokens: int | None = None,
    provider_cache_hit: bool | None = None,
    previous_static_prefix_hash: str | None = None,
) -> PromptAssembly:
    static_tokens = estimate_text_tokens(static_prefix)
    dynamic_budget = budget.total_input_budget_tokens - budget.safety_margin_tokens - static_tokens - tool_schema_tokens_est
    dynamic_budget = max(0, dynamic_budget)
    sorted_items = sorted(items, key=lambda item: (item.priority, item.item_id))

    selected: list[ContextItem] = []
    dropped: list[ContextItem] = []
    used = 0
    protected_overflow = False
    for item in sorted_items:
        if item.protected:
            selected.append(item)
            used += item.tokens_est
            if used > dynamic_budget:
                protected_overflow = True
            continue
        if used + item.tokens_est <= dynamic_budget:
            selected.append(item)
            used += item.tokens_est
        else:
            dropped.append(item)

    minimum_viable = False
    overflow_reason = None
    if protected_overflow:
        minimum_viable = True
        selected = [item for item in selected if item.protected]
        dropped = [item for item in sorted_items if not item.protected]
        overflow_reason = "PROTECTED_CONTEXT_EXCEEDS_DYNAMIC_BUDGET"
    elif dropped:
        overflow_reason = "DROPPED_LOW_PRIORITY_CONTEXT"

    dynamic_suffix = "\n".join(item.text for item in selected)
    static_hash = _hash_text(static_prefix)
    cache_break_reason = None
    if previous_static_prefix_hash and previous_static_prefix_hash != static_hash:
        cache_break_reason = "STATIC_PREFIX_HASH_CHANGED_REQUIRES_PROFILE_VERSION_BUMP"

    return PromptAssembly(
        schema=SCHEMA_VERSION,
        profile_name=profile_name,
        static_prefix_hash=static_hash,
        dynamic_suffix_hash=_hash_text(dynamic_suffix),
        total_tokens_est=static_tokens + tool_schema_tokens_est + used + budget.safety_margin_tokens,
        static_tokens_est=static_tokens,
        dynamic_tokens_est=used,
        selected_items=tuple(selected),
        dropped_items=tuple(dropped),
        minimum_viable_context_used=minimum_viable,
        overflow=bool(dropped or protected_overflow),
        overflow_reason=overflow_reason,
        provider_cache_observable=provider_cache_observable,
        provider_reported_cached_tokens=provider_reported_cached_tokens,
        provider_cache_hit=provider_cache_hit,
        cache_candidate=True,
        cache_break_reason=cache_break_reason,
    )


def default_budget_for_profile(profile_name: str, tool_schema_tokens_est: int) -> ContextBudget:
    if profile_name == "conversation_direct":
        return ContextBudget(
            total_input_budget_tokens=5000,
            static_system_budget_tokens=1800,
            tool_schema_budget_tokens=0,
            memory_packet_budget_tokens=900,
            recent_history_budget_tokens=1200,
            safety_margin_tokens=500,
        )
    if profile_name == "conversation_tools":
        return ContextBudget(
            total_input_budget_tokens=6500,
            static_system_budget_tokens=2000,
            tool_schema_budget_tokens=min(tool_schema_tokens_est, 1500),
            memory_packet_budget_tokens=1000,
            recent_history_budget_tokens=1400,
            safety_margin_tokens=600,
        )
    return ContextBudget(
        total_input_budget_tokens=18000,
        static_system_budget_tokens=2500,
        tool_schema_budget_tokens=tool_schema_tokens_est,
        memory_packet_budget_tokens=1600,
        recent_history_budget_tokens=3500,
        safety_margin_tokens=1000,
    )


def build_budget_proof(
    *,
    profile_name: str,
    tool_schema_tokens_est: int,
    answerability: Mapping[str, Any],
    answer_evidence: Sequence[Mapping[str, Any]],
    forbidden_claims: Sequence[str],
    current_user_message: str,
    supporting_context: Sequence[str] = (),
    recent_history: Sequence[str] = (),
    diagnostics: Sequence[str] = (),
) -> dict[str, Any]:
    budget = default_budget_for_profile(profile_name, tool_schema_tokens_est)
    static_prefix = build_policy_static_prefix(
        profile_name=profile_name,
        tool_profile=profile_name,
        max_claim_strength=str(answerability.get("max_claim_strength", "none")),
        latency_mode="conversation_warm" if profile_name.startswith("conversation") else "heavy_visible_progress",
        forbidden_claims=forbidden_claims,
    )
    items: list[ContextItem] = [
        make_context_item("answerability", "answerability", _canonical_json(answerability)),
        make_context_item("forbidden_claims", "forbidden_claims", _canonical_json(sorted(set(forbidden_claims)))),
        make_context_item("current_user_message", "current_user_message", current_user_message),
    ]
    for idx, evidence in enumerate(answer_evidence):
        items.append(make_context_item(f"answer_evidence:{idx}", "answer_evidence", _canonical_json(evidence)))
    for idx, text in enumerate(supporting_context):
        items.append(make_context_item(f"supporting:{idx}", "supporting_context", text))
    for idx, text in enumerate(recent_history):
        items.append(make_context_item(f"history:{idx}", "recent_history", text))
    for idx, text in enumerate(diagnostics):
        items.append(make_context_item(f"diagnostic:{idx}", "diagnostics", text))

    assembly = compile_context_budget(
        profile_name=profile_name,
        budget=budget,
        static_prefix=static_prefix,
        tool_schema_tokens_est=tool_schema_tokens_est,
        items=items,
        provider_cache_observable=False,
    )
    selected_ids = {item.item_id for item in assembly.selected_items}
    protected_ids = {item.item_id for item in items if item.protected}
    return {
        "schema": SCHEMA_VERSION,
        "profile_name": profile_name,
        "budget": budget.to_dict(),
        "assembly": assembly.to_dict(),
        "protected_ids": sorted(protected_ids),
        "protected_preserved": protected_ids.issubset(selected_ids),
        "tool_schema_budget_zero": tool_schema_tokens_est == 0,
        "provider_cache_observable": False,
    }
