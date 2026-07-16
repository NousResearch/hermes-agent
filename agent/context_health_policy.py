"""Pure policy model for future Context Health governance.

Phase 1 intentionally has no runtime integration.  The functions in this
module only normalize policy data and make deterministic policy decisions that
future pre-turn intake, task-boundary, and provider-payload hooks can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


DEFAULT_LONG_PROMPT_CHAR_THRESHOLD = 8_000
DEFAULT_LONG_PROMPT_LINE_THRESHOLD = 80
DEFAULT_MAX_PROVIDER_CONTEXT_RATIO = 0.75
DEFAULT_BLOCK_PROVIDER_CALL_RATIO = 0.85


@dataclass(frozen=True)
class PreModelIntakePolicy:
    enabled: bool = False
    long_prompt_char_threshold: int = DEFAULT_LONG_PROMPT_CHAR_THRESHOLD
    long_prompt_line_threshold: int = DEFAULT_LONG_PROMPT_LINE_THRESHOLD
    pre_history_required: bool = True
    high_risk_keywords: tuple[str, ...] = (
        "credential",
        "credentials",
        "secret",
        "token",
        "api key",
        "password",
        "private key",
        "connection string",
    )
    sensitive_prompt_action: str = "hold"


@dataclass(frozen=True)
class TaskBoundaryPolicy:
    enabled: bool = False
    default_without_clear_continuation: str = "new_task"
    ambiguous_action: str = "hold"


@dataclass(frozen=True)
class ContextThresholdPolicy:
    max_provider_context_ratio: float = DEFAULT_MAX_PROVIDER_CONTEXT_RATIO
    block_provider_call_ratio: float = DEFAULT_BLOCK_PROVIDER_CALL_RATIO
    allow_model_specific_raise: bool = False


@dataclass(frozen=True)
class ContextHealthPolicy:
    enabled: bool = False
    pre_model_intake: PreModelIntakePolicy = field(default_factory=PreModelIntakePolicy)
    task_boundary: TaskBoundaryPolicy = field(default_factory=TaskBoundaryPolicy)
    thresholds: ContextThresholdPolicy = field(default_factory=ContextThresholdPolicy)
    runtime_behavior_enabled: bool = False


@dataclass(frozen=True)
class PromptIntakeDecision:
    action: str
    reason: str
    pre_history_required: bool = False
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskBoundaryDecision:
    action: str
    reason: str
    defaulted: bool = False
    linked_task_id: str | None = None


@dataclass(frozen=True)
class ThresholdDecision:
    ratio: float
    reason: str
    clamped_model_suggestion: bool = False


def default_context_health_policy() -> ContextHealthPolicy:
    """Return the inert default policy.

    Phase 1 must not alter runtime behavior, so the default policy is disabled
    and `runtime_behavior_enabled` is always false unless a caller explicitly
    opts in through normalized policy data in a future phase.
    """

    return ContextHealthPolicy()


def normalize_context_health_policy(data: Mapping[str, Any] | None) -> ContextHealthPolicy:
    """Normalize a mapping into a typed context health policy.

    Unknown keys are ignored deliberately so future config files can be more
    expressive without breaking this pure Phase 1 model.
    """

    if not data:
        return default_context_health_policy()

    pre_model_raw = _mapping(data.get("pre_model_intake"))
    task_boundary_raw = _mapping(data.get("task_boundary"))
    thresholds_raw = _mapping(data.get("thresholds"))

    enabled = _bool(data.get("enabled"), False)

    pre_model = PreModelIntakePolicy(
        enabled=_bool(pre_model_raw.get("enabled"), False),
        long_prompt_char_threshold=_positive_int(
            pre_model_raw.get("long_prompt_char_threshold"),
            DEFAULT_LONG_PROMPT_CHAR_THRESHOLD,
        ),
        long_prompt_line_threshold=_positive_int(
            pre_model_raw.get("long_prompt_line_threshold"),
            DEFAULT_LONG_PROMPT_LINE_THRESHOLD,
        ),
        pre_history_required=_bool(pre_model_raw.get("pre_history_required"), True),
        high_risk_keywords=_string_tuple(
            pre_model_raw.get("high_risk_keywords"),
            PreModelIntakePolicy().high_risk_keywords,
        ),
        sensitive_prompt_action=_choice(
            pre_model_raw.get("sensitive_prompt_action"),
            {"hold", "force_md_intake"},
            "hold",
        ),
    )

    task_boundary = TaskBoundaryPolicy(
        enabled=_bool(task_boundary_raw.get("enabled"), False),
        default_without_clear_continuation=_choice(
            task_boundary_raw.get("default_without_clear_continuation"),
            {"new_task", "continue_task", "hold"},
            "new_task",
        ),
        ambiguous_action=_choice(
            task_boundary_raw.get("ambiguous_action"),
            {"hold", "new_task"},
            "hold",
        ),
    )

    thresholds = ContextThresholdPolicy(
        max_provider_context_ratio=_ratio(
            thresholds_raw.get("max_provider_context_ratio"),
            DEFAULT_MAX_PROVIDER_CONTEXT_RATIO,
        ),
        block_provider_call_ratio=_ratio(
            thresholds_raw.get("block_provider_call_ratio"),
            DEFAULT_BLOCK_PROVIDER_CALL_RATIO,
        ),
        allow_model_specific_raise=_bool(
            thresholds_raw.get("allow_model_specific_raise"),
            False,
        ),
    )

    return ContextHealthPolicy(
        enabled=enabled,
        pre_model_intake=pre_model,
        task_boundary=task_boundary,
        thresholds=thresholds,
        runtime_behavior_enabled=False,
    )


def classify_prompt_for_intake(
    user_message: str,
    policy: ContextHealthPolicy | None = None,
) -> PromptIntakeDecision:
    """Classify whether a prompt should be handled by future pre-history intake."""

    policy = policy or default_context_health_policy()
    if not policy.enabled or not policy.pre_model_intake.enabled:
        return PromptIntakeDecision(
            action="pass",
            reason="context_health_disabled",
            pre_history_required=False,
        )

    prompt = user_message or ""
    signals = _prompt_signals(prompt, policy.pre_model_intake)
    if "high_risk_keyword" in signals:
        if policy.pre_model_intake.sensitive_prompt_action == "hold":
            return PromptIntakeDecision(
                action="hold",
                reason="sensitive_prompt_requires_review",
                pre_history_required=policy.pre_model_intake.pre_history_required,
                signals=signals,
            )

    if signals:
        return PromptIntakeDecision(
            action="force_md_intake",
            reason="long_prompt" if _has_long_prompt_signal(signals) else "policy_signal",
            pre_history_required=policy.pre_model_intake.pre_history_required,
            signals=signals,
        )

    return PromptIntakeDecision(
        action="pass",
        reason="below_intake_threshold",
        pre_history_required=policy.pre_model_intake.pre_history_required,
    )


def classify_task_boundary(
    *,
    user_message: str,
    active_task_id: str | None,
    closed_task_ids: Sequence[str] = (),
    explicit_continuation_refs: Sequence[str] = (),
    ambiguous_relation: bool = False,
    policy: ContextHealthPolicy | None = None,
) -> TaskBoundaryDecision:
    """Classify task boundary using policy-only deterministic evidence."""

    policy = policy or default_context_health_policy()
    if not policy.enabled or not policy.task_boundary.enabled:
        return TaskBoundaryDecision(action="continue_task", reason="task_boundary_disabled")

    if ambiguous_relation:
        action = policy.task_boundary.ambiguous_action
        return TaskBoundaryDecision(
            action=action,
            reason="ambiguous_task_relation",
            defaulted=False,
        )

    refs = [ref for ref in explicit_continuation_refs if ref]
    if refs:
        return TaskBoundaryDecision(
            action="continue_task",
            reason="explicit_continuation_reference",
            linked_task_id=refs[0],
        )

    lowered = (user_message or "").lower()
    candidate_ids = [task_id for task_id in [active_task_id, *closed_task_ids] if task_id]
    for task_id in candidate_ids:
        if task_id.lower() in lowered and _contains_continuation_language(lowered):
            return TaskBoundaryDecision(
                action="continue_task",
                reason="explicit_continuation_reference",
                linked_task_id=task_id,
            )

    default = policy.task_boundary.default_without_clear_continuation
    if default == "hold":
        return TaskBoundaryDecision(action="hold", reason="no_clear_continuation_evidence", defaulted=True)
    if default == "continue_task":
        return TaskBoundaryDecision(
            action="continue_task",
            reason="no_clear_continuation_evidence",
            defaulted=True,
            linked_task_id=active_task_id,
        )
    return TaskBoundaryDecision(action="new_task", reason="no_clear_continuation_evidence", defaulted=True)


def resolve_effective_threshold(
    *,
    configured_ratio: float,
    model_suggested_ratio: float | None,
    policy: ContextHealthPolicy | None = None,
) -> ThresholdDecision:
    """Resolve a safe threshold under the policy model.

    This is only a pure decision helper in Phase 1. Runtime threshold paths are
    not wired to it yet.
    """

    policy = policy or default_context_health_policy()
    configured = _ratio(configured_ratio, DEFAULT_MAX_PROVIDER_CONTEXT_RATIO)
    model_suggestion = _ratio(model_suggested_ratio, configured) if model_suggested_ratio is not None else None

    if not policy.enabled:
        return ThresholdDecision(ratio=configured, reason="context_health_disabled")

    max_allowed = policy.thresholds.max_provider_context_ratio
    if not policy.thresholds.allow_model_specific_raise:
        safe_ratio = min(configured, max_allowed)
        return ThresholdDecision(
            ratio=safe_ratio,
            reason="model_raise_disallowed" if model_suggestion is not None and model_suggestion > safe_ratio else "configured_threshold_preserved",
            clamped_model_suggestion=model_suggestion is not None and model_suggestion > safe_ratio,
        )

    candidate = model_suggestion if model_suggestion is not None else configured
    if candidate > max_allowed:
        return ThresholdDecision(
            ratio=max_allowed,
            reason="policy_max_provider_context_ratio",
            clamped_model_suggestion=True,
        )
    return ThresholdDecision(ratio=candidate, reason="configured_or_model_threshold")


def _prompt_signals(prompt: str, policy: PreModelIntakePolicy) -> tuple[str, ...]:
    signals: list[str] = []
    if len(prompt) >= policy.long_prompt_char_threshold:
        signals.append("char_count")
    if prompt.count("\n") + 1 >= policy.long_prompt_line_threshold:
        signals.append("line_count")
    if "```" in prompt:
        signals.append("code_block")
    if any(marker in prompt for marker in ("/", "\\")) and any(
        suffix in prompt for suffix in (".py", ".md", ".json", ".yaml", ".yml", ".txt", ".log")
    ):
        signals.append("file_path")
    lowered = prompt.lower()
    if any(keyword.lower() in lowered for keyword in policy.high_risk_keywords):
        signals.append("high_risk_keyword")
    return tuple(dict.fromkeys(signals))


def _has_long_prompt_signal(signals: Sequence[str]) -> bool:
    return any(signal in {"char_count", "line_count", "code_block", "file_path"} for signal in signals)


def _contains_continuation_language(lowered: str) -> bool:
    return any(token in lowered for token in ("continue", "계속", "resume", "이어", "이어서"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool(value: Any, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return default


def _ratio(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and 0 < float(value) <= 1:
        return float(value)
    return float(default)


def _choice(value: Any, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _string_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        items = tuple(item for item in value if isinstance(item, str) and item)
        return items or default
    return default
