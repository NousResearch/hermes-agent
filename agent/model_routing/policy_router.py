"""Dry-run model routing decisions for policy-as-data.

The router is intentionally deterministic and side-effect free. It recommends a
provider/model pair plus governance metadata, but it does not switch the active
Hermes model, call an LLM, write config, or spend OpenRouter credits.
"""

from __future__ import annotations

from .policy import RoutingContext, RoutingDecision, RoutingPolicy


_HIGH_RISK_TASK_TYPES = {
    "strategy",
    "client_facing_draft",
    "security_privacy_review",
    "financial_modeling",
    "legal_contract_drafting",
}

_SENSITIVE_FINAL_TASK_TYPES = {
    "security_privacy_review",
    "financial_modeling",
    "legal_contract_drafting",
}

_COMPLEX_TECHNICAL_TASK_TYPES = {
    "agentic_coding_orchestration",
}


def recommend_model(context: RoutingContext, policy: RoutingPolicy) -> RoutingDecision:
    """Return a dry-run model recommendation for ``context``.

    The function encodes the approved Caelus boundaries:
    free models are draft/low-risk only, strong models require a reason, and
    unknown governed work fails safe to the paid-budget workhorse instead of a
    random or free router.
    """

    warnings: list[str] = []
    route = policy.task_routes.get(context.task_type)

    if route is None:
        model = policy.fallbacks.get("unknown_task", "deepseek/deepseek-v4-flash")
        reason = (
            f"Unknown task type '{context.task_type}' failed safe to the paid-budget "
            "daily workhorse instead of a free or random model."
        )
        escalation_reason = None
        warnings.append(f"Unknown task type: {context.task_type}")
    else:
        model = route["model"]
        reason = route.get("reason") or route.get("escalation_reason") or "Policy route matched."
        escalation_reason = route.get("escalation_reason")

    model, guardrail_warnings = _apply_guardrails(model, context, policy)
    warnings.extend(guardrail_warnings)

    model_config = policy.models.get(model)
    if model_config is None:
        fallback = policy.fallbacks.get("unknown_task", "deepseek/deepseek-v4-flash")
        warnings.append(f"Model '{model}' missing from policy; fell back to {fallback}")
        model = fallback
        model_config = policy.models[model]

    tier = str(model_config["tier"])
    tier_config = policy.tiers[tier]
    cost_class = str(tier_config["cost_class"])
    fallback_model = model_config.get("fallback")
    free_model_allowed = _free_model_allowed(model, tier, context, policy)

    if tier == "S" and not escalation_reason:
        escalation_reason = _derive_escalation_reason(context)
        if policy.modes.get("require_reason_for_strong_model", True) and not escalation_reason:
            warnings.append("Strong model selected without an explicit escalation reason")

    approval_required = _approval_required(context, tier)

    if tier == "S" and escalation_reason:
        reason = escalation_reason

    return RoutingDecision(
        provider="openrouter",
        model=model,
        tier=tier,
        fallback_model=fallback_model,
        free_model_allowed=free_model_allowed,
        reason=reason,
        estimated_cost_class=cost_class,
        approval_required=approval_required,
        policy_warnings=warnings,
        escalation_reason=escalation_reason,
        dry_run=bool(policy.modes.get("dry_run_default", True)),
    )


def _apply_guardrails(
    model: str,
    context: RoutingContext,
    policy: RoutingPolicy,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if model in set(policy.forbidden.get("governed_default_models") or []):
        warnings.append(f"Forbidden governed default '{model}' replaced with budget fallback")
        return policy.fallbacks.get("unknown_task", "deepseek/deepseek-v4-flash"), warnings

    model_config = policy.models.get(model, {})
    tier = model_config.get("tier")
    is_free = tier == "F" or model.endswith(":free") or model == "openrouter/free"

    if is_free and _free_forbidden_for_context(context, policy):
        warnings.append(f"Free model '{model}' not allowed for this context; escalated to strong model")
        return "qwen/qwen3.6-plus", warnings

    if _requires_strong_model(context) and tier != "S":
        warnings.append(f"Task context requires Tier S; escalated from {model} to qwen/qwen3.6-plus")
        return "qwen/qwen3.6-plus", warnings

    return model, warnings


def _free_forbidden_for_context(context: RoutingContext, policy: RoutingPolicy) -> bool:
    if policy.forbidden.get("free_final_authority", True) and context.final_authority:
        return True
    if policy.forbidden.get("free_client_facing_final", True) and context.client_facing:
        return True
    if policy.forbidden.get("free_sensitive_final", True) and context.sensitive_data:
        return True
    return context.risk_level.lower() == "high"


def _requires_strong_model(context: RoutingContext) -> bool:
    if context.client_facing and context.final_authority:
        return True
    if context.sensitive_data and context.final_authority:
        return True
    if context.task_type in _SENSITIVE_FINAL_TASK_TYPES and context.final_authority:
        return True
    if context.task_type in _HIGH_RISK_TASK_TYPES and context.risk_level.lower() == "high":
        return True
    if context.task_type in _COMPLEX_TECHNICAL_TASK_TYPES:
        return True
    return False


def _free_model_allowed(
    model: str,
    tier: str,
    context: RoutingContext,
    policy: RoutingPolicy,
) -> bool:
    if tier != "F":
        return False
    if model in set(policy.forbidden.get("governed_default_models") or []):
        return False
    return not _free_forbidden_for_context(context, policy)


def _approval_required(context: RoutingContext, tier: str) -> bool:
    return any(
        [
            context.client_facing,
            context.sensitive_data,
            context.final_authority,
            tier == "S",
            context.risk_level.lower() == "high",
        ]
    )


def _derive_escalation_reason(context: RoutingContext) -> str | None:
    if context.task_type == "strategy":
        return "Strategy work is a high-leverage business decision requiring Tier S review."
    if context.client_facing:
        return "Client-facing work affects Alan's reputation and requires Tier S review."
    if context.sensitive_data:
        return "Sensitive-data work requires Tier S review and human approval."
    if context.task_type in _COMPLEX_TECHNICAL_TASK_TYPES:
        return "Complex technical orchestration requires Tier S reasoning."
    if context.risk_level.lower() == "high":
        return "High-risk work requires Tier S review."
    return None
