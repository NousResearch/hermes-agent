"""Model-tier router for managed agents.

P3-A creates capability profiles.  P3-B turns those profiles into concrete
model routing suggestions: selected model_ref, fallback chain, timeout, and
context budget.  This module is deterministic and side-effect free; execution
layers can consume its decisions later without duplicating routing rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .capability_matrix import build_capability_matrix, preview_route
from .registry import AgentRegistry, AgentSpec, RiskLevel


TIER_ROLE_PREFERENCES = {
    "quick": (
        "experimental_cheap_task",
        "primary_hermes",
        "experimental_deepseek_pool",
    ),
    "standard": (
        "complex_reasoning",
        "experimental_deepseek_pool",
        "experimental_reasoning",
    ),
    "strong": (
        "complex_coding",
        "primary_claude_code",
        "complex_reasoning",
        "experimental_reasoning",
    ),
    "planner": (
        "experimental_research_and_planning",
        "experimental_chinese_research",
        "complex_reasoning",
        "complex_coding",
        "primary_claude_code",
    ),
}


TIER_FALLBACK_ON = {
    "quick": ("timeout", "rate_limited", "server_error", "empty_final_content"),
    "standard": ("timeout", "rate_limited", "server_error", "empty_final_content"),
    "strong": ("quota_exceeded", "rate_limited", "timeout", "server_error", "empty_final_content"),
    "planner": ("quota_exceeded", "rate_limited", "timeout", "server_error", "empty_final_content"),
}


STATUS_RANK = {
    "active": 0,
    "experimental": 1,
    "deprecated": 2,
    "disabled": 3,
}


@dataclass(frozen=True, slots=True)
class ModelTierDecision:
    agent_id: str
    task_type: str
    risk_level: str
    model_tier: str
    model_ref: str | None
    fallback_chain: tuple[str, ...]
    fallback_on: tuple[str, ...]
    timeout_seconds: int
    max_context_chars: int
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "risk_level": self.risk_level,
            "model_tier": self.model_tier,
            "model_ref": self.model_ref,
            "fallback_chain": list(self.fallback_chain),
            "fallback_on": list(self.fallback_on),
            "timeout_seconds": self.timeout_seconds,
            "max_context_chars": self.max_context_chars,
            "source": self.source,
            "reason": self.reason,
        }


def _model_status(model_ref: str, models_cfg: Mapping[str, Any]) -> str:
    cfg = models_cfg.get(model_ref)
    if not isinstance(cfg, Mapping):
        return "missing"
    return str(cfg.get("status") or "active").strip().lower()


def _active_chain(chain: list[str], models_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    usable = []
    for ref in chain:
        status = _model_status(ref, models_cfg)
        if status in {"disabled", "deprecated", "missing"}:
            continue
        if ref not in usable:
            usable.append(ref)
    return tuple(usable)


def _agent_strategy_chain(agent: AgentSpec, models_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    strategy = agent.model_strategy or {}
    raw_chain = strategy.get("chain") if isinstance(strategy, Mapping) else None
    chain = [str(item).strip() for item in (raw_chain or []) if str(item).strip()]
    primary = str((strategy.get("primary") if isinstance(strategy, Mapping) else "") or agent.model_ref or "").strip()
    if primary and primary not in chain:
        chain.insert(0, primary)
    return _active_chain(chain or ([agent.model_ref] if agent.model_ref else []), models_cfg)


def _role_rank(role: str, tier: str) -> int:
    try:
        return TIER_ROLE_PREFERENCES[tier].index(role)
    except ValueError:
        return 99


def _tier_pool(tier: str, models_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = []
    for model_ref, cfg in models_cfg.items():
        if not isinstance(cfg, Mapping):
            continue
        status = str(cfg.get("status") or "active").strip().lower()
        if status in {"disabled", "deprecated"}:
            continue
        role = str(cfg.get("role") or "").strip()
        rank = _role_rank(role, tier)
        if rank == 99:
            continue
        cost = float(cfg.get("tokens_per_million") or 0)
        candidates.append((rank, STATUS_RANK.get(status, 9), cost, str(model_ref)))
    candidates.sort()
    return tuple(item[3] for item in candidates)


def _ranked_chain_for_tier(chain: tuple[str, ...], tier: str, models_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    ranked = []
    for index, model_ref in enumerate(chain):
        cfg = models_cfg.get(model_ref)
        role = str(cfg.get("role") or "").strip() if isinstance(cfg, Mapping) else ""
        status = _model_status(model_ref, models_cfg)
        cost = float(cfg.get("tokens_per_million") or 0) if isinstance(cfg, Mapping) else 0
        ranked.append((_role_rank(role, tier), STATUS_RANK.get(status, 9), cost, index, model_ref))
    ranked.sort()
    return tuple(item[4] for item in ranked)


def _fallback_on(agent: AgentSpec, tier: str) -> tuple[str, ...]:
    strategy = agent.model_strategy or {}
    raw = strategy.get("fallback_on") if isinstance(strategy, Mapping) else None
    values = tuple(str(item).strip() for item in (raw or []) if str(item).strip())
    return values or TIER_FALLBACK_ON.get(tier, TIER_FALLBACK_ON["standard"])


def resolve_model_tier(
    registry: AgentRegistry,
    models_cfg: Mapping[str, Any],
    *,
    agent_id: str | None = None,
    task_type: str,
    risk_level: str = "R0",
    effectiveness: Mapping[str, Mapping[str, Any]] | None = None,
) -> ModelTierDecision:
    risk = RiskLevel.from_raw(risk_level)
    matrix = build_capability_matrix(registry)
    selected_agent_id = agent_id
    if not selected_agent_id:
        route = preview_route(
            registry,
            task_type=task_type,
            risk_level=risk.value,
            effectiveness=effectiveness,
        )
        selected_agent_id = route.get("primary_agent")
    if not selected_agent_id:
        return ModelTierDecision(
            agent_id="",
            task_type=task_type,
            risk_level=risk.value,
            model_tier="unknown",
            model_ref=None,
            fallback_chain=(),
            fallback_on=(),
            timeout_seconds=0,
            max_context_chars=0,
            source="unresolved",
            reason="no_agent_for_task_type_and_risk",
        )

    resolved_agent_id = registry.resolve_agent_id(selected_agent_id) or selected_agent_id
    agent = registry.get(resolved_agent_id)
    if not agent.allows_risk(risk):
        raise ValueError(f"Agent {resolved_agent_id} does not allow risk {risk.value}")
    profile = matrix[resolved_agent_id]
    tier = profile.model_tier

    strategy_chain = _agent_strategy_chain(agent, models_cfg)
    if strategy_chain:
        chain = _ranked_chain_for_tier(strategy_chain, tier, models_cfg)
        source = "agent_model_strategy"
    else:
        chain = _tier_pool(tier, models_cfg)
        source = "tier_pool"

    return ModelTierDecision(
        agent_id=resolved_agent_id,
        task_type=task_type,
        risk_level=risk.value,
        model_tier=tier,
        model_ref=chain[0] if chain else None,
        fallback_chain=chain,
        fallback_on=_fallback_on(agent, tier) if chain else (),
        timeout_seconds=profile.default_timeout_seconds,
        max_context_chars=profile.max_context_chars,
        source=source,
        reason="agent_strategy_preferred" if source == "agent_model_strategy" else "tier_pool_selected",
    )
