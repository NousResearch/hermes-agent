"""Capability matrix for managed-agent routing previews.

The registry describes who each agent is.  This module turns that metadata into
machine-readable scheduling hints: task types, runtime budgets, model tiers, and
failure policies.  The hints are intentionally deterministic so dashboard and
gate code can explain routing decisions before P3's adaptive router arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .registry import AgentRegistry, AgentSpec, RiskLevel


DEFAULT_FAILURE_POLICIES = {
    "review_only": "record_ineffective_then_fallback_to_codex",
    "external_cli": "timeout_then_switch_agent",
    "fast_worker": "one_retry_then_escalate",
    "quality_gate": "block_on_failed_or_timeout",
    "managed_worker": "retry_then_escalate",
}


CAPABILITY_TASK_TYPES = {
    "analysis": "analysis",
    "architecture_review": "architecture_review",
    "bug_reproduction": "bugfix",
    "code_edit": "implementation",
    "code_review": "code_review",
    "decision_making": "planning",
    "file_modification": "implementation",
    "implementation_planning": "planning",
    "log_analysis": "debugging",
    "refactor": "code_maintenance",
    "release_gate": "quality_gate",
    "review": "code_review",
    "risk_assessment": "security_review",
    "script_execution": "implementation",
    "small_fix": "small_fix",
    "strategy_decision": "planning",
    "technical_decomposition": "planning",
    "test_generation": "tests",
    "test_run": "tests",
    "validation": "quality_gate",
    "web_research": "research",
}


ROLE_TASK_TYPES = {
    "lead_implementer": ("implementation", "code_maintenance", "tests"),
    "fast_worker": ("small_fix", "tests", "debugging"),
    "external_collaboration_worker": ("code_review", "debugging"),
    "principal_engineer": ("code_review", "architecture_review", "planning"),
    "internal_reasoner": ("analysis", "planning", "technical_decomposition"),
    "quality_gate": ("quality_gate", "security_review", "code_review"),
    "research_analyst": ("research", "analysis"),
}


TASK_MODEL_TIERS = {
    "architecture_review": "strong",
    "code_maintenance": "strong",
    "code_review": "strong",
    "debugging": "standard",
    "implementation": "strong",
    "planning": "planner",
    "quality_gate": "strong",
    "research": "planner",
    "security_review": "strong",
    "small_fix": "quick",
    "technical_decomposition": "planner",
    "tests": "quick",
}


TASK_TIMEOUTS = {
    "small_fix": 180,
    "tests": 180,
    "debugging": 240,
    "code_review": 300,
    "implementation": 600,
    "code_maintenance": 600,
    "architecture_review": 600,
    "planning": 420,
    "technical_decomposition": 420,
    "quality_gate": 300,
    "security_review": 420,
    "research": 600,
    "analysis": 300,
}


TASK_CONTEXT_BUDGETS = {
    "small_fix": 8000,
    "tests": 8000,
    "debugging": 12000,
    "code_review": 12000,
    "implementation": 18000,
    "code_maintenance": 18000,
    "architecture_review": 24000,
    "planning": 20000,
    "technical_decomposition": 20000,
    "quality_gate": 12000,
    "security_review": 16000,
    "research": 24000,
    "analysis": 12000,
}


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    agent_id: str
    role: str
    runtime: str
    task_types: tuple[str, ...]
    default_timeout_seconds: int
    max_context_chars: int
    model_tier: str
    failure_policy: str
    risk_allowed: tuple[str, ...]
    strengths: tuple[str, ...] = field(default_factory=tuple)
    weak_spots: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "runtime": self.runtime,
            "task_types": list(self.task_types),
            "default_timeout_seconds": self.default_timeout_seconds,
            "max_context_chars": self.max_context_chars,
            "model_tier": self.model_tier,
            "failure_policy": self.failure_policy,
            "risk_allowed": list(self.risk_allowed),
            "strengths": list(self.strengths),
            "weak_spots": list(self.weak_spots),
        }


def _ordered_unique(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _task_types_for(agent: AgentSpec) -> tuple[str, ...]:
    task_types = list(ROLE_TASK_TYPES.get(agent.role, ()))
    for capability in agent.capabilities:
        mapped = CAPABILITY_TASK_TYPES.get(capability)
        if mapped:
            task_types.append(mapped)
    return _ordered_unique(task_types or ["analysis"])


def _profile_class(agent: AgentSpec) -> str:
    if agent.role == "quality_gate":
        return "quality_gate"
    if agent.role == "fast_worker":
        return "fast_worker"
    if agent.runtime:
        return "external_cli"
    if agent.permission.value == "read_only":
        return "review_only"
    return "managed_worker"


def _model_tier(agent: AgentSpec, task_types: tuple[str, ...]) -> str:
    if agent.role == "fast_worker":
        return "quick"
    tiers = [TASK_MODEL_TIERS.get(task_type, "standard") for task_type in task_types]
    if "planner" in tiers:
        return "planner"
    if "strong" in tiers:
        return "strong"
    if "standard" in tiers:
        return "standard"
    return "quick"


def _max_for_task_types(task_types: tuple[str, ...], table: Mapping[str, int], default: int) -> int:
    return max((table.get(task_type, default) for task_type in task_types), default=default)


def _strengths(agent: AgentSpec, task_types: tuple[str, ...]) -> tuple[str, ...]:
    values = list(task_types[:4])
    if agent.runtime:
        values.append("external_cli_runtime")
    if agent.permission.value == "read_only":
        values.append("low_mutation_risk")
    return _ordered_unique(values)


def _weak_spots(agent: AgentSpec) -> tuple[str, ...]:
    weak: list[str] = []
    if agent.runtime:
        weak.append("requires_external_cli_health")
    if agent.permission.value == "read_only":
        weak.append("cannot_apply_changes")
    if RiskLevel.R3 not in agent.risk_allowed and RiskLevel.R4 not in agent.risk_allowed:
        weak.append("not_for_high_risk_tasks")
    return tuple(weak)


def build_capability_profile(agent: AgentSpec) -> CapabilityProfile:
    task_types = _task_types_for(agent)
    profile_class = _profile_class(agent)
    return CapabilityProfile(
        agent_id=agent.agent_id,
        role=agent.role,
        runtime=agent.runtime or "native",
        task_types=task_types,
        default_timeout_seconds=_max_for_task_types(task_types, TASK_TIMEOUTS, 300),
        max_context_chars=_max_for_task_types(task_types, TASK_CONTEXT_BUDGETS, 12000),
        model_tier=_model_tier(agent, task_types),
        failure_policy=DEFAULT_FAILURE_POLICIES[profile_class],
        risk_allowed=tuple(sorted(level.value for level in agent.risk_allowed)),
        strengths=_strengths(agent, task_types),
        weak_spots=_weak_spots(agent),
    )


def build_capability_matrix(registry: AgentRegistry) -> dict[str, CapabilityProfile]:
    return {
        agent_id: build_capability_profile(agent)
        for agent_id, agent in registry.agents.items()
    }


def _effectiveness_for(
    effectiveness: Mapping[str, Mapping[str, Any]] | None,
    agent_id: str,
) -> Mapping[str, Any]:
    if not effectiveness:
        return {}
    item = effectiveness.get(agent_id)
    return item if isinstance(item, Mapping) else {}


def _adaptive_sort_key(
    profile: CapabilityProfile,
    effectiveness: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[float, float, int, str]:
    item = _effectiveness_for(effectiveness, profile.agent_id)
    observed = int(item.get("run_count") or 0) + int(item.get("handoff_count") or 0)
    if observed <= 0:
        adaptive_penalty = 0.0
    else:
        score = float(item.get("effectiveness_score") or 0.0)
        timeout_rate = float(item.get("timeout_rate") or 0.0)
        failed_rate = float(item.get("failed_rate") or 0.0)
        revision_count = float(item.get("revision_needed_count") or 0.0)
        adaptive_penalty = max(0.0, 100.0 - score) + timeout_rate * 0.5 + failed_rate * 0.75 + revision_count * 5.0
    return (
        adaptive_penalty,
        0 if profile.failure_policy != "block_on_failed_or_timeout" else 1,
        profile.default_timeout_seconds,
        profile.agent_id,
    )


def preview_route(
    registry: AgentRegistry,
    *,
    task_type: str,
    risk_level: str = "R0",
    effectiveness: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    risk = RiskLevel.from_raw(risk_level)
    matrix = build_capability_matrix(registry)
    candidates = []
    for agent_id, profile in matrix.items():
        agent = registry.agents[agent_id]
        if task_type not in profile.task_types:
            continue
        if not agent.allows_risk(risk):
            continue
        candidates.append(profile)

    candidates.sort(key=lambda profile: _adaptive_sort_key(profile, effectiveness))
    primary = candidates[0].agent_id if candidates else None
    return {
        "task_type": task_type,
        "risk_level": risk.value,
        "primary_agent": primary,
        "candidate_agents": [profile.agent_id for profile in candidates],
        "reason": (
            "adaptive_effectiveness_and_capability_match"
            if primary and effectiveness
            else "capability_and_risk_match"
            if primary
            else "no_capability_match"
        ),
    }
