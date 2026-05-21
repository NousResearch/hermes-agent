"""Managed agents router for structured task delegation.

The first rollout keeps routing deterministic and narrow:
- load routing rules from the managed-agents YAML bundle
- resolve task_category + risk_level + user_override
- produce a structured decision suitable for TaskCard updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .policy import PolicyDecision, PolicyEngine
from .registry import AgentRegistry, AgentRegistryError, RiskLevel, load_agent_registry


class ManagedAgentRouterError(ValueError):
    """Raised when router input is malformed or route resolution fails."""


def _normalize_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        return [values.strip()] if values.strip() else []
    if isinstance(values, Iterable):
        return [str(item).strip() for item in values if str(item).strip()]
    return [str(values).strip()] if str(values).strip() else []


def _normalize_risk(value: Any) -> RiskLevel:
    return RiskLevel.from_raw(value)


def _rule_matches(conditions: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    for key, expected in conditions.items():
        if key == "task_category":
            if task.get("task_category") != expected:
                return False
            continue
        if key == "risk_level":
            if task.get("risk_level") != expected:
                return False
            continue
        if key == "risk_max":
            try:
                if RiskLevel.from_raw(task.get("risk_level", "R0")).value > RiskLevel.from_raw(expected).value:
                    return False
            except Exception:
                return False
            continue
        if key == "max_files_changed":
            try:
                if int(task.get("max_files_changed", 0)) > int(expected):
                    return False
            except Exception:
                return False
            continue
        if task.get(key) != expected:
            return False
    return True


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    task_id: str
    mode: str
    agents: list[str] = field(default_factory=list)
    reason: str = ""
    matched_rule: str | None = None
    routing_basis: list[str] = field(default_factory=list)
    fallback_used: str | None = None
    risk_level: str = "R0"
    require_gate: str | None = None
    requires_plan: bool = False
    requires_review: bool = False
    requires_human_approval: bool = False

    def to_execution_plan(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "agents": list(self.agents),
            "delegation_reason": self.reason,
            "require_gate": self.require_gate,
        }

    def to_task_card_updates(self) -> dict[str, Any]:
        return {
            "execution_plan": self.to_execution_plan(),
            "routing_basis": list(self.routing_basis),
            "fallback_used": self.fallback_used,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "agents": list(self.agents),
            "reason": self.reason,
            "matched_rule": self.matched_rule,
            "routing_basis": list(self.routing_basis),
            "fallback_used": self.fallback_used,
            "risk_level": self.risk_level,
            "require_gate": self.require_gate,
            "requires_plan": self.requires_plan,
            "requires_review": self.requires_review,
            "requires_human_approval": self.requires_human_approval,
        }


@dataclass(slots=True)
class ManagedAgentRouter:
    registry: AgentRegistry
    version: str
    default_route: dict[str, Any]
    fallback_route: dict[str, Any]
    rules: tuple[dict[str, Any], ...]
    source_path: Path | None = None
    policy_engine: PolicyEngine | None = None

    def route(self, task: Mapping[str, Any]) -> RoutingDecision:
        task_id = str(task.get("task_id") or "unknown").strip() or "unknown"
        task_category = str(task.get("task_category") or "other").strip() or "other"
        risk = _normalize_risk(task.get("risk_level", "R0"))
        risk_value = risk.value
        policy_decision = self._policy_decision(task)

        if policy_decision.outcome == "deny":
            return RoutingDecision(
                task_id=task_id,
                mode="review_only",
                agents=_normalize_list(self.fallback_route.get("agents")),
                reason="policy_denied",
                matched_rule=None,
                routing_basis=["policy:deny", f"risk:{risk_value}"],
                fallback_used="policy_denied",
                risk_level=risk_value,
                require_gate="review",
                requires_plan=policy_decision.requires_plan,
                requires_review=True,
                requires_human_approval=policy_decision.requires_human_approval,
            )

        matched_rule = self._match_rule(task_category, risk_value, task)
        user_override = str(task.get("user_override") or "").strip() or None
        base_routing_basis: list[str] = []

        if matched_rule is not None:
            base_routing_basis.append(f"rule:{matched_rule['id']}")
        base_routing_basis.append(f"risk:{risk_value}")

        if matched_rule is not None:
            decision = self._decision_from_rule(task_id, risk_value, matched_rule, base_routing_basis)
        else:
            decision = self._decision_from_default(task_id, risk_value, base_routing_basis)

        if user_override:
            decision = self._apply_user_override(
                decision,
                task_id,
                risk_value,
                user_override,
                matched_rule,
                base_routing_basis,
            )

        return self._apply_policy_decision(decision, policy_decision)

    def _policy_decision(self, task: Mapping[str, Any]) -> PolicyDecision:
        if self.policy_engine is None:
            self.policy_engine = PolicyEngine(
                version="managed-agents-router-default",
                priority_order=(
                    "safety",
                    "user_explicit_instruction",
                    "soul_global_policy",
                    "managed_agents_policy",
                    "router_policy",
                    "skill_policy",
                    "agent_preference",
                ),
                rules=(),
            )
        return self.policy_engine.evaluate(task)

    def _apply_policy_decision(
        self,
        decision: RoutingDecision,
        policy_decision: PolicyDecision,
    ) -> RoutingDecision:
        return RoutingDecision(
            task_id=decision.task_id,
            mode=decision.mode,
            agents=list(decision.agents),
            reason=decision.reason,
            matched_rule=decision.matched_rule,
            routing_basis=list(decision.routing_basis),
            fallback_used=decision.fallback_used,
            risk_level=decision.risk_level,
            require_gate=decision.require_gate,
            requires_plan=decision.requires_plan or policy_decision.requires_plan,
            requires_review=decision.requires_review or policy_decision.requires_review,
            requires_human_approval=(
                decision.requires_human_approval
                or policy_decision.requires_human_approval
            ),
        )

    def _match_rule(self, task_category: str, risk_value: str, task: Mapping[str, Any]) -> dict[str, Any] | None:
        for rule in self.rules:
            conditions = rule.get("when") or {}
            if not isinstance(conditions, Mapping):
                continue
            if _rule_matches(conditions, task):
                return rule

        return None

    def _decision_from_rule(
        self,
        task_id: str,
        risk_value: str,
        rule: Mapping[str, Any],
        routing_basis: list[str],
    ) -> RoutingDecision:
        mode = str(rule.get("mode") or self.default_route.get("mode") or "single_agent")
        agents = _normalize_list(rule.get("agents") or rule.get("owner_agent"))
        reason = str(rule.get("reason") or rule.get("delegation_reason") or "")
        require_gate = rule.get("require_gate")
        if require_gate is not None:
            require_gate = str(require_gate)
        requires_human_approval = bool(rule.get("requires_human_approval", False))
        requires_plan = mode == "pipeline" or risk_value in {"R3", "R4"}
        requires_review = mode in {"pipeline", "review_only"} or risk_value in {"R2", "R3", "R4"}

        return RoutingDecision(
            task_id=task_id,
            mode=mode,
            agents=agents,
            reason=reason,
            matched_rule=str(rule.get("id") or "") or None,
            routing_basis=routing_basis,
            fallback_used=None,
            risk_level=risk_value,
            require_gate=require_gate,
            requires_plan=requires_plan,
            requires_review=requires_review,
            requires_human_approval=requires_human_approval,
        )

    def _decision_from_default(
        self,
        task_id: str,
        risk_value: str,
        routing_basis: list[str],
    ) -> RoutingDecision:
        return RoutingDecision(
            task_id=task_id,
            mode="review_only",
            agents=["codex"],
            reason="unresolved_route_requires_review",
            matched_rule=None,
            routing_basis=["fallback:unmatched_route", f"risk:{risk_value}"],
            fallback_used="unmatched_route",
            risk_level=risk_value,
            require_gate=None,
            requires_plan=False,
            requires_review=True,
            requires_human_approval=risk_value == "R4",
        )

    def _apply_user_override(
        self,
        decision: RoutingDecision,
        task_id: str,
        risk_value: str,
        user_override: str,
        matched_rule: Mapping[str, Any] | None,
        routing_basis: list[str],
    ) -> RoutingDecision:
        if risk_value == "R4":
            return RoutingDecision(
                task_id=task_id,
                mode="pipeline",
                agents=["codex", "claude", "nesta"],
                reason="risk_escalation",
                matched_rule="high_risk_pipeline",
                routing_basis=[f"risk_escalation:{risk_value}", f"user_override:{user_override}", "rule:high_risk_pipeline"],
                fallback_used=None,
                risk_level=risk_value,
                require_gate="review",
                requires_plan=True,
                requires_review=True,
                requires_human_approval=True,
            )

        agent = self.registry.agents.get(user_override)
        if agent is None:
            return RoutingDecision(
                task_id=task_id,
                mode="review_only",
                agents=["codex"],
                reason="unresolved_route_requires_review",
                matched_rule=decision.matched_rule,
                routing_basis=[f"fallback:unknown_user_override", f"user_override:{user_override}"] + list(routing_basis),
                fallback_used="unknown_user_override",
                risk_level=risk_value,
                require_gate=None,
                requires_plan=False,
                requires_review=True,
                requires_human_approval=False,
            )

        if not agent.allows_risk(_normalize_risk(risk_value)):
            return RoutingDecision(
                task_id=task_id,
                mode="review_only",
                agents=["codex"],
                reason="unresolved_route_requires_review",
                matched_rule=decision.matched_rule,
                routing_basis=[f"fallback:user_override_risk_mismatch", f"user_override:{user_override}"] + list(routing_basis),
                fallback_used="user_override_risk_mismatch",
                risk_level=risk_value,
                require_gate=None,
                requires_plan=False,
                requires_review=True,
                requires_human_approval=False,
            )

        return RoutingDecision(
            task_id=task_id,
            mode="single_agent" if len(decision.agents) <= 1 else decision.mode,
            agents=[user_override],
            reason="user_explicit_instruction",
            matched_rule="user_override",
            routing_basis=[f"user_override:{user_override}"] + [entry for entry in routing_basis if entry != f"user_override:{user_override}"],
            fallback_used=None,
            risk_level=risk_value,
            require_gate=decision.require_gate,
            requires_plan=decision.requires_plan,
            requires_review=decision.requires_review,
            requires_human_approval=decision.requires_human_approval,
        )


def load_managed_agent_router(path: str | Path) -> ManagedAgentRouter:
    router_path = Path(path)
    data = yaml.safe_load(router_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ManagedAgentRouterError("Managed agents config must be a mapping")

    version = str(data.get("version") or "").strip()
    if not version:
        raise ManagedAgentRouterError("Managed agents config is missing version")

    try:
        registry = AgentRegistry.from_yaml({"version": version, "agents": data.get("agents", [])}, source_path=router_path)
    except AgentRegistryError as exc:
        raise ManagedAgentRouterError(str(exc)) from exc

    routing = data.get("routing")
    if not isinstance(routing, Mapping):
        default_route, fallback_route, normalized_rules = _default_routing(registry)
        return ManagedAgentRouter(
            registry=registry,
            version=version,
            default_route=default_route,
            fallback_route=fallback_route,
            rules=tuple(normalized_rules),
            source_path=router_path,
        )

    default_route = routing.get("default_route") or {}
    fallback_route = routing.get("fallback_route") or {}
    rules = routing.get("rules") or []
    if not isinstance(default_route, Mapping) or not isinstance(fallback_route, Mapping) or not isinstance(rules, list):
        raise ManagedAgentRouterError("Managed agents routing must contain mapping routes and a rule list")

    normalized_rules: list[dict[str, Any]] = []
    for raw_rule in rules:
        if not isinstance(raw_rule, Mapping):
            raise ManagedAgentRouterError("Each routing rule must be a mapping")
        rule_id = str(raw_rule.get("id") or "").strip()
        if not rule_id:
            raise ManagedAgentRouterError("Each routing rule requires an id")
        _validate_agent_references(raw_rule, registry)
        normalized_rules.append(dict(raw_rule))

    _validate_agent_references(default_route, registry)
    _validate_agent_references(fallback_route, registry)

    return ManagedAgentRouter(
        registry=registry,
        version=version,
        default_route=dict(default_route),
        fallback_route=dict(fallback_route),
        rules=tuple(normalized_rules),
        source_path=router_path,
    )


def _default_routing(registry: AgentRegistry) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    default_route = {
        "mode": "review_only",
        "agents": ["codex"] if "codex" in registry.agents else [],
        "reason": "unresolved_route_requires_review",
    }
    fallback_route = dict(default_route)
    rules: list[dict[str, Any]] = []

    if "codex" in registry.agents and "claude" in registry.agents:
        rules.append(
            {
                "id": "high_risk_code_pipeline",
                "when": {"risk_level": "R3"},
                "mode": "pipeline",
                "agents": ["codex", "claude"],
                "require_gate": "review",
                "reason": "risk_escalation",
            }
        )
        rules.append(
            {
                "id": "code_maintenance",
                "when": {"task_category": "code_maintenance"},
                "mode": "single_agent",
                "agents": ["claude"],
                "reason": "default_code_implementation",
            }
        )

    if "deepseek-tui" in registry.agents:
        rules.append(
            {
                "id": "test_generation",
                "when": {"task_category": "tests"},
                "mode": "single_agent",
                "agents": ["deepseek-tui"],
                "reason": "targeted_test_generation",
            }
        )

    if "codex" in registry.agents:
        rules.append(
            {
                "id": "architecture_review",
                "when": {"task_category": "architecture_review"},
                "mode": "single_agent",
                "agents": ["codex"],
                "reason": "design_review",
            }
        )

    return default_route, fallback_route, rules


def _validate_agent_references(route: Mapping[str, Any], registry: AgentRegistry) -> None:
    for key in ("agents", "owner_agent", "planner", "support_agents", "reviewers"):
        value = route.get(key)
        for agent_id in _normalize_list(value):
            if agent_id not in registry.agents:
                raise ManagedAgentRouterError(f"Unknown agent reference: {agent_id}")
