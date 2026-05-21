"""Policy engine for managed agents.

This module keeps the first rollout narrow:
- priority order validation
- risk-driven defaults
- explicit rule matching from a declarative YAML file
- audit-friendly decision records
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .registry import RiskLevel


class PolicyEngineError(ValueError):
    """Raised when a policy document is malformed."""


_PRIORITY_CANONICAL = (
    "safety",
    "user_explicit_instruction",
    "soul_global_policy",
    "managed_agents_policy",
    "router_policy",
    "skill_policy",
    "agent_preference",
)

_DESTRUCTIVE_ACTION_TYPES = {
    "delete_file",
    "deploy",
    "modify_env",
    "database_migration",
    "permission_change",
}


def _normalize_risk(value: Any) -> RiskLevel:
    return RiskLevel.from_raw(value)


def _match_rule(conditions: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    for key, expected in conditions.items():
        if key == "action_type":
            if task.get("action_type") != expected:
                return False
            continue
        if key == "source":
            if task.get("source") != expected:
                return False
            continue
        if key == "user_override":
            if task.get("user_override") != expected:
                return False
            continue
        if task.get(key) != expected:
            return False
    return True


def _rule_priority(rule: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    if rule.get("priority"):
        return str(rule.get("priority"))
    conditions = rule.get("when") or {}
    if isinstance(conditions, Mapping):
        if conditions.get("action_type") in _DESTRUCTIVE_ACTION_TYPES:
            return "safety"
        if "user_override" in conditions:
            return "user_explicit_instruction"
        if conditions.get("source") == "soul":
            return "soul_global_policy"
        if conditions.get("source") == "skill":
            return "skill_policy"
    if task.get("risk_level") in {"R3", "R4"}:
        return "managed_agents_policy"
    return "managed_agents_policy"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    task_id: str
    risk_level: RiskLevel
    outcome: str
    winner: str
    reason: str = ""
    requires_human_approval: bool = False
    requires_review: bool = False
    requires_plan: bool = False
    record: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyEngine:
    version: str
    priority_order: tuple[str, ...]
    rules: tuple[dict[str, Any], ...]
    source_path: Path | None = None

    def evaluate(self, task: Mapping[str, Any]) -> PolicyDecision:
        task_id = str(task.get("task_id") or "").strip() or "unknown"
        risk = _normalize_risk(task.get("risk_level", "R0"))
        requires_human_approval = risk is RiskLevel.R4
        requires_plan = risk in {RiskLevel.R3, RiskLevel.R4}
        requires_review = risk in {RiskLevel.R2, RiskLevel.R3, RiskLevel.R4}

        matched = self._highest_priority_match(task)
        if matched and matched.get("decision") == "deny":
            return self._build_decision(
                task_id,
                risk,
                outcome="deny",
                winner=str(matched.get("priority") or "managed_agents_policy"),
                matched_rule=str(matched.get("id") or ""),
                requires_human_approval=requires_human_approval,
                requires_review=requires_review,
                requires_plan=requires_plan,
            )

        if matched and matched.get("priority") == "safety":
            return self._build_decision(
                task_id,
                risk,
                outcome="deny",
                winner="safety",
                matched_rule=str(matched.get("id") or ""),
                requires_human_approval=requires_human_approval,
                requires_review=requires_review,
                requires_plan=requires_plan,
            )

        if matched and matched.get("priority") == "user_explicit_instruction":
            return self._build_decision(
                task_id,
                risk,
                outcome="allow",
                winner="user_explicit_instruction",
                matched_rule=str(matched.get("id") or ""),
                requires_human_approval=requires_human_approval,
                requires_review=requires_review,
                requires_plan=requires_plan,
            )

        if matched and matched.get("priority") == "soul_global_policy":
            return self._build_decision(
                task_id,
                risk,
                outcome="allow",
                winner="soul_global_policy",
                matched_rule=str(matched.get("id") or ""),
                requires_human_approval=requires_human_approval,
                requires_review=requires_review,
                requires_plan=requires_plan,
            )

        return self._build_decision(
            task_id,
            risk,
            outcome="allow",
            winner="managed_agents_policy",
            matched_rule=str(matched.get("id") or "") if matched else None,
            requires_human_approval=requires_human_approval,
            requires_review=requires_review,
            requires_plan=requires_plan,
        )

    def _highest_priority_match(self, task: Mapping[str, Any]) -> Mapping[str, Any] | None:
        ranked: list[tuple[int, Mapping[str, Any]]] = []
        for rule in self.rules:
            if rule.get("decision") not in {"allow", "deny"}:
                continue
            if _match_rule(rule.get("when") or {}, task):
                priority = _rule_priority(rule, task)
                try:
                    index = _PRIORITY_CANONICAL.index(priority)
                except ValueError:
                    index = len(_PRIORITY_CANONICAL)
                ranked.append((index, dict(rule, priority=priority)))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    def _build_decision(
        self,
        task_id: str,
        risk: RiskLevel,
        *,
        outcome: str,
        winner: str,
        matched_rule: str | None,
        requires_human_approval: bool,
        requires_review: bool,
        requires_plan: bool,
    ) -> PolicyDecision:
        record = {
            "task_id": task_id,
            "risk_level": risk.value,
            "outcome": outcome,
            "winner": winner,
            "matched_rule": matched_rule,
            "requires_human_approval": requires_human_approval,
            "requires_review": requires_review,
            "requires_plan": requires_plan,
        }
        return PolicyDecision(
            task_id=task_id,
            risk_level=risk,
            outcome=outcome,
            winner=winner,
            requires_human_approval=requires_human_approval,
            requires_review=requires_review,
            requires_plan=requires_plan,
            record=record,
        )


def load_policy_engine(path: str | Path) -> PolicyEngine:
    policy_path = Path(path)
    data = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise PolicyEngineError("Policy document must be a mapping")

    version = str(data.get("version") or "").strip()
    if not version:
        raise PolicyEngineError("Policy document is missing version")

    priority_order = tuple(data.get("priority_order") or ())
    if priority_order != _PRIORITY_CANONICAL:
        raise PolicyEngineError("priority_order must match the canonical policy order")

    rules = data.get("rules") or []
    if not isinstance(rules, list):
        raise PolicyEngineError("rules must be a list")

    return PolicyEngine(
        version=version,
        priority_order=priority_order,
        rules=tuple(rules),
        source_path=policy_path,
    )
