"""
cosmos Policy Engine

"Every action must be approved by at least three forms filled in triplicate!"

Policy as Code enforcement engine.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
import re
from loguru import logger


class PolicyEffect(Enum):
    """Policy decision effects."""
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    REQUIRE_APPROVAL = "require_approval"


class PolicyScope(Enum):
    """Policy application scope."""
    GLOBAL = "global"
    SERVICE = "service"
    TEAM = "team"
    USER = "user"
    RESOURCE = "resource"


@dataclass
class PolicyCondition:
    """A condition in a policy rule."""
    field: str  # The field to check
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains, matches
    value: Any  # The value to compare against


@dataclass
class PolicyRule:
    """A single rule within a policy."""
    id: str
    name: str
    description: str = ""
    conditions: List[PolicyCondition] = field(default_factory=list)
    effect: PolicyEffect = PolicyEffect.DENY
    priority: int = 100  # Lower = higher priority
    enabled: bool = True
    message: str = ""  # Message to show when rule triggers

    def evaluate(self, context: Dict) -> Optional[PolicyEffect]:
        """Evaluate this rule against a context."""
        if not self.enabled:
            return None

        all_conditions_met = True
        for condition in self.conditions:
            if not self._evaluate_condition(condition, context):
                all_conditions_met = False
                break

        if all_conditions_met:
            return self.effect
        return None

    def _evaluate_condition(
        self,
        condition: PolicyCondition,
        context: Dict,
    ) -> bool:
        """Evaluate a single condition."""
        value = self._get_nested_value(context, condition.field)

        if value is None:
            return False

        op = condition.operator
        expected = condition.value

        if op == "eq":
            return value == expected
        elif op == "ne":
            return value != expected
        elif op == "gt":
            return value > expected
        elif op == "lt":
            return value < expected
        elif op == "gte":
            return value >= expected
        elif op == "lte":
            return value <= expected
        elif op == "in":
            return value in expected
        elif op == "not_in":
            return value not in expected
        elif op == "contains":
            return expected in value
        elif op == "matches":
            return bool(re.match(expected, str(value)))
        elif op == "exists":
            return value is not None
        elif op == "not_exists":
            return value is None

        return False

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get a nested value from a dictionary using dot notation."""
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None

        return value


@dataclass
class Policy:
    """A policy with multiple rules."""
    id: str
    name: str
    description: str
    scope: PolicyScope
    rules: List[PolicyRule] = field(default_factory=list)

    # Targeting
    applies_to: List[str] = field(default_factory=list)  # Services, teams, etc.
    excludes: List[str] = field(default_factory=list)

    # Metadata
    version: str = "1.0"
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    enabled: bool = True

    def evaluate(self, context: Dict) -> Dict[str, Any]:
        """Evaluate all rules in this policy."""
        if not self.enabled:
            return {"effect": PolicyEffect.ALLOW, "reason": "Policy disabled"}

        # Check targeting
        target = context.get("target", context.get("service", ""))
        if self.applies_to and target not in self.applies_to:
            return {"effect": PolicyEffect.ALLOW, "reason": "Not in scope"}
        if target in self.excludes:
            return {"effect": PolicyEffect.ALLOW, "reason": "Excluded"}

        # Evaluate rules in priority order
        sorted_rules = sorted(
            [r for r in self.rules if r.enabled],
            key=lambda r: r.priority,
        )

        for rule in sorted_rules:
            effect = rule.evaluate(context)
            if effect:
                return {
                    "effect": effect,
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "message": rule.message,
                }

        # Default allow
        return {"effect": PolicyEffect.ALLOW, "reason": "No rules matched"}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scope": self.scope.value,
            "version": self.version,
            "rule_count": len(self.rules),
            "enabled": self.enabled,
            "owner": self.owner,
            "tags": self.tags,
        }


@dataclass
class PolicyDecision:
    """Result of a policy evaluation."""
    allowed: bool
    effect: PolicyEffect
    policy_id: str
    rule_id: str = ""
    message: str = ""
    context: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PolicyEngine:
    """
    Policy as Code enforcement engine.

    Features:
    - Define policies in YAML or JSON
    - Evaluate actions against policies
    - Support for multiple policy scopes
    - Rule prioritization
    - Decision logging
    """

    def __init__(
        self,
        storage_path: Path = None,
    ):
        self.storage_path = Path(storage_path) if storage_path else Path("./data/policies")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.policies: Dict[str, Policy] = {}
        self.decision_log: List[PolicyDecision] = []
        self.custom_functions: Dict[str, Callable] = {}

        self._load_builtin_policies()

    def _load_builtin_policies(self):
        """Load built-in security policies."""
        # Resource access policy
        resource_policy = Policy(
            id="resource-access",
            name="Resource Access Control",
            description="Control access to protected resources",
            scope=PolicyScope.GLOBAL,
            rules=[
                PolicyRule(
                    id="deny-admin-without-mfa",
                    name="Require MFA for Admin",
                    description="Admin access requires MFA",
                    conditions=[
                        PolicyCondition(field="action", operator="eq", value="admin"),
                        PolicyCondition(field="user.mfa_enabled", operator="eq", value=False),
                    ],
                    effect=PolicyEffect.DENY,
                    message="Admin actions require MFA to be enabled",
                ),
                PolicyRule(
                    id="deny-prod-without-approval",
                    name="Production Changes Need Approval",
                    description="Changes to production require approval",
                    conditions=[
                        PolicyCondition(field="environment", operator="eq", value="production"),
                        PolicyCondition(field="action", operator="in", value=["deploy", "delete", "update"]),
                        PolicyCondition(field="approved", operator="eq", value=False),
                    ],
                    effect=PolicyEffect.REQUIRE_APPROVAL,
                    priority=50,
                    message="Production changes require approval",
                ),
            ],
        )
        self.policies["resource-access"] = resource_policy

        # Data protection policy
        data_policy = Policy(
            id="data-protection",
            name="Data Protection",
            description="Protect sensitive data access",
            scope=PolicyScope.GLOBAL,
            rules=[
                PolicyRule(
                    id="deny-pii-export",
                    name="Restrict PII Export",
                    description="Prevent bulk PII export",
                    conditions=[
                        PolicyCondition(field="action", operator="eq", value="export"),
                        PolicyCondition(field="data_type", operator="eq", value="pii"),
                        PolicyCondition(field="record_count", operator="gt", value=100),
                    ],
                    effect=PolicyEffect.DENY,
                    message="Bulk PII export is restricted",
                ),
                PolicyRule(
                    id="warn-sensitive-access",
                    name="Warn on Sensitive Access",
                    description="Warn when accessing sensitive data",
                    conditions=[
                        PolicyCondition(field="data_classification", operator="in", value=["confidential", "restricted"]),
                    ],
                    effect=PolicyEffect.WARN,
                    priority=200,
                    message="You are accessing sensitive data",
                ),
            ],
        )
        self.policies["data-protection"] = data_policy

        # Secret management policy
        secret_policy = Policy(
            id="secret-management",
            name="Secret Management",
            description="Control secret access and rotation",
            scope=PolicyScope.SERVICE,
            rules=[
                PolicyRule(
                    id="deny-secret-without-rotation",
                    name="Require Secret Rotation",
                    description="Secrets must have rotation enabled",
                    conditions=[
                        PolicyCondition(field="action", operator="eq", value="create_secret"),
                        PolicyCondition(field="rotation_enabled", operator="eq", value=False),
                    ],
                    effect=PolicyEffect.WARN,
                    message="Consider enabling automatic rotation for this secret",
                ),
            ],
        )
        self.policies["secret-management"] = secret_policy

    # =========================================================================
    # POLICY MANAGEMENT
    # =========================================================================

    def add_policy(self, policy: Policy):
        """Add a policy."""
        self.policies[policy.id] = policy
        logger.info(f"Added policy: {policy.name}")

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        return self.policies.get(policy_id)

    def list_policies(
        self,
        scope: PolicyScope = None,
        enabled_only: bool = True,
    ) -> List[Policy]:
        """List policies with optional filters."""
        policies = list(self.policies.values())

        if scope:
            policies = [p for p in policies if p.scope == scope]
        if enabled_only:
            policies = [p for p in policies if p.enabled]

        return policies

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        if policy_id in self.policies:
            del self.policies[policy_id]
            return True
        return False

    def load_policy_from_yaml(self, yaml_path: Path) -> Optional[Policy]:
        """Load a policy from YAML file."""
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)

            rules = []
            for rule_data in data.get("rules", []):
                conditions = []
                for cond_data in rule_data.get("conditions", []):
                    conditions.append(PolicyCondition(
                        field=cond_data["field"],
                        operator=cond_data["operator"],
                        value=cond_data["value"],
                    ))

                rule = PolicyRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    description=rule_data.get("description", ""),
                    conditions=conditions,
                    effect=PolicyEffect(rule_data.get("effect", "deny")),
                    priority=rule_data.get("priority", 100),
                    enabled=rule_data.get("enabled", True),
                    message=rule_data.get("message", ""),
                )
                rules.append(rule)

            policy = Policy(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                scope=PolicyScope(data.get("scope", "global")),
                rules=rules,
                applies_to=data.get("applies_to", []),
                excludes=data.get("excludes", []),
                version=data.get("version", "1.0"),
                owner=data.get("owner", ""),
                tags=data.get("tags", []),
            )

            self.add_policy(policy)
            return policy

        except Exception as e:
            logger.error(f"Failed to load policy from {yaml_path}: {e}")
            return None

    def export_policy_to_yaml(self, policy_id: str, output_path: Path):
        """Export a policy to YAML file."""
        policy = self.policies.get(policy_id)
        if not policy:
            return

        data = {
            "id": policy.id,
            "name": policy.name,
            "description": policy.description,
            "scope": policy.scope.value,
            "version": policy.version,
            "owner": policy.owner,
            "tags": policy.tags,
            "applies_to": policy.applies_to,
            "excludes": policy.excludes,
            "rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "description": rule.description,
                    "effect": rule.effect.value,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "message": rule.message,
                    "conditions": [
                        {
                            "field": c.field,
                            "operator": c.operator,
                            "value": c.value,
                        }
                        for c in rule.conditions
                    ],
                }
                for rule in policy.rules
            ],
        }

        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    # =========================================================================
    # EVALUATION
    # =========================================================================

    def evaluate(
        self,
        context: Dict,
        policy_ids: List[str] = None,
    ) -> PolicyDecision:
        """Evaluate context against policies."""
        # Get applicable policies
        if policy_ids:
            policies = [self.policies[pid] for pid in policy_ids if pid in self.policies]
        else:
            policies = [p for p in self.policies.values() if p.enabled]

        # Evaluate each policy
        for policy in policies:
            result = policy.evaluate(context)
            effect = result["effect"]

            if effect == PolicyEffect.DENY:
                decision = PolicyDecision(
                    allowed=False,
                    effect=effect,
                    policy_id=policy.id,
                    rule_id=result.get("rule_id", ""),
                    message=result.get("message", "Access denied by policy"),
                    context=context,
                )
                self._log_decision(decision)
                return decision

            if effect == PolicyEffect.REQUIRE_APPROVAL:
                decision = PolicyDecision(
                    allowed=False,
                    effect=effect,
                    policy_id=policy.id,
                    rule_id=result.get("rule_id", ""),
                    message=result.get("message", "Requires approval"),
                    context=context,
                )
                self._log_decision(decision)
                return decision

        # All policies passed
        decision = PolicyDecision(
            allowed=True,
            effect=PolicyEffect.ALLOW,
            policy_id="",
            message="Allowed",
            context=context,
        )
        self._log_decision(decision)
        return decision

    def check_permission(
        self,
        user_id: str,
        action: str,
        resource: str,
        **kwargs,
    ) -> PolicyDecision:
        """Check if a user has permission for an action."""
        context = {
            "user_id": user_id,
            "action": action,
            "resource": resource,
            **kwargs,
        }
        return self.evaluate(context)

    def _log_decision(self, decision: PolicyDecision):
        """Log a policy decision."""
        self.decision_log.append(decision)

        # Keep only recent decisions
        if len(self.decision_log) > 10000:
            self.decision_log = self.decision_log[-5000:]

    # =========================================================================
    # CUSTOM FUNCTIONS
    # =========================================================================

    def register_function(self, name: str, func: Callable):
        """Register a custom function for policy evaluation."""
        self.custom_functions[name] = func

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_decision_log(
        self,
        policy_id: str = None,
        allowed: bool = None,
        limit: int = 100,
    ) -> List[PolicyDecision]:
        """Get recent policy decisions."""
        decisions = self.decision_log

        if policy_id:
            decisions = [d for d in decisions if d.policy_id == policy_id]
        if allowed is not None:
            decisions = [d for d in decisions if d.allowed == allowed]

        return sorted(decisions, key=lambda d: d.timestamp, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get policy evaluation statistics."""
        total = len(self.decision_log)
        if total == 0:
            return {"total_decisions": 0}

        allowed = sum(1 for d in self.decision_log if d.allowed)
        denied = sum(1 for d in self.decision_log if not d.allowed)

        by_policy = {}
        for decision in self.decision_log:
            pid = decision.policy_id or "default"
            if pid not in by_policy:
                by_policy[pid] = {"allowed": 0, "denied": 0}
            if decision.allowed:
                by_policy[pid]["allowed"] += 1
            else:
                by_policy[pid]["denied"] += 1

        return {
            "total_decisions": total,
            "allowed": allowed,
            "denied": denied,
            "denial_rate": round(denied / total * 100, 2) if total > 0 else 0,
            "by_policy": by_policy,
        }


# Singleton instance
policy_engine = PolicyEngine()
