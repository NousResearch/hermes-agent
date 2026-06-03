"""Audit policy following K8s audit.k8s.io/v1 PolicyRule pattern."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from audit.levels import AuditLevel


@dataclass
class GroupVersionResource:
    """K8s-style resource reference."""

    group: str = ""
    resource: str = ""
    resource_names: Optional[List[str]] = None

    def matches(self, resource: str, group: str = "") -> bool:
        """Check if this rule matches the given resource."""
        if self.resource != "*" and self.resource != resource:
            return False
        if self.group != "*" and self.group != group:
            return False
        return True


@dataclass
class PolicyRule:
    """
    K8s Audit PolicyRule structure.
    First-match-wins evaluation order.
    """

    # Audit level for this rule
    level: AuditLevel = AuditLevel.METADATA

    # Resource targeting
    verbs: Optional[List[str]] = None  # ["create", "update", "delete", "exec"]
    resources: Optional[List[GroupVersionResource]] = None
    namespaces: Optional[List[str]] = None

    # User/group targeting
    users: Optional[List[str]] = None
    groups: Optional[List[str]] = None

    # Operation type targeting
    operation_types: Optional[List[str]] = None  # ["Mutate", "Read"]

    # Channel targeting
    channels: Optional[List[str]] = None

    # Stages (None means all stages)
    stages: Optional[List[str]] = None

    def matches(
        self,
        verb: str,
        resource: str,
        user: str = "",
        user_groups: Optional[List[str]] = None,
        namespace: str = "",
        op_type: str = "",
        channel: str = "",
        group: str = "",
    ) -> bool:
        """
        Check if this rule matches the given operation context.
        All non-None fields must match for a match.
        """
        user_groups = user_groups or []

        # Verb check
        if self.verbs and verb not in self.verbs:
            return False

        # Resource check
        if self.resources:
            matched = False
            for res in self.resources:
                if res.matches(resource, group):
                    matched = True
                    break
            if not matched:
                return False

        # User check
        if self.users and user not in self.users:
            # Support regex patterns
            for pattern in self.users:
                if pattern.startswith("regex:"):
                    if re.match(pattern[7:], user):
                        matched = True
                        break
            else:
                return False

        # Groups check
        if self.groups and not any(g in self.groups for g in user_groups):
            return False

        # Namespace check
        if self.namespaces and namespace not in self.namespaces:
            if "*" not in self.namespaces:
                return False

        # Operation type check
        if self.operation_types and op_type not in self.operation_types:
            return False

        # Channel check
        if self.channels and channel not in self.channels:
            if "*" not in self.channels:
                return False

        return True


@dataclass
class AuditPolicy:
    """
    Audit policy with configurable rules.
    Loads from YAML config or uses defaults.
    """

    api_version: str = "audit.k8s.io/v1"
    kind: str = "AuditPolicy"
    rules: List[PolicyRule] = field(default_factory=list)

    # Backend settings
    log_path: str = "~/.hermes/logs/audit/"
    max_size_mb: int = 100
    max_age_hours: int = 24
    max_backups: int = 720
    compress: bool = True
    timezone_name: str = "Asia/Shanghai"

    # Tamper protection
    tamper_protection_enabled: bool = True

    def evaluate_level(
        self,
        verb: str,
        resource: str,
        user: str = "",
        user_groups: Optional[List[str]] = None,
        namespace: str = "",
        op_type: str = "",
        channel: str = "",
    ) -> AuditLevel:
        """
        First-match-wins policy evaluation.
        Returns the AuditLevel for the first matching rule, or NONE if no match.
        """
        user_groups = user_groups or []
        for rule in self.rules:
            if rule.matches(verb, resource, user, user_groups, namespace, op_type, channel):
                return rule.level
        return AuditLevel.NONE

    @classmethod
    def from_dict(cls, data: dict) -> "AuditPolicy":
        """Load policy from dict (e.g., parsed YAML)."""
        rules = []
        for rule_data in data.get("policy", []):
            level_str = rule_data.get("level", "Metadata")
            level = AuditLevel.from_string(level_str)

            resources = []
            for res in rule_data.get("resources", []):
                if isinstance(res, dict):
                    resources.append(
                        GroupVersionResource(
                            group=res.get("group", ""),
                            resource=res.get("resource", ""),
                            resource_names=res.get("resourceNames"),
                        )
                    )
                elif isinstance(res, str):
                    resources.append(GroupVersionResource(resource=res))

            rules.append(
                PolicyRule(
                    level=level,
                    verbs=rule_data.get("verbs"),
                    resources=resources if resources else None,
                    namespaces=rule_data.get("namespaces"),
                    users=rule_data.get("users"),
                    groups=rule_data.get("groups"),
                    operation_types=rule_data.get("operationTypes"),
                    channels=rule_data.get("channels"),
                    stages=rule_data.get("stages"),
                )
            )

        log_config = data.get("log", {})
        tamper = data.get("tamper_protection", {})

        return cls(
            rules=rules,
            log_path=log_config.get("path", "~/.hermes/logs/audit/"),
            max_size_mb=log_config.get("rotation", {}).get("max_size_mb", 100),
            max_age_hours=log_config.get("rotation", {}).get("max_age_hours", 24),
            max_backups=log_config.get("rotation", {}).get("max_backups", 720),
            compress=log_config.get("rotation", {}).get("compress", True),
            timezone_name=log_config.get("rotation", {}).get("timezone_name", "Asia/Shanghai"),
            tamper_protection_enabled=tamper.get("enabled", True),
        )

    @classmethod
    def default_policy(cls) -> "AuditPolicy":
        """Create a default policy that audits all operations at Metadata level."""
        return cls(
            rules=[
                # Full audit for mutations
                PolicyRule(
                    level=AuditLevel.REQUESTRESPONSE,
                    verbs=["create", "update", "delete", "exec", "execute", "patch"],
                    operation_types=["Mutate"],
                ),
                # Metadata for reads
                PolicyRule(
                    level=AuditLevel.METADATA,
                    verbs=["get", "list", "watch"],
                ),
            ],
        )