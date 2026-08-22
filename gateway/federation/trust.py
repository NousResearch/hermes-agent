"""CRITICAL-2: Trust levels + task sensitivity classification.

Defense-in-depth: a compromised node claiming a `critical` task can
be blocked by Trust 评级 + Task 敏感度 routing.

Pillar 4: Authorization (SECURITY-BASELINE.md)
Pillar 5: Resilience (limited by 3-failure rule, CRITICAL-3)

The classification is enforced by:
- gateway/federation/cluster.py: ClusterCoordinator.pick_node_for_task()
- gateway/federation/relay.py: ClusterRelayCoordinator._evaluate()

Effect: a `critical` task CANNOT be claimed by an `unknown` peer,
no matter how confident the AI evaluator is.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class TrustLevel(str, enum.Enum):
    """Per-node trust level, escalating.

    unknown   — newly joined, not verified yet
    verified  — Ed25519 challenge-response completed
    trusted   — long-term partnership, user-approved
    admin     — primary node, full control
    """
    UNKNOWN = "unknown"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    ADMIN = "admin"

    # Numeric ordering for >, < comparisons
    @property
    def rank(self) -> int:
        return {
            TrustLevel.UNKNOWN: 0,
            TrustLevel.VERIFIED: 1,
            TrustLevel.TRUSTED: 2,
            TrustLevel.ADMIN: 3,
        }[self]


def _compare_trust(a: "TrustLevel", b: "TrustLevel") -> int:
    """Return -1, 0, 1 for a < b, a == b, a > b."""
    if a.rank < b.rank:
        return -1
    if a.rank > b.rank:
        return 1
    return 0


# Make comparisons work via free functions — TrustLevel still enum.


class TaskSensitivity(str, enum.Enum):
    """Per-task sensitivity classification.

    low       — read-only, public info
    medium    — normal task, can be relayed freely
    high      — contains user PII, limited relay
    critical  — destructive / privileged, admin-only
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    # Inverse mapping: minimum trust required to claim
    @property
    def min_trust(self) -> TrustLevel:
        return {
            TaskSensitivity.LOW: TrustLevel.VERIFIED,
            TaskSensitivity.MEDIUM: TrustLevel.VERIFIED,
            TaskSensitivity.HIGH: TrustLevel.TRUSTED,
            TaskSensitivity.CRITICAL: TrustLevel.ADMIN,
        }[self]


@dataclass
class TrustPolicy:
    """Authorization policy for the cluster.

    Configurable via config.yaml:
        cluster:
          trust_policy:
            default_trust: verified
            require_trust_approval: true
    """
    # Default trust when a node first joins (must be verified before upgrade)
    default_trust: TrustLevel = TrustLevel.VERIFIED

    # Whether user must explicitly approve trust upgrades
    require_trust_approval: bool = True

    # When a peer is verified, what trust do they get?
    # Default: VERIFIED (challenge-response passed)
    on_verified_trust: TrustLevel = TrustLevel.VERIFIED

    # When a verified peer has been alive for X seconds, auto-upgrade to TRUSTED
    trusted_after_alive_s: int = 86400  # 24 hours

    # Capability requirements per sensitivity
    sensitivity_rules: Dict[str, str] = field(default_factory=lambda: {
        "low": "verified",
        "medium": "verified",
        "high": "trusted",
        "critical": "admin",
    })

    def can_claim(self, peer_trust: TrustLevel, task_sensitivity: TaskSensitivity) -> bool:
        """Centralized authorization check.

        Returns True if a peer with `peer_trust` can claim a task with
        `task_sensitivity`. Defense-in-depth check — call this BEFORE
        any task claim operation.
        """
        if not isinstance(peer_trust, TrustLevel):
            try:
                peer_trust = TrustLevel(peer_trust)
            except ValueError:
                return False  # unknown trust string = no access
        if not isinstance(task_sensitivity, TaskSensitivity):
            try:
                task_sensitivity = TaskSensitivity(task_sensitivity)
            except ValueError:
                return False  # unknown sensitivity = no access (fail-closed)
        return peer_trust.rank >= task_sensitivity.min_trust.rank

    def should_alert(self, peer_trust: TrustLevel, task_sensitivity: TaskSensitivity) -> bool:
        """Should this attempt be SECURITY-AUDIT logged as an alert?"""
        # Always alert if denied, or if a critical task is touched
        if not self.can_claim(peer_trust, task_sensitivity):
            return True
        if task_sensitivity == TaskSensitivity.CRITICAL:
            return True
        return False


# === Auto-inference helpers ===

_KEYWORDS_HIGH = (
    "release", "rollback", "publish", "npm publish",
    "git push", "merge", "apply", "migrate", "schema",
)
_KEYWORDS_CRITICAL = (
    "delete", "drop", "wipe", "rm -rf", "destroy", "kill",
    "drop database", "drop table", "format", "truncate",
    "rollback prod", "production deploy", "force push",
    "deploy production", "deploy prod",
)


def infer_sensitivity(task_title: str, task_description: str = "") -> TaskSensitivity:
    """Infer task sensitivity from title/description.

    Used by ClusterEvaluator to seed task sensitivity. Users can override
    via config or explicit API.
    """
    text = (task_title + " " + task_description).lower()
    for kw in _KEYWORDS_CRITICAL:
        if kw in text:
            return TaskSensitivity.CRITICAL
    for kw in _KEYWORDS_HIGH:
        if kw in text:
            return TaskSensitivity.HIGH
    if "user" in text or "email" in text or "personal" in text:
        return TaskSensitivity.HIGH
    return TaskSensitivity.MEDIUM


# === Cluster membership ===

@dataclass
class NodeTrustRecord:
    """Per-node trust record stored in cluster registry."""
    node_id: str
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    verified_at: Optional[float] = None  # when verified challenge passed
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    approved_by_user: bool = False

    def effective_trust(self) -> TrustLevel:
        """Compute effective trust, considering time + approval."""
        if not self.approved_by_user:
            # User must approve trust upgrade
            return TrustLevel.VERIFIED if self.verified_at else TrustLevel.UNKNOWN
        # Time-based auto-upgrade
        if self.last_seen_at - self.first_seen_at > 86400:
            return TrustLevel.TRUSTED
        return self.trust_level


__all__ = [
    "TrustLevel",
    "TaskSensitivity",
    "TrustPolicy",
    "NodeTrustRecord",
    "infer_sensitivity",
]
