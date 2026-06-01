"""Component 5 — Graduated Autonomy Gate.

Implements five autonomy levels (L0–L4) with cognitive forcing functions.

Research foundations:
- CUGA: 5-policy governance (Intent Guard → Playbook → Tool Guide → Tool Approvals → Output Formatter)
- Anthropic RCT: 17% comprehension drop among AI-assisted engineers
- Pimenova et al.: trust regulates delegation-to-co-creation continuum
- Bainbridge (1983) / Endsley (1995): automation degrades operator competence

Hard rule: destructive actions require human approval regardless of confidence.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class AutonomyLevel(enum.IntEnum):
    """Five-level autonomy spectrum."""

    L0_FULL_HUMAN_CONTROL = 0
    L1_HUMAN_APPROVED = 1
    L2_HUMAN_NOTIFIED = 2
    L3_HUMAN_ESCALATED = 3
    L4_FULL_AUTONOMY = 4


# Tools that are always destructive regardless of context
_DESTRUCTIVE_TOOLS: frozenset = frozenset({
    "delete_file",
    "rm",
    "drop_table",
    "delete_database",
    "remove_user",
    "git_reset",
    "git_clean",
    "docker_prune",
})

# High-risk tool families (require L1 approval below L3)
_HIGH_RISK_TOOLS: frozenset = frozenset({
    "write_file",
    "patch",
    "edit_file",
    "terminal",
    "execute_command",
    "git_commit",
    "git_push",
    "deploy",
})


@dataclass
class PermissionCheck:
    """Result of an autonomy gate permission check."""

    allowed: bool
    required_level: int
    current_level: int
    reason: str
    comprehension_question: Optional[str] = None


class GraduatedAutonomyGate:
    """Governance gate that regulates agent autonomy based on trust, risk, and competence."""

    def __init__(
        self,
        level: AutonomyLevel = AutonomyLevel.L1_HUMAN_APPROVED,
        enabled: bool = True,
    ):
        self.level = level
        self.enabled = enabled
        self._override_count = 0
        self._auto_approved_count = 0
        self._comprehension_checks_passed = 0
        self._comprehension_checks_failed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_tool_permission(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """Check whether a tool call is permitted at the current autonomy level.

        Returns ``None`` if allowed, or a blocking reason string if denied.
        """
        if not self.enabled:
            return None

        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        # HARD RULE: destructive tools always require human approval
        if tool_name in _DESTRUCTIVE_TOOLS:
            return (
                f"DESTRUCTIVE_ACTION_BLOCKED: '{tool_name}' requires human approval "
                f"regardless of autonomy level (hard rule)."
            )

        # Level 0: no agent execution
        if self.level == AutonomyLevel.L0_FULL_HUMAN_CONTROL:
            return (
                f"L0_FULL_CONTROL: Agent execution is disabled. "
                f"Tool '{tool_name}' was not executed."
            )

        # Level 1: all high-risk tools require approval
        if self.level == AutonomyLevel.L1_HUMAN_APPROVED and tool_name in _HIGH_RISK_TOOLS:
            return (
                f"L1_APPROVAL_REQUIRED: '{tool_name}' is a high-risk tool. "
                f"Human approval required before execution."
            )

        # Level 2: notify on high-risk, allow routine
        if self.level == AutonomyLevel.L2_HUMAN_NOTIFIED and tool_name in _HIGH_RISK_TOOLS:
            # In L2, high-risk tools are allowed but flagged for async review
            # We do NOT block here; the harness logs the notification
            return None

        # Level 3: escalate on uncertainty (handled by confidence calibration upstream)
        # Level 4: full autonomy with audit trail
        return None

    def should_notify(self, tool_call: Dict[str, Any]) -> bool:
        """Determine if a notification should be sent (for L2 and above)."""
        if not self.enabled or self.level < AutonomyLevel.L2_HUMAN_NOTIFIED:
            return False
        tool_name = tool_call.get("name", "")
        return tool_name in _HIGH_RISK_TOOLS

    def get_comprehension_question(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """Generate a cognitive forcing function question for high-risk approvals.

        Returns ``None`` if no question is needed at the current level.
        """
        if not self.enabled:
            return None

        tool_name = tool_call.get("name", "")
        if self.level <= AutonomyLevel.L1_HUMAN_APPROVED and tool_name in _HIGH_RISK_TOOLS:
            return (
                f"Before approving '{tool_name}': Can you explain what this tool does "
                f"and what the rollback procedure is if it fails?"
            )
        return None

    def record_override(self) -> None:
        """Record a human override of an auto-approved action."""
        self._override_count += 1

    def record_auto_approval(self) -> None:
        """Record an action that was auto-approved."""
        self._auto_approved_count += 1

    def record_comprehension_check(self, passed: bool) -> None:
        """Record the result of a comprehension check."""
        if passed:
            self._comprehension_checks_passed += 1
        else:
            self._comprehension_checks_failed += 1

    def get_override_rate(self) -> float:
        """Return the override rate (for the 5% threshold alarm)."""
        total = self._auto_approved_count + self._override_count
        if total == 0:
            return 0.0
        return self._override_count / total

    def should_escalate(self, confidence: float) -> bool:
        """Determine if the system should escalate to human review based on confidence."""
        if self.level >= AutonomyLevel.L3_HUMAN_ESCALATED:
            return confidence < 0.40
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return a JSON-serializable status snapshot."""
        return {
            "level": self.level.name,
            "level_value": int(self.level),
            "auto_approved": self._auto_approved_count,
            "overrides": self._override_count,
            "override_rate": round(self.get_override_rate(), 4),
            "comprehension_passed": self._comprehension_checks_passed,
            "comprehension_failed": self._comprehension_checks_failed,
        }
