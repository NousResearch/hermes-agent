"""Governance gate between the model's tool calls and the tool handler.

Every tool call passes through ``GovernanceGate.evaluate()`` before reaching
the handler. The gate produces a typed ``GovernanceDecision`` per call:
allow / deny / require_approval. Denied calls never execute. The full set
of decisions is recorded on the ``ActionStep`` so the audit trail is
complete — even for calls the model proposed but governance refused.

The default policy is ``DenyAllGovernance`` (fail-closed). Callers must
explicitly opt into a permissive policy. There is no implicit fallback —
absence of a gate is itself denial.
"""

from __future__ import annotations

from typing import Iterable

from .interfaces import GovernanceContext, GovernanceDecision, GovernanceProtocol
from .steps import ToolCall


class DenyAllGovernance:
    """Fail-closed default.

    Refuses every tool call. The presence of this policy on a freshly
    constructed ``MultiStepLoop`` is intentional: callers must make the
    explicit choice to permit anything.
    """

    policy_id = "deny-all-default"

    def decide(self, call: ToolCall, context: GovernanceContext) -> GovernanceDecision:
        return GovernanceDecision(
            call_id=call.id,
            tool_name=call.name,
            verdict="deny",
            reason="no governance policy configured (default deny-all)",
            policy=self.policy_id,
        )


class AllowAllGovernance:
    """Permissive policy — explicit opt-in only.

    Suitable for trusted environments (CI fixtures, single-user development
    runs). Never the default; never silently substituted.
    """

    policy_id = "allow-all"

    def decide(self, call: ToolCall, context: GovernanceContext) -> GovernanceDecision:
        return GovernanceDecision(
            call_id=call.id,
            tool_name=call.name,
            verdict="allow",
            reason="permissive policy",
            policy=self.policy_id,
        )


class AllowListGovernance:
    """Permits a fixed set of tool names; denies everything else.

    The most common useful policy. Build it with the set of tools you
    intend to expose, and the gate refuses any call to a tool not in the
    allowlist.
    """

    policy_id = "allow-list"

    def __init__(self, allowed: Iterable[str]) -> None:
        self._allowed = frozenset(allowed)

    def decide(self, call: ToolCall, context: GovernanceContext) -> GovernanceDecision:
        if call.name in self._allowed:
            return GovernanceDecision(
                call_id=call.id,
                tool_name=call.name,
                verdict="allow",
                reason="tool on allowlist",
                policy=self.policy_id,
            )
        return GovernanceDecision(
            call_id=call.id,
            tool_name=call.name,
            verdict="deny",
            reason=f"tool not on allowlist: {call.name}",
            policy=self.policy_id,
        )


class GovernanceGate:
    """Thin wrapper that runs a policy over a batch of tool calls.

    The loop never calls the policy directly — it goes through the gate so
    decisions are surfaced as a single batch result.
    """

    def __init__(self, policy: GovernanceProtocol) -> None:
        self._policy = policy

    def evaluate(self, calls: list[ToolCall], context: GovernanceContext) -> list[GovernanceDecision]:
        return [self._policy.decide(call, context) for call in calls]

    @property
    def policy(self) -> GovernanceProtocol:
        return self._policy
