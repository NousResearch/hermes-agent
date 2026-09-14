"""The policy decision. One function, used in two places.

This module is **copied verbatim into the runtime** by the runtime adapter, so the code
that explains a decision in the control plane and the code that enforces it inside a
worker are the same code. Two implementations of a security decision will eventually
disagree, and the disagreement will be discovered by a customer.

It therefore depends on nothing but the standard library, and takes the compiled policy
as a plain dictionary rather than a NOVA type.

**Order matters, and deny is strongest.** An explicit denial beats everything, including
the worker baseline: a customer who writes ``deny: [terminal]`` means it, and if that
breaks a worker the compiler warns at build time rather than the runtime overriding the
customer at execution time.

The tool-call ceiling sits *after* the baseline check, so an agent that has exhausted its
budget can still close its own task. A ceiling that silenced task reporting would produce
work that runs and never completes — the same failure that kept positive tool scoping out
of Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

ALLOW = "allow"
DENY = "deny"
REQUIRE_APPROVAL = "require_approval"

#: Schema version of the compiled document. The enforcement point refuses a document it
#: does not understand rather than guessing — a policy it cannot read must not silently
#: become "allow everything".
POLICY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Decision:
    """What policy says about one tool call, and why.

    ``reason`` is written for a human reading an audit trail during an incident or a
    security review, not for a developer reading a stack trace.
    """

    effect: str
    reason: str
    tool: str = ""
    action: str = ""
    rule: str = ""

    @property
    def allowed(self) -> bool:
        return self.effect == ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "effect": self.effect,
            "reason": self.reason,
            "tool": self.tool,
            "action": self.action,
            "rule": self.rule,
        }


def decide(
    policy: Optional[Mapping[str, Any]], tool: str, *, calls_used: int = 0
) -> Decision:
    """Resolve one tool call against a compiled policy document.

    ``calls_used`` is how many budget-consuming calls this run has already made. The
    counter lives with the caller so this function stays pure and testable; the
    enforcement point owns the count.

    A missing or unreadable policy is not an implicit allow-all: an agent materialized
    without a policy has no restrictions to apply, which is different from an agent whose
    policy failed to load. The first is ``allow`` with a reason saying so; the second is
    ``deny``, because a governance control that fails open is not a control.
    """
    tool = (tool or "").strip()
    if not tool:
        return Decision(DENY, "no tool name given", rule="malformed-request")

    if policy is None:
        return Decision(
            DENY,
            "no policy document was loaded; refusing rather than failing open",
            tool=tool,
            rule="policy-missing",
        )

    version = policy.get("schema_version")
    if version != POLICY_SCHEMA_VERSION:
        return Decision(
            DENY,
            f"policy document schema {version!r} is not supported by this enforcement "
            f"point (expected {POLICY_SCHEMA_VERSION}); refusing rather than guessing",
            tool=tool,
            rule="policy-unsupported",
        )

    denied = set(policy.get("deny") or ())
    if tool in denied:
        return Decision(
            DENY, f"{tool} is explicitly denied to this agent", tool=tool, rule="explicit-deny"
        )

    if tool in set(policy.get("baseline") or ()):
        return Decision(
            ALLOW,
            f"{tool} is part of the baseline every agent needs to report its own work",
            tool=tool,
            rule="baseline",
        )

    # HARD BOUNDARY: the per-run tool-call ceiling. Checked after the baseline so an
    # agent out of budget can still report its outcome, and before approval so an
    # exhausted agent does not queue work for a human it can no longer perform.
    ceiling = policy.get("max_tool_calls_per_run")
    if isinstance(ceiling, int) and ceiling > 0 and calls_used >= ceiling:
        return Decision(
            DENY,
            f"this run has already made {calls_used} tool calls, reaching its ceiling of "
            f"{ceiling}; only task-reporting tools remain available",
            tool=tool,
            rule="tool-call-ceiling",
        )

    approval_actions = policy.get("approval_actions") or {}
    for action_name, tools in approval_actions.items():
        if tool in set(tools or ()):
            return Decision(
                REQUIRE_APPROVAL,
                f"{tool} performs '{action_name}', which this deployment requires a human "
                "to approve",
                tool=tool,
                action=action_name,
                rule="approval-required",
            )

    allowed = policy.get("allow")
    if allowed:  # an allow-list is in force
        if tool in set(allowed):
            return Decision(
                ALLOW, f"{tool} is granted to this agent", tool=tool, rule="explicit-allow"
            )
        return Decision(
            DENY,
            f"{tool} is not granted to this agent, and this agent runs under an allow-list",
            tool=tool,
            rule="not-in-allowlist",
        )

    default = policy.get("unlisted_tool") or ALLOW
    if default == DENY:
        return Decision(
            DENY,
            f"{tool} matches no rule and this deployment denies unlisted tools",
            tool=tool,
            rule="default-deny",
        )
    return Decision(
        ALLOW,
        f"{tool} matches no rule and this deployment allows unlisted tools",
        tool=tool,
        rule="default-allow",
    )
