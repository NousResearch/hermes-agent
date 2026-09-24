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


# -- delegation --------------------------------------------------------------
#
# Who may put work on whose queue. An agent declares ``delegation.may_assign_to``; these two
# decisions are that declaration enforced, once where work is created and once where it is
# received. Both are needed: the creating side sees only the agent's own tools, and the
# receiving side is the one place every route converges — the tool, a shell-out to the CLI,
# and the runtime's own decomposer, which routes a triage card's children by asking a model.

#: The tool through which an agent creates work, and the argument naming who does it.
ASSIGNING_TOOL = "kanban_create"
ASSIGNEE_ARG = "assignee"

#: What NOVA's own supervisor writes as a task's creator. Its routing is checked against
#: the same declarations *before* it writes (nova/supervisor/route.py), so it is trusted here.
SUPERVISOR = "nova-supervisor"

#: The tools a refused task may still use: enough to read the card and say why it is being
#: refused, and nothing that does or finishes the work.
REFUSED_TASK_TOOLS = frozenset({"kanban_block", "kanban_show", "kanban_comment", "kanban_heartbeat"})


def _may_assign_to(policy: Optional[Mapping[str, Any]]) -> set[str]:
    return set((policy or {}).get("may_assign_to") or ())


def decide_assignment(policy: Optional[Mapping[str, Any]], assignee: Any) -> Decision:
    """May this agent create work for ``assignee``? Its own queue always; others if declared."""
    self_id = (policy or {}).get("agent_id") or ""
    target = assignee.strip() if isinstance(assignee, str) else ""
    if not target or target == self_id or target in _may_assign_to(policy):
        return Decision(ALLOW, f"{self_id} may assign work to {target or 'itself'}",
                        tool=ASSIGNING_TOOL, rule="delegation-declared")
    declared = ", ".join(sorted(_may_assign_to(policy))) or "no other agent"
    return Decision(
        DENY,
        f"{self_id} may not hand work to {target}: its delegation declares {declared}. "
        f"Add {target} to delegation.may_assign_to in {self_id}'s agent file to allow it",
        tool=ASSIGNING_TOOL,
        rule="delegation-not-declared",
    )


def decide_acceptance(
    policy: Optional[Mapping[str, Any]],
    *,
    authorizer: str,
    authorizer_policy: Optional[Mapping[str, Any]],
    via_decomposer: bool,
) -> Decision:
    """May this agent work a task that ``authorizer`` put on its queue?

    ``authorizer`` is whoever is answerable for the assignment: the task's creator, or —
    for a child the runtime's decomposer routed — whoever owns the card it was split from.
    ``authorizer_policy`` is that authorizer's compiled policy when it is a NOVA agent,
    else None (a person, or a process that is not an agent).
    """
    self_id = (policy or {}).get("agent_id") or ""
    if authorizer == SUPERVISOR:
        return Decision(ALLOW, "routed by the NOVA supervisor, which checked delegation before creating it",
                        rule="supervisor-routed")
    if authorizer == self_id:
        return Decision(ALLOW, f"{self_id} assigned this work to itself", rule="self-assigned")
    if authorizer_policy is not None:
        if self_id in _may_assign_to(authorizer_policy):
            return Decision(ALLOW, f"{authorizer} declares it may hand work to {self_id}",
                            rule="delegation-declared")
        return Decision(
            DENY,
            f"{authorizer} handed this task to {self_id}, but {authorizer}'s delegation does not "
            f"include {self_id}. Add {self_id} to delegation.may_assign_to in {authorizer}'s agent "
            "file, or reassign the task",
            rule="delegation-not-declared",
        )
    if via_decomposer:
        return Decision(
            DENY,
            f"the runtime's decomposer routed this task to {self_id} from a card no NOVA agent owns "
            f"(owner: {authorizer or 'nobody'}), so no delegation declaration covers the choice. "
            "Give the card an agent as its assignee before it is decomposed, or submit the work "
            "as a NOVA objective",
            rule="decomposer-unowned",
        )
    return Decision(ALLOW, f"assigned by {authorizer or 'an operator'}, a person rather than an agent",
                    rule="operator-directed")


# -- file access -------------------------------------------------------------
#
# The runtime's file tools can read and write anywhere its OS user can, and its own guard
# says so: "defense-in-depth, NOT a security boundary" (agent/file_safety.py). It keeps an
# agent out of credential files, not out of *other agents'* conversation history, the
# shared board, the audit log — or its own compiled policy, which a write_file would
# replace. Found on a live run: a worker searching for refund data listed every state.db
# on the host. So file tools are held to an allow-list: the task's workspace, read and
# write; the agent's own profile, read only; nothing else.

#: Tool -> whether it writes. A tool not listed here is not a file tool.
FILE_TOOLS = {"read_file": False, "search_files": False, "write_file": True, "patch": True}


def _under(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root + "/")


def decide_paths(
    tool: str,
    paths: "list[str]",
    *,
    writable_roots: "list[str]",
    readable_roots: "list[str]",
) -> Decision:
    """May ``tool`` touch every one of ``paths``? Paths and roots arrive fully resolved."""
    writes = FILE_TOOLS.get(tool, False)
    allowed = list(writable_roots) + ([] if writes else list(readable_roots))
    for path in paths:
        if not any(_under(path, root) for root in allowed if root):
            where = "the task's workspace" + ("" if writes else " or this agent's own profile")
            return Decision(
                DENY,
                f"{tool} may only {'write' if writes else 'read'} inside {where}; {path} is "
                "outside it. Work on copies inside the workspace, or use the knowledge tool "
                "for documents",
                tool=tool,
                rule="file-outside-scope",
            )
    return Decision(ALLOW, f"{tool} stays inside its scope", tool=tool, rule="file-in-scope")


# -- monthly budgets ---------------------------------------------------------
#
# A spend ceiling the runtime cannot enforce on model calls — no plugin can veto one — but
# can enforce where NOVA has a veto: a task is refused before it starts, and every tool
# call but the ones that close the task is refused, once month-to-date spend reaches the
# agent's budget or the tenant's. A reply already in flight can land; the overshoot is at
# most that reply, and the numbers are the runtime's own cost estimate, not an invoice.


def decide_budget(
    policy: Optional[Mapping[str, Any]], *, agent_spend: float, tenant_spend: float
) -> Decision:
    """Is there budget left this month? ``*_spend`` are month-to-date USD estimates."""
    agent_id = (policy or {}).get("agent_id") or "this agent"
    for scope, limit, spent in (
        (f"{agent_id}'s", (policy or {}).get("monthly_budget_usd") or 0, agent_spend),
        ("the tenant's", (policy or {}).get("tenant_monthly_budget_usd") or 0, tenant_spend),
    ):
        if isinstance(limit, (int, float)) and limit > 0 and spent >= limit:
            return Decision(
                DENY,
                f"{scope} monthly budget of ${limit:,.2f} is spent (${spent:,.2f} so far this "
                "month). New work and tool calls stop until the month turns or an "
                "administrator raises the budget",
                rule="budget-exhausted",
            )
    return Decision(ALLOW, "within budget", rule="within-budget")
