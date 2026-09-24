"""Fold an agent's spec and the tenant policy into one compiled document.

Compilation is where declarations become enforceable. It is also where contradictions
surface: an agent denying a tool its own permission grants, or denying a baseline tool it
needs to report work. Those are reported as warnings at build time, because the
alternative is discovering them when a task runs and never closes.

The compiled document is deliberately a plain dictionary of tool names — no NOVA types,
no indirection — so the enforcement point inside the runtime can read it with nothing but
the standard library.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from nova.policy.decide import POLICY_SCHEMA_VERSION
from nova.policy.model import PolicySpec
from nova.spec import AgentSpec

#: Fields a runtime adapter injects into a compiled document at materialization time.
#: They are plumbing, not meaning: moving the audit log does not change what an agent is
#: allowed to do, so they are excluded from the agent's identity.
RUNTIME_INJECTED_KEYS = frozenset({"audit_log", "tenant_id", "tenant_monthly_budget_usd"})

#: The tool NOVA installs for an agent granted knowledge sources. Named here rather
#: than imported from the adapter: the compiler is runtime-agnostic, and this is the
#: NOVA-side name of the capability, not one runtime's implementation of it.
KNOWLEDGE_TOOL = "knowledge_search"


@dataclass(frozen=True)
class CompiledPolicy:
    """One agent's enforceable policy, plus what compiling it revealed."""

    agent_id: str
    document: dict[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_allowlist(self) -> bool:
        return bool(self.document.get("allow"))

    def to_dict(self) -> dict[str, Any]:
        return dict(self.document)


def compile_policy(
    spec: AgentSpec,
    policy: PolicySpec,
    *,
    toolset_tools: Optional[Mapping[str, Sequence[str]]] = None,
) -> CompiledPolicy:
    """Compile ``spec`` against ``policy``.

    An agent that declares neither permissions nor an allow-list runs without an
    allow-list, so Phase 1 bundles keep working unchanged: adding governance must not
    silently restrict agents that predate it.

    ``toolset_tools`` maps each toolset the agent declares to the tools it contains, as the
    runtime resolves them (:meth:`AgentRuntime.compile_policy` supplies it). A declared
    toolset is a grant of its tools: without this, ``toolsets: [web, file]`` under
    ``unlisted_tool: deny`` let the runtime pin the worker to those tools while the policy
    hook refused every one of them.
    """
    warnings: list[str] = []

    denied = set(spec.tools.deny)
    baseline = set(policy.baseline_tools)

    # Tools this agent is granted, from its permissions and its explicit allow list.
    granted = policy.tools_for_permissions(spec.permissions) | set(spec.tools.allow)

    if spec.tools.toolsets:
        if toolset_tools is None:
            if policy.unlisted_tool == "deny":
                warnings.append(
                    f"toolset(s) {', '.join(spec.tools.toolsets)} could not be resolved to "
                    "tools here, so under unlisted_tool: deny their tools are not granted"
                )
        else:
            unknown_toolsets = [name for name in spec.tools.toolsets if name not in toolset_tools]
            if unknown_toolsets:
                warnings.append(
                    f"toolset(s) {', '.join(sorted(unknown_toolsets))} are not defined by the "
                    "runtime, so they grant nothing"
                )
            for name in spec.tools.toolsets:
                granted.update(toolset_tools.get(name, ()))

    # Declaring knowledge sources IS the authorization to search them. Requiring a second,
    # separate tool permission would mean a tenant grants an agent a corpus, NOVA installs
    # the search tool, and NOVA's own policy then refuses every call to it — which is
    # exactly what happened in a live worker before this line existed. A trap that only
    # fires under an allow-list, where the failure reads as a knowledge bug rather than a
    # policy one. An explicit deny still wins below, so a tenant can revoke it.
    if spec.knowledge.sources:
        granted.add(KNOWLEDGE_TOOL)

    unknown_permissions = [name for name in spec.permissions if name not in policy.permissions]
    if unknown_permissions:
        warnings.append(
            f"permission(s) {', '.join(sorted(unknown_permissions))} are not defined in the "
            "tenant policy, so they grant nothing"
        )

    # Actions this agent must escalate: the tenant default, plus its own additions.
    approval_actions: dict[str, list[str]] = {}
    for name, action in policy.actions.items():
        required = action.requires_approval or name in spec.approval.required_for
        if required:
            approval_actions[name] = sorted(action.tools)

    unknown_actions = [
        name for name in spec.approval.required_for if name not in policy.actions
    ]
    if unknown_actions:
        warnings.append(
            f"approval is required for {', '.join(sorted(unknown_actions))}, which the tenant "
            "policy does not define as an action — nothing will trigger it. Define the action "
            "and the tools that perform it"
        )

    # Contradictions worth surfacing before they reach a running agent.
    denied_baseline = sorted(denied & baseline)
    if denied_baseline:
        warnings.append(
            f"denies baseline tool(s) {', '.join(denied_baseline)} — this agent may be unable "
            "to report its own task outcome, leaving work that runs and never closes"
        )

    denied_granted = sorted(denied & granted)
    if denied_granted:
        warnings.append(
            f"denies {', '.join(denied_granted)} which its own permissions grant — deny wins, "
            "so the grant has no effect"
        )

    denied_approval = sorted(
        denied.intersection(*[set(t) for t in approval_actions.values()])
        if approval_actions
        else set()
    )
    if denied_approval:
        warnings.append(
            f"denies {', '.join(denied_approval)}, which is also marked for approval — it will "
            "be denied outright and never reach a human"
        )

    document: dict[str, Any] = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "agent_id": spec.id,
        "deny": sorted(denied),
        "baseline": sorted(baseline),
        "approval_actions": {name: approval_actions[name] for name in sorted(approval_actions)},
        "allow": sorted(granted),
        "unlisted_tool": policy.unlisted_tool,
        # HARD BOUNDARY (nova/policy/limits.py). Absent or zero means no ceiling.
        "max_tool_calls_per_run": spec.limits.max_tool_calls_per_run or 0,
        # The agents this one may hand work to. Enforced in the worker twice: on the tool
        # that creates work for another agent, and on the receiving side, where a task no
        # declared delegation put there is refused (decide.decide_acceptance).
        "may_assign_to": sorted(spec.delegation.may_assign_to),
        # HARD BOUNDARY at task start and at every tool call (decide.decide_budget). Absent
        # or zero means no budget. The tenant-wide figure is injected at apply time.
        "monthly_budget_usd": spec.limits.monthly_budget_usd or 0,
    }
    return CompiledPolicy(agent_id=spec.id, document=document, warnings=tuple(warnings))


def policy_identity(document: dict[str, Any]) -> dict[str, Any]:
    """The part of a compiled document that defines what an agent may do."""
    return {key: value for key, value in document.items() if key not in RUNTIME_INJECTED_KEYS}


def agent_digest(
    spec: AgentSpec,
    policy: Optional[CompiledPolicy],
    knowledge: Optional[Mapping[str, Any]] = None,
    deployment: Optional[Mapping[str, Any]] = None,
) -> str:
    """Stable identity of an agent as materialized: its spec, its policy, its knowledge grant.

    The single definition, used by the materializer when it writes provenance and by the
    control plane when it reports drift. Two definitions would disagree, and the
    disagreement would show as an agent permanently "out of sync".

    The knowledge grant belongs here for a reason that is easy to miss: an agent's spec names
    its corpora by id, but the corpus *titles and descriptions* come from ``knowledge.yaml``
    and are compiled into the tool description the model reads. Editing a title with the
    spec untouched changes what the model sees, so it has to change the digest too — or the
    next apply would report the agent up to date while its tool still describes a corpus by
    its old name. ``index_path`` and ``audit_log`` are stripped for the same reason
    :data:`RUNTIME_INJECTED_KEYS` are: where the files live is deployment, not identity.

    ``deployment`` is the tenant-level part of the agent's runtime configuration — the
    resolved provider and ``runtime_config`` from ``deployment.yaml`` — for the same reason:
    switching the tenant's model with every agent spec untouched changes which model the
    agent calls. Left out, an apply reports every agent ``unchanged`` and the profiles keep
    the old model; that is how a move to a Bedrock inference profile never reached the
    runtime. Empty for a tenant that declares none, so those digests do not move.
    """
    if policy is None and not knowledge and not deployment:
        return spec.digest()
    payload: dict[str, Any] = {"spec": spec.to_dict()}
    if policy is not None:
        payload["policy"] = policy_identity(policy.document)
    if knowledge:
        payload["knowledge"] = {
            key: value
            for key, value in knowledge.items()
            if key not in RUNTIME_INJECTED_KEYS and key not in ("index_path",)
        }
    if deployment:
        payload["deployment"] = dict(deployment)
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
