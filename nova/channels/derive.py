"""Per-channel approval, expressed as something this runtime can actually enforce.

The requirement is easy to state — *"a refund needs a human when it comes in on WhatsApp,
but not on the internal Slack"* — and the obvious implementation does not work here. The
policy hook the runtime calls before a tool runs receives ``tool_name``, ``args``,
``task_id``, ``session_id``, ``turn_id`` and ``tool_call_id``. **It is not told which channel
the turn arrived on.** The session *key* encodes the platform; the session *id* the hook
receives does not. So a plugin cannot branch on the channel, and a per-channel rule
implemented inside the hook would be a rule that never fires — enforcement in name only,
which this project has a standing rule against shipping.

What the runtime *can* do is enforce a different policy per profile, and it already does:
one compiled policy per agent, read by a fail-closed hook. And a profile is precisely what
the channel layer already routes to.

So a channel that tightens approval derives an agent. ``customer-support`` reached over
``acme-support-telegram`` becomes ``customer-support__acme-support-telegram``: the same
agent — same instructions, same tools, same knowledge — with the channel's extra approval
requirements compiled into its own policy, and the channel's routes pointed at it. The
enforcement is the existing enforcement. Nothing new has to be trusted.

**The cost is real and is not hidden.** A derived agent is a separate profile, so it has its
own session and memory namespace: the same person talking to "the support agent" on Telegram
and in a ticket is talking to two profiles, and they do not share conversation history. For a
channel strict enough to need extra approvals that separation is usually right — an external
conversation and an internal one are different trust contexts — but it is a consequence a
customer should meet in a document rather than discover in a transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

from nova.errors import SpecError

#: Separator between the base agent and the channel it was derived for. Double underscore
#: because the runtime's profile grammar allows it and a hand-written agent id is very
#: unlikely to contain one, so a derived profile is recognisable on sight and in a log.
DERIVED_SEPARATOR = "__"

#: The runtime's own profile grammar (``hermes_cli/profiles.py``), restated so NOVA can
#: refuse an over-long derived name at declaration time rather than at materialization.
PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class DerivedAgent:
    """One agent as reached over one channel, with that channel's approvals applied."""

    id: str
    base_agent: str
    channel_id: str
    #: Actions this variant escalates that the base agent does not. Recorded so the
    #: dashboard can say *why* the variant exists rather than showing a second agent with
    #: no explanation.
    added_approvals: tuple[str, ...]
    #: Approvals the channel asked for that this agent cannot perform, and which were
    #: therefore dropped. Carried so the reason is reportable rather than invisible.
    dropped_approvals: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        out = {
            "id": self.id,
            "base_agent": self.base_agent,
            "channel_id": self.channel_id,
            "added_approvals": list(self.added_approvals),
        }
        if self.dropped_approvals:
            out["dropped_approvals"] = list(self.dropped_approvals)
        return out


def derived_id(base_agent: str, channel_id: str) -> str:
    """The profile name for one agent reached over one channel."""
    name = f"{base_agent}{DERIVED_SEPARATOR}{channel_id}"
    if not PROFILE_NAME.match(name):
        raise SpecError(
            f"the derived agent name {name!r} is not a valid profile id "
            f"([a-z0-9][a-z0-9_-]{{0,63}}). Shorten the channel id or the agent id — a "
            f"channel that tightens approval creates one profile per granted agent"
        )
    return name


def additional_approvals(
    channel_required: Sequence[str], agent_required: Iterable[str], tenant_always: Iterable[str]
) -> tuple[str, ...]:
    """What this channel adds beyond what the agent already escalates everywhere.

    Empty means the channel asks for nothing the agent does not already do, and no variant
    is derived — deriving one anyway would double a profile to change nothing, and an
    operator reading the agent list deserves every entry in it to have a reason.
    """
    already = set(agent_required) | set(tenant_always)
    return tuple(sorted(set(channel_required) - already))


def reachable_approvals(
    added: Sequence[str], actions: Mapping[str, object], granted: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split added approvals into those the agent can actually perform, and those it cannot.

    **This is a security check, not tidiness.** ``nova/policy/decide.py`` tests approval
    actions *before* the allow-list, so a tool that is not granted but is named by an
    approval action resolves to ``require_approval`` rather than ``deny``. Declaring an
    approval for an action an agent cannot perform would therefore *grant* it — behind a
    human gate, but granted — and a channel declaration that widens an agent's reach is
    exactly what ``ChannelApproval`` promises is impossible.

    So an unreachable approval is dropped and reported, never compiled. Dropped rather than
    refused because one channel usually grants several agents and only some of them hold the
    tools: refusing would make the common case unwritable.
    """
    reachable: list[str] = []
    unreachable: list[str] = []
    allowed = set(granted)
    for name in added:
        tools = set(getattr(actions.get(name), "tools", ()) or ())
        (reachable if tools & allowed else unreachable).append(name)
    return tuple(reachable), tuple(unreachable)


def plan_derivations(bundle) -> tuple[DerivedAgent, ...]:
    """Every (agent, channel) pair that needs its own profile, in a stable order."""
    from nova.policy import compile_policy

    policy = bundle.policy
    tenant_always = (
        {name for name, action in policy.actions.items() if action.requires_approval}
        if policy is not None
        else set()
    )
    by_id = {spec.id: spec for spec in bundle.agents}

    out: list[DerivedAgent] = []
    for channel in bundle.channels:
        required = channel.approval.required_for
        if not required or not channel.enabled:
            continue
        if policy is None:
            raise SpecError(
                f"channel {channel.id!r} requires approval for "
                f"{', '.join(required)}, but this bundle declares no policy — there is "
                f"nothing to escalate and no enforcement plugin would be installed",
                field=f"channels[{channel.id}].approval.required_for",
                source=channel.source,
            )
        unknown = sorted(set(required) - set(policy.actions))
        if unknown:
            raise SpecError(
                f"requires approval for {', '.join(unknown)}, which the tenant policy does "
                f"not define as an action — nothing would trigger it. Define the action and "
                f"the tools that perform it in policy.yaml",
                field=f"channels[{channel.id}].approval.required_for",
                source=channel.source,
            )

        for agent_id in channel.allowed_agents:
            spec = by_id.get(agent_id)
            if spec is None:  # check_agents_exist already refuses this; belt and braces
                continue
            added = additional_approvals(required, spec.approval.required_for, tenant_always)
            if not added:
                continue
            granted = compile_policy(spec, policy).document.get("allow") or ()
            added, dropped = reachable_approvals(added, policy.actions, granted)
            if not added:
                # Everything the channel asked for is unreachable for this agent. No
                # variant, because a profile whose only difference would widen access is
                # worse than no profile at all.
                continue
            out.append(
                DerivedAgent(
                    id=derived_id(agent_id, channel.id),
                    base_agent=agent_id,
                    channel_id=channel.id,
                    added_approvals=added,
                    dropped_approvals=dropped,
                )
            )
    return tuple(out)


def derive_specs(bundle) -> tuple:
    """The :class:`~nova.spec.agent.AgentSpec` objects for every derived variant.

    Built with ``dataclasses.replace`` off the base agent, so a variant inherits everything
    — instructions, tools, knowledge, limits, delegation — by construction. A variant that
    could drift from its base would be a second agent wearing the first one's name, and the
    drift would show up as a customer getting different answers on different channels.
    """
    from nova.spec.agent import ApprovalSpec

    by_id = {spec.id: spec for spec in bundle.agents}
    specs = []
    for derived in plan_derivations(bundle):
        base = by_id[derived.base_agent]
        specs.append(
            replace(
                base,
                id=derived.id,
                approval=ApprovalSpec(
                    required_for=tuple(
                        sorted(set(base.approval.required_for) | set(derived.added_approvals))
                    )
                ),
            )
        )
    return tuple(specs)


def route_target(
    derivations: Sequence[DerivedAgent], channel_id: str, agent_id: str
) -> str:
    """Where a route should actually point: the derived variant when one exists.

    The indirection lives here rather than in the compiler so that "which profile does this
    conversation reach" has exactly one answer, and both the compiler and the dashboard read
    it from the same function.
    """
    for derived in derivations:
        if derived.channel_id == channel_id and derived.base_agent == agent_id:
            return derived.id
    return agent_id


def display_names(derivations: Sequence[DerivedAgent]) -> Mapping[str, str]:
    """Derived id -> how it should read in a dashboard, so it is never a mystery profile."""
    return {
        d.id: f"{d.base_agent} (on {d.channel_id})" for d in derivations
    }
