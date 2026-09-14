"""Turn a declared automation into a runtime cron job — or refuse to.

This is the chokepoint Phase 11 said had to exist before create could be offered. Every
automation, whether it came from a bundle file or over the control API, passes through
:func:`compile_automation`. There is no second path to the runtime's scheduler.

**What is enforced here, and what is enforced elsewhere.** Being precise about this
matters more than the checks themselves:

* *Here, at compile time:* the automation names a real agent; its declared permissions,
  corpora and channels are a **subset** of what that agent was already granted; the
  schedule parses; the objective is present and bounded.
* *At runtime, by Hermes:* what the agent may actually do. The automation's work executes
  inside the owning agent's profile, so the agent's compiled ``nova-policy.json`` and the
  fail-closed ``pre_tool_call`` hook are the real ceiling.

The consequence is worth stating plainly rather than implying otherwise: declaring fewer
permissions on an automation than its agent holds is a **declaration, not a narrowing**.
The runtime will still allow the agent everything the agent may do. The subset check
stops an automation being a privilege-escalation path — it cannot reach past its agent —
but it does not sandbox an automation below its agent. Doing that would need a derived
profile per automation, the way Phase 10 derives one per channel grant, and that is a
separate piece of work.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from nova.errors import SpecError
from nova.spec.automation import AutomationSpec


@dataclass(frozen=True)
class CompiledAutomation:
    """An automation validated against its tenant and ready for the runtime.

    ``job_kwargs`` is exactly what :func:`cron.jobs.create_job` will be called with —
    nothing is added at the call site, so what was reviewed is what runs.
    """

    spec: AutomationSpec
    agent_id: str
    tenant_id: str
    digest: str
    job_kwargs: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "digest": self.digest,
        }


def _digest(spec: AutomationSpec, tenant_id: str) -> str:
    """A stable fingerprint of the declaration.

    Recorded with the automation so a later reader can tell whether the thing running is
    the thing that was reviewed. Tenant-salted so identical declarations in two tenants
    are not the same artifact.
    """
    payload = json.dumps(
        {"tenant": tenant_id, **spec.to_dict()}, sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _granted_permissions(agent) -> set[str]:
    return set(getattr(agent, "permissions", ()) or ())


def _granted_knowledge(agent) -> set[str]:
    knowledge = getattr(agent, "knowledge", None)
    return set(getattr(knowledge, "sources", ()) or ())


def _channels_granting(bundle, agent_id: str) -> set[str]:
    """Channel ids whose ``allowed_agents`` includes this agent.

    Read from the channel's grant rather than from any list on the agent: the grant is
    declared on the connection, and Phase 9 made that the single place it lives.
    """
    granting: set[str] = set()
    for channel in getattr(bundle, "channels", ()) or ():
        if agent_id in (getattr(channel, "allowed_agents", ()) or ()):
            granting.add(getattr(channel, "id", ""))
    return granting - {""}


def _refuse(spec: AutomationSpec, message: str) -> SpecError:
    return SpecError(f"automation {spec.id!r}: {message}", source=spec.source)


#: Signature of a schedule validator: takes the phrase, raises SpecError if the runtime
#: would not accept it.
ScheduleValidator = Callable[[str], Any]


def compile_automation(
    spec: AutomationSpec, bundle, *, validate_schedule: Optional[ScheduleValidator] = None
) -> CompiledAutomation:
    """Validate ``spec`` against its tenant and produce the runtime call.

    Raises :class:`~nova.errors.SpecError` naming the automation and the field, in the
    same shape as every other spec failure, so a bad automation fails a bundle load the
    way a bad agent does.

    ``validate_schedule`` is **injected**, not imported. NOVA does not implement a second
    schedule grammar — the runtime's parser is the one that decides when things run — but
    this package may not name the runtime either (``tests/platform/test_boundaries.py``
    holds that line, and caught the first version of this file doing it). So the adapter
    supplies its parser and the compiler calls it. Omitted means the schedule is checked
    later, at the adapter boundary, rather than not at all.
    """
    agents = {a.id: a for a in getattr(bundle, "agents", ()) or ()}
    agent = agents.get(spec.agent)
    if agent is None:
        known = ", ".join(sorted(agents)) or "(none)"
        raise _refuse(spec, f"names agent {spec.agent!r}, which this tenant does not have. Declared: {known}")
    if not getattr(agent, "enabled", True):
        raise _refuse(
            spec,
            f"is owned by agent {spec.agent!r}, which is disabled. A schedule pointing at "
            "a disabled agent is work that will never run",
        )

    # Permissions: subset of the agent's grant. An automation that could name a
    # permission its agent lacks would be a way to ask for something the tenant never
    # granted, on a timer.
    ungranted = sorted(set(spec.permissions) - _granted_permissions(agent))
    if ungranted:
        raise _refuse(
            spec,
            f"declares permission(s) {', '.join(ungranted)} that agent {spec.agent!r} "
            "does not hold. An automation may only ask for what its agent was already "
            "granted",
        )

    unreadable = sorted(set(spec.knowledge) - _granted_knowledge(agent))
    if unreadable:
        raise _refuse(
            spec,
            f"declares knowledge source(s) {', '.join(unreadable)} that agent "
            f"{spec.agent!r} may not read",
        )

    unreachable = sorted(set(spec.channels) - _channels_granting(bundle, spec.agent))
    if unreachable:
        raise _refuse(
            spec,
            f"declares channel(s) {', '.join(unreachable)} that do not grant agent "
            f"{spec.agent!r}. A channel grant is declared on the connection",
        )

    # Validated with the runtime's own parser when one was supplied, and the result
    # discarded: the runtime parses it again at create time. Doing it here means an
    # invalid schedule fails review rather than landing in the store half-formed.
    if validate_schedule is not None:
        validate_schedule(spec.schedule)

    tenant_id = getattr(getattr(bundle, "organization", None), "tenant_id", "") or ""
    return CompiledAutomation(
        spec=spec,
        agent_id=spec.agent,
        tenant_id=tenant_id,
        digest=_digest(spec, tenant_id),
        job_kwargs=_job_kwargs(spec),
    )


def _job_kwargs(spec: AutomationSpec) -> dict[str, Any]:
    """Exactly what ``cron.jobs.create_job`` is called with.

    ``paused_reason`` is only included when actually pausing: the runtime refuses
    ``paused_reason`` without ``paused=True``, and passing an empty string is not the
    same as passing nothing.
    """
    kwargs: dict[str, Any] = {
        "name": spec.title,
        "prompt": spec.objective,
        "schedule": spec.schedule,
        # Created paused when declared disabled, so a job is never briefly live between
        # being written and being paused.
        "paused": not spec.enabled,
        # Deliver locally. The runtime defaults ``deliver`` to 'origin' whenever an
        # origin is passed, which would route output back through a channel NOVA never
        # reviewed for this automation.
        "deliver": "local",
    }
    if not spec.enabled:
        kwargs["paused_reason"] = "declared disabled"
    return kwargs


def compile_all(
    bundle, *, validate_schedule: Optional[ScheduleValidator] = None
) -> tuple[CompiledAutomation, ...]:
    """Every automation in a bundle, compiled. Raises on the first that will not."""
    return tuple(
        compile_automation(spec, bundle, validate_schedule=validate_schedule)
        for spec in (getattr(bundle, "automations", ()) or ())
    )


def check_automation_references(bundle) -> None:
    """Bundle-load hook: refuse a bundle whose automations do not compile.

    Called from :func:`nova.spec.bundle.load_bundle`, which has no runtime, so the
    schedule phrase is **not** checked here — it is checked where a runtime exists
    (``nova validate``/``apply`` and the control API, both of which pass the adapter's
    parser). Everything that can be decided from the bundle alone is decided here.
    """
    compile_all(bundle)
