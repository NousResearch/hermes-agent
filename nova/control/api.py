"""Route handlers for the Control API.

Pure functions of (path, query) -> :class:`Response`. No HTTP framework, no sockets, no
globals — the transport is a thin adapter in :mod:`nova.control.server`, so the API's
behaviour is tested directly and the transport can be replaced without touching any of
this.

Phase 1 served five read-only routes. Every response is JSON-serialisable data assembled
from the runtime adapter and the tenant bundle; nothing here knows which runtime is
underneath.

Phase 8 added writes, behind their own dispatcher. :meth:`ControlAPI.handle` still serves
only reads and cannot reach a write handler however it is called — the two entry points are
separate functions rather than one function branching on a method string, because a
branch is a thing somebody eventually gets the wrong way round.

Every write takes a :class:`~nova.control.auth.Principal`. Not a name, a principal: the
route is checked against the caller's role *here*, not only in the transport, so a second
transport cannot forget to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from nova import __version__
from nova.agents import AGENT_FIELDS
from nova.audit import AuditLog, new_correlation_id
from nova.errors import NovaError
from nova.policy import agent_digest, compile_policy, decide
from nova.policy.limits import ENFORCING_CLASSES
from nova.runtime.base import AgentRuntime
from nova.spec import TenantBundle

API_PREFIX = "/platform/v1"


@dataclass(frozen=True)
class Response:
    """One API response: an HTTP status and a JSON-serialisable body."""

    status: int
    body: Any
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


def _write_route(tail: str) -> Optional[str]:
    """The declared write route a concrete path belongs to, or None.

    Concrete paths carry an identifier (``/work/abc123/decide``); the permission table is
    keyed on the shape (``/work/decide``). Matching is exact on both ends rather than by
    prefix, so a path that merely starts the right way cannot borrow the permission.
    """
    parts = [part for part in tail.split("/") if part]
    # Item-level: /work/<id>/decide, /objectives/<id>/submit
    if len(parts) == 3 and parts[0] in ("work", "objectives", "automations"):
        return f"/{parts[0]}/{parts[2]}"
    # Collection-level: /channels/apply. Applying channels is one act over the whole
    # declaration — routes are compiled together, and a per-connection apply would leave
    # the runtime holding half of a routing table.
    if len(parts) == 2 and parts[0] == "channels":
        return f"/{parts[0]}/{parts[1]}"
    # Collection-level create: POST /automations. Declaring a new governed object is an
    # act on the collection, not on a member that does not exist yet.
    if parts == ["automations"]:
        return "/automations/create"
    if parts == ["agents"]:
        return "/agents/create"
    # /agents/<id>/<action>. Kept item-level and enumerated rather than matched by prefix,
    # so a new action has to be added here before it can be called.
    if len(parts) == 3 and parts[0] == "agents" and parts[2] in AGENT_ACTIONS:
        return f"/agents/{parts[2]}"
    return None


#: What an administrator may do to a declared agent.
#:
#: ``soul`` rewrites the persona. ``archive`` and ``restore`` flip ``enabled`` and are
#: reversible; ``delete`` removes the declaration and is not. They are separate routes
#: rather than one "update" taking a field name, so the audit log records which act
#: happened rather than which key changed.
#: Ceiling on a persona. Matches the automation objective cap: both are text the model
#: sees on every turn, and an unbounded one is a per-turn cost nobody reviewed.
MAX_INSTRUCTIONS_CHARS = 20000

AGENT_ACTIONS = ("update", "soul", "duplicate", "archive", "restore", "delete")


#: What an administrator may do to an automation the runtime already holds.
#:
#: Pause and resume move an existing schedule between two states the runtime defines.
#: Delete removes it. Creating is NOT here: it is a collection-level act that must go
#: through the compiler (``/automations/create``), because an automation created from a
#: raw prompt would be an instruction no policy reviewed, running on a timer.
AUTOMATION_ACTIONS = ("pause", "resume", "delete", "update")


def _error(status: int, message: str, **extra: Any) -> Response:
    """A problem the caller can act on, in one consistent shape."""
    return Response(status, {"error": {"status": status, "message": message, **extra}})


class ControlAPI:
    """Read-only control plane over one tenant's deployment.

    Constructed with the declared bundle and a runtime adapter. The bundle is the
    declared truth and the runtime is the applied truth; where they differ the API says
    so rather than presenting one as the other.
    """

    def __init__(
        self,
        bundle: TenantBundle,
        runtime: AgentRuntime,
        *,
        audit: Optional[AuditLog] = None,
    ) -> None:
        self.bundle = bundle
        self.runtime = runtime
        self.audit = audit

    # -- writes ---------------------------------------------------------------

    def write(self, path: str, principal, payload: Mapping[str, Any]) -> Response:
        """Dispatch one write. Separate from :meth:`handle` so a read cannot become one.

        The principal is checked against :data:`~nova.control.auth.WRITE_ROUTES` here as
        well as in the transport. Belt and braces on the one surface where being wrong
        means an unauthorised *action* rather than an unauthorised read.
        """
        if not path.startswith(API_PREFIX):
            return _error(404, f"no such route: {path}")
        tail = path[len(API_PREFIX) :].rstrip("/") or "/"

        route = _write_route(tail)
        if route is None:
            return _error(404, f"no such write route: {path}")
        if not principal.may_write(route):
            return _error(
                403, f"role {principal.role!r} may not call this route", route=route
            )
        if self.audit is None:
            # Refused, not degraded. A write path that can run without leaving a record is
            # one somebody will run without leaving a record, and the whole value of an
            # approval is the record that it happened.
            return _error(
                503,
                "this control plane has no audit log, so it will not accept writes. "
                "Start it with a runtime home NOVA can write an audit log into",
            )

        if route == "/work/decide":
            return self._decide_work(tail, principal, payload)
        if route == "/automations/create":
            return self._create_automation(principal, payload)
        if route == "/automations/decide":
            return self._decide_automation(tail, principal, payload)
        if route == "/channels/apply":
            return self._apply_channels(principal, payload)
        if route == "/agents/create":
            return self._create_agent(principal, payload)
        if route.startswith("/agents/"):
            return self._agent_action(tail, route.rsplit("/", 1)[1], principal, payload)
        return self._submit_objective(tail, principal, payload)

    def _decide_work(self, tail: str, principal, payload: Mapping[str, Any]) -> Response:
        from nova.runtime.base import WORK_ACTIONS

        if not self.runtime.capabilities.work_decisions:
            return _error(
                501,
                f"runtime {self.runtime.name!r} cannot act on work items, so this control "
                "plane will not offer a button that does nothing",
            )

        task_id = tail[len("/work/") :].rsplit("/", 1)[0]
        action = str(payload.get("action") or "").strip()
        if action not in WORK_ACTIONS:
            return _error(
                400,
                f"action must be one of {', '.join(WORK_ACTIONS)}",
                given=action or None,
            )
        reason = str(payload.get("reason") or "").strip()
        note = str(payload.get("note") or "").strip()

        decision = self.runtime.decide_work(
            task_id,
            action,
            actor=principal.name,
            # Written as the human, not as the server. An auditor filtering on `actor`
            # must see who decided; a process name there answers the wrong question.
            audit=self.audit.with_actor(principal.name),
            correlation_id=new_correlation_id(),
            reason=reason,
            note=note,
        )
        body = {**decision.to_dict(), "actor": principal.name}
        # 409, not 400: the request was well-formed and the world disagreed. An operator
        # whose second click is told "bad request" goes looking for a bug in the button.
        return Response(200 if decision.applied else 409, body)

    def _submit_objective(self, tail: str, principal, payload: Mapping[str, Any]) -> Response:
        from nova.supervisor import submit_objective

        objective_id = tail[len("/objectives/") :].rsplit("/", 1)[0]
        try:
            objective = next(o for o in self.bundle.objectives if o.id == objective_id)
        except StopIteration:
            known = ", ".join(sorted(o.id for o in self.bundle.objectives)) or "(none)"
            return _error(404, f"no objective {objective_id!r}; declared: {known}")

        dry_run = bool(payload.get("dry_run"))
        report = submit_objective(
            objective,
            self.bundle.agents,
            self.runtime,
            audit=self.audit.with_actor(principal.name),
            tenant_id=self.bundle.tenant_id,
            dry_run=dry_run,
        )
        body = {**report.to_dict(), "actor": principal.name, "dry_run": dry_run}
        return Response(200 if report.submitted or dry_run else 409, body)

    def _apply_channels(self, principal, payload: Mapping[str, Any]) -> Response:
        if not self.runtime.capabilities.channel_delivery:
            return _error(
                501,
                f"runtime {self.runtime.name!r} cannot deliver channels, so this control "
                "plane will not offer a connect button that does nothing",
            )
        from nova.channels.derive import plan_derivations

        dry_run = bool(payload.get("dry_run"))
        result = self.runtime.apply_channels(
            self.bundle.channels,
            audit=self.audit.with_actor(principal.name),
            correlation_id=new_correlation_id(),
            derivations=plan_derivations(self.bundle),
            dry_run=dry_run,
        )
        return Response(200, {**result, "actor": principal.name, "dry_run": dry_run})

    def _compiled(self, agent_id: str):
        """The compiled policy for one agent, or None when no policy is declared."""
        if self.bundle.policy is None:
            return None
        return compile_policy(self.bundle.agent(agent_id), self.bundle.policy)

    # -- routing --------------------------------------------------------------

    def handle(self, path: str, query: Optional[Mapping[str, str]] = None) -> Response:
        """Dispatch one GET. Unknown paths are 404, write methods never reach here."""
        query = query or {}
        if not path.startswith(API_PREFIX):
            return _error(404, f"no such route: {path}")
        tail = path[len(API_PREFIX) :].rstrip("/") or "/"

        if tail == "/health":
            return self.health()
        if tail == "/identity":
            return self.identity()
        if tail == "/agents":
            return self.agents()
        if tail == "/tasks":
            return self.tasks(query)
        if tail.startswith("/tasks/"):
            task_id = tail[len("/tasks/") :]
            return self.task(task_id) if task_id else _error(404, "no task id given")
        if tail == "/policy":
            return self.policy()
        if tail == "/policy/simulate":
            return self.simulate(query)
        if tail == "/decisions":
            return self.decisions(query)
        if tail == "/budget":
            return self.budget()
        if tail == "/knowledge":
            return self.knowledge()
        if tail == "/objectives":
            return self.objectives()
        if tail == "/automations":
            return self.automations()
        if tail == "/channels":
            return self.channels()
        if tail.startswith("/agents/") and tail.endswith("/soul"):
            return self.agent_soul(tail[len("/agents/") : -len("/soul")])
        if tail.startswith("/agents/") and tail.endswith("/automations"):
            return self.agent_automations(tail[len("/agents/") : -len("/automations")])
        if tail.startswith("/agents/") and tail.endswith("/config"):
            return self.agent_config(tail[len("/agents/") : -len("/config")])
        return _error(404, f"no such route: {path}")

    def agent_config(self, agent_id: str) -> Response:
        """An agent's editable declaration, plus the options a form can offer.

        Deliberately **not** the same shape as ``/agents``, which is a presentation view
        carrying display names, digests and sync state. This returns the fields
        :data:`nova.agents.AGENT_FIELDS` accepts and nothing else, so what a form reads is
        exactly what it may write back. The alternative — one payload serving both — meant
        a form having to know which of its keys the write route would silently drop.

        ``choices`` carries option lists rather than making a form guess them: permissions
        come from the tenant's own policy, toolsets from the runtime's registry, corpora
        from the knowledge catalogue, channels from the channel catalogue. Every one is
        read from the thing that owns it, so an option that disappeared upstream disappears
        here rather than lingering and failing at save time.

        Admin-only, because the declaration carries the agent's permissions.
        """
        from nova.channels.providers import catalogue

        spec = next((a for a in self.bundle.agents if a.id == agent_id), None)
        if spec is None:
            return _error(404, f"no agent {agent_id!r}")

        policy = self.bundle.policy
        return Response(
            200,
            {
                "id": spec.id,
                "fields": {
                    "name": spec.name,
                    "role": spec.role,
                    "description": spec.description,
                    "enabled": spec.enabled,
                    "model": spec.model.to_dict(),
                    "tools": spec.tools.to_dict(),
                    "knowledge": spec.knowledge.to_dict(),
                    "permissions": list(spec.permissions),
                    "approval": spec.approval.to_dict(),
                    "limits": spec.limits.to_dict(),
                    "delegation": spec.delegation.to_dict(),
                },
                "settable": list(AGENT_FIELDS),
                "choices": {
                    "permissions": sorted(policy.permissions) if policy else [],
                    "actions": sorted(policy.actions) if policy else [],
                    "toolsets": [dict(t) for t in self.runtime.toolsets()],
                    "knowledge": sorted(self.bundle.knowledge.ids),
                    "channels": [
                        {"id": p.id, "label": p.label} for p in catalogue()
                    ],
                    "agents": sorted(
                        a.id for a in self.bundle.agents if a.id != agent_id
                    ),
                },
            },
        )

    def agent_soul(self, agent_id: str) -> Response:
        """The agent's persona as declared, and where an edit will be written.

        Reading the *bundle's* copy rather than the materialised ``SOUL.md``. The profile's
        file carries tenant branding and a knowledge briefing that the materialiser adds;
        showing that in an editor would invite someone to edit the generated parts, and the
        next apply would discard exactly those edits.
        """
        from nova import agents as agent_ops

        try:
            return Response(200, agent_ops.instructions_of(self.bundle.root, agent_id))
        except NovaError as exc:
            return _error(404, str(exc))

    def agent_automations(self, agent_id: str) -> Response:
        """One agent's schedules, with the executions the runtime actually recorded."""
        if not any(a.id == agent_id for a in self.bundle.agents):
            return _error(404, f"no agent {agent_id!r}")
        rows = [
            row for row in self.runtime.list_automations() if row.agent_id == agent_id
        ]
        out = []
        for row in rows:
            record = row.to_dict()
            record["executions"] = [
                run.to_dict()
                for run in self.runtime.automation_executions(
                    agent_id, row.automation_id, limit=10
                )
            ]
            out.append(record)
        return Response(
            200,
            {
                "agent_id": agent_id,
                "automations": out,
                "scheduler": (self.runtime.scheduler_health(agent_id).to_dict()
                              if hasattr(self.runtime, "scheduler_health") else {}),
            },
        )

    # -- routes ---------------------------------------------------------------

    def channels(self) -> Response:
        """What is connected, which agents each connection may reach, and what it still needs.

        Carries **no credential and no credential value** — only the variable names the
        provider's adapter reads, and which of them are still absent per agent. That is the
        most a control plane should ever be able to say about a secret.
        """
        from nova.channels.derive import plan_derivations
        from nova.channels.providers import catalogue

        derivations = plan_derivations(self.bundle)
        readiness = {
            row["id"]: row
            for row in self.runtime.channel_readiness(self.bundle.channels, derivations)
        }
        rows = []
        for channel in self.bundle.channels:
            ready = readiness.get(channel.id, {})
            provider = channel.catalogue
            rows.append(
                {
                    "id": channel.id,
                    "provider": channel.provider,
                    "provider_label": provider.label,
                    "display_name": channel.display_name or provider.label,
                    "enabled": channel.enabled,
                    "transport": provider.transport.value,
                    "needs_public_endpoint": provider.needs_public_endpoint,
                    "verification": provider.verification.value,
                    "caveat": provider.caveat,
                    "allowed_agents": list(channel.allowed_agents),
                    "routes": [route.to_dict() for route in channel.routes],
                    "approval_required_for": list(channel.approval.required_for),
                    "derived_agents": [
                        d.to_dict() for d in derivations if d.channel_id == channel.id
                    ],
                    "required_env": ready.get("required_env", list(provider.required_env)),
                    "missing_by_agent": ready.get("missing_by_agent", {}),
                    "status": (
                        "disabled" if not channel.enabled
                        else "connected" if ready.get("ready") else "needs_credentials"
                    ),
                }
            )
        return Response(
            200,
            {
                "declared": bool(self.bundle.channels),
                "channel_delivery": self.runtime.capabilities.channel_delivery,
                "channels": rows,
                "catalogue": [p.to_dict() for p in catalogue()],
            },
        )

    def health(self) -> Response:
        """Platform and runtime health. Never 503 for an absent work store — that is normal."""
        runtime_health = self.runtime.health()
        return Response(
            200,
            {
                "platform": {"version": __version__, "tenant_id": self.bundle.tenant_id},
                "runtime": {**self.runtime.describe(), **runtime_health.to_dict()},
                "bundle": {"digest": self.bundle.digest(), "agents": len(self.bundle.agents)},
            },
        )

    def identity(self) -> Response:
        """Resolved branding for display surfaces.

        The dashboard themes itself from this at boot, which is what makes one build
        serve differently-branded deployments.
        """
        identity = self.bundle.identity
        return Response(
            200,
            {
                "product_name": identity.product_name,
                "company_name": identity.company_name,
                "logo": identity.logo,
                "favicon": identity.favicon,
                "theme": identity.theme.to_dict(),
                "support": identity.support.to_dict(),
                "welcome": identity.welcome,
                "tenant_id": self.bundle.tenant_id,
            },
        )

    def agents(self) -> Response:
        """Declared agents, each with its applied state and whether the two agree."""
        applied = {agent.agent_id: agent for agent in self.runtime.list_agents()}
        identity = self.bundle.identity

        rows: list[dict[str, Any]] = []
        for spec in self.bundle.agents:
            live = applied.get(spec.id)
            declared_digest = self.runtime.expected_digest(
                spec, policy=self._compiled(spec.id), knowledge=self.bundle.knowledge
            )
            rows.append(
                {
                    "id": spec.id,
                    "display_name": identity.display_name_for(spec.id, spec.name),
                    "role": spec.role,
                    "description": spec.description,
                    "enabled": spec.enabled,
                    "model": spec.model.to_dict(),
                    "materialized": live is not None,
                    "in_sync": bool(live and live.digest == declared_digest),
                    "declared_digest": declared_digest,
                    "applied_digest": live.digest if live else "",
                    "limits": spec.limits.to_dict(),
                    "approval_required_for": list(spec.approval.required_for),
                    "knowledge_sources": list(spec.knowledge.sources),
                    "may_assign_to": list(spec.delegation.may_assign_to),
                }
            )

        # Agents present in the runtime but absent from the bundle are surfaced rather
        # than hidden: an operator needs to see everything that can run.
        declared = {spec.id for spec in self.bundle.agents}
        unmanaged = [
            {
                "id": agent.agent_id,
                "display_name": agent.agent_id,
                "declared": False,
                "managed_by_nova": agent.managed_by_nova,
            }
            for agent in applied.values()
            if agent.agent_id not in declared
        ]

        return Response(200, {"agents": rows, "undeclared": unmanaged})

    def tasks(self, query: Mapping[str, str]) -> Response:
        """Work the runtime holds, newest first, with a per-state tally."""
        agent_id = (query.get("agent") or "").strip()
        try:
            limit = int(query.get("limit") or 100)
        except ValueError:
            return _error(400, "limit must be a whole number")
        if limit < 1:
            return _error(400, "limit must be at least 1")

        views = self.runtime.list_tasks(agent_id=agent_id, limit=limit)
        counts: dict[str, int] = {}
        for view in views:
            counts[view.state] = counts.get(view.state, 0) + 1

        return Response(
            200,
            {
                "tasks": [view.to_dict() for view in views],
                "counts": counts,
                "needs_attention": sum(1 for view in views if view.needs_attention),
                "filtered_by_agent": agent_id,
            },
        )

    def automations(self) -> Response:
        """Recurring work the runtime holds, per agent, with whether it will actually fire.

        The scheduler answer is not decoration. Hermes runs its ticker inside the gateway
        and has no standalone cron daemon, so a deployment can hold a perfectly correct
        schedule that nothing ever executes — the runtime's own CLI calls this their
        most common support report. Listing schedules without saying whether a scheduler
        is attached would present intentions as commitments.
        """
        if not self.runtime.capabilities.scheduling:
            return Response(
                200,
                {
                    "scheduling": False,
                    "automations": [],
                    "detail": (
                        f"runtime {self.runtime.name!r} does not hold scheduled work"
                    ),
                },
            )

        rows = self.runtime.list_automations()
        # Liveness is per agent because the store is: each profile keeps its own ticker
        # markers. Asked once per agent that actually has automations, not per row.
        agents = sorted({row.agent_id for row in rows if row.agent_id})
        health = {
            agent: self.runtime.scheduler_health(agent).to_dict() for agent in agents
        }
        # Same identity projection the Agents screen uses, so one agent is not called two
        # different things on two screens.
        identity = self.bundle.identity
        display = {a.id: identity.display_name_for(a.id, a.name) for a in self.bundle.agents}
        return Response(
            200,
            {
                "scheduling": True,
                "automations": [
                    {**row.to_dict(), "agent_display_name": display.get(row.agent_id, row.agent_id)}
                    for row in rows
                ],
                "scheduler_health": health,
                "counts": {
                    "total": len(rows),
                    "enabled": sum(1 for r in rows if r.enabled),
                    "paused": sum(1 for r in rows if not r.enabled),
                },
                # Provenance for the ones NOVA declared. An automation missing from here
                # was created outside NOVA — shown as that, not given invented origins.
                "governance": self._automation_governance(rows),
                # Agents the create form may target. Sent rather than inferred from the
                # automation list, which would only ever name agents that already have
                # one.
                "agents": [
                    {"id": a.id, "display_name": display.get(a.id, a.id)}
                    for a in self.bundle.agents
                    if a.enabled
                ],
            },
        )

    def _automation_governance(self, rows) -> dict[str, Any]:
        """NOVA's record for each listed automation, where one exists."""
        from nova.automations import registry

        found: dict[str, Any] = {}
        for row in rows:
            try:
                entry = registry.governance(
                    self.runtime.state_location,
                    row.automation_id,
                    tenant_id=self.bundle.tenant_id,
                )
            except Exception:  # noqa: BLE001 — provenance is a nicety, not the listing
                entry = None
            if entry is not None:
                found[row.automation_id] = entry
        return found

    def _forget_automation(self, job_id: str) -> None:
        """Drop NOVA's provenance record, best effort.

        Best effort on purpose: the runtime is the source of truth for existence, and a
        registry write that failed must not turn a successful delete into an error the
        operator has to reconcile.
        """
        from nova.automations import registry

        try:
            registry.forget(self.runtime.state_location, job_id)
        except Exception:  # noqa: BLE001 — provenance is a record, not the act
            pass

    # -- agents ---------------------------------------------------------------

    def _reload_bundle(self):
        """Re-read the bundle from disk after an edit, so later reads see the new state.

        The API holds a bundle it was constructed with. An edit that changed the files but
        not that object would leave the Control Centre showing the old configuration until
        the process restarted — the "did my change apply?" failure this whole feature
        exists to remove.
        """
        from nova.spec import load_bundle

        self.bundle = load_bundle(self.bundle.root)
        return self.bundle

    def _apply_to_runtime(self, correlation_id: str, actor: str) -> dict[str, Any]:
        """Push the edited bundle into the runtime and report what actually happened.

        A bundle edit changes a declaration. Until it is applied, the running agent still
        has the old persona, so reporting success on the write alone would be reporting
        that a file changed — which is not what anybody asked.

        A failure here is **not** rolled back, and the response says so. The declaration is
        valid and saved; what failed is materialising it. Silently reverting a saved edit
        because a later step failed would lose the operator's work.
        """
        from nova.apply import apply_bundle

        try:
            result = apply_bundle(
                self.bundle,
                self.runtime,
                audit=self.audit.with_actor(actor),
                dry_run=False,
                # Same id as the edit, so the audit log shows one act rather than an edit
                # and an unrelated apply that happened to follow it.
                correlation_id=correlation_id,
            )
        except NovaError as exc:
            return {"applied": False, "error": str(exc)}
        summary = {
            "applied": True,
            "created": list(getattr(result, "created", ()) or ()),
            "changed": list(getattr(result, "changed", ()) or ()),
            "unchanged": list(getattr(result, "unchanged", ()) or ()),
            "warnings": list(getattr(result, "warnings", ()) or ()),
        }
        return summary

    def _agent_write(
        self,
        principal,
        *,
        kind: str,
        subject: str,
        detail: dict[str, Any],
        operation,
    ) -> Response:
        """One agent mutation: intent, edit the bundle, apply, committed.

        Every agent route funnels through here so the audit shape cannot vary between them,
        and so every one of them reloads and applies rather than leaving that to be
        remembered per route.
        """
        correlation_id = new_correlation_id()
        audit = self.audit.with_actor(principal.name)
        audit.record(
            kind=kind, phase="intent", subject=subject,
            correlation_id=correlation_id, detail={**detail, "actor": principal.name},
        )
        try:
            _bundle, changed = operation()
        except NovaError as exc:
            audit.record(
                kind=kind, phase="failed", subject=subject,
                correlation_id=correlation_id, detail={"error": str(exc)},
            )
            # 400: the edit was refused by validation, which names the field.
            return _error(400, str(exc))
        except Exception as exc:  # noqa: BLE001 — an intent must always reach a terminal phase
            audit.record(
                kind=kind, phase="failed", subject=subject,
                correlation_id=correlation_id, detail={"error": repr(exc)},
            )
            raise

        self._reload_bundle()
        applied = self._apply_to_runtime(correlation_id, principal.name)
        audit.record(
            kind=kind,
            phase="committed" if applied.get("applied") else "failed",
            subject=subject,
            correlation_id=correlation_id,
            detail={"files": changed, "applied": applied.get("applied", False)},
        )
        return Response(
            200,
            {
                "ok": True,
                "actor": principal.name,
                "agent_id": subject,
                "files_changed": changed,
                # Separate keys on purpose: the declaration is saved either way, and the
                # screen must be able to say "saved, but not yet running" rather than
                # collapsing both into one tick.
                "saved": True,
                "runtime": applied,
                "correlation_id": correlation_id,
            },
        )

    def _create_agent(self, principal, payload: Mapping[str, Any]) -> Response:
        from nova import agents as agent_ops

        agent_id = str(payload.get("id") or "").strip()
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            fields = {k: v for k, v in payload.items() if k not in ("id", "instructions")}
        instructions = str(payload.get("instructions") or "")

        return self._agent_write(
            principal,
            kind="agent.created",
            subject=agent_id,
            detail={"fields": sorted(fields), "has_instructions": bool(instructions.strip())},
            operation=lambda: agent_ops.create_agent(
                self.bundle.root, agent_id=agent_id, fields=fields, instructions=instructions
            ),
        )

    def _agent_action(
        self, tail: str, action: str, principal, payload: Mapping[str, Any]
    ) -> Response:
        from nova import agents as agent_ops

        parts = [part for part in tail.split("/") if part]
        agent_id = parts[1] if len(parts) > 2 else ""
        root = self.bundle.root

        if action == "update":
            fields = payload.get("fields")
            if not isinstance(fields, Mapping):
                return _error(400, "send the changes as an object under 'fields'")
            return self._agent_write(
                principal, kind="agent.updated", subject=agent_id,
                detail={"fields": sorted(fields)},
                operation=lambda: agent_ops.update_agent(root, agent_id, fields),
            )

        if action == "soul":
            if "instructions" not in payload:
                return _error(400, "send the persona as 'instructions'")
            text = str(payload.get("instructions") or "")
            if len(text) > MAX_INSTRUCTIONS_CHARS:
                return _error(
                    400,
                    f"a persona may not exceed {MAX_INSTRUCTIONS_CHARS} characters; this "
                    f"one is {len(text)}. It is prepended to every turn this agent takes",
                )
            return self._agent_write(
                principal, kind="agent.soul_changed", subject=agent_id,
                # The persona itself is NOT recorded: the audit log says who changed the
                # instructions and when, and the bundle holds what they changed them to.
                # Copying the text into a second store doubles what a leak would expose.
                detail={"characters": len(text)},
                operation=lambda: agent_ops.set_instructions(root, agent_id, text),
            )

        if action == "duplicate":
            new_id = str(payload.get("new_id") or "").strip()
            name = str(payload.get("name") or "")
            return self._agent_write(
                principal, kind="agent.duplicated", subject=agent_id,
                detail={"new_id": new_id},
                operation=lambda: agent_ops.duplicate_agent(root, agent_id, new_id, name=name),
            )

        if action in ("archive", "restore"):
            enabled = action == "restore"
            return self._agent_write(
                principal, kind=f"agent.{action}d", subject=agent_id,
                detail={"enabled": enabled},
                operation=lambda: agent_ops.archive_agent(root, agent_id, enabled=enabled),
            )

        if action == "delete":
            return self._agent_write(
                principal, kind="agent.deleted", subject=agent_id, detail={},
                operation=lambda: agent_ops.delete_agent(root, agent_id),
            )

        return _error(404, f"no such agent action: {action!r}")

    def _create_automation(self, principal, payload: Mapping[str, Any]) -> Response:
        """Declare a new automation: compile it, then let the runtime schedule it.

        The body is an :class:`~nova.spec.automation.AutomationSpec` document — the same
        shape a bundle file uses — and it goes through the same compiler. There is no
        path from here to ``cron.jobs.create_job`` that skips that: an automation created
        from a raw prompt would be a recurring instruction no policy reviewed, which is
        precisely what Phase 11 refused to build.
        """
        from nova.automations.compile import compile_automation
        from nova.spec.automation import AutomationSpec

        if not self.runtime.capabilities.scheduling:
            return _error(
                501,
                f"runtime {self.runtime.name!r} does not hold scheduled work, so this "
                "control plane will not offer a button that does nothing",
            )

        try:
            spec = AutomationSpec.parse(payload)
            compiled = compile_automation(
                spec, self.bundle, validate_schedule=self.runtime.validate_schedule
            )
        except NovaError as exc:
            # 400, not 500: the declaration was refused by validation, and the message
            # names the automation and the field the way every other spec failure does.
            return _error(400, str(exc))

        correlation_id = new_correlation_id()
        audit = self.audit.with_actor(principal.name)
        audit.record(
            kind="automation.declared",
            phase="intent",
            subject=compiled.agent_id,
            digest=compiled.digest,
            correlation_id=correlation_id,
            detail={
                "automation_id": spec.id, "title": spec.title, "schedule": spec.schedule,
                "permissions": list(spec.permissions), "knowledge": list(spec.knowledge),
                "channels": list(spec.channels), "reason": spec.reason,
                "actor": principal.name,
            },
        )
        try:
            created = self.runtime.create_automation(compiled.agent_id, compiled)
        except Exception as exc:  # noqa: BLE001 — an intent must always reach a terminal phase
            # Without this, a runtime that raises leaves an ``intent`` row with no
            # ``committed`` and no ``failed`` beside it, and the log reads as though the
            # act might have happened. The write-ahead invariant is only worth anything
            # if every intent is closed.
            audit.record(
                kind="automation.declared",
                phase="failed",
                subject=compiled.agent_id,
                digest=compiled.digest,
                correlation_id=correlation_id,
                detail={"automation_id": spec.id, "actor": principal.name},
                error=f"{type(exc).__name__}: {exc}",
            )
            return _error(
                502,
                f"the runtime refused this automation: {exc}",
                automation_id=spec.id,
            )
        if created is None:
            audit.record(
                kind="automation.declared",
                phase="failed",
                subject=compiled.agent_id,
                digest=compiled.digest,
                correlation_id=correlation_id,
                detail={"automation_id": spec.id, "actor": principal.name},
            )
            return Response(
                409,
                {
                    "applied": False, "automation_id": spec.id,
                    "detail": (
                        f"the runtime did not create this automation — agent "
                        f"{compiled.agent_id!r} may not be materialized yet"
                    ),
                },
            )

        from nova.automations import registry

        try:
            registry.record(
                self.runtime.state_location,
                job_id=created.automation_id,
                tenant_id=self.bundle.tenant_id,
                compiled=compiled,
                actor=principal.name,
                correlation_id=correlation_id,
                created_at=created.created_at or "",
            )
        except Exception:  # noqa: BLE001 — provenance is a record, not the act
            pass

        audit.record(
            kind="automation.declared",
            phase="committed",
            subject=compiled.agent_id,
            digest=compiled.digest,
            correlation_id=correlation_id,
            detail={
                "automation_id": spec.id, "runtime_job_id": created.automation_id,
                "actor": principal.name,
            },
        )
        return Response(
            201,
            {
                "applied": True, "actor": principal.name,
                "declaration": compiled.to_dict(),
                "automation": created.to_dict(),
            },
        )

    def _decide_automation(
        self, tail: str, principal, payload: Mapping[str, Any]
    ) -> Response:
        """Pause or resume one automation.

        The owning agent is resolved here, not taken from the caller. An automation id
        alone does not say whose it is, and asking the client to supply the pair would
        make the client's claim part of the lookup.

        Create and delete are deliberately absent: creating an automation hands an agent
        an instruction NOVA never compiled and no policy reviewed. See
        docs/audits/PHASE_11_AUTOMATIONS.md.
        """
        if not self.runtime.capabilities.scheduling:
            return _error(
                501,
                f"runtime {self.runtime.name!r} does not hold scheduled work, so this "
                "control plane will not offer a button that does nothing",
            )

        automation_id = tail[len("/automations/") :].rsplit("/", 1)[0]
        action = str(payload.get("action") or "").strip()
        if action not in AUTOMATION_ACTIONS:
            return _error(
                400,
                f"action must be one of {', '.join(AUTOMATION_ACTIONS)}",
                given=action or None,
            )
        reason = str(payload.get("reason") or "").strip()

        owner = next(
            (
                row.agent_id
                for row in self.runtime.list_automations()
                if row.automation_id == automation_id
            ),
            None,
        )
        if owner is None:
            # 404 for an automation this tenant does not own, exactly as for one that
            # does not exist: distinguishing them confirms other tenants' ids.
            return _error(404, f"no automation {automation_id!r}")

        correlation_id = new_correlation_id()
        audit = self.audit.with_actor(principal.name)
        # Written ahead of the act, so an attempt that fails midway is still on the
        # record. "Model-visible means logged" applies to operator actions too: the
        # thing that changes the world gets an intent row before it changes it.
        audit.record(
            kind="automation.decision",
            phase="intent",
            subject=owner,
            correlation_id=correlation_id,
            detail={
                "automation_id": automation_id, "action": action,
                "reason": reason, "actor": principal.name,
            },
        )
        if action == "update":
            updates = payload.get("updates")
            if not isinstance(updates, Mapping) or not updates:
                audit.record(
                    kind="automation.decision", phase="failed", subject=owner,
                    correlation_id=correlation_id,
                    detail={"automation_id": automation_id, "action": action},
                    error="no updates supplied",
                )
                return _error(400, "send the changes as an object under 'updates'")
            try:
                updated = self.runtime.update_automation(
                    owner, automation_id, dict(updates)
                )
            except NovaError as exc:
                audit.record(
                    kind="automation.decision", phase="failed", subject=owner,
                    correlation_id=correlation_id,
                    detail={"automation_id": automation_id, "action": action},
                    error=str(exc),
                )
                return _error(400, str(exc))
            except Exception as exc:  # noqa: BLE001 — close the intent, always
                audit.record(
                    kind="automation.decision", phase="failed", subject=owner,
                    correlation_id=correlation_id,
                    detail={"automation_id": automation_id, "action": action},
                    error=f"{type(exc).__name__}: {exc}",
                )
                return _error(502, f"the runtime refused this edit: {exc}")
            audit.record(
                kind="automation.decision",
                phase="committed" if updated else "failed",
                subject=owner,
                correlation_id=correlation_id,
                detail={
                    "automation_id": automation_id, "action": action,
                    # The fields that changed, never their values: a schedule is harmless
                    # but the shape of this record should not depend on that staying true.
                    "fields": sorted(updates), "reason": reason,
                },
            )
            if updated is None:
                return _error(404, f"no automation {automation_id!r}")
            return Response(
                200,
                {"applied": True, "action": action, "actor": principal.name,
                 "automation": updated.to_dict()},
            )

        if action == "delete":
            try:
                removed = self.runtime.delete_automation(owner, automation_id)
            except Exception as exc:  # noqa: BLE001 — close the intent, always
                audit.record(
                    kind="automation.decision", phase="failed", subject=owner,
                    correlation_id=correlation_id,
                    detail={"automation_id": automation_id, "action": action,
                            "actor": principal.name},
                    error=f"{type(exc).__name__}: {exc}",
                )
                return _error(502, f"the runtime refused this delete: {exc}")
            if removed:
                # Drop NOVA's provenance record too, so the registry does not accumulate
                # entries for automations the runtime no longer has.
                self._forget_automation(automation_id)
            audit.record(
                kind="automation.decision",
                phase="committed" if removed else "failed",
                subject=owner,
                correlation_id=correlation_id,
                detail={
                    "automation_id": automation_id, "action": action,
                    "actor": principal.name,
                },
            )
            if not removed:
                return Response(
                    409,
                    {
                        "applied": False, "automation_id": automation_id, "action": action,
                        "detail": "the runtime did not remove this automation",
                    },
                )
            return Response(
                200,
                {"applied": True, "action": action, "actor": principal.name,
                 "automation_id": automation_id},
            )

        try:
            updated = self.runtime.set_automation_enabled(
                owner, automation_id, enabled=action == "resume", reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 — close the intent, always
            audit.record(
                kind="automation.decision", phase="failed", subject=owner,
                correlation_id=correlation_id,
                detail={"automation_id": automation_id, "action": action,
                        "actor": principal.name},
                error=f"{type(exc).__name__}: {exc}",
            )
            return _error(502, f"the runtime refused this transition: {exc}")
        audit.record(
            kind="automation.decision",
            phase="committed" if updated else "failed",
            subject=owner,
            correlation_id=correlation_id,
            detail={
                "automation_id": automation_id, "action": action,
                "enabled": bool(updated and updated.enabled), "actor": principal.name,
            },
        )
        if updated is None:
            # 409: the request was well formed and the runtime declined.
            return Response(
                409,
                {
                    "applied": False, "automation_id": automation_id, "action": action,
                    "detail": "the runtime did not apply this transition",
                },
            )
        return Response(
            200,
            {
                "applied": True, "action": action, "actor": principal.name,
                "automation": updated.to_dict(),
            },
        )

    def objectives(self) -> Response:
        """Declared objectives: how each routes, and where each has got to.

        Routing and progress together, because separately neither answers the question an
        operator has. "Blocked" is not useful without knowing whether it is blocked on a
        failing step or on a delegation the tenant never authorised — those need opposite
        responses, and only one of them is anyone's fault.
        """
        from nova.supervisor import collect, route_objective

        if not self.bundle.objectives:
            return Response(
                200,
                {
                    "declared": False,
                    "work_submission": self.runtime.capabilities.work_submission,
                    "objectives": [],
                    "detail": "no objectives/ in this tenant bundle",
                },
            )

        # One read of the board for every objective rather than one per objective.
        tasks = self.runtime.list_tasks(limit=1000)

        rows: list[dict[str, Any]] = []
        for spec in self.bundle.objectives:
            decision = route_objective(spec, self.bundle.agents)
            report = collect(spec, self.runtime, tasks=tasks)
            rows.append(
                {
                    "id": spec.id,
                    "title": spec.title,
                    "owner": spec.owner,
                    "owner_display_name": self.bundle.identity.display_name_for(
                        spec.owner, spec.owner
                    ),
                    "description": spec.description,
                    "acceptance": spec.acceptance,
                    "enabled": spec.enabled,
                    "routing_allowed": decision.allowed,
                    "refusals": [step.to_dict() for step in decision.refusals],
                    "ungoverned_steps": list(decision.ungoverned_steps),
                    "warnings": list(decision.warnings) + list(report.warnings),
                    **report.to_dict(),
                }
            )

        return Response(
            200,
            {
                "declared": True,
                "work_submission": self.runtime.capabilities.work_submission,
                "detail": (
                    ""
                    if self.runtime.capabilities.work_submission
                    else f"runtime {self.runtime.name!r} cannot accept submitted work"
                ),
                "objectives": rows,
            },
        )

    def knowledge(self) -> Response:
        """Declared corpora, what is actually indexed, and which agents can read each.

        Three facts that live in three places — the bundle, the index, and each agent's
        grant — and are only useful together. "Who can read the handbook" and "is the
        handbook actually indexed" are the two questions asked about a knowledge
        deployment, and neither is answerable from one source alone.
        """
        declared = self.bundle.knowledge.sources
        readers: dict[str, list[str]] = {}
        for spec in self.bundle.agents:
            for source_id in spec.knowledge.sources:
                readers.setdefault(source_id, []).append(spec.id)

        indexed: dict[str, dict[str, int]] = {}
        index_detail = ""
        index_path = self.runtime.knowledge_index_path
        if index_path.is_file():
            try:
                from nova.knowledge import KnowledgeIndex

                with KnowledgeIndex.open(index_path, create=False) as index:
                    indexed = index.stats()
            except NovaError as exc:
                index_detail = str(exc)
        else:
            index_detail = "no index yet — run `nova knowledge ingest`"

        sources = [
            {
                "id": source.id,
                "title": source.display_title,
                "description": source.description,
                "classification": source.classification,
                "root": str(source.root),
                "readable_by": sorted(readers.get(source.id, [])),
                "indexed": source.id in indexed,
                "documents": indexed.get(source.id, {}).get("documents", 0),
                "chunks": indexed.get(source.id, {}).get("chunks", 0),
                "bytes": indexed.get(source.id, {}).get("bytes", 0),
            }
            for source in declared
        ]

        # A corpus in the index that the bundle no longer declares is still searchable by
        # any agent whose grant was not re-applied. Surfaced rather than filtered out.
        undeclared = sorted(set(indexed) - {source.id for source in declared})

        return Response(
            200,
            {
                "retrieval_enabled": self.runtime.capabilities.knowledge_retrieval,
                "document_extraction": self.runtime.capabilities.document_extraction,
                "index_path": str(index_path),
                "index_detail": index_detail,
                "sources": sources,
                "undeclared_in_index": undeclared,
                "agents": [
                    {
                        "id": spec.id,
                        "display_name": self.bundle.identity.display_name_for(spec.id, spec.name),
                        "sources": list(spec.knowledge.sources),
                    }
                    for spec in self.bundle.agents
                ],
            },
        )

    def task(self, task_id: str) -> Response:
        """One task with the attempts, notes and artifacts the runtime already keeps.

        404 — not 403 — for a task belonging to another tenant. A control plane that
        distinguishes "forbidden" from "not found" confirms other tenants' task ids to
        anyone who guesses one, and ids are short.

        Artifacts carry no path: the runtime stores an absolute host path on every
        attachment row, and it stops at the adapter (see
        :class:`nova.runtime.base.ArtifactView`).
        """
        detail = self.runtime.task_detail(task_id)
        if detail is None:
            return _error(404, f"no task {task_id!r}")
        return Response(200, detail.to_dict())

    # -- governance -----------------------------------------------------------

    def policy(self) -> Response:
        """The tenant policy and what it compiles to per agent.

        A security reviewer reads this to answer "what can this agent actually do?"
        without reading YAML across several files or trusting a summary.
        """
        if self.bundle.policy is None:
            return Response(
                200,
                {
                    "declared": False,
                    "enforced": False,
                    "detail": (
                        "no policy.yaml in this tenant bundle; agents run with no "
                        "platform-level restrictions"
                    ),
                    "actions": {},
                    "permissions": {},
                    "agents": [],
                },
            )

        enforced = self.runtime.capabilities.policy_enforcement
        agents = []
        for spec in self.bundle.agents:
            compiled = compile_policy(spec, self.bundle.policy)
            agents.append(
                {
                    "id": spec.id,
                    "display_name": self.bundle.identity.display_name_for(spec.id, spec.name),
                    "allow": compiled.document["allow"],
                    "deny": compiled.document["deny"],
                    "approval_actions": sorted(compiled.document["approval_actions"]),
                    "unlisted_tool": compiled.document["unlisted_tool"],
                    "has_allowlist": compiled.has_allowlist,
                    "warnings": list(compiled.warnings),
                }
            )

        return Response(
            200,
            {
                "declared": True,
                "enforced": enforced,
                "detail": (
                    ""
                    if enforced
                    else f"runtime {self.runtime.name!r} cannot enforce policy; these rules are inert"
                ),
                "actions": {
                    name: action.to_dict()
                    for name, action in sorted(self.bundle.policy.actions.items())
                },
                "permissions": {
                    name: permission.to_dict()
                    for name, permission in sorted(self.bundle.policy.permissions.items())
                },
                "baseline_tools": list(self.bundle.policy.baseline_tools),
                "agents": agents,
            },
        )

    def simulate(self, query: Mapping[str, str]) -> Response:
        """Explain what policy would do for one agent and one tool.

        Answers the question a customer's security team actually asks — "what happens if
        this agent calls that?" — using the same decision function the runtime enforces
        with, so the answer cannot drift from the behaviour.
        """
        agent_id = (query.get("agent") or "").strip()
        tool = (query.get("tool") or "").strip()
        if not agent_id or not tool:
            return _error(400, "both 'agent' and 'tool' are required")

        known = {spec.id for spec in self.bundle.agents}
        if agent_id not in known:
            return _error(404, f"no agent {agent_id!r}", known_agents=sorted(known))

        compiled = self._compiled(agent_id)
        if compiled is None:
            return Response(
                200,
                {
                    "agent": agent_id,
                    "tool": tool,
                    "decision": {
                        "effect": "allow",
                        "reason": "no policy is declared for this tenant",
                        "tool": tool,
                        "action": "",
                        "rule": "no-policy",
                    },
                    "enforced": False,
                },
            )

        decision = decide(compiled.document, tool)
        return Response(
            200,
            {
                "agent": agent_id,
                "tool": tool,
                "decision": decision.to_dict(),
                "enforced": self.runtime.capabilities.policy_enforcement,
            },
        )

    def decisions(self, query: Mapping[str, str]) -> Response:
        """Refusals and escalations the runtime has recorded.

        Permitted calls are deliberately absent: they are the overwhelming majority, and
        including them would bury what a reviewer is looking for.
        """
        try:
            limit = int(query.get("limit") or 100)
        except ValueError:
            return _error(400, "limit must be a whole number")
        if limit < 1:
            return _error(400, "limit must be at least 1")

        if self.audit is None:
            return Response(
                200,
                {"decisions": [], "detail": "no audit log is attached to this control plane"},
            )

        agent_id = (query.get("agent") or "").strip()
        rows = [
            {
                "ts": event.ts,
                "agent_id": event.subject,
                "tool": event.detail.get("tool", ""),
                "effect": event.detail.get("effect", ""),
                "reason": event.detail.get("reason", ""),
                "rule": event.detail.get("rule", ""),
                "action": event.detail.get("action", ""),
                "correlation_id": event.correlation_id,
            }
            for event in self.audit.read()
            if event.kind == "policy.decision"
            and (not agent_id or event.subject == agent_id)
        ]
        rows.reverse()  # newest first
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["effect"]] = counts.get(row["effect"], 0) + 1
        return Response(200, {"decisions": rows[:limit], "counts": counts, "total": len(rows)})

    # -- budget ---------------------------------------------------------------

    def budget(self) -> Response:
        """Limits and reported usage, with the two kept structurally apart.

        ``controls`` are things that stop an agent. ``observed`` are measurements that do
        not. They are separate keys rather than one list with a flag, because a flag is
        easy to drop in a UI and a missing key is not — and presenting a measurement as a
        ceiling is the specific failure this route is shaped to prevent.
        """
        identity = self.bundle.identity
        # Enforcement is the runtime's claim, not the platform's assumption.
        facts = {fact.key: fact for fact in self.runtime.limit_facts()}

        controls: list[dict[str, Any]] = []
        advisory: list[dict[str, Any]] = []
        recorded: list[dict[str, Any]] = []

        for spec in self.bundle.agents:
            limits = spec.limits.to_dict()
            display = identity.display_name_for(spec.id, spec.name)

            def _row(key: str, value: Any) -> dict[str, Any]:
                fact = facts.get(key)
                return {
                    "agent_id": spec.id,
                    "display_name": display,
                    "key": key,
                    "value": value,
                    "enforcement": fact.enforcement if fact else "observed_only",
                    "summary": fact.summary if fact else "",
                    "compiles_to": fact.compiles_to if fact else "",
                }

            for key, value in limits.items():
                if key == "delegation":
                    for sub_key, sub_value in value.items():
                        row = _row(f"delegation.{sub_key}", sub_value)
                        (controls if row["enforcement"] in ENFORCING_CLASSES else recorded).append(row)
                    continue
                row = _row(key, value)
                if row["enforcement"] in ENFORCING_CLASSES:
                    controls.append(row)
                elif row["enforcement"] == "soft_advisory":
                    advisory.append(row)
                else:
                    recorded.append(row)

        observed = [self.runtime.usage(spec.id).to_dict() for spec in self.bundle.agents]

        return Response(
            200,
            {
                # Things that actually stop an agent.
                "controls": controls,
                # Asks the agent to finish. Not a limit.
                "advisory": advisory,
                # Carried to the runtime but not enforced by anything NOVA controls.
                "recorded": recorded,
                # Measurements. Never a ceiling.
                "observed": observed,
                "observed_caveat": (
                    "Reported usage is observation, not a limit. No plugin can veto a model "
                    "call in this runtime, figures lag a background writer, and costs are "
                    "the runtime's estimate rather than an invoice."
                ),
                "enforcement_classes": [fact.to_dict() for fact in self.runtime.limit_facts()],
            },
        )
