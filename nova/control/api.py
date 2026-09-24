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
    """One API response: an HTTP status and a JSON-serialisable body.

    ``raw`` is the exception, and there is exactly one caller: a tenant's logo. Base64ing
    an image through JSON would inflate it by a third and put it inside a payload the
    dashboard polls, and a data: URI large enough to matter is a data: URI on every poll.
    When ``raw`` is set the transport sends those bytes with ``content_type`` and ignores
    ``body``.
    """

    status: int
    body: Any
    headers: Mapping[str, str] = field(default_factory=dict)
    raw: Optional[bytes] = None
    content_type: str = ""

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
    # /settings/<what>. Enumerated, so a new settings write has to be declared here before
    # it can be called.
    if len(parts) == 2 and parts[0] == "settings" and parts[1] in SETTINGS_ACTIONS:
        return f"/settings/{parts[1]}"
    # /knowledge/<source>/<action>. Enumerated, like every other member action.
    if len(parts) == 3 and parts[0] == "knowledge" and parts[2] in KNOWLEDGE_ACTIONS:
        return f"/knowledge/{parts[2]}"
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

#: What an administrator may do to a corpus. ``reindex`` is separate from upload so an
#: operator who added files on the host by hand can make them searchable without one.
#: ``sync`` only means anything for a corpus that declares an origin.
KNOWLEDGE_ACTIONS = ("upload", "remove", "reindex", "sync")

#: What an administrator may change about the tenant itself.
SETTINGS_ACTIONS = ("organization", "identity", "logo", "agent-name")

AGENT_ACTIONS = ("update", "soul", "duplicate", "archive", "restore", "delete", "credentials",
                 "mcp", "plugins")


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
        if route.startswith("/settings/"):
            return self._settings_write(route.rsplit("/", 1)[1], principal, payload)
        if route.startswith("/knowledge/"):
            return self._knowledge_write(tail, route.rsplit("/", 1)[1], principal, payload)
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
        if not decision.applied:
            # Where every client reads a failed write's explanation. The reason was only
            # under "reason", so the dashboard showed a second click as a bare "HTTP 409".
            unknown = not decision.resulting_status and "no such work item" in decision.reason.lower()
            body["error"] = {
                "status": 404 if unknown else 409,
                "message": f"Nothing changed: {decision.reason or 'the task is not in a state that allows it'}.",
            }
            if unknown:
                return Response(404, body)
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
        if not (report.submitted or dry_run):
            # The refusal detail is in ``routing``; this is the sentence a client shows. The
            # dashboard reads ``error.message`` from every failed write, and without it a
            # refused objective read as a bare "HTTP 409".
            refusals = "; ".join(
                f"step {step.step_id}: {step.detail or step.reason}"
                for step in report.routing.steps if not step.allowed
            ) or "its routing was refused"
            body["error"] = {
                "status": 409,
                "message": f"Nothing was started: {refusals}. Fix the owner's delegation "
                           "(may_assign_to) or the step's agent, then start it again.",
            }
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
        return self.runtime.compile_policy(self.bundle.agent(agent_id), self.bundle.policy)

    # -- routing --------------------------------------------------------------

    def handle(self, path: str, query: Optional[Mapping[str, str]] = None) -> Response:
        """Dispatch one GET. Unknown paths are 404, write methods never reach here."""
        query = query or {}
        if not path.startswith(API_PREFIX):
            return _error(404, f"no such route: {path}")
        tail = path[len(API_PREFIX) :].rstrip("/") or "/"

        if tail == "/health":
            return self.health()
        if tail == "/model":
            return self.model()
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
        if tail.startswith("/agents/") and tail.endswith("/logs"):
            return self.agent_logs(tail[len("/agents/") : -len("/logs")], query)
        if tail.startswith("/agents/") and tail.endswith("/credentials"):
            return self.agent_credentials(tail[len("/agents/") : -len("/credentials")])
        if tail.startswith("/agents/") and tail.endswith("/extensions"):
            return self.agent_extensions(tail[len("/agents/") : -len("/extensions")])
        if tail == "/extensions":
            return self.extensions()
        if tail.startswith("/agents/") and tail.endswith("/activity"):
            return self.agent_activity(tail[len("/agents/") : -len("/activity")], query)
        if tail.startswith("/knowledge/") and tail.endswith("/documents"):
            return self.knowledge_documents(tail[len("/knowledge/") : -len("/documents")])
        if tail == "/settings":
            return self.settings()
        if tail in ("/branding/logo", "/branding/favicon"):
            return self.branding_image(tail.rsplit("/", 1)[1])
        return _error(404, f"no such route: {path}")

    def settings(self) -> Response:
        """Who this deployment serves, and what the workforce is called.

        The two declarations behind every branded surface, returned in the shape the write
        routes accept so a form reads and writes the same keys. ``tenant_id`` is included
        and marked immutable rather than omitted: an operator looking for where to change
        it deserves an answer, not an absence.
        """
        from nova.branding import IDENTITY_FIELDS, IMMUTABLE, ORGANIZATION_FIELDS

        organization = self.bundle.organization
        identity = self.bundle.identity
        return Response(
            200,
            {
                "organization": {
                    "tenant_id": organization.tenant_id,
                    "legal_name": organization.legal_name,
                    "region": organization.region,
                    "timezone": organization.timezone,
                    "contact_email": organization.contact_email,
                },
                "identity": {
                    "product_name": identity.product_name,
                    "company_name": identity.company_name,
                    "theme": identity.theme.to_dict(),
                    "support": identity.support.to_dict(),
                    "messages": {"welcome": identity.welcome, "goodbye": identity.goodbye},
                    "agents": dict(identity.agent_display_names),
                },
                # Whether an image is stored, not the image: the bytes come from
                # /branding/<kind>, which the page loads as an ordinary same-origin image.
                "logo": {
                    "logo": bool(identity.logo),
                    "favicon": bool(identity.favicon),
                },
                "settable": {
                    "organization": list(ORGANIZATION_FIELDS),
                    "identity": list(IDENTITY_FIELDS),
                },
                "immutable": list(IMMUTABLE),
            },
        )

    def branding_image(self, kind: str) -> Response:
        """The tenant's stored logo, served from this origin.

        Not a redirect and not a data: URI. The Control Centre runs under
        ``img-src 'self' data:``, so an external URL would be blocked by the browser and
        show as a broken image with no explanation; and a data: URI large enough to be a
        real logo would ride along on every poll of the payload that carried it.
        """
        from nova.branding import logo_bytes

        found = logo_bytes(self.bundle, kind)
        if found is None:
            return _error(404, f"this tenant has no {kind}")
        raw, content_type = found
        return Response(
            200, None, raw=raw, content_type=content_type,
            # Revalidated rather than cached: a logo changes rarely, but when it changes the
            # operator who just uploaded it is the one looking at the screen.
            headers={"Cache-Control": "no-cache"},
        )

    def _known_agent(self, agent_id: str):
        return next((a for a in self.bundle.agents if a.id == agent_id), None)

    def agent_logs(self, agent_id: str, query: Mapping[str, str]) -> Response:
        """A bounded tail of one of this agent's log files.

        Admin-only, and that is load-bearing rather than incidental: a log line can carry
        anything the runtime wrote — a prompt, a tool argument, part of a document NOVA
        never saw. No attempt is made to sanitise it, because a sanitiser that misses one
        pattern is worse than a clear statement of who may read.

        Without ``stream`` this lists what exists rather than guessing which log was meant.
        """
        if self._known_agent(agent_id) is None:
            return _error(404, f"no agent {agent_id!r}")

        stream = str(query.get("stream") or "").strip()
        if not stream:
            return Response(
                200,
                {"agent_id": agent_id, "streams": [dict(s) for s in self.runtime.log_streams(agent_id)]},
            )
        try:
            lines = int(query.get("lines") or 200)
        except (TypeError, ValueError):
            return _error(400, "lines must be a number")
        try:
            body = self.runtime.read_log(agent_id, stream, lines=lines)
        except NovaError as exc:
            return _error(400, str(exc))
        return Response(200, {"agent_id": agent_id, **body})

    def extensions(self) -> Response:
        """Everything this deployment could grant an agent.

        Read from the runtime's own MCP catalogue and plugin registry. NOVA keeps no second
        list: a hand-maintained copy of a registry the runtime owns drifts, and the drift is
        invisible until somebody picks an option that no longer exists.
        """
        from nova.extensions import catalogue

        known = catalogue()
        return Response(
            200,
            {
                "mcp": [entry.to_dict() for entry in known.mcp],
                # Platforms are excluded: they are channels, they have their own screen, and
                # a second switch for the same thing would eventually disagree with the
                # first. The count is reported so their absence is explained rather than
                # looking like a gap.
                "plugins": [entry.to_dict() for entry in known.plugins if entry.grantable],
                "channels_elsewhere": sum(
                    1 for entry in known.plugins if entry.kind == "platform"
                ),
                "detail": known.detail,
            },
        )

    def agent_extensions(self, agent_id: str) -> Response:
        """What one agent is granted, and what granting it actually achieves.

        Every row carries the honest runtime consequence rather than a tick. An OAuth MCP
        server that has been granted and applied still cannot be called until somebody
        authorizes it on the host, and a bundled backend plugin loads whether or not
        anything here says "enabled" — both are stated, because a control plane that
        implies otherwise is worse than one that says nothing.
        """
        from nova.extensions import catalogue

        agent = next((a for a in self.bundle.agents if a.id == agent_id), None)
        if agent is None:
            return _error(404, f"no agent {agent_id!r} in this bundle")

        known = catalogue()
        granted = set(agent.extensions.mcp)
        enabled = set(agent.extensions.plugins_enable)
        disabled = set(agent.extensions.plugins_disable)

        mcp = []
        for entry in known.mcp:
            row = entry.to_dict()
            row["granted"] = entry.id in granted
            mcp.append(row)

        plugins = []
        for entry in known.plugins:
            if not entry.grantable:
                continue
            row = entry.to_dict()
            row["state"] = (
                "disable" if entry.id in disabled
                else "enable" if entry.id in enabled
                else "default"
            )
            # What "default" means for this plugin, which is not the same answer for all
            # of them: a bundled backend loads, everything else does not.
            row["loads_by_default"] = entry.auto_loads
            plugins.append(row)

        # A grant that has been saved but not applied is a real and common state — the
        # bundle is edited here and the profile is written by apply. Reported the same way
        # every other saved-vs-applied pair in this control plane is.
        applied = self._applied_extensions(agent_id)
        return Response(
            200,
            {
                "agent_id": agent_id,
                "mcp": mcp,
                "plugins": plugins,
                "granted": {
                    "mcp": sorted(granted),
                    "enable": sorted(enabled),
                    "disable": sorted(disabled),
                },
                "applied": applied,
                "detail": known.detail,
            },
        )

    def _applied_extensions(self, agent_id: str) -> dict[str, Any]:
        """What the running profile actually has, as opposed to what the bundle declares.

        Read from the materialized ``config.yaml``. Two different facts, reported apart:
        an operator who granted a server and has not applied should see that, not a tick
        that means "we wrote it down".
        """
        path = getattr(self.runtime, "paths", None)
        config_path = path.config_path(agent_id) if path is not None else None
        if config_path is None or not config_path.is_file():
            return {"known": False, "mcp": [], "enabled": [], "disabled": [],
                    "detail": "this agent has not been applied to the runtime yet"}
        try:
            import yaml

            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as exc:
            return {"known": False, "mcp": [], "enabled": [], "disabled": [],
                    "detail": f"the applied configuration could not be read: {exc}"}
        if not isinstance(data, dict):
            return {"known": False, "mcp": [], "enabled": [], "disabled": [],
                    "detail": "the applied configuration is not a mapping"}
        servers = data.get("mcp_servers")
        plugins = data.get("plugins")
        plugins = plugins if isinstance(plugins, dict) else {}
        return {
            "known": True,
            "mcp": sorted(servers) if isinstance(servers, dict) else [],
            "enabled": [str(n) for n in (plugins.get("enabled") or [])],
            "disabled": [str(n) for n in (plugins.get("disabled") or [])],
            "detail": "",
        }

    def agent_credentials(self, agent_id: str) -> Response:
        """Which credentials this agent needs, and which are set. Never a value.

        The list is derived from the tenant's own declaration — the channels that grant
        this agent, its model, the deployment default — so revoking a channel grant also
        removes its credentials from this screen and from what may be written.
        """
        from nova.credentials import slots_for_agent

        if self._known_agent(agent_id) is None:
            return _error(404, f"no agent {agent_id!r}")

        slots = slots_for_agent(self.bundle, agent_id)
        present = self.runtime.credential_presence(agent_id, tuple(s.name for s in slots))
        return Response(
            200,
            {
                "agent_id": agent_id,
                "credentials": [s.to_dict(present=bool(present.get(s.name))) for s in slots],
                "writable": self.runtime.capabilities.credential_isolation,
            },
        )

    def agent_activity(self, agent_id: str, query: Mapping[str, str]) -> Response:
        """What this agent has actually done, from the records the runtime kept.

        Three real sources, no synthesis: work the runtime holds for this agent, the
        executions its schedules recorded, and the policy decisions made on its behalf.
        An empty section means the runtime recorded nothing, which is reported as that
        rather than softened.
        """
        if self._known_agent(agent_id) is None:
            return _error(404, f"no agent {agent_id!r}")
        try:
            limit = max(1, min(int(query.get("limit") or 25), 200))
        except (TypeError, ValueError):
            return _error(400, "limit must be a number")

        # The runtime filters by agent itself — doing it here would pull the whole board
        # across the boundary to throw most of it away.
        tasks = [t.to_dict() for t in self.runtime.list_tasks(agent_id=agent_id, limit=limit)]

        executions: list[dict[str, Any]] = []
        for row in self.runtime.list_automations():
            if row.agent_id != agent_id:
                continue
            for run in self.runtime.automation_executions(agent_id, row.automation_id, limit=limit):
                record = run.to_dict()
                record["automation_id"] = row.automation_id
                record["automation_name"] = row.name
                executions.append(record)

        # Reuse the /decisions route rather than a second reader over the same log: it
        # already filters by agent, and two implementations of "what was refused" would
        # eventually disagree about which is authoritative.
        decided = self.decisions({"limit": str(limit), "agent": agent_id})
        decisions = decided.body.get("decisions", []) if decided.status == 200 else []

        return Response(
            200,
            {
                "agent_id": agent_id,
                "tasks": tasks,
                "executions": executions[:limit],
                "decisions": decisions,
                "logs": [dict(s) for s in self.runtime.log_streams(agent_id)],
            },
        )

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
        live_status = self.runtime.channel_live_status()
        rows = []
        for channel in self.bundle.channels:
            ready = readiness.get(channel.id, {})
            provider = channel.catalogue
            # Credential readiness says a connection *could* work; the gateway's own record
            # says whether it *does*. Both are reported, never merged into one tick.
            if live_status is None:
                live = {"state": "unknown", "detail": "this runtime does not report connections"}
            elif live_status["gateway"] != "running":
                live = {"state": "disconnected",
                        "detail": f"the gateway is {live_status['gateway']}"}
            else:
                entry = live_status["platforms"].get(channel.provider)
                live = (
                    {"state": "unknown", "detail": "the gateway has not reported this connection"}
                    if entry is None else
                    {"state": "connected" if entry["state"] == "connected" else "disconnected",
                     "detail": entry["error"] or entry["state"], "since": entry["updated_at"]}
                )
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
                    "live": live,
                    "capabilities": {
                        key: cap.to_dict() for key, cap in sorted(provider.capabilities.items())
                    },
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

    def model(self) -> Response:
        """Which model the workforce is configured to call, and whether calls are succeeding.

        The verdict comes from the runtime's own record of recent calls, not from a probe:
        the control plane holds no model credential and spends no tokens on a status page.
        A failure carries the provider's raw message and a plain-language reading of it,
        including who can fix it — the account admin, for instance, when the cloud account
        has not been granted the model, which no change to NOVA will fix.
        """
        from nova.runtime.model_errors import ModelError  # noqa: F401 — shape documented there

        declared = self.bundle.deployment.provider
        agents = []
        for spec in self.bundle.agents:
            resolved = self.bundle.provider_for(spec.id)
            agents.append({
                "id": spec.id,
                "display_name": self.bundle.identity.display_name_for(spec.id, spec.name),
                "provider": resolved.provider, "model": resolved.model,
            })
        return Response(
            200,
            {
                "configured": {
                    "declared": declared.declared,
                    "provider": declared.provider,
                    "model": declared.model,
                    "region": declared.region,
                },
                "agents": agents,
                **self.runtime.model_status(),
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
                spec,
                policy=self._compiled(spec.id),
                knowledge=self.bundle.knowledge,
                deployment=self.bundle.deployment,
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
                    # None for a disabled agent: apply deliberately leaves its profile as it
                    # was, so a digest comparison can only ever say "drifted" — and the UI's
                    # "re-apply to reconcile" advice would be a loop. "Not in service" is the
                    # whole truth about it.
                    "in_sync": (bool(live and live.digest == declared_digest) if spec.enabled else None),
                    "declared_digest": declared_digest,
                    "applied_digest": live.digest if live else "",
                    "limits": spec.limits.to_dict(),
                    "approval_required_for": list(spec.approval.required_for),
                    "knowledge_sources": list(spec.knowledge.sources),
                    "may_assign_to": list(spec.delegation.may_assign_to),
                    # For the Team chart: the systems each agent reaches, beside who it may
                    # hand work to. Permissions stay on /agents/<id>/config — this list is a
                    # presentation view and is kept light.
                    "integrations": list(spec.extensions.mcp),
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
        """Work the runtime holds, newest first, with a per-state tally.

        Carries ``execution`` for the same reason :meth:`automations` carries
        ``scheduler_health``: a list of pending tasks with nothing claiming them looks
        exactly like a list of tasks about to run, and the screen is where somebody
        decides whether the platform is working.
        """
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

        from nova.runtime.model_errors import summarize_task_error

        # One reading of the model record for the whole page. A failed task whose run
        # overlaps the newest model failure is explained by it: the runtime's own error for
        # such a task only says the worker ended without reporting, which is the symptom.
        from nova.runtime.model_errors import is_nova_failure_block

        model_failure = self.runtime.model_status().get("last_failure") if any(
            view.last_error or view.detail.get("block_reason") for view in views
        ) else None
        rows = [self._task_row(view, model_failure, summarize_task_error, is_nova_failure_block)
                for view in views]

        return Response(
            200,
            {
                "tasks": rows,
                "counts": counts,
                "needs_attention": sum(1 for view in views if view.needs_attention),
                "filtered_by_agent": agent_id,
                # Board-wide, so it is not filtered by the agent query above: "is anything
                # running this work" is a property of the deployment, not of a filter.
                "execution": self.runtime.work_execution_health().to_dict(),
            },
        )

    @staticmethod
    def _task_row(view, model_failure, summarize_task_error, is_nova_failure_block) -> dict:
        """One task as the Work screen shows it, with what is wrong and what kind of wrong."""
        row = view.to_dict()
        block_reason = str(view.detail.get("block_reason") or "")
        # The problem shown: a crash's error, else the reason the task was blocked. A block
        # reason is also why a task is stuck, and it was the half that never reached here.
        problem = view.last_error or block_reason
        failed = (
            bool(view.last_error) or bool(view.consecutive_failures)
            or is_nova_failure_block(block_reason)
        )
        # A failure asks for a fix and a retry; a held or review item asks for a
        # decision. Mixing them made every crash look like an approval request.
        row["attention_kind"] = (
            "failed" if view.needs_attention and failed
            else "decision" if view.needs_attention else ""
        )
        if problem:
            row["last_error"] = problem
            summary = summarize_task_error(problem)
            if model_failure and view.started_at and model_failure["at"] >= view.started_at:
                summary["cause"] = model_failure["error"]
            row["error_summary"] = summary
        if block_reason:
            row["block_reason"] = block_reason
        return row

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

    # -- knowledge ------------------------------------------------------------

    def _source(self, source_id: str):
        return next((s for s in self.bundle.knowledge.sources if s.id == source_id), None)

    def knowledge_documents(self, source_id: str) -> Response:
        """What is in one corpus, and what the index has of it.

        Two facts that are only useful together. A document on disk that the index has
        never seen is invisible to every agent, and that is precisely the state an upload
        that skipped reindexing would leave behind — so the screen can say it.
        """
        from nova.knowledge.store import list_documents

        source = self._source(source_id)
        if source is None:
            return _error(404, f"no knowledge source {source_id!r}")

        indexed: dict[str, Any] = {}
        detail = ""
        index_path = self.runtime.knowledge_index_path
        if index_path.is_file():
            try:
                from nova.knowledge import KnowledgeIndex

                with KnowledgeIndex.open(index_path, create=False) as index:
                    indexed = index.stats().get(source_id, {})
            except NovaError as exc:
                detail = str(exc)
        else:
            detail = "no index yet — nothing in this corpus is searchable"

        return Response(
            200,
            {
                "id": source.id,
                "title": source.display_title,
                "root": str(source.root),
                "classification": source.classification,
                "accepts": list(source.include),
                "excludes": list(source.exclude),
                "max_file_bytes": source.max_file_bytes,
                # Present only for a mirrored corpus. Its presence is what tells the screen
                # to offer Sync instead of Upload — the two are mutually exclusive, because
                # a document uploaded into a mirror survives only until the next sync.
                "origin": source.origin.to_dict() if source.origin else None,
                "documents": [dict(row) for row in list_documents(source)],
                "indexed": {
                    "documents": indexed.get("documents", 0),
                    "chunks": indexed.get("chunks", 0),
                    "detail": detail,
                },
                "readable_by": sorted(
                    a.id for a in self.bundle.agents if source_id in a.knowledge.sources
                ),
            },
        )

    def _reindex(self, source_id: str, correlation_id: str, actor: str) -> dict[str, Any]:
        """Re-ingest one corpus so what is on disk is what agents can find.

        Reported separately from the file write, for the same reason a bundle edit reports
        "saved" and "applied" apart: a document stored but not indexed is a real state, and
        one combined tick would let it read as done.
        """
        from nova.knowledge import ingest

        try:
            report = ingest(
                self.bundle.knowledge,
                self.runtime.knowledge_index_path,
                # The runtime's extractor, as `nova knowledge ingest` uses: without it an
                # uploaded PDF or Office file was stored, reported saved, and never indexed.
                extractor=self.runtime,
                source_ids=[source_id],
                audit=self.audit.with_actor(actor) if self.audit else None,
                correlation_id=correlation_id,
            )
        except NovaError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — an index failure must not lose the upload
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        for source_report in getattr(report, "sources", ()) or ():
            if getattr(source_report, "source_id", "") == source_id:
                return {
                    "ok": True,
                    # `indexed` on the report is a count of documents written this run;
                    # `unchanged` are the ones already current. Both are reported, because
                    # "0 indexed" after an upload is alarming and "0 indexed, 12 unchanged"
                    # is not.
                    "documents": getattr(source_report, "indexed", 0),
                    "unchanged": getattr(source_report, "unchanged", 0),
                    "chunks": getattr(source_report, "chunks", 0),
                    "removed": list(getattr(source_report, "removed", ()) or ()),
                    # Pairs of (document, reason) — a file the ingester refused, which is
                    # the one thing an operator needs to see after an upload.
                    "skipped": [
                        {"document": str(item[0]), "reason": str(item[1])}
                        if isinstance(item, (tuple, list)) and len(item) == 2
                        else {"document": str(item), "reason": ""}
                        for item in (getattr(source_report, "skipped", ()) or ())
                    ],
                }
        return {"ok": True, "documents": 0, "unchanged": 0, "chunks": 0,
                "removed": [], "skipped": []}

    def _sync_origin(
        self, source, principal, payload: Mapping[str, Any], correlation_id: str, audit
    ) -> Response:
        """Mirror a corpus's bucket into its root, then reindex what arrived.

        Two reports rather than one, for the same reason upload separates ``saved`` from
        ``index``: a sync that downloaded twelve documents and an index that then refused
        them is a real state, and a single tick would let it read as done.
        """
        from nova.knowledge.origin import sync as sync_origin

        if source.origin is None:
            return _error(
                409,
                f"{source.id!r} has no declared origin. Add an `origin:` block to its entry "
                "in knowledge.yaml to mirror it from a bucket.",
            )

        dry_run = bool(payload.get("dry_run"))
        kind = "knowledge.synced"
        audit.record(kind=kind, phase="intent", subject=source.id,
                     correlation_id=correlation_id,
                     detail={"location": source.origin.location, "dry_run": dry_run,
                             "actor": principal.name})
        try:
            report = sync_origin(source, dry_run=dry_run)
        except NovaError as exc:
            audit.record(kind=kind, phase="failed", subject=source.id,
                         correlation_id=correlation_id, detail={}, error=str(exc))
            return _error(400, str(exc))

        result = report.to_dict()
        if not report.ok:
            audit.record(kind=kind, phase="failed", subject=source.id,
                         correlation_id=correlation_id, detail=result, error=report.error)
            return Response(200, {"ok": False, "source": source.id, "sync": result,
                                  "index": {"ok": False, "error": "not attempted"}})

        # A dry run touched nothing, so there is nothing to reindex.
        index = ({"ok": True, "documents": 0, "unchanged": 0, "chunks": 0,
                  "removed": [], "skipped": []} if dry_run
                 else self._reindex(source.id, correlation_id, principal.name))
        audit.record(kind=kind, phase="committed", subject=source.id,
                     correlation_id=correlation_id, detail={**result, "index": index})
        return Response(200, {"ok": True, "source": source.id, "sync": result,
                              "index": index})

    def _knowledge_write(
        self, tail: str, action: str, principal, payload: Mapping[str, Any]
    ) -> Response:
        """Add to, remove from, or rebuild a corpus."""
        import base64
        import binascii

        from nova.knowledge.store import remove_document, store_document

        parts = [part for part in tail.split("/") if part]
        source_id = parts[1] if len(parts) > 2 else ""
        source = self._source(source_id)
        if source is None:
            return _error(404, f"no knowledge source {source_id!r}")

        if not self.runtime.capabilities.knowledge_retrieval:
            return _error(
                501,
                f"runtime {self.runtime.name!r} cannot retrieve knowledge, so a document "
                "added here would never be read",
            )

        correlation_id = new_correlation_id()
        audit = self.audit.with_actor(principal.name)

        if action == "reindex":
            audit.record(kind="knowledge.reindexed", phase="intent", subject=source_id,
                         correlation_id=correlation_id, detail={"actor": principal.name})
            result = self._reindex(source_id, correlation_id, principal.name)
            audit.record(
                kind="knowledge.reindexed",
                phase="committed" if result.get("ok") else "failed",
                subject=source_id, correlation_id=correlation_id, detail=result,
            )
            return Response(200, {"ok": True, "source": source_id, "index": result})

        if action == "sync":
            return self._sync_origin(source, principal, payload, correlation_id, audit)

        if source.origin is not None:
            # Both remaining actions write into the corpus root, and the next sync would
            # undo them. Refusing is the honest answer; the bucket is where the documents
            # are managed.
            return _error(
                409,
                f"{source_id!r} is mirrored from {source.origin.location} — add or remove "
                "documents there and sync. A change made here would be reverted by the "
                "next sync.",
            )

        if action == "remove":
            name = str(payload.get("name") or "")
            audit.record(kind="knowledge.document_removed", phase="intent", subject=source_id,
                         correlation_id=correlation_id,
                         detail={"name": name, "actor": principal.name})
            try:
                removed = remove_document(source, name)
            except NovaError as exc:
                audit.record(kind="knowledge.document_removed", phase="failed",
                             subject=source_id, correlation_id=correlation_id,
                             detail={"name": name}, error=str(exc))
                return _error(400, str(exc))
            if not removed:
                audit.record(kind="knowledge.document_removed", phase="failed",
                             subject=source_id, correlation_id=correlation_id,
                             detail={"name": name}, error="not found")
                return _error(404, f"{name!r} is not in {source_id!r}")
            index = self._reindex(source_id, correlation_id, principal.name)
            audit.record(kind="knowledge.document_removed", phase="committed",
                         subject=source_id, correlation_id=correlation_id,
                         detail={"name": name, "index": index})
            return Response(200, {"ok": True, "source": source_id, "removed": name,
                                  "index": index})

        # upload
        filename = str(payload.get("filename") or "")
        raw = payload.get("data")
        if not isinstance(raw, str) or not raw:
            return _error(400, "send the document as base64 text under 'data'")
        if raw.startswith("data:"):
            _, _, raw = raw.partition(",")
        try:
            content = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            return _error(400, "the document must be base64-encoded")

        audit.record(
            kind="knowledge.document_added", phase="intent", subject=source_id,
            correlation_id=correlation_id,
            # The filename and its size, never the contents: a corpus holds the customer's
            # documents, and copying one into the audit log doubles what a leak exposes.
            detail={"filename": filename, "bytes": len(content), "actor": principal.name},
        )
        try:
            stored = store_document(
                source, filename=filename, data=content,
                replace=bool(payload.get("replace")),
            )
        except NovaError as exc:
            audit.record(kind="knowledge.document_added", phase="failed", subject=source_id,
                         correlation_id=correlation_id, detail={"filename": filename},
                         error=str(exc))
            return _error(400, str(exc))
        except OSError as exc:
            audit.record(kind="knowledge.document_added", phase="failed", subject=source_id,
                         correlation_id=correlation_id, detail={"filename": filename},
                         error=str(exc))
            return _error(500, f"the document could not be written: {exc}")

        index = self._reindex(source_id, correlation_id, principal.name)
        audit.record(
            kind="knowledge.document_added",
            phase="committed" if index.get("ok") else "failed",
            subject=source_id, correlation_id=correlation_id,
            detail={"name": stored["name"], "bytes": stored["bytes"], "index": index},
        )
        return Response(
            200,
            {
                "ok": True, "source": source_id, "stored": stored,
                # Separate keys on purpose: the document is on disk either way, and a
                # failed index means no agent can find it yet.
                "saved": True, "index": index,
            },
        )

    def _settings_write(self, what: str, principal, payload: Mapping[str, Any]) -> Response:
        """Change the tenant's own identity.

        Reuses :meth:`_agent_write` because the shape is identical — intent, edit the
        bundle, apply, committed — and the only difference is the subject. A second
        implementation would eventually disagree with the first about what "applied" means.
        """
        from nova import branding

        root = self.bundle.root

        if what == "organization":
            fields = payload.get("fields")
            if not isinstance(fields, Mapping) or not fields:
                return _error(400, "send the changes as an object under 'fields'")
            return self._agent_write(
                principal, kind="settings.organization_changed",
                subject=self.bundle.tenant_id, detail={"fields": sorted(fields)},
                operation=lambda: branding.update_organization(root, fields),
            )

        if what == "identity":
            fields = payload.get("fields")
            if not isinstance(fields, Mapping) or not fields:
                return _error(400, "send the changes as an object under 'fields'")
            return self._agent_write(
                principal, kind="settings.identity_changed",
                subject=self.bundle.tenant_id, detail={"fields": sorted(fields)},
                operation=lambda: branding.update_identity(root, fields),
            )

        if what == "agent-name":
            agent_id = str(payload.get("agent_id") or "").strip()
            display_name = str(payload.get("display_name") or "")
            return self._agent_write(
                principal, kind="settings.agent_name_changed", subject=agent_id,
                detail={"display_name": display_name},
                operation=lambda: branding.set_agent_display_name(root, agent_id, display_name),
            )

        if what == "logo":
            kind = str(payload.get("kind") or "logo").strip()
            data = payload.get("data")
            if data in (None, ""):
                return self._agent_write(
                    principal, kind="settings.logo_cleared",
                    subject=self.bundle.tenant_id, detail={"kind": kind},
                    operation=lambda: branding.clear_logo(root, kind),
                )
            if not isinstance(data, str):
                return _error(400, "the image must be base64 text")
            content_type = str(payload.get("content_type") or "").strip()
            return self._agent_write(
                principal, kind="settings.logo_changed",
                subject=self.bundle.tenant_id,
                # The bytes are not recorded — only that an image of this type was stored.
                detail={"kind": kind, "content_type": content_type, "bytes": len(data)},
                operation=lambda: branding.set_logo(
                    root, data=data, content_type=content_type, kind=kind
                ),
            )

        return _error(404, f"no such settings route: {what!r}")

    def _unknown_toolsets(self, fields: Mapping[str, Any]) -> str:
        """A refusal sentence when ``fields`` names toolsets this runtime does not have.

        Toolsets are runtime vocabulary, so the bundle cannot check them the way it checks a
        knowledge source; the compiler only warns that an unknown one "grants nothing". An
        agent created that way looks equipped and is not — refuse it where it is entered.
        """
        tools = fields.get("tools")
        named = tools.get("toolsets") if isinstance(tools, Mapping) else None
        if not isinstance(named, (list, tuple)) or not named:
            return ""
        try:
            known = {str(t["id"]) for t in self.runtime.toolsets()}
        except Exception:  # noqa: BLE001 — a runtime that cannot list them is not a reason to refuse
            return ""
        if not known:  # a runtime that publishes no registry cannot be checked against
            return ""
        unknown = sorted({str(n) for n in named} - known)
        if not unknown:
            return ""
        return (
            f"tools.toolsets: this runtime has no toolset named {', '.join(unknown)}. "
            "Pick from the toolsets listed in the agent editor"
        )

    def _create_agent(self, principal, payload: Mapping[str, Any]) -> Response:
        from nova import agents as agent_ops

        agent_id = str(payload.get("id") or "").strip()
        fields = payload.get("fields")
        if not isinstance(fields, Mapping):
            fields = {k: v for k, v in payload.items() if k not in ("id", "instructions")}
        instructions = str(payload.get("instructions") or "")
        unknown = self._unknown_toolsets(fields)
        if unknown:
            return _error(400, unknown)

        return self._agent_write(
            principal,
            kind="agent.created",
            subject=agent_id,
            detail={"fields": sorted(fields), "has_instructions": bool(instructions.strip())},
            operation=lambda: agent_ops.create_agent(
                self.bundle.root, agent_id=agent_id, fields=fields, instructions=instructions
            ),
        )

    def _write_credentials(self, agent_id: str, principal, payload: Mapping[str, Any]) -> Response:
        """Set or clear this agent's credentials.

        The one path in NOVA that writes a secret, and everything about it is narrower than
        the routes around it.

        *It does not go through the bundle.* A credential is not a declaration — it does not
        belong in version control, and ``apply`` must keep being unable to overwrite one.
        ``.env`` therefore stays on the materialiser's ``NEVER_WRITE`` list and this writes
        the profile directly.

        *It does not accept an arbitrary name.* ``.env`` is loaded into the environment of
        the process that runs the agent, so a write path taking any name could set
        ``LD_PRELOAD`` or ``PYTHONPATH`` and become code execution. Names are checked
        against what the tenant's own declaration says this agent needs.

        *It does not record the value.* The audit log gets which names changed and who
        changed them. Copying the secret into a second store would double what a leak
        exposes, and the whole point of the audit is to be safely readable.
        """
        from nova.credentials import check_writable

        if self._known_agent(agent_id) is None:
            return _error(404, f"no agent {agent_id!r}")

        raw = payload.get("values")
        if not isinstance(raw, Mapping) or not raw:
            return _error(400, "send the credentials as an object under 'values'")

        values: dict[str, Optional[str]] = {}
        for name, value in raw.items():
            name = str(name).strip()
            if not name:
                return _error(400, "a credential name may not be blank")
            if value is None:
                values[name] = None            # explicit clear
                continue
            if not isinstance(value, str):
                return _error(400, f"{name}: a credential value must be text")
            # A blank string is a clear, not a credential set to nothing: an operator who
            # empties the field means "remove this", and storing "" would leave the adapter
            # believing it had one.
            values[name] = value if value.strip() else None

        try:
            check_writable(self.bundle, agent_id, values)
        except NovaError as exc:
            return _error(400, str(exc))

        correlation_id = new_correlation_id()
        audit = self.audit.with_actor(principal.name)
        setting = sorted(n for n, v in values.items() if v is not None)
        clearing = sorted(n for n, v in values.items() if v is None)
        audit.record(
            kind="agent.credentials_changed", phase="intent", subject=agent_id,
            correlation_id=correlation_id,
            # Names only. There is no branch of this method that puts a value in a record.
            detail={"set": setting, "cleared": clearing, "actor": principal.name},
        )
        try:
            changed = self.runtime.write_credentials(agent_id, values)
        except NovaError as exc:
            audit.record(
                kind="agent.credentials_changed", phase="failed", subject=agent_id,
                correlation_id=correlation_id, detail={"set": setting, "cleared": clearing},
                error=str(exc),
            )
            return _error(400, str(exc))
        except Exception as exc:  # noqa: BLE001 — an intent always reaches a terminal phase
            audit.record(
                kind="agent.credentials_changed", phase="failed", subject=agent_id,
                correlation_id=correlation_id, detail={"set": setting, "cleared": clearing},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        audit.record(
            kind="agent.credentials_changed", phase="committed", subject=agent_id,
            correlation_id=correlation_id, detail={"changed": list(changed)},
        )
        return Response(
            200,
            {
                "ok": True, "actor": principal.name, "agent_id": agent_id,
                "changed": list(changed),
                # No apply: a credential is read by the agent's process at start, not
                # materialised from a declaration, so there is nothing for apply to do.
                "runtime": {"applied": True},
                "detail": (
                    "" if changed
                    else "nothing changed — the values sent already matched what was stored"
                ),
            },
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
            unknown = self._unknown_toolsets(fields)
            if unknown:
                return _error(400, unknown)
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

        if action == "credentials":
            return self._write_credentials(agent_id, principal, payload)

        if action == "mcp":
            from nova.extensions import manage as extension_ops

            server_id = str(payload.get("server") or "").strip()
            granted = bool(payload.get("granted"))
            if not server_id:
                return _error(400, "name the MCP server under 'server'")
            return self._agent_write(
                principal, kind="agent.mcp_changed", subject=agent_id,
                detail={"server": server_id, "granted": granted},
                operation=lambda: extension_ops.set_mcp(
                    root, agent_id, server_id, granted=granted
                ),
            )

        if action == "plugins":
            from nova.extensions import manage as extension_ops

            plugin_id = str(payload.get("plugin") or "").strip()
            state = str(payload.get("state") or "").strip()
            if not plugin_id:
                return _error(400, "name the plugin under 'plugin'")
            return self._agent_write(
                principal, kind="agent.plugin_changed", subject=agent_id,
                detail={"plugin": plugin_id, "state": state},
                operation=lambda: extension_ops.set_plugin(
                    root, agent_id, plugin_id, state=state
                ),
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
            compiled = self.runtime.compile_policy(spec, self.bundle.policy)
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

        # Month to date against each budget, counted the way the stop counts it, so the
        # screen shows how close an agent is before it stops rather than after.
        spend_rows = []
        tenant_spent: Optional[float] = 0.0
        for spec in self.bundle.agents:
            spent = self.runtime.spend_this_month(spec.id)
            tenant_spent = None if (spent is None or tenant_spent is None) else tenant_spent + spent
            spend_rows.append({
                "agent_id": spec.id,
                "display_name": identity.display_name_for(spec.id, spec.name),
                "spent_usd": spent,
                "budget_usd": spec.limits.monthly_budget_usd,
            })
        tenant_budget = self.bundle.deployment.monthly_budget_usd if self.bundle.deployment else None

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
                "this_month": {
                    "agents": spend_rows,
                    "tenant": {"spent_usd": tenant_spent, "budget_usd": tenant_budget},
                    "caveat": (
                        "Spend is the runtime's own cost estimate, month to date (UTC). At a "
                        "budget, new work and tool calls stop; a reply already in flight can "
                        "still land."
                    ),
                },
            },
        )
