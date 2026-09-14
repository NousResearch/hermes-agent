"""HermesRuntime — the AgentRuntime implementation for the Hermes runtime."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from nova.audit import AuditLog
from nova.errors import RuntimeAdapterError
from nova.knowledge.sources import KnowledgeCatalog
from nova.spec.deployment import DeploymentSpec
from nova.runtime.hermes.skin import build_skin, skin_filename
from nova.runtime.base import (
    AgentRuntime,
    MaterializedAgent,
    MaterializeResult,
    RuntimeCapabilities,
    ExtractedDocument,
    RuntimeHealth,
    SubmitResult,
    TaskView,
    UsageSummary,
    WorkDecision,
    WorkItem,
)
from nova.runtime.hermes import materialize as _materialize
from nova.runtime.hermes import submit as _submit
from nova.runtime.hermes import decide as _decide
from nova.runtime.hermes import channels as _channels
from nova.runtime.hermes import readiness as _readiness
from nova.runtime.hermes import provider as _provider
from nova.runtime.hermes import compat as _compat
from nova.runtime.hermes import extract as _extract
from nova.runtime.hermes import usage as _usage
from nova.runtime.hermes.limits import LIMIT_FACTS
from nova.runtime.hermes import work as _work
from nova.runtime.hermes.paths import HermesPaths
from nova.policy import CompiledPolicy
from nova.spec import AgentSpec, IdentitySpec

#: What the Hermes runtime provides, as established by the Phase 0 audit.
#:
#: ``tool_scoping`` is True because tool DENIALS compile to the runtime's unconditional
#: deny list and are genuinely enforced ahead of any bypass. Positive scoping
#: (toolsets/allow) is not yet compiled — see ``materialize.warnings_for`` for why — and
#: the materializer warns whenever a spec declares it.
#: ``knowledge_retrieval`` is True: a granted agent gets a ``knowledge_search`` tool
#: installed as a per-agent plugin, scoped in SQL to the corpora its tenant granted it.
#: Plugin toolsets are enabled by default (``hermes_cli/tools_config.py``), so the tool
#: reaches the model without NOVA writing ``platform_toolsets`` — the key it deliberately
#: does not touch.
#: ``document_extraction`` is True because the runtime ships extractors for PDF, Office and
#: OpenDocument formats, which NOVA borrows rather than reimplements (see ``extract.py``).
#: ``work_submission`` is True: NOVA creates tasks through ``kanban_db.create_task`` — the
#: runtime's own API for its own shared board, which is a different database from the
#: conversation and credential state on ``materialize.NEVER_WRITE``.
#: ``policy_enforcement`` is True: policy compiles to a plugin on the runtime's documented
#: pre-tool-call hook, which vetoes a call or escalates it to the same human gate that
#: guards dangerous shell commands — and that gate fails closed with no human present.
#: ``channel_delivery`` is True: the runtime ships 22 messaging platform adapters behind a
#: documented plugin seam, routes a conversation to a profile through ``gateway.profile_routes``,
#: and refuses a route whose target profile it does not serve. NOVA compiles its declaration
#: into that configuration and writes no adapter, no transport and no protocol code.
#: ``work_decisions`` is True: the runtime already has a review gate and an operator
#: promotion path (``request_changes``, ``promote_task``, ``unblock_task``, ``add_comment``),
#: each with its own state machine and its own event rows. NOVA asks for a transition and
#: lets the runtime refuse an illegal one, rather than writing a status itself.
HERMES_CAPABILITIES = RuntimeCapabilities(
    durable_tasks=True,
    worktree_isolation=True,
    process_isolation=True,
    credential_isolation=True,
    tool_scoping=True,
    knowledge_retrieval=True,
    document_extraction=True,
    work_submission=True,
    policy_enforcement=True,
    work_decisions=True,
    channel_delivery=True,
    brand_projection=True,
    # Hermes holds recurring work per profile in cron/jobs.json, with its own execution
    # ledger. True says the runtime HOLDS schedules — not that anything is running them;
    # the ticker lives in the gateway, so liveness is asked per agent via
    # ``scheduler_health``.
    scheduling=True,
)


class HermesRuntime(AgentRuntime):
    """Adapts NOVA onto Hermes.

    A NOVA agent is a Hermes profile; NOVA identity is a Hermes skin. Nothing here
    imports a Hermes module: the adapter works through the runtime's on-disk extension
    points, which is what keeps the core patch budget at zero.
    """

    name = "hermes"

    def __init__(
        self,
        *,
        home: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        tenant_id: str = "",
    ) -> None:
        self.paths = HermesPaths(home=Path(home)) if home is not None else HermesPaths.resolve(env)
        self.tenant_id = tenant_id

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return HERMES_CAPABILITIES

    @property
    def state_location(self) -> Path:
        return self.paths.home

    def limit_facts(self) -> tuple:
        return LIMIT_FACTS

    # -- agents ---------------------------------------------------------------

    def materialize_agent(
        self,
        spec: AgentSpec,
        *,
        audit: AuditLog,
        correlation_id: str,
        identity: Optional[IdentitySpec] = None,
        policy: Optional[CompiledPolicy] = None,
        knowledge: Optional[KnowledgeCatalog] = None,
        deployment: Optional[DeploymentSpec] = None,
        dry_run: bool = False,
    ) -> MaterializeResult:
        deployment = deployment or DeploymentSpec()
        resolved = deployment.provider.merged_with(spec.model.deployment)
        runtime_config = deployment.runtime_config
        grant = _materialize.build_knowledge_config(
            spec,
            self.paths,
            knowledge,
            tenant_id=audit.tenant_id,
            audit_log=audit.path,
        )
        if dry_run:
            # A dry run changes nothing, so it is not a model-visible change. It is
            # still recorded: knowing what an operator previewed is useful during an
            # incident, and a plain record() carries no intent/commit pair.
            result = _materialize.materialize(
                spec, self.paths, identity=identity, policy=policy, knowledge=grant,
                provider=resolved, runtime_config=runtime_config,
                tenant_id=self.tenant_id, dry_run=True,
            )
            audit.record(
                "agent.materialize_preview",
                correlation_id=correlation_id,
                subject=spec.id,
                digest=result.digest,
                detail=result.to_detail(),
            )
            return result

        with audit.model_visible_change(
            "agent.materialized",
            correlation_id=correlation_id,
            subject=spec.id,
            digest=_materialize._combined_digest(spec, policy, grant),
            detail={
                "runtime": self.name,
                "agent_name": spec.name,
                "policy": bool(policy),
                "knowledge_sources": [
                    entry["id"] for entry in (grant or {}).get("sources", [])
                ],
            },
        ) as outcome:
            result = _materialize.materialize(
                spec, self.paths, identity=identity, policy=policy, knowledge=grant,
                provider=resolved, runtime_config=runtime_config,
                tenant_id=self.tenant_id, dry_run=False,
            )
            outcome.update(result.to_detail())
        return result

    def submit_work(
        self,
        items: Sequence[WorkItem],
        *,
        audit: AuditLog,
        correlation_id: str,
        dry_run: bool = False,
    ) -> SubmitResult:
        detail = {
            "runtime": self.name,
            "items": len(items),
            "assignees": sorted({item.assignee for item in items}),
        }
        if dry_run:
            result = _submit.submit(self.paths.home, items, dry_run=True)
            audit.record(
                "work.submit_preview",
                correlation_id=correlation_id,
                subject=items[0].key.split(":")[0] if items else "",
                detail={**detail, **result.to_dict()},
            )
            return result

        with audit.model_visible_change(
            "work.submitted",
            correlation_id=correlation_id,
            subject=items[0].key.split(":")[0] if items else "",
            detail=detail,
        ) as outcome:
            result = _submit.submit(self.paths.home, items, dry_run=False)
            outcome.update(result.to_dict())
        return result

    def decide_work(
        self,
        task_id: str,
        action: str,
        *,
        actor: str,
        audit: AuditLog,
        correlation_id: str,
        reason: str = "",
        note: str = "",
    ) -> WorkDecision:
        detail = {"runtime": self.name, "action": action, "actor": actor, "task_id": task_id}
        if reason:
            detail["reason"] = reason

        # A note becomes part of what a worker reads. Everything else changes *when* a
        # worker runs, not *what it reads* — so only the first is write-ahead. Blurring
        # that line in either direction makes "model-visible means logged" mean less.
        if action == "annotate":
            with audit.model_visible_change(
                "work.annotated",
                correlation_id=correlation_id,
                subject=task_id,
                detail={**detail, "note_length": len(note)},
            ) as outcome:
                decision = _decide.decide(
                    self.paths.home, task_id, action, actor=actor, reason=reason, note=note
                )
                outcome.update(decision.to_dict())
            return decision

        decision = _decide.decide(
            self.paths.home, task_id, action, actor=actor, reason=reason, note=note
        )
        audit.record(
            "work.decided",
            correlation_id=correlation_id,
            subject=task_id,
            detail={**detail, **decision.to_dict()},
        )
        return decision

    def apply_channels(
        self,
        channels: Sequence[Any],
        *,
        audit: AuditLog,
        correlation_id: str,
        derivations: Sequence[Any] = (),
        dry_run: bool = False,
    ) -> dict[str, Any]:
        detail = {
            "runtime": self.name,
            "connections": len(channels),
            "providers": sorted({c.provider for c in channels}),
            "agents_granted": sorted({a for c in channels for a in c.allowed_agents}),
            "derived_agents": [d.id for d in derivations],
        }
        if dry_run:
            plan = _channels.plan(channels, derivations)
            audit.record(
                "channel.plan",
                correlation_id=correlation_id,
                subject=self.tenant_id,
                detail={**detail, **plan.to_dict()},
            )
            return plan.to_dict()

        with audit.model_visible_change(
            "channel.connected",
            correlation_id=correlation_id,
            subject=self.tenant_id,
            detail=detail,
        ) as outcome:
            plan = _channels.apply(
                self.paths.home, channels, derivations=derivations, dry_run=False
            )
            outcome.update(plan.to_dict())
        return plan.to_dict()

    def channel_readiness(
        self, channels: Sequence[Any], derivations: Sequence[Any] = ()
    ) -> list[dict[str, Any]]:
        return _channels.readiness(channels, home=self.paths.home, derivations=derivations)

    def never_archive(self) -> tuple[str, ...]:
        """The runtime's own state, taken from the list the materializer already refuses
        to write — one definition of "not NOVA's", used for both writing and archiving."""
        return tuple(sorted(_materialize.NEVER_WRITE)) + (
            "kanban.db", "kanban.db-wal", "kanban.db-shm",
        )

    def compatibility(self) -> list[str]:
        """Warnings about running against an unverified runtime version.

        A warning rather than a refusal: making a patch release of the runtime an outage
        would be a worse failure than the one this prevents. What an operator needs is to
        learn they are outside the verified range from NOVA, not from a worker that will
        not start.
        """
        return _compat.check()

    def deployment_readiness(
        self,
        spec: AgentSpec,
        deployment: Optional[DeploymentSpec] = None,
    ) -> dict[str, Any]:
        deployment = deployment or DeploymentSpec()
        resolved = deployment.provider.merged_with(spec.model.deployment)
        provider_config, warnings = _provider.build_provider_config(resolved)
        required = _provider.required_env(resolved, dict(deployment.runtime_config or {}))
        report = _readiness.check(
            spec.id, required, profile_dir=self.paths.profile_dir(spec.id)
        ).to_dict()
        # Warnings ride along rather than being fetched separately: a caller that had to
        # import the adapter to ask for them would be encoding which runtime it is talking
        # to, which is the one thing the contract exists to prevent.
        report["warnings"] = warnings
        report["provider"] = resolved.to_dict()
        return report

    @property
    def knowledge_index_path(self) -> Path:
        return self.paths.knowledge_index

    def expected_digest(
        self,
        spec: AgentSpec,
        *,
        policy: Optional[CompiledPolicy] = None,
        knowledge: Optional[KnowledgeCatalog] = None,
    ) -> str:
        """As the contract, plus this agent's resolved knowledge grant.

        The grant is resolved rather than taken from the spec because the corpus titles the
        tool description carries come from the tenant catalog, not from the agent.
        """
        grant = _materialize.build_knowledge_config(spec, self.paths, knowledge)
        return _materialize._combined_digest(spec, policy, grant)

    def list_agents(self) -> list[MaterializedAgent]:
        profiles_dir = self.paths.profiles_dir
        if not profiles_dir.is_dir():
            return []

        agents: list[MaterializedAgent] = []
        for entry in sorted(profiles_dir.iterdir()):
            if not entry.is_dir():
                continue
            provenance = _materialize.Provenance.read(
                self.paths.provenance_path(entry.name)
            )
            agents.append(
                MaterializedAgent(
                    agent_id=entry.name,
                    display_name=self._display_name(entry.name),
                    enabled=True,
                    digest=provenance.digest if provenance else "",
                    location=entry,
                    managed_by_nova=provenance is not None,
                    detail={"runtime": self.name},
                )
            )
        return agents

    def _display_name(self, agent_id: str) -> str:
        """Best-effort display name read back from the materialized profile."""
        config_path = self.paths.config_path(agent_id)
        if config_path.is_file():
            try:
                data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                return agent_id
            nova_section = data.get("nova")
            if isinstance(nova_section, dict):
                name = nova_section.get("display_name")
                if isinstance(name, str) and name.strip():
                    return name.strip()
        return agent_id

    def remove_agent(
        self, agent_id: str, *, audit: AuditLog, correlation_id: str, dry_run: bool = False
    ) -> bool:
        profile_dir = self.paths.profile_dir(agent_id)
        if not profile_dir.is_dir():
            return False

        provenance = _materialize.Provenance.read(self.paths.provenance_path(agent_id))
        if provenance is None:
            raise RuntimeAdapterError(
                f"profile {profile_dir} was not created by NOVA; refusing to remove it"
            )

        protected = sorted(
            child.name
            for child in profile_dir.iterdir()
            if child.name in _materialize.NEVER_WRITE
        )
        if protected:
            raise RuntimeAdapterError(
                f"profile {profile_dir} holds customer state ({', '.join(protected)}); "
                "refusing to remove it. Move that data out first if the agent is truly finished."
            )

        if dry_run:
            audit.record(
                "agent.remove_preview",
                correlation_id=correlation_id,
                subject=agent_id,
                detail={"runtime": self.name, "location": str(profile_dir)},
            )
            return True

        with audit.model_visible_change(
            "agent.removed",
            correlation_id=correlation_id,
            subject=agent_id,
            digest=provenance.digest,
            detail={"runtime": self.name, "location": str(profile_dir)},
        ):
            shutil.rmtree(profile_dir)
        return True

    # -- work read models -----------------------------------------------------

    def list_tasks(self, *, agent_id: str = "", limit: int = 200) -> list[TaskView]:
        # Scoped to this deployment's tenant. The board is a shared surface — the runtime's
        # own CLI and dashboard write to it too — so "every row in kanban.db" is not the
        # same question as "this tenant's work", and answering the first while being asked
        # the second is how one tenant's task titles reach another's dashboard.
        return _work.list_tasks(
            self.paths.home, agent_id=agent_id, limit=limit, tenant_id=self.tenant_id
        )

    def get_task(self, task_id: str) -> Optional[TaskView]:
        return _work.get_task(self.paths.home, task_id, tenant_id=self.tenant_id)

    def task_detail(self, task_id: str):
        """The task plus the attempts, notes and artifacts the runtime already keeps.

        Same tenant scope as :meth:`get_task` — a foreign id is None, not a 403.
        """
        return _work.task_detail(self.paths.home, task_id, tenant_id=self.tenant_id)

    # -- automations ----------------------------------------------------------

    def list_automations(self, *, agent_id: str = "") -> list:
        """Automations across this tenant's agents, or one agent's.

        An automation lives in its agent's profile store, so "this tenant's" means
        "across the profiles this deployment materialized" — there is no board-wide
        query that could return another tenant's schedules even by accident.
        """
        from nova.runtime.hermes import automations as _automations

        wanted = [agent_id] if agent_id else [a.agent_id for a in self.list_agents()]
        found: list = []
        for name in wanted:
            profile = self.paths.profile_dir(name)
            if not profile.is_dir():
                continue
            found.extend(_automations.list_automations(profile, name))
        return found

    def scheduler_health(self, agent_id: str):
        from nova.runtime.hermes import automations as _automations

        return _automations.scheduler_health(self.paths.profile_dir(agent_id))

    def set_automation_enabled(
        self, agent_id: str, automation_id: str, *, enabled: bool, reason: str = "",
    ):
        from nova.runtime.hermes import automations as _automations

        profile = self.paths.profile_dir(agent_id)
        if not profile.is_dir():
            return None
        return _automations.set_enabled(
            profile, agent_id, automation_id, enabled=enabled, reason=reason,
        )

    def validate_schedule(self, schedule: str) -> None:
        from nova.runtime.hermes import automations as _automations

        _automations.validate_schedule(schedule)

    def create_automation(self, agent_id: str, compiled):
        """Write a compiled automation into the owning agent's cron store.

        Takes a :class:`~nova.automations.compile.CompiledAutomation`, never a raw
        prompt: the compiler is the only path to the scheduler, and a method that
        accepted free text would reopen the hole Phase 11 left deliberately closed.
        """
        from nova.runtime.hermes import automations as _automations

        profile = self.paths.profile_dir(agent_id)
        if not profile.is_dir():
            return None
        return _automations.create(profile, agent_id, compiled)

    def delete_automation(self, agent_id: str, automation_id: str) -> bool:
        from nova.runtime.hermes import automations as _automations

        profile = self.paths.profile_dir(agent_id)
        if not profile.is_dir():
            return False
        return _automations.delete(profile, agent_id, automation_id)

    def update_automation(self, agent_id: str, automation_id: str, updates: dict):
        from nova.runtime.hermes import automations as _automations

        profile = self.paths.profile_dir(agent_id)
        if not profile.is_dir():
            return None
        return _automations.update(profile, agent_id, automation_id, updates)

    def automation_executions(self, agent_id: str, automation_id: str, *, limit: int = 20):
        from nova.runtime.hermes import automations as _automations

        profile = self.paths.profile_dir(agent_id)
        if not profile.is_dir():
            return ()
        return _automations.executions(profile, automation_id, limit=limit)

    def health(self) -> RuntimeHealth:
        present, detail = _work.store_status(self.paths.home)
        home_exists = self.paths.home.is_dir()
        return RuntimeHealth(
            reachable=home_exists,
            detail=detail if home_exists else f"runtime home {self.paths.home} does not exist",
            work_store_present=present,
            agent_count=len(self.list_agents()),
        )

    def extract_text(self, path: Path) -> ExtractedDocument:
        """Delegates to the runtime's extractor; see :mod:`nova.runtime.hermes.extract`."""
        return _extract.extract(path)

    def usage(self, agent_id: str) -> UsageSummary:
        """Reported usage for one agent. Observation only — see the returned caveats."""
        return _usage.read_usage(self.paths.profile_dir(agent_id), agent_id)

    # -- identity -------------------------------------------------------------

    def apply_identity(
        self,
        identity: IdentitySpec,
        *,
        audit: AuditLog,
        correlation_id: str,
        dry_run: bool = False,
    ) -> MaterializeResult:
        tenant = self.tenant_id or "nova"
        skin = build_skin(identity, tenant_id=tenant)
        target = self.paths.skins_dir / skin_filename(tenant)
        text = yaml.safe_dump(skin, sort_keys=True, default_flow_style=False, allow_unicode=True)

        existing = target.read_text(encoding="utf-8") if target.is_file() else None
        changed = existing != text

        if dry_run:
            audit.record(
                "identity.apply_preview",
                correlation_id=correlation_id,
                subject=tenant,
                detail={"runtime": self.name, "path": str(target), "changed": changed},
            )
            return MaterializeResult(
                agent_id=tenant,
                created=existing is None,
                changed=changed and existing is not None,
                digest="",
                paths_written=(target,),
                location=target,
            )

        if not changed:
            return MaterializeResult(
                agent_id=tenant, created=False, changed=False, digest="", location=target
            )

        with audit.model_visible_change(
            "identity.applied",
            correlation_id=correlation_id,
            subject=tenant,
            detail={"runtime": self.name, "product_name": identity.product_name},
        ) as outcome:
            _materialize.atomic_write(target, text)
            outcome.update({"path": str(target)})

        return MaterializeResult(
            agent_id=tenant,
            created=existing is None,
            changed=existing is not None,
            digest="",
            paths_written=(target,),
            location=target,
        )
