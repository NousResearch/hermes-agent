"""Apply a tenant bundle to a runtime.

The one orchestration entry point above the adapter: take a fully validated bundle and
make a runtime match it. Everything it does is expressed against
:class:`~nova.runtime.base.AgentRuntime`, so it works unchanged on any future adapter.

Ordering matters and is deliberate. Identity is applied **before** agents, because an
agent's persona embeds its branded display name — projecting identity second would leave
every agent carrying the previous tenant's name until the next materialize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from nova.audit import AuditLog, new_correlation_id
from nova.observability import operation, set_correlation_id
from nova.policy import compile_policy
from nova.runtime.base import AgentRuntime, MaterializeResult
from nova.spec import TenantBundle


@dataclass(frozen=True)
class ApplyReport:
    """What applying a bundle did, per agent and overall."""

    tenant_id: str
    runtime: str
    correlation_id: str
    bundle_digest: str
    dry_run: bool
    identity: Optional[MaterializeResult] = None
    agents: tuple[MaterializeResult, ...] = ()
    skipped: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def created(self) -> tuple[str, ...]:
        return tuple(r.agent_id for r in self.agents if r.created)

    @property
    def changed(self) -> tuple[str, ...]:
        return tuple(r.agent_id for r in self.agents if r.changed and not r.created)

    @property
    def unchanged(self) -> tuple[str, ...]:
        return tuple(r.agent_id for r in self.agents if r.unchanged)

    def summary(self) -> str:
        parts = [
            f"tenant={self.tenant_id}",
            f"runtime={self.runtime}",
            f"created={len(self.created)}",
            f"changed={len(self.changed)}",
            f"unchanged={len(self.unchanged)}",
        ]
        if self.skipped:
            parts.append(f"skipped={len(self.skipped)}")
        if self.dry_run:
            parts.append("dry-run")
        return " ".join(parts)


def apply_bundle(
    bundle: TenantBundle,
    runtime: AgentRuntime,
    *,
    audit: AuditLog,
    dry_run: bool = False,
    include_disabled: bool = False,
    correlation_id: Optional[str] = None,
) -> ApplyReport:
    """Make ``runtime`` match ``bundle``. See :func:`_apply_bundle` for the behaviour.

    This wrapper exists only to own the operational trace. Driving the context manager by
    hand around the body would log the start and, on an exception, never log the failure —
    which is the one thing the trace is for.
    """
    # Set before the trace opens, so the "started" line carries it too — otherwise a log
    # shipper cannot group the pair, which is the one thing the id is for.
    correlation_id = correlation_id or new_correlation_id()
    set_correlation_id(correlation_id)

    with operation(
        "apply", tenant_id=bundle.tenant_id, runtime=runtime.name, dry_run=dry_run
    ) as trace:
        return _apply_bundle(
            bundle,
            runtime,
            audit=audit,
            dry_run=dry_run,
            include_disabled=include_disabled,
            correlation_id=correlation_id,
            trace=trace,
        )


def _apply_bundle(
    bundle: TenantBundle,
    runtime: AgentRuntime,
    *,
    audit: AuditLog,
    dry_run: bool,
    include_disabled: bool,
    correlation_id: Optional[str],
    trace: operation,
) -> ApplyReport:
    """Make ``runtime`` match ``bundle``.

    Disabled agents are skipped by default rather than materialized-and-ignored: leaving
    no profile behind is the unambiguous state, and a disabled agent that still has a
    profile is one dispatcher configuration change away from running.

    This function never removes agents. Deletion is destructive and stays an explicit
    operator action through :meth:`AgentRuntime.remove_agent`; a bundle that no longer
    names an agent produces a warning here, not a removal.
    """
    # Already set by the caller above; the same id the audit log stamps, so an operational
    # trace and a governance record line up without guessing from timestamps.
    correlation_id = correlation_id or new_correlation_id()
    warnings: list[str] = []

    # Asked before anything is materialized: an adapter reaching into a schema that has
    # moved should say so while the operator is still watching, not when a worker fails.
    warnings.extend(runtime.compatibility())

    capability_gaps = runtime.capabilities.missing_for(["durable_tasks", "process_isolation"])
    if capability_gaps:
        warnings.append(
            f"runtime {runtime.name!r} does not provide: {', '.join(capability_gaps)} — "
            "unattended agent work may not survive a restart"
        )

    if bundle.policy is not None and not runtime.capabilities.policy_enforcement:
        warnings.append(
            f"a policy is declared but runtime {runtime.name!r} cannot enforce it — every "
            "rule would be inert. Refusing to present governance that does not exist"
        )

    knowledge_users = [spec.id for spec in bundle.agents if spec.knowledge.sources]
    if knowledge_users and not runtime.capabilities.knowledge_retrieval:
        warnings.append(
            f"knowledge sources are declared by {', '.join(knowledge_users)} but runtime "
            f"{runtime.name!r} cannot retrieve them yet; the declaration is recorded and inert"
        )
    if knowledge_users and not bundle.knowledge.sources:
        warnings.append(
            f"{', '.join(knowledge_users)} name knowledge sources but the bundle declares no "
            "corpora — no knowledge tool will be installed"
        )

    audit.record(
        "bundle.apply_started",
        correlation_id=correlation_id,
        subject=bundle.tenant_id,
        digest=bundle.digest(),
        detail={
            "runtime": runtime.name,
            "dry_run": dry_run,
            "agents": [spec.id for spec in bundle.agents],
            "policy_declared": bundle.policy is not None,
            "knowledge_sources": [source.id for source in bundle.knowledge.sources],
            # Variable NAMES only. A credential must never reach an audit record.
            "required_env": list(bundle.deployment.required_env),
            "warnings": warnings,
        },
    )

    identity_result = runtime.apply_identity(
        bundle.identity, audit=audit, correlation_id=correlation_id, dry_run=dry_run
    )

    results: list[MaterializeResult] = []
    skipped: list[str] = []

    # Channel-scoped agent variants. A channel that tightens approval cannot be enforced by
    # the policy hook — the hook is never told which channel it is serving — so the tighter
    # posture becomes its own profile with its own compiled policy, which the hook does
    # enforce. See nova/channels/derive.py for why this is the honest shape rather than a
    # rule that would silently never fire.
    from nova.channels.derive import derive_specs, plan_derivations

    derivations = plan_derivations(bundle)
    derived = derive_specs(bundle)
    for entry in derivations:
        warnings.append(
            f"{entry.base_agent}: reached over {entry.channel_id!r} as {entry.id!r}, which "
            f"additionally escalates {', '.join(entry.added_approvals)}. It is a separate "
            f"profile, so it does not share conversation history with {entry.base_agent!r}"
        )

    all_specs = tuple(bundle.agents) + derived
    selected = all_specs if include_disabled else tuple(s for s in all_specs if s.enabled)
    for spec in all_specs:
        if spec not in selected:
            skipped.append(spec.id)
            continue
        compiled = compile_policy(spec, bundle.policy) if bundle.policy is not None else None
        if compiled is not None:
            warnings.extend(f"{spec.id}: {note}" for note in compiled.warnings)
            # Runtime-specific fields the portable compiler cannot know: where to record a
            # refusal, and which tenant it belongs to. Injected here, where both are in hand.
            compiled.document["tenant_id"] = bundle.tenant_id
            compiled.document["audit_log"] = str(audit.path)
        # Warnings the materializer produced — a spec the runtime cannot fully honour, a
        # profile being adopted by this tenant — were reaching the audit record and not the
        # operator. A warning nobody sees is the same class of problem as a control nobody
        # enforces, so they are surfaced here rather than only being archived.
        results.append(
            runtime.materialize_agent(
                spec,
                audit=audit,
                correlation_id=correlation_id,
                identity=bundle.identity,
                policy=compiled,
                knowledge=bundle.knowledge,
                deployment=bundle.deployment,
                dry_run=dry_run,
            )
        )

    # Readiness is reported, never fixed: NOVA writes the NAME of every credential and
    # none of the values, so an agent can be perfectly materialized and still unable to
    # start. Saying so at apply time is the whole difference between finding out now and
    # finding out when the first task runs.
    for spec in selected:
        report_row = runtime.deployment_readiness(spec, bundle.deployment)
        if not report_row.get("ready", True):
            missing = ", ".join(report_row.get("missing", []))
            location = report_row.get("env_file") or "the agent's .env"
            warnings.append(
                f"{spec.id}: cannot run yet — {missing} is not set. Add it to {location}; "
                "NOVA never writes credentials, so that file is yours and survives apply"
            )

    for result in results:
        warnings.extend(f"{result.agent_id}: {note}" for note in result.warnings)

    trace.add(
        created=len([r for r in results if r.created]),
        changed=len([r for r in results if r.changed and not r.created]),
        unchanged=len([r for r in results if r.unchanged]),
        skipped=len(skipped),
        warnings=len(warnings),
    )

    orphans = _orphans(bundle, runtime)
    if orphans:
        warnings.append(
            f"runtime still holds NOVA-managed agent(s) no longer in the bundle: "
            f"{', '.join(orphans)} — remove them explicitly if that is intended"
        )

    report = ApplyReport(
        tenant_id=bundle.tenant_id,
        runtime=runtime.name,
        correlation_id=correlation_id,
        bundle_digest=bundle.digest(),
        dry_run=dry_run,
        identity=identity_result,
        agents=tuple(results),
        skipped=tuple(skipped),
        warnings=tuple(warnings),
    )

    audit.record(
        "bundle.apply_finished",
        correlation_id=correlation_id,
        subject=bundle.tenant_id,
        digest=bundle.digest(),
        detail={
            "runtime": runtime.name,
            "dry_run": dry_run,
            "created": list(report.created),
            "changed": list(report.changed),
            "unchanged": list(report.unchanged),
            "skipped": list(report.skipped),
            "warnings": list(report.warnings),
        },
    )
    return report


def _orphans(bundle: TenantBundle, runtime: AgentRuntime) -> list[str]:
    """NOVA-managed agents present in the runtime but absent from the bundle.

    Channel-scoped variants count as declared: they exist because a channel declared an
    approval requirement, and reporting them as orphans would tell an operator to delete the
    profiles their own channel policy depends on.
    """
    from nova.channels.derive import plan_derivations

    declared = {spec.id for spec in bundle.agents}
    declared |= {entry.id for entry in plan_derivations(bundle)}
    return sorted(
        agent.agent_id
        for agent in runtime.list_agents()
        if agent.managed_by_nova and agent.agent_id not in declared
    )
