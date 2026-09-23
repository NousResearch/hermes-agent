"""What an interrupted run left behind, and what to do about it.

NOVA's audit log is write-ahead: an intent is recorded before a model-visible change and a
commit after it. That design exists to make a crash *recoverable*, and until now only half
of it was built — the CLI counted unfinished intents and printed a warning. Nothing
established what had actually happened, so an operator was told "a previous run was
interrupted" and left to work out what that meant for a customer's deployment.

The recovery is deliberately **diagnosis, not repair**. An unfinished intent means the
outcome is unknown, and the honest response to an unknown is to find out — not to guess and
act. So this compares the intent against the current state and says which of three things is
true:

``resolved``    the change is present and correct; the crash was between the write and the
                commit record, and re-running would be a no-op.
``incomplete``  the change is absent or partial; re-running will finish it.
``unknown``     the subject no longer exists, or the kind is not one that can be checked
                against state.

Only the second is a call to action, and the action is ``nova apply`` — which is idempotent,
already audited, and the thing an operator would have run anyway. NOVA does not need a
second, less-tested write path to clean up after the first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from nova.audit import AuditEvent, AuditLog
from nova.policy import agent_digest, compile_policy
from nova.runtime.base import AgentRuntime
from nova.spec import TenantBundle

RESOLVED = "resolved"
INCOMPLETE = "incomplete"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Finding:
    """One unfinished intent, and what the current state says about it."""

    kind: str
    subject: str
    correlation_id: str
    started_at: str
    state: str
    detail: str = ""

    @property
    def needs_action(self) -> bool:
        return self.state == INCOMPLETE

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "correlation_id": self.correlation_id,
            "started_at": self.started_at,
            "state": self.state,
            "detail": self.detail,
            "needs_action": self.needs_action,
        }


@dataclass
class Reconciliation:
    """Everything an interrupted run left unfinished."""

    findings: tuple[Finding, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def actionable(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.needs_action)

    def summary(self) -> str:
        if self.clean:
            return "no unfinished changes in the audit log"
        actionable = len(self.actionable)
        if not actionable:
            return (
                f"{len(self.findings)} unfinished intent(s), all resolved — the changes "
                "landed and only the commit record was lost"
            )
        return (
            f"{len(self.findings)} unfinished intent(s), {actionable} needing action. "
            "Re-run `nova apply` — it is idempotent and will finish them"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "findings": [f.to_dict() for f in self.findings],
            "warnings": list(self.warnings),
        }


def reconcile(
    audit: AuditLog,
    bundle: Optional[TenantBundle] = None,
    runtime: Optional[AgentRuntime] = None,
) -> Reconciliation:
    """Diagnose every unfinished intent against the current state.

    ``bundle`` and ``runtime`` are optional: without them every finding is ``unknown``,
    which is still better than a bare count, because it names what was in flight.
    """
    intents = audit.open_intents()
    if not intents:
        return Reconciliation()

    applied = {}
    if runtime is not None:
        try:
            applied = {agent.agent_id: agent for agent in runtime.list_agents()}
        except Exception as exc:  # noqa: BLE001 — diagnosis must not fail on a bad runtime
            return Reconciliation(
                findings=tuple(_undiagnosed(event) for event in intents),
                warnings=(f"could not read the runtime to diagnose these: {exc}",),
            )

    return Reconciliation(
        findings=tuple(_diagnose(event, bundle, runtime, applied) for event in intents)
    )


def _undiagnosed(event: AuditEvent) -> Finding:
    return Finding(
        kind=event.kind,
        subject=event.subject,
        correlation_id=event.correlation_id,
        started_at=event.ts,
        state=UNKNOWN,
        detail="no runtime available to check the current state against",
    )


def _diagnose(
    event: AuditEvent,
    bundle: Optional[TenantBundle],
    runtime: Optional[AgentRuntime],
    applied: dict,
) -> Finding:
    if event.kind == "agent.materialized" and bundle is not None and runtime is not None:
        return _diagnose_agent(event, bundle, runtime, applied)
    if event.kind == "knowledge.indexed":
        # The index is derived data and ingestion is idempotent, so an interrupted one is
        # finished by re-running rather than repaired.
        return Finding(
            kind=event.kind, subject=event.subject, correlation_id=event.correlation_id,
            started_at=event.ts, state=INCOMPLETE,
            detail="re-run `nova knowledge ingest`; it is incremental and will finish",
        )
    if event.kind == "work.submitted":
        return Finding(
            kind=event.kind, subject=event.subject, correlation_id=event.correlation_id,
            started_at=event.ts, state=INCOMPLETE,
            detail=(
                "re-submit the objective; work items are idempotent on their key, so a "
                "partially-submitted plan is completed rather than duplicated"
            ),
        )
    return _undiagnosed(event)


def _diagnose_agent(
    event: AuditEvent, bundle: TenantBundle, runtime: AgentRuntime, applied: dict
) -> Finding:
    """Whether the agent this intent covers matches what the bundle declares.

    The digest comparison is the same one the control plane uses for drift, asked of a
    different question: not "has the bundle moved on?" but "did this change land?".
    """
    live = applied.get(event.subject)
    if live is None:
        return Finding(
            kind=event.kind, subject=event.subject, correlation_id=event.correlation_id,
            started_at=event.ts, state=INCOMPLETE,
            detail="the agent was never created; `nova apply` will create it",
        )

    try:
        spec = bundle.agent(event.subject)
    except Exception:  # noqa: BLE001 — an agent dropped from the bundle since the crash
        return Finding(
            kind=event.kind, subject=event.subject, correlation_id=event.correlation_id,
            started_at=event.ts, state=UNKNOWN,
            detail="the bundle no longer declares this agent; remove it explicitly if that was intended",
        )

    policy = compile_policy(spec, bundle.policy) if bundle.policy is not None else None
    expected = runtime.expected_digest(
        spec, policy=policy, knowledge=bundle.knowledge, deployment=bundle.deployment
    )
    if live.digest == expected:
        return Finding(
            kind=event.kind, subject=event.subject, correlation_id=event.correlation_id,
            started_at=event.ts, state=RESOLVED,
            detail="the agent matches the bundle; only the commit record was lost",
        )
    return Finding(
        kind=event.kind, subject=event.subject, correlation_id=event.correlation_id,
        started_at=event.ts, state=INCOMPLETE,
        detail="the agent does not match the bundle; `nova apply` will bring it into line",
    )
