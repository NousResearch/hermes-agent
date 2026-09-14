"""The runtime adapter contract.

Deliberately expressed in NOVA's vocabulary and no one else's. There is no ``profile``
in this interface, because a profile is a Hermes concept; there is no ``session``,
because that is a DSH concept. An adapter maps NOVA's *agent* onto whatever its runtime
calls the same idea.

Two rules keep this boundary honest, and both are checked by
``tests/platform/test_runtime_contract.py``:

1. Nothing in this module may import a runtime.
2. No runtime-specific vocabulary may appear in the names here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from nova.audit import AuditLog
from nova.errors import RuntimeAdapterError
from nova.policy import CompiledPolicy, agent_digest
from nova.knowledge.sources import KnowledgeCatalog
from nova.spec import AgentSpec, IdentitySpec
from nova.spec.deployment import DeploymentSpec


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What a runtime can actually do.

    NOVA asks rather than assumes. A capability a runtime lacks is reported here so the
    platform can degrade honestly — refusing the operation, or warning — instead of
    silently producing an agent that will not behave as its spec says.

    The fields are chosen to be the ones that differ between real runtimes, and each is
    a property a workforce deployment depends on.
    """

    #: Work survives a process restart and runs with no interactive session attached.
    durable_tasks: bool = False
    #: The runtime holds recurring/scheduled work per agent, and can report it.
    #:
    #: **It does not mean anything is running those schedules.** In Hermes the ticker
    #: lives inside the gateway and there is no standalone cron daemon, so a deployment
    #: can hold a correct schedule that never fires. That is a separate question, asked
    #: per agent via :meth:`AgentRuntime.scheduler_health`.
    scheduling: bool = False
    #: Concurrent agents get isolated working directories.
    worktree_isolation: bool = False
    #: Each agent runs in its own OS process.
    process_isolation: bool = False
    #: Each agent resolves credentials from its own per-agent store, so two agents given
    #: different keys get different keys.
    #:
    #: **It does not mean an agent cannot see another's credentials.** A worker inherits its
    #: parent's process environment, and in a dispatcher-spawned worker the runtime's
    #: multiplex guard is inactive, so a credential exported into the host environment is
    #: readable by every agent on that host. Verified: a worker reads a variable set only in
    #: the parent.
    #:
    #: Isolation therefore holds exactly when credentials live only in the per-agent store
    #: and never in the host environment. That is a deployment property, not a runtime
    #: guarantee, which is why :meth:`AgentRuntime.deployment_readiness` reports where each
    #: credential actually resolved from rather than only whether it resolved.
    credential_isolation: bool = False
    #: Per-agent tool restriction is enforced by the runtime.
    tool_scoping: bool = False
    #: An agent can search a declared corpus from inside a run. False means a declared
    #: knowledge source is carried and recorded but never reaches a model.
    knowledge_retrieval: bool = False
    #: The runtime can pull text out of formats beyond UTF-8 (PDF, Office, OpenDocument),
    #: through :meth:`AgentRuntime.extract_text`. Ingestion still works without it — it
    #: just skips what it cannot read, and says which files it skipped and why.
    document_extraction: bool = False
    #: NOVA can place work on the runtime's own board, with dependencies between items.
    #: False means an objective can be planned and routed but never submitted.
    work_submission: bool = False
    #: Per-agent tool policy is enforced inside the runtime, including escalation of
    #: business actions to a human. False means a declared policy would be inert.
    policy_enforcement: bool = False
    #: A human can act on work that is already on the board: release it, send it back,
    #: resume it, or leave a note a worker will read. False means the control plane stays
    #: read-only, because an approval nobody can act on is a button that lies.
    work_decisions: bool = False
    #: External communication channels can be connected, and inbound conversations routed to
    #: named agents. False means a declared channel would be carried and never delivered,
    #: which is worse than not offering channels at all.
    channel_delivery: bool = False
    #: Branding can be projected into the runtime's own display surfaces.
    brand_projection: bool = False

    def missing_for(self, required: Sequence[str]) -> list[str]:
        """Which of ``required`` this runtime does not provide."""
        return [name for name in required if not getattr(self, name, False)]


#: NOVA's canonical work states. Deliberately a small, generic set — a runtime whose own
#: vocabulary is richer maps into these and keeps its native value in
#: :attr:`TaskView.runtime_status`, so the UI is portable and nothing is lost.
TASK_STATES = ("pending", "running", "blocked", "review", "done", "archived")


@dataclass(frozen=True)
class TaskView:
    """One unit of work, as the control plane sees it.

    A read model: it is assembled for display and never written back. ``runtime_status``
    carries the runtime's own word for the state, because an operator debugging a stuck
    task needs the runtime's vocabulary, not a translation of it.
    """

    task_id: str
    title: str
    state: str
    runtime_status: str = ""
    agent_id: str = ""
    created_at: Optional[int] = None
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    priority: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    tenant_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_attention(self) -> bool:
        """Blocked, or failing repeatedly. What an operator should look at first."""
        return self.state == "blocked" or self.consecutive_failures > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "state": self.state,
            "runtime_status": self.runtime_status,
            "agent_id": self.agent_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "priority": self.priority,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "needs_attention": self.needs_attention,
        }


@dataclass(frozen=True)
class TaskRunView:
    """One attempt at a task.

    A task that failed twice and succeeded on the third try is three rows here, and an
    operator asking "why did this take all morning?" is asking about these, not about the
    task's current state.
    """

    run_id: int
    status: str
    outcome: str = ""
    started_at: Optional[int] = None
    ended_at: Optional[int] = None
    summary: str = ""
    error: str = ""
    agent_id: str = ""


@dataclass(frozen=True)
class TaskNoteView:
    """A note on a task — from a person or from the worker itself."""

    author: str
    body: str
    created_at: Optional[int] = None


@dataclass(frozen=True)
class ArtifactView:
    """A file a task produced.

    Deliberately carries no path. The runtime stores an absolute host path on every
    attachment row; it is attacker-useful and of no use to a browser, so it stops at the
    adapter. A download, when one exists, must be mediated by NOVA and re-checked against
    the tenant at request time.
    """

    artifact_id: int
    filename: str
    content_type: str = ""
    size_bytes: int = 0
    uploaded_by: str = ""
    created_at: Optional[int] = None


@dataclass(frozen=True)
class TaskDetail:
    """Everything the control plane can say about one task.

    Assembled from the runtime's own durable records — attempts, notes and artifacts all
    already exist; nothing here is derived or estimated.
    """

    task: TaskView
    runs: tuple[TaskRunView, ...] = ()
    notes: tuple[TaskNoteView, ...] = ()
    artifacts: tuple[ArtifactView, ...] = ()
    depends_on: tuple[str, ...] = ()
    blocks: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "runs": [
                {
                    "run_id": r.run_id, "status": r.status, "outcome": r.outcome,
                    "started_at": r.started_at, "ended_at": r.ended_at,
                    "summary": r.summary, "error": r.error, "agent_id": r.agent_id,
                }
                for r in self.runs
            ],
            "notes": [
                {"author": n.author, "body": n.body, "created_at": n.created_at}
                for n in self.notes
            ],
            "artifacts": [
                {
                    "artifact_id": a.artifact_id, "filename": a.filename,
                    "content_type": a.content_type, "size_bytes": a.size_bytes,
                    "uploaded_by": a.uploaded_by, "created_at": a.created_at,
                }
                for a in self.artifacts
            ],
            "depends_on": list(self.depends_on),
            "blocks": list(self.blocks),
        }



@dataclass(frozen=True)
class AutomationRunView:
    """One attempt at a scheduled automation, from the runtime's execution ledger."""

    run_id: str
    status: str
    claimed_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: str = ""


@dataclass(frozen=True)
class SchedulerHealth:
    """Whether anything is actually going to run this agent's schedules.

    Separate from :class:`RuntimeHealth` because it answers a different question, and a
    dangerous one to get wrong: a list of schedules with no scheduler attached looks
    identical to a working automation suite right up until nothing happens.

    ``running`` is "the loop is iterating"; ``healthy`` is "and it is completing ticks
    without raising". A ticker stuck failing keeps the first true and the second false.
    """

    running: bool = False
    healthy: bool = False
    heartbeat_age_seconds: Optional[float] = None
    success_age_seconds: Optional[float] = None
    last_error: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "healthy": self.healthy,
            "heartbeat_age_seconds": self.heartbeat_age_seconds,
            "success_age_seconds": self.success_age_seconds,
            "last_error": self.last_error,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AutomationView:
    """A recurring or scheduled piece of work the runtime holds for one agent.

    A read model. ``schedule_display`` is the runtime's own rendering of the schedule —
    re-deriving it from the expression would be a second implementation that drifts.

    Deliberately carries no prompt or script. What an automation *tells an agent to do*
    is instruction text; it is not needed to answer "what runs, when, and did it work",
    and every field a control plane returns is a field that can leak.
    """

    automation_id: str
    name: str
    agent_id: str
    schedule_display: str = ""
    schedule_kind: str = ""
    schedule_expression: str = ""
    enabled: bool = True
    state: str = ""
    next_run_at: Optional[str] = None
    last_run_at: Optional[str] = None
    last_status: str = ""
    last_error: str = ""
    failure_streak: int = 0
    paused_reason: str = ""
    created_at: Optional[str] = None
    runs: tuple[AutomationRunView, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "automation_id": self.automation_id,
            "name": self.name,
            "agent_id": self.agent_id,
            "schedule_display": self.schedule_display,
            "schedule_kind": self.schedule_kind,
            "schedule_expression": self.schedule_expression,
            "enabled": self.enabled,
            "state": self.state,
            "next_run_at": self.next_run_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "failure_streak": self.failure_streak,
            "paused_reason": self.paused_reason,
            "created_at": self.created_at,
            "runs": [
                {
                    "run_id": r.run_id, "status": r.status, "claimed_at": r.claimed_at,
                    "started_at": r.started_at, "finished_at": r.finished_at,
                    "error": r.error,
                }
                for r in self.runs
            ],
        }


@dataclass(frozen=True)
class RuntimeHealth:
    """Whether the runtime is present and readable.

    ``reachable`` false is a normal state, not an error: a fresh deployment has no work
    store until the runtime first runs. The control plane reports that plainly rather
    than failing.
    """

    reachable: bool
    detail: str = ""
    work_store_present: bool = False
    agent_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "detail": self.detail,
            "work_store_present": self.work_store_present,
            "agent_count": self.agent_count,
        }


@dataclass(frozen=True)
class ModelUsage:
    """Reported usage for one model."""

    model: str
    provider: str = ""
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cost_status: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "actual_cost_usd": round(self.actual_cost_usd, 6),
            "cost_status": self.cost_status,
        }


@dataclass(frozen=True)
class UsageSummary:
    """One agent's reported usage, with its own caveats attached.

    ``enforcement`` is fixed at ``observed_only`` and is part of the payload so no
    consumer can render these numbers as a budget without contradicting the data it was
    handed.
    """

    agent_id: str
    available: bool
    models: tuple[ModelUsage, ...] = ()
    detail: str = ""
    enforcement: str = "observed_only"
    #: Why these figures must not be treated as a ceiling or an invoice.
    caveats: tuple[str, ...] = field(
        default_factory=lambda: (
            "Not a spending limit: no plugin can veto a model call in this runtime.",
            "Lagging: usage is written by a coalescing background thread.",
            "Estimated: reconcile against the provider's billing before charging anyone.",
        )
    )

    @property
    def total_tokens(self) -> int:
        return sum(usage.total_tokens for usage in self.models)

    @property
    def estimated_cost_usd(self) -> float:
        return sum(usage.estimated_cost_usd for usage in self.models)

    @property
    def api_calls(self) -> int:
        return sum(usage.api_calls for usage in self.models)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "available": self.available,
            "detail": self.detail,
            "enforcement": self.enforcement,
            "caveats": list(self.caveats),
            "api_calls": self.api_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "models": [usage.to_dict() for usage in self.models],
        }


@dataclass(frozen=True)
class ExtractedDocument:
    """One document's text, as pulled out of whatever format it arrived in."""

    path: str
    text: str
    extracted: bool
    #: How the text was obtained: ``native`` (plain text read by NOVA) or the adapter's
    #: own extractor. Recorded so an operator can tell a real PDF extraction from a
    #: fallback that read the bytes as text.
    method: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "extracted": self.extracted,
            "method": self.method,
            "detail": self.detail,
            "characters": len(self.text),
        }


@dataclass(frozen=True)
class WorkItem:
    """One unit of work NOVA is asking a runtime to schedule.

    ``depends_on`` names other items by their NOVA ``key``, not by any runtime id: the
    caller does not know what the runtime will call them, and in a dry run there are no
    runtime ids at all. The adapter resolves keys to its own identifiers as it goes, which
    is why items arrive in dependency order.

    ``key`` is also the idempotency key. Submitting the same objective twice must produce
    one set of work items, not two — a supervisor that duplicates a month-end close on a
    retry is worse than one that does nothing.
    """

    key: str
    title: str
    assignee: str
    body: str = ""
    depends_on: tuple[str, ...] = ()
    priority: int = 0
    max_runtime_seconds: Optional[int] = None
    max_retries: Optional[int] = None
    #: Ask the runtime to decompose this item rather than run it as a single unit. An
    #: adapter with no decomposer treats it as an ordinary item and says so in the result.
    decompose: bool = False
    tenant_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "assignee": self.assignee,
            "depends_on": list(self.depends_on),
            "priority": self.priority,
            "decompose": self.decompose,
        }


@dataclass(frozen=True)
class SubmittedItem:
    """One work item as the runtime now holds it."""

    key: str
    task_id: str
    created: bool
    assignee: str = ""
    state: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "task_id": self.task_id,
            "created": self.created,
            "assignee": self.assignee,
            "state": self.state,
            "detail": self.detail,
        }


#: What a human can do to a work item that already exists. Deliberately four, and
#: deliberately not "set status to X": a control plane that can write any state can write an
#: inconsistent one, and the runtime's own transitions are the only ones that keep its
#: accounting straight.
WORK_ACTIONS = ("release", "reject", "resume", "annotate")


@dataclass(frozen=True)
class WorkDecision:
    """The outcome of one human decision about one work item.

    ``applied=False`` with a ``reason`` is the normal, expected answer, not an error: a task
    somebody already approved, or one that moved on while the operator was reading it, must
    say so plainly rather than raising. The alternative is an operator clicking twice and
    being told the second click crashed.
    """

    action: str
    task_id: str
    applied: bool
    #: Why it did not apply, in words an operator can act on. Empty when it did.
    reason: str = ""
    #: The runtime's own word for where the item ended up, when it moved.
    resulting_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = {"action": self.action, "task_id": self.task_id, "applied": self.applied}
        if self.reason:
            out["reason"] = self.reason
        if self.resulting_status:
            out["resulting_status"] = self.resulting_status
        return out


@dataclass(frozen=True)
class SubmitResult:
    """What a submission did."""

    items: tuple[SubmittedItem, ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)
    dry_run: bool = False

    @property
    def created(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.items if item.created)

    @property
    def existing(self) -> tuple[str, ...]:
        """Items a previous submission already created. Re-submitting is a no-op, by design."""
        return tuple(item.key for item in self.items if not item.created)

    def task_id(self, key: str) -> str:
        for item in self.items:
            if item.key == key:
                return item.task_id
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "created": list(self.created),
            "existing": list(self.existing),
            "items": [item.to_dict() for item in self.items],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class MaterializedAgent:
    """An agent as it currently exists inside a runtime.

    ``digest`` is the spec digest NOVA recorded when it wrote this agent; comparing it
    with a freshly loaded spec's digest detects drift — either the bundle changed, or
    someone edited the runtime directly.
    """

    agent_id: str
    display_name: str
    enabled: bool
    digest: str = ""
    location: Optional[Path] = None
    managed_by_nova: bool = True
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterializeResult:
    """What a materialization actually did."""

    agent_id: str
    created: bool
    changed: bool
    digest: str
    paths_written: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    location: Optional[Path] = None

    @property
    def unchanged(self) -> bool:
        return not self.created and not self.changed

    def to_detail(self) -> dict[str, Any]:
        """Audit-friendly summary."""
        return {
            "created": self.created,
            "changed": self.changed,
            "paths_written": [str(p) for p in self.paths_written],
            "warnings": list(self.warnings),
            "location": str(self.location) if self.location else "",
        }


class AgentRuntime(ABC):
    """Adapter onto one agent runtime.

    Implementations own every runtime-specific decision — where agents live on disk,
    what their configuration is called, how branding reaches the display. Callers get
    the same five operations regardless.
    """

    #: Short stable identifier, e.g. ``"hermes"``.
    name: str = ""

    @property
    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        """What this runtime supports. Never assume; always ask."""

    @abstractmethod
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
        """Create or update one agent inside the runtime.

        Must be idempotent: materializing an unchanged spec twice leaves the runtime
        identical and reports ``unchanged``. Must route the change through
        ``audit.model_visible_change`` — an agent's instructions and tools are
        model-visible by definition.

        ``policy`` is the agent's compiled policy, or None when the tenant declares none.
        An adapter that cannot enforce policy must say so through
        :attr:`RuntimeCapabilities.policy_enforcement` rather than accepting it silently.

        ``knowledge`` is the tenant's whole declared catalog, not this agent's slice of it.
        The adapter narrows it by ``spec.knowledge.sources`` and installs whatever retrieval
        mechanism it has. Passing the catalog rather than a resolved grant keeps the caller
        free of any assumption about how a corpus reaches an agent — a file beside a
        profile here, something else entirely in the next adapter.
        """

    def expected_digest(
        self,
        spec: AgentSpec,
        *,
        policy: Optional[CompiledPolicy] = None,
        knowledge: Optional[KnowledgeCatalog] = None,
    ) -> str:
        """The digest :meth:`materialize_agent` would record for this spec, without writing.

        The drift check and the materializer must agree on what an agent's identity *is*, and
        only the adapter knows everything that goes into it — an adapter that compiles extra
        runtime-specific state into an agent has to fold that state in here too. Two
        definitions of this would disagree the moment one of them grew a field, and the
        symptom is an agent that reports "out of sync" forever and re-materializes on every
        apply.

        The default covers the portable inputs; :func:`nova.policy.agent_digest` is the one
        implementation underneath.
        """
        return agent_digest(spec, policy)

    @abstractmethod
    def list_agents(self) -> list[MaterializedAgent]:
        """Every agent present in the runtime, NOVA-managed or not.

        Includes unmanaged ones deliberately: an operator needs to see what is there,
        and NOVA needs to refuse to overwrite what it does not own.
        """

    @abstractmethod
    def remove_agent(
        self, agent_id: str, *, audit: AuditLog, correlation_id: str, dry_run: bool = False
    ) -> bool:
        """Remove a NOVA-managed agent. Returns False when it was not present.

        Must refuse to remove an agent NOVA did not create, and must never delete
        customer data such as credentials or conversation history.
        """

    @abstractmethod
    def apply_identity(
        self,
        identity: IdentitySpec,
        *,
        audit: AuditLog,
        correlation_id: str,
        dry_run: bool = False,
    ) -> MaterializeResult:
        """Project branding into the runtime's display surfaces."""

    @abstractmethod
    def list_tasks(self, *, agent_id: str = "", limit: int = 200) -> list[TaskView]:
        """Work the runtime currently holds, newest first.

        A read model over whatever the runtime uses to track work. Implementations must
        open that store read-only: the control plane observes, it does not write.
        Returns an empty list when the runtime has no work store yet.
        """

    @abstractmethod
    def get_task(self, task_id: str) -> Optional[TaskView]:
        """One task, or None when it is not present."""

    def task_detail(self, task_id: str) -> Optional[TaskDetail]:
        """A task with its attempts, notes and artifacts, or None.

        Default returns the task alone, so a runtime that keeps no attempt history is
        still answerable — it reports what it has rather than obliging every adapter to
        invent a run model.
        """
        task = self.get_task(task_id)
        return None if task is None else TaskDetail(task=task)

    @abstractmethod
    def health(self) -> RuntimeHealth:
        """Whether the runtime is present and readable."""

    def list_automations(self, *, agent_id: str = "") -> list["AutomationView"]:
        """Recurring and scheduled work the runtime holds, per agent.

        Empty by default: a runtime with no scheduler reports none rather than obliging
        every adapter to model one. Check ``capabilities.scheduling`` to tell "none
        declared" from "this runtime cannot schedule".
        """
        return []

    def scheduler_health(self, agent_id: str) -> "SchedulerHealth":
        """Whether this agent's schedules will actually fire.

        A schedule the runtime holds and a schedule something is running are different
        facts, and the gap between them is invisible until work silently stops. Default
        reports not-running with a reason rather than implying health.
        """
        return SchedulerHealth(
            running=False,
            detail=f"runtime {self.name!r} does not report scheduler liveness",
        )

    def validate_schedule(self, schedule: str) -> None:
        """Raise ``SpecError`` when this runtime would not accept ``schedule``.

        Supplied to the automation compiler, which must not name a runtime itself. A
        runtime with no scheduler accepts anything here and refuses at create time.
        """
        return None

    def create_automation(self, agent_id: str, compiled) -> Optional["AutomationView"]:
        """Create a runtime job from a **compiled** automation; None when unsupported.

        Deliberately takes a compiled object rather than a prompt and a schedule. A
        runtime method that accepted free text would let any caller schedule an
        instruction that no policy reviewed, which is the thing the compiler exists to
        prevent.
        """
        return None

    def delete_automation(self, agent_id: str, automation_id: str) -> bool:
        """Remove one automation; False when absent or not this agent's."""
        return False

    def set_automation_enabled(
        self, agent_id: str, automation_id: str, *, enabled: bool, reason: str = "",
    ) -> Optional["AutomationView"]:
        """Pause or resume one automation; None when it is not this agent's.

        Only the enabled transition, and only through the runtime's own API — the
        runtime owns what pausing means (clearing the marker, recomputing the next run).
        """
        return None

    def extract_text(self, path: Path) -> ExtractedDocument:
        """Pull readable text out of a document.

        Document extraction is a *runtime capability*: a mature runtime already handles
        PDF, Office and OpenDocument formats, and re-implementing that would be weeks of
        work and a permanent liability. Adapters therefore expose theirs here rather than
        the platform carrying its own.

        The default reads UTF-8 text and nothing else, so a runtime that offers no
        extractor still ingests plain text and markdown instead of failing.
        """
        try:
            return ExtractedDocument(
                path=str(path),
                text=path.read_text(encoding="utf-8"),
                extracted=True,
                method="native",
            )
        except (OSError, UnicodeDecodeError) as exc:
            return ExtractedDocument(
                path=str(path),
                text="",
                extracted=False,
                method="native",
                detail=f"not readable as UTF-8 text and this runtime offers no extractor: {exc}",
            )

    def submit_work(
        self,
        items: Sequence[WorkItem],
        *,
        audit: AuditLog,
        correlation_id: str,
        dry_run: bool = False,
    ) -> SubmitResult:
        """Place work on the runtime's board, in dependency order.

        Items arrive already ordered so every item's dependencies precede it; an adapter
        resolves each ``depends_on`` key to the identifier it minted moments earlier.

        Must be idempotent on ``WorkItem.key``. Re-submitting an objective after a partial
        failure has to finish it, not duplicate the half that succeeded.

        Creating work is model-visible — a worker will read the title and body this places
        on the board — so an implementation routes it through
        ``audit.model_visible_change``.

        The default refuses. An adapter that cannot schedule work must not silently accept
        a plan and drop it; it reports ``work_submission=False`` and this is what happens
        if something calls it anyway.
        """
        raise RuntimeAdapterError(
            f"runtime {self.name!r} cannot accept submitted work "
            "(capabilities.work_submission is False)"
        )

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
        """Apply one human decision to one work item.

        ``action`` is one of :data:`WORK_ACTIONS`:

        ``release``   a queued or held item is let through to run.
        ``reject``    an item awaiting review goes back to whoever produced it, with a reason.
        ``resume``    a held item returns to whatever phase it was in.
        ``annotate``  a note is attached that a worker will read.

        ``actor`` is the authenticated human, and an implementation must carry it into
        whatever record the runtime keeps. An approval that cannot name who approved is not
        an approval, and this is the parameter that makes it one.

        ``annotate`` is **model-visible**: the note becomes part of what a worker reads as
        its instructions, exactly as a work item's body does. It goes through
        ``audit.model_visible_change``. ``release``, ``reject`` and ``resume`` change *when*
        a worker runs rather than *what it reads*, so they are recorded, not write-ahead —
        the distinction is the whole basis of the audit's honesty, and blurring it in either
        direction would make the log mean less.

        The default refuses, for the same reason :meth:`submit_work` does: a runtime that
        cannot act on work must not accept a decision and drop it.
        """
        raise RuntimeAdapterError(
            f"runtime {self.name!r} cannot act on work items "
            "(capabilities.work_decisions is False)"
        )

    def apply_channels(
        self,
        channels: Sequence[Any],
        *,
        audit: AuditLog,
        correlation_id: str,
        derivations: Sequence[Any] = (),
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Make the runtime deliver declared conversations to the agents that were granted.

        Connecting a channel is **model-visible in the most direct sense there is**: an
        inbound message becomes the text a worker reads, exactly as a work item's body does.
        So an implementation routes this through ``audit.model_visible_change``, and the
        write-ahead record is what lets an operator answer "when did this channel start
        reaching that agent" after the fact.

        An implementation must compile the declared grant into whatever the runtime uses to
        decide which agents it serves, so that a route escaping validation still cannot
        deliver. NOVA's parser refusing is the first gate; the runtime refusing is the one
        that holds when somebody edits configuration by hand.

        The default refuses. A runtime that cannot deliver a channel must not accept a
        declaration and drop it — a connected-looking channel that silently goes nowhere is
        the worst failure this layer has.
        """
        raise RuntimeAdapterError(
            f"runtime {self.name!r} cannot deliver channels "
            "(capabilities.channel_delivery is False)"
        )

    def channel_readiness(
        self, channels: Sequence[Any], derivations: Sequence[Any] = ()
    ) -> list[dict[str, Any]]:
        """Which credential variables each connection still needs, per granted agent.

        Names, never values — the same rule the model-provider seam runs on. A connection
        whose credential is absent is not broken configuration; it is configuration waiting
        for the operator step NOVA deliberately cannot take on their behalf.
        """
        return []

    @abstractmethod
    def usage(self, agent_id: str) -> UsageSummary:
        """Reported model usage for one agent.

        **Observation, not control.** An implementation must not present these figures as
        a budget: this runtime family cannot veto a model call, so a token or cost ceiling
        is not enforceable. The returned summary carries that caveat in its own payload.
        """

    @property
    @abstractmethod
    def state_location(self) -> Path:
        """Where this runtime keeps the state NOVA materializes into.

        Named in NOVA's terms because every runtime has one; what it contains and how it
        is laid out is the adapter's business and nobody else's.
        """

    def never_archive(self) -> tuple[str, ...]:
        """File and directory names inside the state location that NOVA must not archive.

        The runtime's own databases and the customer's conversation state: things NOVA
        neither wrote nor can safely restore over. Empty by default, because an adapter
        that names nothing is taken to own nothing — the safe reading, since the cost of
        forgetting is copying a customer's session history into a backup.
        """
        return ()

    def compatibility(self) -> list[str]:
        """Warnings about the runtime version this adapter is talking to.

        Empty by default: an adapter that makes no version claim is taken to make none,
        which is the safe reading and the honest one.
        """
        return []

    def deployment_readiness(
        self,
        spec: AgentSpec,
        deployment: Optional[DeploymentSpec] = None,
    ) -> dict[str, Any]:
        """Which credentials this agent still needs, and where they must go.

        On the contract because the answer is runtime-shaped — *where* an operator puts a
        credential differs per runtime — while the question is not. A platform that could
        only ask this of one adapter would report a second runtime as permanently ready.

        The default reports nothing required, which is the honest answer for an adapter that
        does not resolve credentials at all.
        """
        return {
            "agent_id": spec.id,
            "ready": True,
            "required": [],
            "missing": [],
            "warnings": [],
            "provider": {},
        }

    @property
    def knowledge_index_path(self) -> Path:
        """Where this runtime's deployment keeps the corpus index.

        On the contract rather than computed by callers, for the same reason
        :attr:`state_location` is: a CLI or a control plane that built this path itself
        would be encoding one adapter's directory layout, and would quietly point at the
        wrong file the first time an adapter laid its state out differently.
        """
        return self.state_location / "nova-knowledge.db"

    def limit_facts(self) -> tuple:
        """What each NOVA limit actually does on this runtime.

        Enforcement is a claim about a runtime, not about a limit, so each adapter
        declares its own. The default is empty: an adapter that says nothing is taken to
        enforce nothing, which is the safe reading.
        """
        return ()

    def describe(self) -> dict[str, Any]:
        """Adapter summary for operator tooling and the Control API."""
        return {
            "runtime": self.name,
            "state_location": str(self.state_location),
            "capabilities": {
                key: getattr(self.capabilities, key)
                for key in RuntimeCapabilities.__dataclass_fields__
            },
        }
