"""Control Room domain contract — backend-neutral DTOs.

This module is the single semantic model shared by all four Control Room
surfaces (CLI, Ink TUI, native Hermes Desktop, Kensei Dashboard). It is
deliberately pure: no I/O, no gateway imports, no database access. Providers
compose authoritative runtime state into these shapes; renderers consume them.

Contract versioning:
- ``SNAPSHOT_VERSION`` must be bumped on any breaking shape change.
- Every row carries ``source`` metadata and server-calculated
  ``available_actions`` — renderers must NOT guess capability.
- A partial/unavailable source yields a typed unavailable/degraded item,
  never an empty count pretending nothing exists.

The TypeScript mirror is generated deterministically by
``export_ts_schema.py`` into ``schema/control-room-v1.ts`` (see the strategy
docstring there). Python remains authoritative.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

SNAPSHOT_VERSION = 1

# ---------------------------------------------------------------------------
# Attention severity — ordering is contract: lower value = higher priority.
# ---------------------------------------------------------------------------


class AttentionSeverity(IntEnum):
    """Deterministic priority used to rank attention items and derive counts.

    Ordering contract (CR-103):
    1. approvals / held messages first  (``critical``)
    2. errors / stalled work            (``error``)
    3. blocked / review tasks           (``warning``)
    4. informational running/ready      (``info``)
    """

    critical = 0
    error = 1
    warning = 2
    info = 3


class AttentionKind(str, Enum):
    """Typed attention source. Kept separate from severity so a blocked task
    can be ``warning`` while a held peer message is ``critical``."""

    approval = "approval"
    held_message = "held_message"
    error = "error"
    stalled = "stalled"
    blocked_task = "blocked_task"
    review_task = "review_task"
    running = "running"
    ready = "ready"
    system = "system"
    info = "info"


# ---------------------------------------------------------------------------
# Capability flags — server-computed per snapshot; renderers never guess.
# ---------------------------------------------------------------------------


class Capabilities(BaseModel):
    """Which action families are genuinely wired for the scoped profile.

    A capability is ``True`` only when the authoritative backend path exists
    end-to-end in this runtime. Anything else is ``False`` and the renderer
    must show the action as unavailable — never a dead control.
    """

    approvals: bool = False
    peer_messages: bool = False
    kanban_actions: bool = False
    process_control: bool = False
    delegation_control: bool = False


# ---------------------------------------------------------------------------
# Row types
# ---------------------------------------------------------------------------


class SourceMeta(BaseModel):
    """Provenance for every row. ``state`` mirrors the provider's health:
    ``ok``, ``degraded``, or ``unavailable``. An unavailable provider yields a
    typed placeholder row (see ``unavailable_*`` constructors), never silence.
    """

    provider: str
    state: Literal["ok", "degraded", "unavailable"] = "ok"
    detail: str = ""


class AttentionItem(BaseModel):
    """One thing that may need the human. Row identity is ``kind:id``."""

    kind: AttentionKind
    id: str
    severity: AttentionSeverity
    title: str
    detail: str = ""
    profile: str = ""
    source: SourceMeta = Field(default_factory=lambda: SourceMeta(provider="unknown"))
    # ISO-8601 UTC timestamp used for deterministic tie-break ordering.
    updated_at: str = ""
    # Server-calculated; see ControlRoomAction. Renderers must not guess.
    available_actions: List[str] = Field(default_factory=list)

    @property
    def stable_id(self) -> str:
        return f"{self.kind.value}:{self.id}"


class AgentRow(BaseModel):
    """Live foreground agent, async delegation/subagent, or background process."""

    kind: Literal["agent", "delegation", "process"] = "agent"
    id: str
    name: str
    status: str = "running"
    detail: str = ""
    profile: str = ""
    source: SourceMeta = Field(default_factory=lambda: SourceMeta(provider="unknown"))
    available_actions: List[str] = Field(default_factory=list)

    @property
    def stable_id(self) -> str:
        return f"{self.kind}:{self.id}"


class TaskRow(BaseModel):
    """Kanban task summary by attention/running/ready state."""

    id: str
    title: str
    state: str = "unknown"
    board: str = ""
    owner: str = ""
    profile: str = ""
    latest_run: str = ""  # short human summary, not raw payload
    source: SourceMeta = Field(default_factory=lambda: SourceMeta(provider="unknown"))
    available_actions: List[str] = Field(default_factory=list)

    @property
    def stable_id(self) -> str:
        return f"task:{self.id}"


class MessageRow(BaseModel):
    """Hermes Peer inbox entry or request summary.

    Contract rule (CR-007): aggregated status counts must NEVER embed message
    bodies. This row carries titles/summaries only; full bodies are fetched
    by detail actions through the Peer public API.
    """

    kind: Literal["peer_message", "peer_request"] = "peer_message"
    id: str
    title: str
    state: str = "queued"
    sender: str = ""
    profile: str = ""
    source: SourceMeta = Field(default_factory=lambda: SourceMeta(provider="unknown"))
    available_actions: List[str] = Field(default_factory=list)

    @property
    def stable_id(self) -> str:
        return f"{self.kind}:{self.id}"


class SystemSummary(BaseModel):
    """Compact gateway/runtime health plus a small actionable error summary."""

    state: str = "unknown"  # healthy | degraded | down | unknown
    severity: AttentionSeverity = AttentionSeverity.info
    detail: str = ""
    errors: List[str] = Field(default_factory=list)
    source: SourceMeta = Field(default_factory=lambda: SourceMeta(provider="unknown"))
    available_actions: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class SnapshotCounts(BaseModel):
    """Aggregate counts. Must agree across every surface for the same
    profile/scope. Message counts are counts only — no bodies, ever."""

    needs_you: int = 0
    agents_active: int = 0
    tasks_running: int = 0
    messages_unread: int = 0
    system_severity: AttentionSeverity = AttentionSeverity.info


class ControlRoomSnapshot(BaseModel):
    """Profile-scoped read model returned to all renderers."""

    version: int = SNAPSHOT_VERSION
    profile: str = ""
    generated_at: str = ""
    attention: List[AttentionItem] = Field(default_factory=list)
    counts: SnapshotCounts = Field(default_factory=SnapshotCounts)
    agents: List[AgentRow] = Field(default_factory=list)
    tasks: List[TaskRow] = Field(default_factory=list)
    messages: List[MessageRow] = Field(default_factory=list)
    system: SystemSummary = Field(default_factory=SystemSummary)
    capabilities: Capabilities = Field(default_factory=Capabilities)

    def validate_invariants(self) -> List[str]:
        """Contract checks (CR-007): return a list of violated invariants.

        - counts.needs_you equals the count of attention items whose severity
          is ``critical`` or ``error`` (things that need the human).
        - every attention/agent/task/message row has a stable ``kind:id``
          identity.
        - no message body can leak into the snapshot (row fields are fixed;
          this validator re-checks the JSON payload contains no ``body`` key).
        - no cross-profile leakage: every row's ``profile`` is either the
          snapshot profile or explicitly labelled multi-profile.
        """
        violations: List[str] = []

        need_you_items = [
            a for a in self.attention
            if a.severity in (AttentionSeverity.critical, AttentionSeverity.error)
        ]
        if len(need_you_items) != self.counts.needs_you:
            violations.append(
                f"counts.needs_you={self.counts.needs_you} != attention critical/error={len(need_you_items)}"
            )

        for row in list(self.attention) + list(self.agents) + list(self.tasks) + list(self.messages):
            if not row.stable_id or ":" not in row.stable_id:
                violations.append(f"row missing stable kind:id identity: {row!r}")

        payload = self.model_dump(mode="json")
        if "body" in payload:
            violations.append("snapshot payload contains a 'body' key (message body leak)")

        return violations


# ---------------------------------------------------------------------------
# Action envelope and result
# ---------------------------------------------------------------------------


class ActionTarget(BaseModel):
    kind: str  # e.g. "approval", "process", "subagent", "delegation", "task", "peer_message", "peer_request"
    id: str
    profile: Optional[str] = None


class ControlRoomAction(BaseModel):
    id: str
    target: ActionTarget
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confirmation: Literal["none", "required"] = "required"
    # Optional revision guard: when set, the server must re-fetch the target
    # and refuse to mutate if the live revision differs (CR-201).
    expected_revision: Optional[str] = None


class ControlRoomActionResult(BaseModel):
    status: Literal[
        "completed",
        "confirmation_required",
        "rejected",
        "stale",
        "unavailable",
        "failed",
    ] = "failed"
    receipt: Optional[Dict[str, Any]] = None
    message: str = ""


# ---------------------------------------------------------------------------
# Error codes (machine-readable, stable across surfaces)
# ---------------------------------------------------------------------------


class ErrorCode(str, Enum):
    SNAPSHOT_UNAVAILABLE = "snapshot_unavailable"
    STALE_TARGET = "stale_target"
    CROSS_PROFILE = "cross_profile"
    UNAUTHORIZED = "unauthorized"
    UNKNOWN_ACTION = "unknown_action"
    UNKNOWN_TARGET = "unknown_target"
    BACKEND_FAILED = "backend_failed"
    CONFIRMATION_REQUIRED = "confirmation_required"


class ControlRoomError(BaseModel):
    code: ErrorCode
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Typed unavailable constructors — a partial source must surface as a
# typed unavailable/degraded item, never as an empty count pretending
# nothing exists (CR-104).
# ---------------------------------------------------------------------------


def unavailable_source(
    provider: str,
    detail: str = "provider unavailable in this runtime",
) -> SourceMeta:
    return SourceMeta(provider=provider, state="unavailable", detail=detail)


def unavailable_attention(
    provider: str,
    kind: AttentionKind = AttentionKind.info,
    detail: str = "provider unavailable in this runtime",
) -> AttentionItem:
    return AttentionItem(
        kind=kind,
        id="unavailable",
        severity=AttentionSeverity.info,
        title=f"{provider} unavailable",
        detail=detail,
        source=unavailable_source(provider, detail),
        available_actions=[],
    )


def unavailable_system(provider: str, detail: str = "provider unavailable in this runtime") -> SystemSummary:
    return SystemSummary(
        state="unknown",
        severity=AttentionSeverity.info,
        detail=detail,
        source=unavailable_source(provider, detail),
        available_actions=[],
    )
