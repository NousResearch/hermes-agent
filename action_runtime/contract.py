"""Action Runtime contract — the single schema for "the Orchestration Core asks
the Action Runtime to do something, and the Runtime reports back honestly".

This is the Phase 2 choke point from docs/architecture/central-brain-openclaw.md
(§5.3 ExecutionTask / §5.4 ExecutionResult). The gateway's exec handlers
(shell.exec, cli.exec, slash.exec, …) build an ``ExecutionResult`` via a thin
per-handler adapter and render it back to the existing JSON-RPC wire dict, so
the schema lives in ONE place while the wire stays byte-compatible (additive
only — see §12 compat rule).

Pure dataclasses with no dependency on the gateway: ``tui_gateway.server``
imports THIS module, never the reverse (no import cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Status(str, Enum):
    """Semantic outcome of an action — what actually happened, not whether the
    RPC round-tripped. Clients key off this, not on prose."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"
    NEEDS_INPUT = "needs_input"  # paused on a human-in-loop prompt
    BLOCKED = "blocked"          # refused by a pre-execution guard/policy


class ErrorType(str, Enum):
    """Coarse failure class. ``retryable`` (on ExecError) drives the Core's
    retry-vs-replan decision (§8); the type lets it pick a strategy."""

    TIMEOUT = "timeout"
    DENIED = "denied"                # dangerous-command guard, blocked argv
    NOT_FOUND = "not_found"
    PROVIDER_ERROR = "provider_error"  # plugin handler raised, model switch rejected
    TRANSPORT = "transport"          # broken pipe / worker died — typically retryable
    NONZERO_EXIT = "nonzero_exit"    # subprocess returned a non-zero exit code
    INTERNAL = "internal"


@dataclass
class ExecError:
    type: ErrorType
    retryable: bool
    message: str


@dataclass
class SideEffect:
    """Something the action changed in the world (a model switch, a compress,
    a killed process). ``applied`` is False when the side effect was attempted
    but did NOT take (e.g. a backend-rejected /model switch)."""

    kind: str
    detail: str = ""
    applied: bool = True
    target: Optional[str] = None


@dataclass
class NeedsInput:
    """Set when ``status == NEEDS_INPUT``: the action paused for a human."""

    kind: str  # "approval" | "clarify" | "secret"
    prompt: str


@dataclass
class Constraints:
    """Bounds the Runtime must not exceed. Phase 2 populates timeouts only;
    the rest are reserved for Phase 4's policy-aware execution."""

    timeout_s: Optional[int] = None
    network: str = "allow"     # "allow" | "deny"
    filesystem: str = "rw"     # "ro" | "rw" | "none"
    budget_tokens: Optional[int] = None


@dataclass
class ExecutionTask:
    """What the Core hands the Runtime. Phase 2 only fills task_id / inputs /
    constraints; intent/goal/success_criteria are reserved for Phase 4's
    intent-based ``task.submit`` so the shape need not change later."""

    task_id: str
    idempotency_key: Optional[str] = None  # re-submit safely (Broken-pipe cure, §8)
    intent: Optional[str] = None           # Phase 4: high-level, tool-agnostic
    goal: Any = None
    inputs: dict[str, Any] = field(default_factory=dict)
    constraints: Constraints = field(default_factory=Constraints)
    success_criteria: Optional[str] = None
    context_ref: Optional[str] = None      # session_id / memory pointer
    # §12 observability: correlates ONE logical request across
    # task → result → registry record. None = a pre-trace caller; the
    # Runtime never synthesizes one.
    trace_id: Optional[str] = None


@dataclass
class ExecutionResult:
    """What the Runtime reports back. ``outputs`` is the lossless payload bag —
    each adapter writes the handler's native keys here and reads them back out
    in ``*_to_wire`` so the JSON-RPC result stays byte-identical."""

    task_id: Optional[str]
    status: Status
    outputs: dict[str, Any] = field(default_factory=dict)
    error: Optional[ExecError] = None
    side_effects: list[SideEffect] = field(default_factory=list)
    needs_input: Optional[NeedsInput] = None
    # Echo of ExecutionTask.trace_id (§12): correlates one logical request
    # across task → result → registry record. None = a pre-trace caller;
    # the rich wire renders it ONLY when set.
    trace_id: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == Status.SUCCEEDED
