"""Public, plugin-safe lifecycle API for delegated Hermes subagents.

This module deliberately exposes immutable contracts, not ``AIAgent`` objects.
It is the supported boundary for plugins that need to supervise fresh child
sessions; plugins must obtain it from ``PluginContext.subagent_lifecycle``.
"""

from __future__ import annotations

import contextvars
import dataclasses
import enum
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, Mapping, Optional

from agent.interrupt_compat import request_hard_interrupt
from agent.worker_contract import (
    JobContract,
    WorkforceGovernance,
    WorkerConstitution,
)

PUBLIC_CONTRACT_VERSION = 1
_MAX_GOAL_CHARS = 16_000
_MAX_CONTEXT_CHARS = 32_000
_MAX_METADATA_BYTES = 8_192
_MAX_RESULT_CHARS = 32_000
_TERMINAL_RETENTION_SECONDS = 3_600


class SubagentLifecycleError(ValueError):
    """A request cannot be safely accepted by the public lifecycle API."""


class SubagentState(str, enum.Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


@dataclasses.dataclass(frozen=True)
class SubagentLaunchRequest:
    """Validated inputs for one plugin-owned child worker.

    ``constitution`` describes the worker profile and non-negotiable rules.
    ``job_contract`` adds a scoped, expiring assignment; when supplied it must
    be active at launch and match the constitution profile when both are set.
    """

    goal: str
    context: Optional[str] = None
    role: str = "leaf"
    model: Optional[str] = None
    allowed_toolsets: Optional[tuple[str, ...]] = None
    blocked_tools: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    parent_session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    constitution: Optional[WorkerConstitution] = None
    job_contract: Optional[JobContract] = None
    governance: Optional[WorkforceGovernance] = None


@dataclasses.dataclass(frozen=True)
class SubagentHandle:
    contract_version: int
    subagent_id: str
    parent_session_id: Optional[str]
    correlation_id: Optional[str]
    created_at: float
    provider: Optional[str]
    model: Optional[str]
    role: str
    depth: int
    capability: str
    worker_profile: Optional[str] = None
    constitution: Optional[WorkerConstitution] = None
    job_contract: Optional[JobContract] = None
    governance: Optional[WorkforceGovernance] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "subagent_id": self.subagent_id,
            "parent_session_id": self.parent_session_id,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at,
            "provider": self.provider,
            "model": self.model,
            "role": self.role,
            "depth": self.depth,
            "capability": self.capability,
            "worker_profile": self.worker_profile,
            "constitution": (
                self.constitution.to_dict() if self.constitution is not None else None
            ),
            "job_contract": (
                self.job_contract.to_dict() if self.job_contract is not None else None
            ),
            "governance": (
                self.governance.to_dict() if self.governance is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentHandle":
        try:
            data = dict(value)
            constitution = data.get("constitution")
            if constitution is not None:
                constitution = WorkerConstitution(
                    profile=constitution["profile"],
                    values=tuple(constitution["values"]),
                    authority=constitution["authority"],
                    forbidden_actions=tuple(constitution.get("forbidden_actions", ())),
                    required_evidence=tuple(constitution.get("required_evidence", ())),
                    escalation_path=constitution.get(
                        "escalation_path", "operator-review"
                    ),
                )
                constitution.validate()
            job_contract = data.get("job_contract")
            if job_contract is not None:
                job_contract = JobContract(
                    name=job_contract["name"],
                    worker_profile=job_contract["worker_profile"],
                    job=job_contract["job"],
                    scope=tuple(job_contract["scope"]),
                    authority=job_contract["authority"],
                    obligations=tuple(job_contract.get("obligations", ())),
                    granted_at=job_contract["granted_at"],
                    expires_at=job_contract["expires_at"],
                    requires_review=job_contract.get("requires_review", True),
                )
                job_contract.validate()
            governance = data.get("governance")
            if governance is not None:
                governance = WorkforceGovernance.from_dict(governance)
            data["constitution"] = constitution
            data["job_contract"] = job_contract
            data["governance"] = governance
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise SubagentLifecycleError("Malformed subagent handle.") from exc


@dataclasses.dataclass(frozen=True)
class SubagentStatus:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentTerminalState:
    handle: SubagentHandle
    state: SubagentState
    completed: bool
    timed_out: bool = False
    diagnostic: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentCancelResult:
    accepted: bool
    already_terminal: bool = False
    unknown_handle: bool = False
    unsupported: bool = False
    state: SubagentState = SubagentState.UNKNOWN


@dataclasses.dataclass(frozen=True)
class SubagentResult:
    handle: SubagentHandle
    terminal_state: SubagentState
    ready: bool
    summary: Optional[str] = None
    structured_payload: Optional[Mapping[str, Any]] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_classification: Optional[str] = None
    error_message: Optional[str] = None
    usage_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    tool_execution_summary: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    result_hash: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class SubagentReconnectResult:
    connected: bool
    state: SubagentState
    diagnostic: Optional[str] = None


@dataclasses.dataclass
class _Record:
    handle: SubagentHandle
    state: SubagentState
    updated_at: float
    agent: Any = None
    future: Optional[Future] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[SubagentResult] = None


class _Registry:
    """Thread-safe terminal-retention registry; never returns live records."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.records: dict[str, _Record] = {}
        self.correlations: dict[tuple[Optional[str], str], str] = {}


_REGISTRY = _Registry()
# Daemon worker pool: a wedged/abandoned child must never block interpreter
# exit at atexit-join time (same rationale as _run_single_child's timeout
# executor and the async-delegation registry pool).
from tools.daemon_pool import DaemonThreadPoolExecutor as _DaemonExecutor

_EXECUTOR = _DaemonExecutor(max_workers=8, thread_name_prefix="hermes-lifecycle")
_SECRET = secrets.token_bytes(32)
_ACTIVE_PARENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "hermes_subagent_lifecycle_parent", default=None
)


@contextmanager
def bind_subagent_parent(parent_agent: Any):
    """Bind the host-owned parent for the current agent turn."""
    token = _ACTIVE_PARENT_AGENT.set(parent_agent)
    try:
        yield
    finally:
        _ACTIVE_PARENT_AGENT.reset(token)


def get_active_subagent_parent() -> Any:
    """Return the parent bound to this execution context, if any."""
    return _ACTIVE_PARENT_AGENT.get()


class SubagentLifecycleService:
    """Stable public service returned by :attr:`PluginContext.subagent_lifecycle`.

    Running children are in-process only.  Completed results remain available
    until process exit; ``reconnect`` accurately reports that a serialized
    handle cannot reconnect after a restart instead of launching work again.
    """

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver

    def launch(self, request: SubagentLaunchRequest) -> SubagentHandle:
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError(
                "No active Hermes parent session is available."
            )
        self._validate_request(request, parent)
        parent_session_id = str(getattr(parent, "session_id", "") or "") or None
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError(
                "parent_session_id does not match the active session."
            )
        correlation_key = (parent_session_id, request.correlation_id or "")
        with _REGISTRY.lock:
            self._cleanup_locked()
            if request.correlation_id and correlation_key in _REGISTRY.correlations:
                raise SubagentLifecycleError(
                    "Duplicate correlation_id for this parent session."
                )

        # Delegate construction remains internal so plugin code never imports
        # private delegation helpers or manipulates the active-child registry.
        from tools.delegate_tool import (
            _build_child_preserving_parent_tools,
            DEFAULT_MAX_ITERATIONS,
        )

        child = _build_child_preserving_parent_tools(
            task_index=0,
            goal=request.goal,
            context=self._workforce_context(request),
            toolsets=list(request.allowed_toolsets)
            if request.allowed_toolsets
            else None,
            model=request.model,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            task_count=1,
            parent_agent=parent,
            role=request.role,
        )
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        if not subagent_id:
            raise SubagentLifecycleError("Hermes failed to assign a child identity.")
        child._worker_constitution = request.constitution
        child._job_contract = request.job_contract
        child._workforce_governance = request.governance
        created = time.time()
        handle = SubagentHandle(
            PUBLIC_CONTRACT_VERSION,
            subagent_id,
            parent_session_id,
            request.correlation_id,
            created,
            getattr(child, "provider", None),
            getattr(child, "model", None),
            getattr(child, "_delegate_role", request.role),
            int(getattr(child, "_delegate_depth", 1) or 1),
            self._capability(subagent_id, parent_session_id, created),
            (
                request.job_contract.worker_profile
                if request.job_contract is not None
                else request.constitution.profile
                if request.constitution is not None
                else None
            ),
            request.constitution,
            request.job_contract,
            request.governance,
        )
        record = _Record(handle, SubagentState.PENDING, created, agent=child)
        with _REGISTRY.lock:
            _REGISTRY.records[subagent_id] = record
            if request.correlation_id:
                _REGISTRY.correlations[correlation_key] = subagent_id
        record.future = _EXECUTOR.submit(self._run, record, request.goal, parent)
        return handle

    @staticmethod
    def _workforce_context(request: SubagentLaunchRequest) -> Optional[str]:
        """Add validated assignment rules to the child-facing task context."""
        if (
            request.constitution is None
            and request.job_contract is None
            and request.governance is None
        ):
            return request.context

        assignment: dict[str, Any] = {}
        if request.constitution is not None:
            assignment["constitution"] = request.constitution.to_dict()
        if request.job_contract is not None:
            assignment["job_contract"] = request.job_contract.to_dict()
        if request.governance is not None:
            assignment["governance"] = request.governance.to_dict()
        rendered = json.dumps(assignment, sort_keys=True)
        block = (
            "GOVERNED WORKFORCE ASSIGNMENT:\n"
            "The following host-validated rules describe this worker's role. "
            "Treat them as task constraints; Hermes tool permissions remain "
            "the authority boundary.\n"
            f"{rendered}"
        )
        if request.context and request.context.strip():
            return f"{request.context.rstrip()}\n\n{block}"
        return block

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            return SubagentStatus(
                handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE"
            )
        with _REGISTRY.lock:
            return SubagentStatus(record.handle, record.state, record.updated_at)

    def wait(
        self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None
    ) -> SubagentTerminalState:
        record = self._record(handle)
        if record is None:
            return SubagentTerminalState(
                handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE"
            )
        future = record.future
        if future is not None:
            try:
                future.result(timeout=timeout_seconds)
            except TimeoutError:
                return SubagentTerminalState(record.handle, record.state, False, True)
            except Exception:
                pass
        with _REGISTRY.lock:
            return SubagentTerminalState(
                record.handle, record.state, record.result is not None
            )

    def cancel(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
        record = self._record(handle)
        if record is None:
            return SubagentCancelResult(False, unknown_handle=True)
        with _REGISTRY.lock:
            if record.result is not None:
                return SubagentCancelResult(
                    False, already_terminal=True, state=record.state
                )
            agent = record.agent
            record.state = SubagentState.CANCEL_REQUESTED
            record.updated_at = time.time()
        if agent is None:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        try:
            accepted = request_hard_interrupt(
                agent,
                f"Lifecycle cancellation requested: {reason[:500]}",
                tool_reason="subagent cancellation requested",
            )
        except Exception:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        if not accepted:
            return SubagentCancelResult(
                False, unsupported=True, state=SubagentState.CANCEL_REQUESTED
            )
        return SubagentCancelResult(True, state=SubagentState.CANCEL_REQUESTED)

    def result(self, handle: SubagentHandle) -> SubagentResult:
        record = self._record(handle)
        if record is None:
            return SubagentResult(
                handle,
                SubagentState.UNKNOWN,
                False,
                error_classification="UNKNOWN_HANDLE",
            )
        with _REGISTRY.lock:
            if record.result is not None:
                return record.result
            return SubagentResult(
                record.handle, record.state, False, error_classification="NOT_READY"
            )

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            return SubagentReconnectResult(
                False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE"
            )
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def _record(self, handle: SubagentHandle) -> Optional[_Record]:
        if (
            not isinstance(handle, SubagentHandle)
            or type(handle.contract_version) is not int
            or handle.contract_version != PUBLIC_CONTRACT_VERSION
        ):
            return None
        if (
            not isinstance(handle.subagent_id, str)
            or not handle.subagent_id
            or (
                handle.parent_session_id is not None
                and not isinstance(handle.parent_session_id, str)
            )
            or (
                handle.correlation_id is not None
                and not isinstance(handle.correlation_id, str)
            )
            or isinstance(handle.created_at, bool)
            or not isinstance(handle.created_at, (int, float))
            or not math.isfinite(handle.created_at)
            or (handle.provider is not None and not isinstance(handle.provider, str))
            or (handle.model is not None and not isinstance(handle.model, str))
            or not isinstance(handle.role, str)
            or type(handle.depth) is not int
            or not isinstance(handle.capability, str)
        ):
            return None
        if not hmac.compare_digest(
            handle.capability,
            self._capability(
                handle.subagent_id, handle.parent_session_id, handle.created_at
            ),
        ):
            return None
        parent = self._parent_agent_resolver()
        active_parent_id = str(getattr(parent, "session_id", "") or "") or None
        if active_parent_id != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            return _REGISTRY.records.get(handle.subagent_id)

    @staticmethod
    def _cleanup_locked() -> None:
        """Retain terminal snapshots for a bounded period, never live work."""
        cutoff = time.time() - _TERMINAL_RETENTION_SECONDS
        expired = [
            subagent_id
            for subagent_id, record in _REGISTRY.records.items()
            if record.result is not None
            and record.completed_at is not None
            and record.completed_at < cutoff
        ]
        for subagent_id in expired:
            record = _REGISTRY.records.pop(subagent_id)
            if record.handle.correlation_id:
                _REGISTRY.correlations.pop(
                    (record.handle.parent_session_id, record.handle.correlation_id),
                    None,
                )

    def _run(self, record: _Record, goal: str, parent: Any) -> None:
        with _REGISTRY.lock:
            if record.state is not SubagentState.CANCEL_REQUESTED:
                record.state = SubagentState.RUNNING
            record.started_at = time.time()
            record.updated_at = record.started_at
        try:
            from tools.delegate_tool import _run_child_lifecycle

            raw = _run_child_lifecycle(0, goal, record.agent, parent)
            status = (
                str(raw.get("status", "error")) if isinstance(raw, dict) else "error"
            )
            if status == "completed":
                state = SubagentState.SUCCEEDED
            elif status == "interrupted":
                state = (
                    SubagentState.CANCELLED
                    if record.state == SubagentState.CANCEL_REQUESTED
                    else SubagentState.INTERRUPTED
                )
            else:
                state = SubagentState.FAILED
            summary = raw.get("summary") if isinstance(raw, dict) else None
            summary = str(summary)[:_MAX_RESULT_CHARS] if summary is not None else None
            error = raw.get("error") if isinstance(raw, dict) else None
            result = SubagentResult(
                record.handle,
                state,
                True,
                summary=summary,
                completed_at=time.time(),
                started_at=record.started_at,
                error_classification=None
                if state == SubagentState.SUCCEEDED
                else status.upper(),
                error_message=str(error)[:_MAX_RESULT_CHARS] if error else None,
                usage_metadata={"api_calls": raw.get("api_calls", 0)}
                if isinstance(raw, dict)
                else {},
                tool_execution_summary={
                    "duration_seconds": raw.get("duration_seconds", 0)
                }
                if isinstance(raw, dict)
                else {},
            )
        except Exception as exc:
            result = SubagentResult(
                record.handle,
                SubagentState.FAILED,
                True,
                started_at=record.started_at,
                completed_at=time.time(),
                error_classification=type(exc).__name__,
                error_message=str(exc)[:_MAX_RESULT_CHARS],
            )
        payload = dataclasses.asdict(result)
        payload.pop("result_hash", None)
        result = dataclasses.replace(
            result,
            result_hash=hashlib.sha256(
                json.dumps(payload, sort_keys=True, default=str).encode()
            ).hexdigest(),
        )
        with _REGISTRY.lock:
            record.agent = None
            record.result = result
            record.state = result.terminal_state
            record.completed_at = result.completed_at
            record.updated_at = result.completed_at or time.time()

    @staticmethod
    def _capability(
        subagent_id: str, parent_session_id: Optional[str], created_at: float
    ) -> str:
        value = f"{subagent_id}|{parent_session_id or ''}|{created_at:.6f}".encode()
        return hmac.new(_SECRET, value, hashlib.sha256).hexdigest()

    @staticmethod
    def _validate_request(request: SubagentLaunchRequest, parent: Any) -> None:
        if (
            not isinstance(request, SubagentLaunchRequest)
            or not isinstance(request.goal, str)
            or not request.goal.strip()
            or len(request.goal) > _MAX_GOAL_CHARS
        ):
            raise SubagentLifecycleError(
                "goal must be a non-empty string of at most 16000 characters."
            )
        if request.context is not None and (
            not isinstance(request.context, str)
            or len(request.context) > _MAX_CONTEXT_CHARS
        ):
            raise SubagentLifecycleError(
                "context must be a string of at most 32000 characters."
            )
        if request.role not in {"leaf", "orchestrator"}:
            raise SubagentLifecycleError("role must be 'leaf' or 'orchestrator'.")
        if request.timeout_seconds is not None:
            raise SubagentLifecycleError(
                "Per-launch timeout is not supported; configure delegation timeout explicitly."
            )
        if request.working_directory is not None:
            raise SubagentLifecycleError(
                "working_directory is not supported because Hermes delegates use isolated task environments."
            )
        if request.blocked_tools:
            raise SubagentLifecycleError(
                "Per-tool blocking is not supported; use allowed_toolsets. Hermes always blocks unsafe child tools."
            )
        if request.constitution is not None:
            if not isinstance(request.constitution, WorkerConstitution):
                raise SubagentLifecycleError(
                    "constitution must be a WorkerConstitution."
                )
            try:
                request.constitution.validate()
            except ValueError as exc:
                raise SubagentLifecycleError(str(exc)) from exc
        if request.job_contract is not None:
            if not isinstance(request.job_contract, JobContract):
                raise SubagentLifecycleError("job_contract must be a JobContract.")
            try:
                request.job_contract.validate()
                if not request.job_contract.is_active():
                    raise SubagentLifecycleError(
                        "job_contract must be active at launch."
                    )
            except SubagentLifecycleError:
                raise
            except ValueError as exc:
                raise SubagentLifecycleError(str(exc)) from exc
        if request.governance is not None:
            if not isinstance(request.governance, WorkforceGovernance):
                raise SubagentLifecycleError(
                    "governance must be a WorkforceGovernance."
                )
            try:
                request.governance.validate()
            except ValueError as exc:
                raise SubagentLifecycleError(str(exc)) from exc
        if (
            request.constitution is not None
            and request.job_contract is not None
            and request.constitution.profile != request.job_contract.worker_profile
        ):
            raise SubagentLifecycleError(
                "constitution profile must match job_contract worker_profile."
            )
        try:
            metadata_bytes = len(
                json.dumps(dict(request.metadata), sort_keys=True).encode()
            )
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("metadata must be JSON-serializable.") from exc
        if metadata_bytes > _MAX_METADATA_BYTES:
            raise SubagentLifecycleError("metadata exceeds 8192 bytes.")
        if request.allowed_toolsets:
            from toolsets import TOOLSETS

            unknown = set(request.allowed_toolsets) - set(TOOLSETS)
            if unknown:
                raise SubagentLifecycleError(
                    f"Unknown toolsets: {', '.join(sorted(unknown))}."
                )
            enabled = getattr(parent, "enabled_toolsets", None)
            if enabled is not None and not set(request.allowed_toolsets).issubset(
                set(enabled)
            ):
                raise SubagentLifecycleError(
                    "Requested toolsets would broaden parent permissions."
                )
