"""Public, plugin-safe lifecycle API for delegated Hermes subagents: immutable contracts, not ``AIAgent``
objects. Plugins obtain it via ``PluginContext.subagent_lifecycle``."""

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
import contextlib
import weakref
from contextlib import contextmanager
from concurrent.futures import Future, TimeoutError
from typing import Any, Callable, Mapping, Optional, Sequence

from agent.interrupt_compat import request_hard_interrupt

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
    goal: str
    context: Optional[str] = None
    role: str = "leaf"
    model: Optional[str] = None
    provider: Optional[str] = None
    allowed_toolsets: Optional[tuple[str, ...]] = None
    blocked_tools: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    parent_session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    timeout_seconds: Optional[float] = None


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

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SubagentHandle:
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
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
    # Batch identity, host-minted: the display position of this child (``task_index`` of
    # ``task_count``) and the live-transcript manifest its status is written back to.
    task_index: int = 0
    delegation_id: Optional[str] = None
    live_home: Any = None
    live_writer: Any = None


@dataclasses.dataclass
class _Prepared:
    """One validated request of a batch, with its resolved route, before any child is built."""

    request: SubagentLaunchRequest
    routing_cfg: dict[str, Any]
    creds: dict[str, Any]


@dataclasses.dataclass
class _Registry:
    """Thread-safe terminal-retention registry; never returns live records."""

    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    records: dict[str, _Record] = dataclasses.field(default_factory=dict)
    correlations: dict[tuple[Optional[str], str], str] = dataclasses.field(default_factory=dict)


_REGISTRY = _Registry()
from tools.daemon_pool import DaemonThreadPoolExecutor as _DaemonExecutor  # daemon: a wedged child never blocks exit
_EXECUTOR = _DaemonExecutor(max_workers=8, thread_name_prefix="hermes-lifecycle")
_SECRET = secrets.token_bytes(32)
_MANIFEST_LOCK = threading.Lock()
_ACTIVE_PARENT_AGENT: contextvars.ContextVar[Any] = contextvars.ContextVar("hermes_subagent_lifecycle_parent", default=None)


@contextmanager
def bind_subagent_parent(parent_agent: Any):
    """Bind the host-owned parent for the current agent turn.

    Stored as a weakref: every asyncio Handle/Future scheduled from the turn
    (LSP reader loops, kernel pipes, ...) snapshots the Context, and those
    snapshots outlive the turn. A strong ref there pinned finished delegate
    children — each of which binds itself here for its own turn — in the
    parent process heap for the life of the background loop.
    """
    try:
        ref = weakref.ref(parent_agent)
    except TypeError:
        ref = lambda: parent_agent
    token = _ACTIVE_PARENT_AGENT.set(ref)
    try:
        yield
    finally:
        _ACTIVE_PARENT_AGENT.reset(token)


def get_active_subagent_parent() -> Any:
    """Return the parent bound to this execution context, if any."""
    ref = _ACTIVE_PARENT_AGENT.get()
    return ref() if ref is not None else None


def _opt_str(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _session_id_of(agent: Any) -> Optional[str]:
    return str(getattr(agent, "session_id", "") or "") or None


def _clip(value: Any) -> Optional[str]:
    return str(value)[:_MAX_RESULT_CHARS] if value is not None else None


# Per-field shape check applied to a (possibly deserialized) handle before trusting it.
_HANDLE_FIELD_CHECKS: tuple[tuple[str, Callable[[Any], bool]], ...] = (
    ("contract_version", lambda v: type(v) is int and v == PUBLIC_CONTRACT_VERSION),
    ("subagent_id", lambda v: isinstance(v, str) and bool(v)),
    ("parent_session_id", _opt_str),
    ("correlation_id", _opt_str),
    ("created_at", lambda v: not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)),
    ("provider", _opt_str),
    ("model", _opt_str),
    ("role", lambda v: isinstance(v, str)),
    ("depth", lambda v: type(v) is int),
    ("capability", lambda v: isinstance(v, str)),
)

# Launch-request rejections in check order: (predicate, error). The type check leads so later predicates may
# dereference request fields.
_REQUEST_REJECTIONS: tuple[tuple[Callable[[Any], bool], str], ...] = (
    (lambda r: not isinstance(r, SubagentLaunchRequest) or not isinstance(r.goal, str) or not r.goal.strip() or len(r.goal) > _MAX_GOAL_CHARS,
     "goal must be a non-empty string of at most 16000 characters."),
    (lambda r: r.context is not None and (not isinstance(r.context, str) or len(r.context) > _MAX_CONTEXT_CHARS),
     "context must be a string of at most 32000 characters."),
    (lambda r: r.role not in {"leaf", "orchestrator"}, "role must be 'leaf' or 'orchestrator'."),
    (lambda r: r.provider is not None and (not isinstance(r.provider, str) or not r.provider.strip() or len(r.provider) > 64),
     "provider must be a non-empty string of at most 64 characters."),
    (lambda r: r.timeout_seconds is not None, "Per-launch timeout is not supported; configure delegation timeout explicitly."),
    (lambda r: r.working_directory is not None,
     "working_directory is not supported because Hermes delegates use isolated task environments."),
    (lambda r: bool(r.blocked_tools),
     "Per-tool blocking is not supported; use allowed_toolsets. Hermes always blocks unsafe child tools."),
)


def _handle_is_well_formed(handle: Any) -> bool:
    return isinstance(handle, SubagentHandle) and all(check(getattr(handle, field)) for field, check in _HANDLE_FIELD_CHECKS)


def _coerce_request(request: Any) -> SubagentLaunchRequest:
    if isinstance(request, SubagentLaunchRequest):
        return request
    if not isinstance(request, Mapping):
        raise SubagentLifecycleError("request must be a SubagentLaunchRequest or a mapping of its fields.")
    values = dict(request)
    unknown = set(values) - {field.name for field in dataclasses.fields(SubagentLaunchRequest)}
    if unknown:
        raise SubagentLifecycleError(f"Unknown request fields: {', '.join(sorted(map(str, unknown)))}.")
    for name in ("allowed_toolsets", "blocked_tools"):
        if isinstance(values.get(name), list):
            values[name] = tuple(values[name])
    try:
        return SubagentLaunchRequest(**values)
    except TypeError as exc:
        raise SubagentLifecycleError(f"Malformed subagent launch request: {exc}") from exc


def _coerce_handle(handle: Any) -> Optional[SubagentHandle]:
    if isinstance(handle, SubagentHandle):
        return handle
    if isinstance(handle, Mapping):
        try:
            return SubagentHandle.from_dict(handle)
        except SubagentLifecycleError:
            return None
    return None


class SubagentLifecycleService:
    """Stable public service behind :attr:`PluginContext.subagent_lifecycle`. Children run in-process only;
    completed results stay until process exit; ``reconnect`` reports that a serialized handle cannot
    reconnect after a restart instead of launching work again."""

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver

    def launch(self, request: SubagentLaunchRequest | Mapping[str, Any]) -> SubagentHandle:
        """Launch one child: a batch of one (see :meth:`launch_batch`)."""
        return self.launch_batch([request])[0]

    def launch_batch(self, requests: Sequence[SubagentLaunchRequest | Mapping[str, Any]]) -> list[SubagentHandle]:
        """Launch several children as ONE delegation, the way a multi-task ``delegate_task`` call does.

        The host mints the batch identity: one ``delegation_id`` shared by every child (UIs group
        the batch by it), ``task_index`` / ``task_count`` on every relayed ``subagent.*`` event, and
        a live transcript per child (``subagent.tail``). Handles come back in request order. Every
        request is validated and routed before any child is built, so a rejected request starts
        nothing: the whole batch fails with :class:`SubagentLifecycleError`.
        """
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError("No active Hermes parent session is available.")
        if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes, Mapping)) or not requests:
            raise SubagentLifecycleError("requests must be a non-empty sequence of launch requests.")
        parent_session_id = _session_id_of(parent)
        from tools.delegate_tool_config import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        if len(requests) > max_children:
            raise SubagentLifecycleError(
                f"A batch may hold at most {max_children} requests (delegation.max_concurrent_children); got {len(requests)}.")
        prepared = [self._prepare(request, parent, parent_session_id) for request in requests]
        correlations = [p.request.correlation_id for p in prepared if p.request.correlation_id]
        if len(set(correlations)) != len(correlations):
            raise SubagentLifecycleError("Duplicate correlation_id within one batch.")
        with _REGISTRY.lock:
            self._cleanup_locked()
            for correlation_id in correlations:
                if (parent_session_id, correlation_id) in _REGISTRY.correlations:
                    raise SubagentLifecycleError("Duplicate correlation_id for this parent session.")
        # Lazy: delegate construction stays internal, plugins never import private delegation helpers.
        from tools.delegate_tool import (
            _attach_live_delegation, _build_child_preserving_parent_tools, _credential_overrides, _parent_live_home,
            DEFAULT_MAX_ITERATIONS,
        )
        from tools.delegation_live_log import create_live_transcripts

        live_home = _parent_live_home(parent)
        first = prepared[0]
        delegation_id, writers, _paths = create_live_transcripts(
            [{"goal": p.request.goal, "context": p.request.context} for p in prepared],
            model=first.request.model or first.creds.get("model"), provider=first.creds.get("provider"), home=live_home,
        )
        handles: list[SubagentHandle] = []
        records: list[tuple[_Record, _Prepared]] = []
        try:
            for index, item in enumerate(prepared):
                request = item.request
                child = _build_child_preserving_parent_tools(
                    task_index=index, goal=request.goal, context=request.context,
                    toolsets=list(request.allowed_toolsets) if request.allowed_toolsets else None,
                    model=request.model or item.creds["model"], max_iterations=DEFAULT_MAX_ITERATIONS,
                    task_count=len(prepared), parent_agent=parent, role=request.role,
                    **_credential_overrides(item.creds, item.routing_cfg),
                )
                subagent_id = str(getattr(child, "_subagent_id", "") or "")
                if not subagent_id:
                    raise SubagentLifecycleError("Hermes failed to assign a child identity.")
                writer = writers[index] if index < len(writers) else None
                _attach_live_delegation(child, delegation_id, writer)
                created = time.time()
                handle = SubagentHandle(
                    PUBLIC_CONTRACT_VERSION, subagent_id, parent_session_id, request.correlation_id, created,
                    getattr(child, "provider", None), getattr(child, "model", None),
                    getattr(child, "_delegate_role", request.role), int(getattr(child, "_delegate_depth", 1) or 1),
                    self._capability(subagent_id, parent_session_id, created),
                )
                handles.append(handle)
                records.append((_Record(handle, SubagentState.PENDING, created, agent=child, task_index=index,
                                        delegation_id=delegation_id, live_home=live_home, live_writer=writer), item))
        except Exception as exc:
            # A sibling failed to build: the children built so far are attached to the parent and hold a
            # session each, but none has run. Release them so the batch is all-or-nothing in effect too.
            self._discard_built(records, parent, delegation_id, live_home, exc)
            if isinstance(exc, SubagentLifecycleError):
                raise
            raise SubagentLifecycleError(f"{type(exc).__name__}: {exc}") from exc
        with _REGISTRY.lock:
            for record, item in records:
                _REGISTRY.records[record.handle.subagent_id] = record
                if item.request.correlation_id:
                    _REGISTRY.correlations[(parent_session_id, item.request.correlation_id)] = record.handle.subagent_id
        for record, item in records:
            record.future = _EXECUTOR.submit(self._run, record, item.request.goal, parent)
        return handles

    @staticmethod
    def _discard_built(records: list, parent: Any, delegation_id: Optional[str], live_home: Any, exc: Exception) -> None:
        """Best-effort release of children built for a batch that never started."""
        from tools.delegate_tool_child_run import _close_child, _detach_child
        from tools.delegation_live_log import update_manifest_statuses
        entries = []
        for record, _item in records:
            with contextlib.suppress(Exception):
                _detach_child(parent, record.agent)
            _close_child(record.agent, "Could not close a batch child that never ran")
            entry = {"task_index": record.task_index, "status": "error", "exit_reason": "batch_failed",
                     "error": f"a sibling failed to build: {exc}"}
            entries.append(entry)
            if record.live_writer is not None:
                with contextlib.suppress(Exception):
                    record.live_writer.finalize(entry)
        if entries and delegation_id:
            with _MANIFEST_LOCK:
                update_manifest_statuses(delegation_id, entries, home=live_home)

    def _prepare(self, request: Any, parent: Any, parent_session_id: Optional[str]) -> _Prepared:
        """Validate one request and resolve its route; nothing here has side effects."""
        request = _coerce_request(request)
        self._validate_request(request, parent)
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError("parent_session_id does not match the active session.")
        from tools.delegate_tool import _load_config, _resolve_delegation_credentials
        # Routing precedence: request.provider > delegation config > parent.
        routing_cfg = ({"provider": request.provider, "model": request.model or ""} if request.provider is not None
                       else _load_config())
        try:
            creds = _resolve_delegation_credentials(routing_cfg, parent)
        except ValueError as exc:
            raise SubagentLifecycleError(str(exc)) from exc
        return _Prepared(request, routing_cfg, creds)

    def status(self, handle: SubagentHandle | Mapping[str, Any]) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            return SubagentStatus(handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE")
        with _REGISTRY.lock:
            return SubagentStatus(record.handle, record.state, record.updated_at)

    def wait(self, handle: SubagentHandle | Mapping[str, Any], *, timeout_seconds: Optional[float] = None) -> SubagentTerminalState:
        record = self._record(handle)
        if record is None:
            return SubagentTerminalState(handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE")
        try:
            if record.future is not None:
                record.future.result(timeout=timeout_seconds)
        except TimeoutError:
            return SubagentTerminalState(record.handle, record.state, False, True)
        except Exception:
            pass
        with _REGISTRY.lock:
            return SubagentTerminalState(record.handle, record.state, record.result is not None)

    def cancel(self, handle: SubagentHandle | Mapping[str, Any], *, reason: str) -> SubagentCancelResult:
        record = self._record(handle)
        if record is None:
            return SubagentCancelResult(False, unknown_handle=True)
        with _REGISTRY.lock:
            if record.result is not None:
                return SubagentCancelResult(False, already_terminal=True, state=record.state)
            agent = record.agent
            record.state = SubagentState.CANCEL_REQUESTED
            record.updated_at = time.time()
        accepted = False
        if agent is not None:
            with contextlib.suppress(Exception):
                accepted = request_hard_interrupt(
                    agent, f"Lifecycle cancellation requested: {reason[:500]}", tool_reason="subagent cancellation requested",
                )
        return SubagentCancelResult(bool(accepted), unsupported=not accepted, state=SubagentState.CANCEL_REQUESTED)

    def result(self, handle: SubagentHandle | Mapping[str, Any]) -> SubagentResult:
        record = self._record(handle)
        if record is None:
            return SubagentResult(handle, SubagentState.UNKNOWN, False, error_classification="UNKNOWN_HANDLE")
        with _REGISTRY.lock:
            return record.result or SubagentResult(record.handle, record.state, False, error_classification="NOT_READY")

    def reconnect(self, handle: SubagentHandle | Mapping[str, Any]) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            return SubagentReconnectResult(False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE")
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def steer(self, handle: SubagentHandle | Mapping[str, Any], text: str) -> bool:
        """Queue text for the child's next iteration boundary while it accepts steering."""
        record = self._record(handle)
        if record is None or not isinstance(text, str) or not text.strip():
            return False
        with _REGISTRY.lock:
            if record.result is not None:
                return False
        from tools.delegate_tool_registry import steer_subagent
        return steer_subagent(record.handle.subagent_id, text.strip())

    def _record(self, handle: SubagentHandle | Mapping[str, Any]) -> Optional[_Record]:
        """Registry record for a well-formed, capability-verified handle owned by the active parent."""
        handle = _coerce_handle(handle)
        if not _handle_is_well_formed(handle):
            return None
        expected = self._capability(handle.subagent_id, handle.parent_session_id, handle.created_at)
        if not hmac.compare_digest(handle.capability, expected):
            return None
        if _session_id_of(self._parent_agent_resolver()) != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            return _REGISTRY.records.get(handle.subagent_id)

    @staticmethod
    def _cleanup_locked() -> None:
        """Retain terminal snapshots for a bounded period, never live work."""
        cutoff = time.time() - _TERMINAL_RETENTION_SECONDS
        expired = [
            sid for sid, record in _REGISTRY.records.items()
            if record.result is not None and record.completed_at is not None and record.completed_at < cutoff
        ]
        for subagent_id in expired:
            handle = _REGISTRY.records.pop(subagent_id).handle
            if handle.correlation_id:
                _REGISTRY.correlations.pop((handle.parent_session_id, handle.correlation_id), None)

    def _run(self, record: _Record, goal: str, parent: Any) -> None:
        with _REGISTRY.lock:
            if record.state is not SubagentState.CANCEL_REQUESTED:
                record.state = SubagentState.RUNNING
            record.started_at = record.updated_at = time.time()
        try:
            from tools.delegate_tool import _run_child_lifecycle
            raw = _run_child_lifecycle(record.task_index, goal, record.agent, parent)
            is_dict = isinstance(raw, dict)
            raw = raw if is_dict else {}
            if is_dict:
                self._settle_transcript(record, raw)
            status = str(raw.get("status", "error"))
            if status == "interrupted":
                state = SubagentState.CANCELLED if record.state == SubagentState.CANCEL_REQUESTED else SubagentState.INTERRUPTED
            else:
                state = SubagentState.SUCCEEDED if status == "completed" else SubagentState.FAILED
            fields: dict[str, Any] = dict(
                summary=_clip(raw.get("summary")), error_message=_clip(raw.get("error") or None),
                error_classification=None if state == SubagentState.SUCCEEDED else status.upper(),
                usage_metadata={"api_calls": raw.get("api_calls", 0)} if is_dict else {},
                tool_execution_summary={"duration_seconds": raw.get("duration_seconds", 0)} if is_dict else {},
            )
        except Exception as exc:
            state = SubagentState.FAILED
            fields = dict(error_classification=type(exc).__name__, error_message=_clip(exc))
            self._settle_transcript(record, {"task_index": record.task_index, "status": "error",
                                             "exit_reason": "error", "error": f"{type(exc).__name__}: {exc}"})
        result = SubagentResult(record.handle, state, True, started_at=record.started_at, completed_at=time.time(), **fields)
        payload = dataclasses.asdict(result)
        payload.pop("result_hash", None)
        result = dataclasses.replace(result, result_hash=hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest())
        with _REGISTRY.lock:
            record.agent, record.result, record.state = None, result, result.terminal_state
            record.completed_at = record.updated_at = result.completed_at

    @staticmethod
    def _settle_transcript(record: _Record, entry: dict) -> None:
        """Same best-effort write-back as a delegate_task batch: the transcript's end marker and the
        manifest status. Siblings finish on their own threads, so the read-modify-write is serialized."""
        if record.live_writer is not None:
            with contextlib.suppress(Exception):
                record.live_writer.finalize(entry)
        if record.delegation_id:
            from tools.delegation_live_log import update_manifest_statuses
            with _MANIFEST_LOCK:
                update_manifest_statuses(record.delegation_id, [entry], home=record.live_home)

    @staticmethod
    def _capability(subagent_id: str, parent_session_id: Optional[str], created_at: float) -> str:
        value = f"{subagent_id}|{parent_session_id or ''}|{created_at:.6f}".encode()
        return hmac.new(_SECRET, value, hashlib.sha256).hexdigest()

    @staticmethod
    def _validate_request(request: SubagentLaunchRequest, parent: Any) -> None:
        for rejected, message in _REQUEST_REJECTIONS:
            if rejected(request):
                raise SubagentLifecycleError(message)
        try:
            metadata_bytes = len(json.dumps(dict(request.metadata), sort_keys=True).encode())
        except (TypeError, ValueError) as exc:
            raise SubagentLifecycleError("metadata must be JSON-serializable.") from exc
        if metadata_bytes > _MAX_METADATA_BYTES:
            raise SubagentLifecycleError("metadata exceeds 8192 bytes.")
        if not request.allowed_toolsets:
            return
        # Plugin and registry-alias toolsets count as known, as they do for delegate_task.
        from toolsets import validate_toolset
        unknown = {name for name in request.allowed_toolsets if not validate_toolset(name)}
        if unknown:
            raise SubagentLifecycleError(f"Unknown toolsets: {', '.join(sorted(unknown))}.")
        enabled = getattr(parent, "enabled_toolsets", None)
        if enabled is not None and not set(request.allowed_toolsets).issubset(set(enabled)):
            raise SubagentLifecycleError("Requested toolsets would broaden parent permissions.")
