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
from typing import Any, Callable, Mapping, Optional

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
    allowed_toolsets: Optional[tuple[str, ...]] = None
    blocked_tools: tuple[str, ...] = ()
    working_directory: Optional[str] = None
    parent_session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    timeout_seconds: Optional[float] = None
    # Opt in to a private, single-turn child on this profile's configured default route.
    # This deliberately ignores delegation.provider/model and all fallback chains.
    private_default_route: bool = False


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
    def from_dict(cls, value: Mapping[str, Any]) -> "SubagentHandle":
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
    profile_key: Optional[str] = None


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
        ref = lambda: parent_agent  # noqa: E731 — non-weakrefable test doubles
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
    (lambda r: r.timeout_seconds is not None, "Per-launch timeout is not supported; use wait(timeout_seconds=...) and cancel()."),
    (lambda r: type(r.private_default_route) is not bool, "private_default_route must be a boolean."),
    (lambda r: r.private_default_route and (r.model is not None or r.allowed_toolsets is not None or r.role != "leaf"),
     "Private default-route children must be leaf children without model or toolset overrides."),
    (lambda r: r.working_directory is not None,
     "working_directory is not supported because Hermes delegates use isolated task environments."),
    (lambda r: bool(r.blocked_tools),
     "Per-tool blocking is not supported; use allowed_toolsets. Hermes always blocks unsafe child tools."),
)


def _handle_is_well_formed(handle: Any) -> bool:
    return isinstance(handle, SubagentHandle) and all(check(getattr(handle, field)) for field, check in _HANDLE_FIELD_CHECKS)


class SubagentLifecycleService:
    """Stable public service behind :attr:`PluginContext.subagent_lifecycle`. Children run in-process only;
    completed results stay until process exit; ``reconnect`` reports that a serialized handle cannot
    reconnect after a restart instead of launching work again."""

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver

    def launch(self, request: SubagentLaunchRequest) -> SubagentHandle:
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError("No active Hermes parent session is available.")
        self._validate_request(request, parent)
        parent_session_id = _session_id_of(parent)
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError("parent_session_id does not match the active session.")
        from hermes_constants import hermes_home_key
        profile_key = hermes_home_key() if request.private_default_route else None
        correlation_key = (f"profile:{profile_key}:{parent_session_id}" if profile_key else parent_session_id,
                           request.correlation_id or "")
        with _REGISTRY.lock:
            self._cleanup_locked()
            if request.correlation_id and correlation_key in _REGISTRY.correlations:
                raise SubagentLifecycleError("Duplicate correlation_id for this parent session.")
        # Resolve the entire route before child construction (which embeds goal/context in its prompt)
        # and before submission. Never use delegation.provider or its fallback policy here.
        private_route = self._private_route() if request.private_default_route else None
        # Lazy: delegate construction stays internal, plugins never import private delegation helpers.
        from tools.delegate_tool import _build_child_preserving_parent_tools, DEFAULT_MAX_ITERATIONS
        try:
            child = _build_child_preserving_parent_tools(
                task_index=0, goal=request.goal, context=request.context,
                toolsets=list(request.allowed_toolsets) if request.allowed_toolsets else None,
                model=private_route["model"] if private_route else request.model,
                max_iterations=1 if private_route else DEFAULT_MAX_ITERATIONS,
                task_count=1, parent_agent=parent, role=request.role,
                private_default_route=bool(private_route),
                **(private_route["overrides"] if private_route else {}),
            )
        except Exception as exc:
            if private_route:
                raise SubagentLifecycleError("Private default-route child could not be constructed.") from exc
            raise
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        if not subagent_id:
            raise SubagentLifecycleError("Hermes failed to assign a child identity.")
        created = time.time()
        handle = SubagentHandle(
            PUBLIC_CONTRACT_VERSION, subagent_id, parent_session_id, request.correlation_id, created,
            getattr(child, "provider", None), getattr(child, "model", None), getattr(child, "_delegate_role", request.role),
            int(getattr(child, "_delegate_depth", 1) or 1), self._capability(subagent_id, parent_session_id, created),
        )
        record = _Record(handle, SubagentState.PENDING, created, agent=child, profile_key=profile_key)
        with _REGISTRY.lock:
            _REGISTRY.records[subagent_id] = record
            if request.correlation_id:
                _REGISTRY.correlations[correlation_key] = subagent_id
        # Worker threads do not inherit ContextVars automatically. The bound home AND secret
        # scope must travel together, including on A -> B -> A multiplexed hosts.
        ctx = contextvars.copy_context()
        try:
            record.future = _EXECUTOR.submit(ctx.run, self._run, record, request.goal, parent, bool(private_route))
        except BaseException:
            with _REGISTRY.lock:
                _REGISTRY.records.pop(subagent_id, None)
                if request.correlation_id:
                    _REGISTRY.correlations.pop(correlation_key, None)
            child.close()
            raise
        return handle

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            return SubagentStatus(handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE")
        with _REGISTRY.lock:
            return SubagentStatus(record.handle, record.state, record.updated_at)

    def wait(self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None) -> SubagentTerminalState:
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

    def cancel(self, handle: SubagentHandle, *, reason: str) -> SubagentCancelResult:
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

    def result(self, handle: SubagentHandle) -> SubagentResult:
        record = self._record(handle)
        if record is None:
            return SubagentResult(handle, SubagentState.UNKNOWN, False, error_classification="UNKNOWN_HANDLE")
        with _REGISTRY.lock:
            return record.result or SubagentResult(record.handle, record.state, False, error_classification="NOT_READY")

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            return SubagentReconnectResult(False, SubagentState.UNKNOWN, "RECONNECT_UNAVAILABLE")
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def _record(self, handle: SubagentHandle) -> Optional[_Record]:
        """Registry record for a well-formed, capability-verified handle owned by the active parent."""
        if not _handle_is_well_formed(handle):
            return None
        expected = self._capability(handle.subagent_id, handle.parent_session_id, handle.created_at)
        if not hmac.compare_digest(handle.capability, expected):
            return None
        if _session_id_of(self._parent_agent_resolver()) != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            record = _REGISTRY.records.get(handle.subagent_id)
            if record is not None and record.profile_key is not None:
                from hermes_constants import hermes_home_key
                if record.profile_key != hermes_home_key():
                    return None
            return record

    @staticmethod
    def _cleanup_locked() -> None:
        """Retain terminal snapshots for a bounded period, never live work."""
        cutoff = time.time() - _TERMINAL_RETENTION_SECONDS
        expired = [
            sid for sid, record in _REGISTRY.records.items()
            if record.result is not None and record.completed_at is not None and record.completed_at < cutoff
        ]
        for subagent_id in expired:
            expired_record = _REGISTRY.records.pop(subagent_id)
            handle = expired_record.handle
            if handle.correlation_id:
                # Private records namespace correlations by profile; ordinary ones retain
                # their existing session-only key.
                key = (f"profile:{expired_record.profile_key}:{handle.parent_session_id}"
                       if expired_record.profile_key else handle.parent_session_id)
                _REGISTRY.correlations.pop((key, handle.correlation_id), None)

    def _run(self, record: _Record, goal: str, parent: Any, private: bool = False) -> None:
        with _REGISTRY.lock:
            if record.state is not SubagentState.CANCEL_REQUESTED:
                record.state = SubagentState.RUNNING
            record.started_at = record.updated_at = time.time()
        try:
            if private:
                # No delegation registry, progress relay, heartbeat, worktree, transcript,
                # memory finalization, stop hooks, or parent cost mutation.
                from agent.delegation_context import delegated_child_context
                from hermes_logging import private_child_log_scope
                child = record.agent
                try:
                    with private_child_log_scope(), delegated_child_context(str(child.session_id)):
                        response = child.run_conversation(user_message=goal)
                    raw = {
                        "status": "interrupted" if child._interrupt_requested else
                                  "error" if not isinstance(response, dict) or response.get("failed")
                                  or response.get("error") or response.get("completed") is False else "completed",
                        "summary": response.get("final_response") if isinstance(response, dict) else None,
                    }
                finally:
                    with private_child_log_scope():
                        child.close()
            else:
                from tools.delegate_tool import _run_child_lifecycle
                raw = _run_child_lifecycle(0, goal, record.agent, parent)
            is_dict = isinstance(raw, dict)
            raw = raw if is_dict else {}
            status = str(raw.get("status", "error"))
            if status == "interrupted":
                state = SubagentState.CANCELLED if record.state == SubagentState.CANCEL_REQUESTED else SubagentState.INTERRUPTED
            else:
                state = SubagentState.SUCCEEDED if status == "completed" else SubagentState.FAILED
            fields: dict[str, Any] = dict(
                summary=None if private and state is SubagentState.FAILED else _clip(raw.get("summary")),
                error_message=None if private else _clip(raw.get("error") or None),
                error_classification=None if state == SubagentState.SUCCEEDED else status.upper(),
                usage_metadata={"api_calls": raw.get("api_calls", 0)} if is_dict else {},
                tool_execution_summary={"duration_seconds": raw.get("duration_seconds", 0)} if is_dict else {},
            )
        except Exception as exc:
            state = SubagentState.FAILED
            fields = dict(error_classification=type(exc).__name__,
                          error_message="Private child failed." if private else _clip(exc))
        result = SubagentResult(record.handle, state, True, started_at=record.started_at, completed_at=time.time(), **fields)
        payload = dataclasses.asdict(result)
        payload.pop("result_hash", None)
        result = dataclasses.replace(result, result_hash=hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest())
        with _REGISTRY.lock:
            record.agent, record.result, record.state = None, result, result.terminal_state
            record.completed_at = record.updated_at = result.completed_at

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
        from toolsets import TOOLSETS
        unknown = set(request.allowed_toolsets) - set(TOOLSETS)
        if unknown:
            raise SubagentLifecycleError(f"Unknown toolsets: {', '.join(sorted(unknown))}.")
        enabled = getattr(parent, "enabled_toolsets", None)
        if enabled is not None and not set(request.allowed_toolsets).issubset(set(enabled)):
            raise SubagentLifecycleError("Requested toolsets would broaden parent permissions.")

    @staticmethod
    def _private_route() -> dict[str, Any]:
        from hermes_cli.config import load_config_readonly
        from tools.delegate_tool_config import _resolve_delegation_credentials

        model_cfg = (load_config_readonly().get("model") or {})
        provider = model_cfg.get("provider")
        model = model_cfg.get("default")
        if (not isinstance(provider, str) or not provider.strip() or provider.strip().lower() == "auto"
                or not isinstance(model, str) or not model.strip()):
            raise SubagentLifecycleError("Private default route requires explicit model.provider and model.default.")
        try:
            # Reuse the host's runtime provider resolution, but with an isolated routing
            # owner. In particular, delegation.base_url/api_key/provider/fallback cannot win.
            creds = _resolve_delegation_credentials({"provider": provider, "model": model}, None)
        except Exception as exc:
            raise SubagentLifecycleError("Configured default provider is unavailable.") from exc
        # ACP/external processes can execute commands and own their own transcript;
        # they cannot satisfy the in-process no-tools/no-persistence contract.
        if creds.get("command"):
            raise SubagentLifecycleError("Private default route does not support external-process providers.")
        if not creds.get("api_key") or (not creds.get("base_url")
                                        and provider.strip().lower() not in {"bedrock", "vertex", "google", "google-genai"}):
            raise SubagentLifecycleError("Configured default provider has no usable credential or endpoint.")
        return {"model": model, "overrides": {
            "override_provider": creds["provider"], "override_base_url": creds["base_url"],
            "override_api_key": creds["api_key"], "override_api_mode": creds["api_mode"],
            "override_request_overrides": creds.get("request_overrides"),
            "override_acp_command": creds.get("command"), "override_acp_args": creds.get("args"),
            "routing_cfg": {"fallback_providers": []},
        }}
