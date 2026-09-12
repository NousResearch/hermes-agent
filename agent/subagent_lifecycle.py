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
import uuid
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
    profile: Optional[str] = None
    provider: Optional[str] = None
    reasoning_effort: Optional[str] = None


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
    worker_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        # Keep this a shallow scalar projection.  Besides being cheaper than
        # ``asdict``, it never attempts to copy runtime/provider objects when
        # an internal caller is still assembling a handle.
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
            "worker_id": self.worker_id,
            "run_id": self.run_id,
        }

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
    store: Any = None
    owner_session_id: Optional[str] = None
    worker_id: Optional[str] = None
    run_id: Optional[str] = None
    lease_token: Optional[str] = None
    conversation_history: list = dataclasses.field(default_factory=list)
    lease_stop: Any = None
    lease_thread: Any = None
    delivered_message_ids: list[str] = dataclasses.field(default_factory=list)
    max_tool_calls: Optional[int] = None
    tool_calls: int = 0
    max_concurrent: int = 10
    goal: str = ""
    parent_agent: Any = None
    completion_owner: str = "service"
    budget_epoch_id: Optional[str] = None


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


def _text_or_none(value: Any) -> Optional[str]:
    """Keep public identity metadata typed; opaque runtime objects are unknown."""
    return value if isinstance(value, str) and value else None


def _session_id_of(agent: Any) -> Optional[str]:
    return str(getattr(agent, "session_id", "") or "") or None


def _owner_session_id_of(agent: Any) -> Optional[str]:
    """Stable root owner for a worker tree; a nested child's session id is never a new authority."""
    return str(getattr(agent, "_worker_owner_session_id", "") or "") or _session_id_of(agent)


def _clip(value: Any) -> Optional[str]:
    return str(value)[:_MAX_RESULT_CHARS] if value is not None else None


def _prompt_of(agent: Any) -> str:
    return str(
        getattr(agent, "ephemeral_system_prompt", None)
        or getattr(agent, "system_prompt", None)
        or ""
    )


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
    ("worker_id", _opt_str),
    ("run_id", _opt_str),
)

# Launch-request rejections in check order: (predicate, error). The type check leads so later predicates may
# dereference request fields.
_REQUEST_REJECTIONS: tuple[tuple[Callable[[Any], bool], str], ...] = (
    (lambda r: not isinstance(r, SubagentLaunchRequest) or not isinstance(r.goal, str) or not r.goal.strip() or len(r.goal) > _MAX_GOAL_CHARS,
     "goal must be a non-empty string of at most 16000 characters."),
    (lambda r: r.context is not None and (not isinstance(r.context, str) or len(r.context) > _MAX_CONTEXT_CHARS),
     "context must be a string of at most 32000 characters."),
    (lambda r: r.role not in {"leaf", "orchestrator"}, "role must be 'leaf' or 'orchestrator'."),
    (lambda r: r.timeout_seconds is not None, "Per-launch timeout is not supported; configure delegation timeout explicitly."),
    (lambda r: r.working_directory is not None,
     "working_directory is not supported because Hermes delegates use isolated task environments."),
    (lambda r: not isinstance(r.blocked_tools, tuple) or any(
        not isinstance(name, str) or not name for name in r.blocked_tools),
     "blocked_tools must be a tuple of nonempty tool names."),
    (lambda r: any(not _opt_str(value) for value in (r.profile, r.provider, r.model, r.reasoning_effort)),
     "profile, provider, model, and reasoning_effort must be strings when provided."),
)


def _handle_is_well_formed(handle: Any) -> bool:
    return isinstance(handle, SubagentHandle) and all(check(getattr(handle, field)) for field, check in _HANDLE_FIELD_CHECKS)


def _session_db_of(parent: Any) -> Any:
    # Read only attributes that are really present.  Dynamic proxy/test-double
    # ``__getattr__`` values are not durable database authority, and an
    # explicitly disabled ``_session_db = None`` must not fall through to a
    # fabricated public attribute.
    import inspect
    missing = object()
    db = inspect.getattr_static(parent, "_session_db", missing)
    if db is missing:
        db = inspect.getattr_static(parent, "session_db", None)
    if db is None:
        return None
    inner = inspect.getattr_static(db, "_db", missing)
    return getattr(db, "_db") if inner is not missing else db


def _persistent_store(parent: Any) -> Any:
    retained = getattr(parent, "_worker_lifecycle_record", None)
    if isinstance(retained, _Record) and retained.store is not None:
        return retained.store
    db = _session_db_of(parent)
    if db is None or not callable(getattr(type(db), "_execute_write", None)) \
            or not callable(getattr(type(db), "_read_ctx", None)):
        return None
    from agent.worker_store import WorkerStore
    store = WorkerStore(db)
    store.ensure_schema()
    return store


def _profile_policy_snapshot(
    cfg: dict,
    profile: Optional[str],
    creds: Mapping[str, Any],
    *,
    child: Any = None,
) -> tuple[str, dict]:
    """Allowlisted execution contract; volatile discovery metadata is deliberately excluded."""
    from agent.delegation_model_routing import parse_profiles
    spec = parse_profiles(cfg).get(profile) if profile else None
    selected = None
    if spec is not None:
        selected = {
            "name": spec.name,
            "instructions_hash": hashlib.sha256(spec.instructions.encode()).hexdigest(),
            "tool_policy": dataclasses.asdict(spec.tool_policy),
            "workspace_context": dataclasses.asdict(spec.workspace_context),
            "execution_limits": dataclasses.asdict(spec.execution_limits),
            "enabled_routes": [dataclasses.asdict(item) for item in spec.enabled_routes],
        }
    # Empty authority is an explicit denial, not missing metadata. Only an
    # absent execution catalog may fall back to legacy visible schemas.
    tool_names = getattr(child, "_worker_effective_tool_names", None)
    if tool_names is None:
        tool_names = getattr(child, "_executable_tool_names", None)
    if tool_names is None:
        tool_names = getattr(child, "valid_tool_names", None)
    effective_tools = sorted(name for name in (tool_names or ()) if isinstance(name, str))
    from agent.worker_interfaces import frozen_worker_interface_contract

    policy = {
        "profile_contract": selected,
        "effective_tools": effective_tools,
        "worker_interface": frozen_worker_interface_contract(
            getattr(child, "_worker_interface_selection", None)
        ),
        "route": {
            key: creds.get(key) for key in (
                "requested_profile", "requested_provider", "requested_model", "requested_reasoning_effort",
                "resolved_provider", "resolved_model", "resolved_reasoning_effort",
            )
        },
    }
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest(), json.loads(encoded)


def _terminal_status(state: SubagentState) -> str:
    return {
        SubagentState.SUCCEEDED: "SUCCEEDED",
        SubagentState.CANCELLED: "CANCELLED",
        SubagentState.INTERRUPTED: "INTERRUPTED",
    }.get(state, "FAILED")


def _concurrency_limit(cfg: Mapping[str, Any]) -> int:
    value = (cfg or {}).get("max_concurrent_children", 10)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 10


def _iteration_limit(cfg: Mapping[str, Any], creds: Mapping[str, Any], default: int) -> int:
    global_limit = (cfg or {}).get("max_iterations", default)
    if not isinstance(global_limit, int) or isinstance(global_limit, bool) or global_limit < 1:
        global_limit = default
    profile_limit = creds.get("max_iterations")
    return min(global_limit, profile_limit) if isinstance(profile_limit, int) else global_limit


def _tree_budget_limits(cfg: Mapping[str, Any], creds: Mapping[str, Any], default: int) -> dict[str, Any]:
    limits = creds.get("execution_limits")
    profile_timeout = getattr(limits, "timeout_seconds", None)
    from tools.delegate_tool_config import _get_child_timeout
    global_timeout = _get_child_timeout()
    timeout = min(value for value in (global_timeout, profile_timeout) if value is not None) \
        if global_timeout is not None or profile_timeout is not None else None
    return {
        "max_iterations": _iteration_limit(cfg, creds, default),
        "max_tool_calls": getattr(limits, "max_tool_calls", None),
        "timeout_seconds": timeout,
    }


def _parent_budget_epoch(parent: Any) -> Optional[str]:
    record = getattr(parent, "_worker_lifecycle_record", None)
    return record.budget_epoch_id if isinstance(record, _Record) else None


def _attach_tree_budget(record: _Record) -> None:
    if record.agent is None or record.store is None or not record.budget_epoch_id:
        return
    record.agent._worker_budget_epoch_id = record.budget_epoch_id
    snapshot = record.store.budget_snapshot(record.run_id, record.owner_session_id)
    if not snapshot or snapshot.get("deadline_at") is None:
        return
    remaining = max(0.001, float(snapshot["deadline_at"]) - time.time())
    current = getattr(record.agent, "_worker_timeout_seconds", None)
    record.agent._worker_timeout_seconds = min(current, remaining) \
        if isinstance(current, (int, float)) and current > 0 else remaining


def before_worker_provider_attempt(agent: Any) -> None:
    """Reserve one durable tree iteration immediately before provider transport."""
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.store is None or not record.budget_epoch_id:
        return
    try:
        record.store.reserve_iteration(record.run_id, record.owner_session_id, record.lease_token)
    except Exception as exc:
        from agent.worker_store import WorkerBudgetExceeded
        if not isinstance(exc, WorkerBudgetExceeded):
            raise
        agent._worker_budget_termination_reason = exc.reason
        request_hard_interrupt(agent, exc.reason.replace("_", " "), tool_reason=exc.reason)
        raise InterruptedError(exc.reason) from exc


def before_worker_tool(agent: Any, tool_call_id: str) -> None:
    """Fail-closed durable boundary immediately before a worker tool dispatch."""
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.store is None:
        return
    if record.max_tool_calls is not None and record.tool_calls >= record.max_tool_calls:
        raise SubagentLifecycleError("Worker tool-call limit reached for this run.")
    try:
        record.store.mark_tool_boundary(
            record.run_id, record.owner_session_id, record.lease_token,
            tool_call_id=tool_call_id, tool_inflight=True)
    except Exception as exc:
        from agent.worker_store import WorkerBudgetExceeded
        if isinstance(exc, WorkerBudgetExceeded):
            agent._worker_budget_termination_reason = exc.reason
        raise
    record.tool_calls += 1


def before_worker_nested_tool(agent: Any, tool_call_id: str) -> bool:
    """Admit an execute_code RPC under the worker captured for that cell."""
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.store is None:
        return False
    before_worker_tool(agent, tool_call_id)
    return True


def queue_worker_parent_message(agent: Any, content: str) -> Optional[Mapping[str, Any]]:
    """Durably queue a child-to-root-parent message when this worker has a store."""
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.store is None or not record.run_id:
        return None
    return record.store.enqueue_parent_message(record.run_id, record.owner_session_id, content)


def checkpoint_worker_tool_result(
    agent: Any,
    history: list,
    *,
    tool_call_id: Optional[str],
    admitted: Optional[bool] = None,
    settled: bool = True,
) -> None:
    """Atomically persist the tool result/history before clearing uncertainty."""
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.store is None:
        return
    delivered = tuple(record.delivered_message_ids)
    record.store.checkpoint_tool_result(
        record.run_id,
        record.owner_session_id,
        record.lease_token,
        history=history,
        tool_call_id=tool_call_id,
        admitted=admitted,
        settled=settled,
        delivered_message_ids=delivered,
    )
    record.conversation_history = list(history)
    record.delivered_message_ids.clear()


def checkpoint_worker_nested_tool_result(agent: Any, tool_call_id: str) -> None:
    """Settle one nested RPC without replacing the worker's conversation checkpoint."""
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.store is None:
        return
    worker = record.store.get_worker(record.worker_id, record.owner_session_id)
    checkpoint_worker_tool_result(
        agent, list(worker.get("history") or record.conversation_history),
        tool_call_id=tool_call_id, settled=True,
    )


def dispatch_worker_nested_tool(tool_name: str, tool_args: dict, *, task_id: str) -> str:
    """Dispatch one execute_code RPC with the cell's captured worker authority."""
    from model_tools import handle_function_call
    from tools.registry import tool_error

    agent = get_active_subagent_parent()
    tool_call_id = "execute-code-rpc-" + uuid.uuid4().hex
    try:
        admitted = before_worker_nested_tool(agent, tool_call_id)
    except Exception as exc:
        return tool_error(str(exc))
    from agent.worker_interfaces import is_worker_interface_tool

    selection = getattr(agent, "_worker_interface_selection", None)
    if is_worker_interface_tool(selection, tool_name):
        result = agent._dispatch_worker_interface(tool_name, tool_args)
    else:
        result = handle_function_call(tool_name, tool_args, task_id=task_id)
    if admitted:
        checkpoint_worker_nested_tool_result(agent, tool_call_id)
    return result


def worker_tool_calls_remaining(agent: Any) -> Optional[int]:
    record = getattr(agent, "_worker_lifecycle_record", None)
    if not isinstance(record, _Record) or record.max_tool_calls is None:
        return None
    return max(0, record.max_tool_calls - record.tool_calls)


class SubagentLifecycleService:
    """Shared lifecycle behind plugins and ``delegate_task``.

    A parent with ``SessionDB`` gets durable workers, runs, messages and reconnectable terminal
    snapshots. Existing hosts without durable state keep the legacy process-local behavior.
    """

    def __init__(self, parent_agent_resolver: Callable[[], Any]) -> None:
        self._parent_agent_resolver = parent_agent_resolver

    def adopt_delegate_child(
        self,
        child: Any,
        *,
        goal: str,
        context: Optional[str],
        profile: Optional[str],
        creds: Mapping[str, Any],
        cfg: Mapping[str, Any],
    ) -> Optional[SubagentHandle]:
        """Attach the parent tool's child to the same durable lifecycle used by plugins."""
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS
        parent = self._parent_agent_resolver()
        owner = _owner_session_id_of(parent)
        store = _persistent_store(parent) if parent is not None else None
        if store is None or not owner:
            return None
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        if not subagent_id:
            raise SubagentLifecycleError("Hermes failed to assign a child identity.")
        created = time.time()
        capability = self._capability(subagent_id, owner, created)
        config_revision, policy = _profile_policy_snapshot(dict(cfg), profile, creds, child=child)
        policy["launch_allowed_toolsets"] = None
        policy["launch_blocked_tools"] = []
        policy["role"] = getattr(child, "_delegate_role", "leaf")
        policy["capability_digest"] = hashlib.sha256(capability.encode()).hexdigest()
        parent_worker_id = getattr(parent, "_worker_id", None)
        worker = store.create_worker(
            owner,
            profile=profile,
            config_revision=config_revision,
            policy=policy,
            frozen_prompt=_prompt_of(child),
            parent_worker_id=parent_worker_id,
        )
        if isinstance(getattr(child, "_worker_route_receipt", None), dict):
            child._worker_route_receipt["profile_revision"] = config_revision
            child._worker_route_receipt["effective_tools"] = list(policy["effective_tools"])
        queued = store.enqueue_run(
            worker["worker_id"], owner, goal=goal, context=context or "",
            capability_digest=hashlib.sha256(capability.encode()).hexdigest(),
            budget_epoch_id=_parent_budget_epoch(parent),
            budget_limits=_tree_budget_limits(cfg, creds, DEFAULT_MAX_ITERATIONS),
        )
        max_concurrent = _concurrency_limit(cfg)
        active = store.claim_next_run(worker["worker_id"], owner, max_concurrent=max_concurrent)
        if active is None or active["run_id"] != queued["run_id"]:
            raise SubagentLifecycleError("Worker run could not acquire an execution slot.")
        child._worker_id, child._worker_run_id = worker["worker_id"], active["run_id"]
        child._worker_owner_session_id = owner
        handle = SubagentHandle(
            contract_version=PUBLIC_CONTRACT_VERSION,
            subagent_id=subagent_id,
            parent_session_id=owner,
            correlation_id=None,
            created_at=created,
            provider=_text_or_none(getattr(child, "provider", None)),
            model=_text_or_none(getattr(child, "model", None)),
            role=_text_or_none(getattr(child, "_delegate_role", None)) or "leaf",
            depth=int(getattr(child, "_delegate_depth", 1) or 1),
            capability=capability,
            worker_id=worker["worker_id"],
            run_id=active["run_id"],
        )
        record = _Record(
            handle, SubagentState.RUNNING, created, agent=child, started_at=created,
            store=store, owner_session_id=owner, worker_id=worker["worker_id"],
            run_id=active["run_id"], lease_token=active["lease_token"],
            max_tool_calls=getattr(creds.get("execution_limits"), "max_tool_calls", None),
            max_concurrent=max_concurrent, goal=goal, parent_agent=parent,
            completion_owner="delegate",
            budget_epoch_id=active.get("budget_epoch_id"),
        )
        self._bind_tool_boundary(record)
        _attach_tree_budget(record)
        self._start_external_lease(record)
        child._worker_lifecycle_record = record
        with _REGISTRY.lock:
            _REGISTRY.records[subagent_id] = record
        return handle

    @staticmethod
    def _start_external_lease(record: _Record) -> None:
        stop = threading.Event()
        record.lease_stop = stop

        def renew():
            while not stop.wait(15.0):
                try:
                    record.store.heartbeat_run(
                        record.run_id, record.owner_session_id, record.lease_token, lease_seconds=60)
                except Exception:
                    request_hard_interrupt(
                        record.agent, "Durable worker execution lease was lost.", tool_reason="worker lease lost")
                    return

        record.lease_thread = threading.Thread(target=renew, name="hermes-worker-lease", daemon=True)
        record.lease_thread.start()

    @classmethod
    def complete_adopted_child(cls, child: Any, entry: Mapping[str, Any]) -> None:
        record = getattr(child, "_worker_lifecycle_record", None)
        if not isinstance(record, _Record) or record.completion_owner != "delegate":
            return
        status = str(entry.get("status") or "error")
        if status == "completed":
            state = SubagentState.SUCCEEDED
        elif status == "interrupted":
            state = (
                SubagentState.CANCELLED
                if record.state is SubagentState.CANCEL_REQUESTED
                else SubagentState.INTERRUPTED
            )
        else:
            state = SubagentState.FAILED
        cls._publish_result(
            record,
            state,
            {
                "summary": _clip(entry.get("summary")),
                "error_message": _clip(entry.get("error")),
                "error_classification": None if state is SubagentState.SUCCEEDED else status.upper(),
                "usage_metadata": {"api_calls": entry.get("api_calls", 0)},
                "tool_execution_summary": {"duration_seconds": entry.get("duration_seconds", 0)},
            },
            evidence={
                "exit_reason": entry.get("exit_reason"),
                "tokens": entry.get("tokens") or {},
                "cost_usd": entry.get("cost_usd"),
                "cost_status": entry.get("cost_status") or "unknown",
            },
        )

    def launch(self, request: SubagentLaunchRequest) -> SubagentHandle:
        parent = self._parent_agent_resolver()
        if parent is None:
            raise SubagentLifecycleError("No active Hermes parent session is available.")
        self._validate_request(request, parent)
        parent_session_id = _owner_session_id_of(parent)
        if request.parent_session_id and request.parent_session_id != parent_session_id:
            raise SubagentLifecycleError("parent_session_id does not match the active session.")
        from tools.delegate_tool import _validate_spawn_admission
        try:
            _validate_spawn_admission(parent, 1)
        except ValueError as exc:
            raise SubagentLifecycleError(str(exc)) from exc
        correlation_key = (parent_session_id, request.correlation_id or "")
        with _REGISTRY.lock:
            self._cleanup_locked()
            if request.correlation_id and correlation_key in _REGISTRY.correlations:
                raise SubagentLifecycleError("Duplicate correlation_id for this parent session.")
        # Lazy: delegate construction stays internal, plugins never import private delegation helpers.
        from tools.delegate_tool import (
            DEFAULT_MAX_ITERATIONS, _build_child_preserving_parent_tools, _profile_task_overrides,
        )
        from tools.delegate_tool_config import _load_config, _resolve_delegation_credentials
        cfg = _load_config()
        if request.profile:
            creds = _resolve_delegation_credentials(
                cfg, parent, request.profile,
                requested_provider=request.provider,
                requested_model=request.model,
                requested_reasoning_effort=request.reasoning_effort,
            )
            overrides = _profile_task_overrides(creds)
            run_iterations = _iteration_limit(cfg, creds, DEFAULT_MAX_ITERATIONS)
        else:
            if request.provider or request.reasoning_effort:
                raise SubagentLifecycleError("provider/reasoning_effort require a configured worker profile.")
            creds = _resolve_delegation_credentials(cfg, parent)
            overrides = {
                "override_provider": creds["provider"], "override_base_url": creds["base_url"],
                "override_api_key": creds["api_key"], "override_api_mode": creds["api_mode"],
                "override_request_overrides": creds.get("request_overrides"),
                "override_acp_command": creds.get("command"), "override_acp_args": creds.get("args"),
                "routing_cfg": cfg,
            }
            run_iterations = _iteration_limit(cfg, creds, DEFAULT_MAX_ITERATIONS)
        child = _build_child_preserving_parent_tools(
            task_index=0, goal=request.goal, context=request.context,
            toolsets=(
                list(request.allowed_toolsets)
                if request.allowed_toolsets is not None else None
            ),
            model=(request.model or creds.get("model")), max_iterations=run_iterations,
            task_count=1, parent_agent=parent, role=request.role,
            request_blocked_tools=list(request.blocked_tools), **overrides,
        )
        subagent_id = str(getattr(child, "_subagent_id", "") or "")
        if not subagent_id:
            raise SubagentLifecycleError("Hermes failed to assign a child identity.")
        created = time.time()
        capability = self._capability(subagent_id, parent_session_id, created)
        store = _persistent_store(parent)
        worker_id = run_id = lease_token = None
        if store is not None and parent_session_id:
            config_revision, policy = _profile_policy_snapshot(cfg, request.profile, creds, child=child)
            policy["launch_allowed_toolsets"] = (
                list(request.allowed_toolsets)
                if request.allowed_toolsets is not None else None)
            policy["launch_blocked_tools"] = list(request.blocked_tools)
            policy["role"] = getattr(child, "_delegate_role", request.role)
            policy["capability_digest"] = hashlib.sha256(capability.encode()).hexdigest()
            worker = store.create_worker(
                parent_session_id,
                profile=request.profile,
                config_revision=config_revision,
                policy=policy,
                frozen_prompt=_prompt_of(child),
                parent_worker_id=getattr(parent, "_worker_id", None),
            )
            if isinstance(getattr(child, "_worker_route_receipt", None), dict):
                child._worker_route_receipt["profile_revision"] = config_revision
                child._worker_route_receipt["effective_tools"] = list(policy["effective_tools"])
            queued = store.enqueue_run(
                worker["worker_id"], parent_session_id, goal=request.goal,
                context=request.context or "",
                capability_digest=hashlib.sha256(capability.encode()).hexdigest(),
                budget_epoch_id=_parent_budget_epoch(parent),
                budget_limits=_tree_budget_limits(cfg, creds, DEFAULT_MAX_ITERATIONS),
            )
            max_concurrent = _concurrency_limit(cfg)
            active = store.claim_next_run(
                worker["worker_id"], parent_session_id, max_concurrent=max_concurrent)
            active_run = active if active is not None and active["run_id"] == queued["run_id"] else None
            worker_id, run_id = worker["worker_id"], queued["run_id"]
            lease_token = active_run["lease_token"] if active_run else None
            child._worker_id, child._worker_run_id = worker_id, run_id
            child._worker_owner_session_id = parent_session_id
        handle = SubagentHandle(
            contract_version=PUBLIC_CONTRACT_VERSION,
            subagent_id=subagent_id,
            parent_session_id=parent_session_id,
            correlation_id=request.correlation_id,
            created_at=created,
            provider=_text_or_none(getattr(child, "provider", None)),
            model=_text_or_none(getattr(child, "model", None)),
            role=_text_or_none(getattr(child, "_delegate_role", None)) or request.role,
            depth=int(getattr(child, "_delegate_depth", 1) or 1),
            capability=capability,
            worker_id=worker_id,
            run_id=run_id,
        )
        record = _Record(
            handle, SubagentState.PENDING, created,
            agent=child if lease_token is not None or store is None else None,
            store=store,
            owner_session_id=parent_session_id, worker_id=worker_id, run_id=run_id, lease_token=lease_token,
            max_tool_calls=getattr(creds.get("execution_limits"), "max_tool_calls", None),
            max_concurrent=_concurrency_limit(cfg), goal=request.goal, parent_agent=parent,
            budget_epoch_id=(active_run or queued).get("budget_epoch_id") if store is not None else None,
        )
        with _REGISTRY.lock:
            _REGISTRY.records[subagent_id] = record
            if request.correlation_id:
                _REGISTRY.correlations[correlation_key] = subagent_id
        if store is not None and lease_token is not None:
            self._start_record(record)
        elif store is None:
            record.future = _EXECUTOR.submit(self._run, record, request.goal, parent)
        else:
            # A queued run retains identity and policy, never a pre-authorized
            # live agent.  It is rebuilt from current authority immediately
            # before its exact lease is claimed.
            with contextlib.suppress(Exception):
                child.close()
        return handle

    def status(self, handle: SubagentHandle) -> SubagentStatus:
        record = self._record(handle)
        if record is None:
            durable = self._durable_snapshot(handle)
            if durable is None:
                return SubagentStatus(handle, SubagentState.UNKNOWN, time.time(), "UNKNOWN_HANDLE")
            return SubagentStatus(handle, SubagentState(durable[2]["status"]), durable[2]["updated_at"], "DURABLE_SNAPSHOT")
        with _REGISTRY.lock:
            return SubagentStatus(record.handle, record.state, record.updated_at)

    def wait(self, handle: SubagentHandle, *, timeout_seconds: Optional[float] = None) -> SubagentTerminalState:
        record = self._record(handle)
        if record is None:
            durable = self._durable_snapshot(handle)
            if durable is None:
                return SubagentTerminalState(handle, SubagentState.UNKNOWN, True, diagnostic="UNKNOWN_HANDLE")
            store, worker, run = durable
            if run["status"] == "RUNNING" and run.get("lease_expires_at", 0) <= time.time():
                store.recover_expired_runs(handle.parent_session_id, [worker["worker_id"]])
                run = store.get_run(handle.run_id, handle.parent_session_id)
            if run["status"] == "PENDING":
                self._schedule_owner(
                    store, handle.parent_session_id, self._parent_agent_resolver(),
                    worker_id=worker["worker_id"], run_id=run["run_id"],
                )
                deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
                while run["status"] in {"PENDING", "RUNNING"}:
                    if deadline is not None and time.monotonic() >= deadline:
                        break
                    time.sleep(0.01)
                    run = store.get_run(handle.run_id, handle.parent_session_id)
            state = SubagentState(run["status"])
            completed = state in {
                SubagentState.SUCCEEDED, SubagentState.FAILED, SubagentState.INTERRUPTED, SubagentState.CANCELLED,
            }
            return SubagentTerminalState(handle, state, completed, diagnostic="DURABLE_SNAPSHOT")
        if (
            record.state is SubagentState.PENDING
            and record.future is None
            and record.store is not None
            and record.owner_session_id
        ):
            self._schedule_owner(
                record.store, record.owner_session_id, self._parent_agent_resolver(),
                worker_id=record.worker_id, run_id=record.run_id,
            )
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while record.result is None:
            future = record.future
            if future is None:
                if deadline is not None and time.monotonic() >= deadline:
                    return SubagentTerminalState(record.handle, record.state, False, True)
                time.sleep(0.01)
                continue
            try:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                future.result(timeout=remaining)
            except TimeoutError:
                return SubagentTerminalState(record.handle, record.state, False, True)
            except Exception:
                pass
            break
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
            durable = self._durable_snapshot(handle)
            if durable is None:
                return SubagentResult(handle, SubagentState.UNKNOWN, False, error_classification="UNKNOWN_HANDLE")
            run = durable[2]
            state = SubagentState(run["status"])
            payload = run.get("result") or {}
            ready = state in {
                SubagentState.SUCCEEDED, SubagentState.FAILED, SubagentState.INTERRUPTED, SubagentState.CANCELLED,
            }
            return SubagentResult(
                handle,
                state,
                ready,
                summary=payload.get("summary"),
                error_classification=payload.get("error_classification") or (None if state is SubagentState.SUCCEEDED else state.value),
                error_message=payload.get("error_message"),
                result_hash=payload.get("result_hash"),
            )
        with _REGISTRY.lock:
            return record.result or SubagentResult(record.handle, record.state, False, error_classification="NOT_READY")

    def reconnect(self, handle: SubagentHandle) -> SubagentReconnectResult:
        record = self._record(handle)
        if record is None:
            durable = self._durable_snapshot(handle)
            if durable is None:
                return SubagentReconnectResult(False, SubagentState.UNKNOWN, "UNKNOWN_HANDLE")
            _store, _worker, run = durable
            state = SubagentState(run["status"])
            return SubagentReconnectResult(True, state, "DURABLE_SNAPSHOT")
        with _REGISTRY.lock:
            return SubagentReconnectResult(True, record.state)

    def message(self, handle: SubagentHandle, content: str, *, message_id: Optional[str] = None) -> Mapping[str, Any]:
        """Queue one durable FIFO followup for the worker; delivery is acked with a checkpoint."""
        record = self._record(handle)
        parent = self._parent_agent_resolver()
        durable = None if record is not None else self._durable_snapshot(handle)
        store = record.store if record is not None else (durable[0] if durable else None)
        owner = _owner_session_id_of(parent)
        worker_id = handle.worker_id or (record.worker_id if record is not None else None)
        if store is None or not owner or not worker_id:
            raise SubagentLifecycleError("Durable worker messaging is unavailable for this handle.")
        return store.enqueue_message(worker_id, owner, content, message_id=message_id, sender_id=owner)

    @staticmethod
    def _actor_can_target(
        store: Any,
        owner: str,
        actor_worker_id: Optional[str],
        target: Mapping[str, Any],
        *,
        message_only: bool,
        allow_siblings: bool,
    ) -> bool:
        if not actor_worker_id:
            return True
        actor = store.get_worker(actor_worker_id, owner)
        if target["worker_id"] == actor_worker_id:
            return True
        current = target
        while current.get("parent_worker_id"):
            if current["parent_worker_id"] == actor_worker_id:
                return True
            current = store.get_worker(current["parent_worker_id"], owner)
        if message_only and actor.get("parent_worker_id") == target["worker_id"]:
            return True
        return bool(
            message_only
            and allow_siblings
            and actor.get("parent_worker_id")
            and actor.get("parent_worker_id") == target.get("parent_worker_id")
        )

    @staticmethod
    def _subtree_ids(store: Any, owner: str, worker_id: str) -> list[str]:
        workers = store.list_workers(owner)
        selected = {worker_id}
        changed = True
        while changed:
            changed = False
            for worker in workers:
                if worker.get("parent_worker_id") in selected and worker["worker_id"] not in selected:
                    selected.add(worker["worker_id"])
                    changed = True
        return [worker["worker_id"] for worker in workers if worker["worker_id"] in selected]

    def control(
        self,
        action: str,
        *,
        worker_id: Optional[str] = None,
        run_id: Optional[str] = None,
        message: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        reconciliation_disposition: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Parent-tool control by stable IDs; authority is the bound live session, never ID knowledge."""
        parent = self._parent_agent_resolver()
        owner = _owner_session_id_of(parent)
        store = _persistent_store(parent) if parent is not None else None
        if store is None or not owner:
            raise SubagentLifecycleError("Durable worker state is unavailable for this session.")
        normalized = action.strip().lower()
        actor_worker_id = getattr(parent, "_worker_id", None)
        from tools.delegate_tool_config import _load_config
        cfg = _load_config()
        allow_siblings = bool((cfg or {}).get("allow_sibling_messaging", False))
        if normalized == "status":
            if not worker_id:
                workers = store.list_workers(owner)
                visible = [
                    item for item in workers
                    if self._actor_can_target(
                        store, owner, actor_worker_id, item, message_only=False, allow_siblings=allow_siblings)
                ]
                return {"workers": [self._safe_worker_snapshot(store, item, owner) for item in visible]}
            worker = store.get_worker(worker_id, owner)
            if not self._actor_can_target(
                store, owner, actor_worker_id, worker, message_only=False, allow_siblings=allow_siblings
            ):
                raise PermissionError("Worker control target is outside the actor's owned subtree.")
            store.recover_expired_runs(owner, [worker_id])
            worker = store.get_worker(worker_id, owner)
            return self._safe_worker_snapshot(store, worker, owner)
        if normalized == "completions":
            pending = store.pending_completions(owner)
            return {
                "completions": [
                    self._completion_snapshot(item) for item in pending
                    if self._actor_can_target(
                        store,
                        owner,
                        actor_worker_id,
                        store.get_worker(item["worker_id"], owner),
                        message_only=False,
                        allow_siblings=allow_siblings,
                    )
                ]
            }
        if not worker_id:
            raise SubagentLifecycleError(f"action='{normalized}' requires worker_id.")
        worker = store.get_worker(worker_id, owner)
        message_action = normalized == "message"
        if not self._actor_can_target(
            store, owner, actor_worker_id, worker,
            message_only=message_action, allow_siblings=allow_siblings,
        ):
            raise PermissionError("Worker control target is outside the actor's authorized relation.")
        store.recover_expired_runs(owner, [worker_id])
        worker = store.get_worker(worker_id, owner)
        runs = store.list_runs(worker_id, owner)
        selected = next((item for item in runs if item["run_id"] == run_id), None) if run_id else (runs[-1] if runs else None)
        if normalized == "message":
            queued = store.enqueue_message(
                worker_id, owner, message or "", sender_id=actor_worker_id or owner)
            delivery = "NEXT_RUN"
            with _REGISTRY.lock:
                active_record = next((
                    item for item in _REGISTRY.records.values()
                    if item.owner_session_id == owner and item.worker_id == worker_id
                    and item.run_id == (selected or {}).get("run_id")
                    and item.state in {SubagentState.STARTING, SubagentState.RUNNING}
                    and item.agent is not None
                ), None)
            if active_record is not None:
                steer = getattr(active_record.agent, "steer", None)
                if callable(steer) and steer(message or ""):
                    active_record.delivered_message_ids.append(queued["message_id"])
                    delivery = "RUNNING_STEER_PENDING_CHECKPOINT"
            return {
                "worker_id": worker_id,
                "message_id": queued["message_id"],
                "status": queued["status"],
                "delivery": delivery,
            }
        if selected is None:
            raise SubagentLifecycleError("Worker has no matching run.")
        if normalized == "inspect":
            messages = store.list_messages(worker_id, owner)
            parent_messages = store.list_parent_messages(selected["run_id"], owner)
            return {
                **self._safe_worker_snapshot(store, worker, owner),
                "run": self._inspect_run_snapshot(selected),
                # Explicit inspection is the opt-in transcript surface.  Keep
                # routine status/results compact and never expose system
                # prompts, hidden reasoning, or provider/session objects.
                "conversation": self._inspect_conversation(worker.get("history") or []),
                "messages": [
                    {
                        key: item.get(key) for key in (
                            "message_id", "sender_id", "status", "delivered_run_id", "created_at", "delivered_at",
                        )
                    }
                    for item in messages
                ],
                "messages_to_parent": [
                    {key: item.get(key) for key in (
                        "message_id", "content", "status", "created_at", "published_at", "acknowledged_at")}
                    for item in parent_messages
                ],
            }
        if normalized == "wait":
            timeout = 0.0 if timeout_seconds is None else float(timeout_seconds)
            if timeout < 0 or timeout > 60:
                raise SubagentLifecycleError("timeout_seconds must be between 0 and 60.")
            self._schedule_owner(
                store, owner, parent, worker_id=worker_id, run_id=selected["run_id"])
            selected = store.get_run(selected["run_id"], owner)
            deadline = time.monotonic() + timeout
            while selected["status"] in {"PENDING", "RUNNING"} and time.monotonic() < deadline:
                if selected["status"] == "RUNNING" and selected.get("lease_expires_at", 0) <= time.time():
                    store.recover_expired_runs(owner, [worker_id])
                    self._schedule_owner(
                        store, owner, parent, worker_id=worker_id, run_id=selected["run_id"])
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                selected = store.get_run(selected["run_id"], owner)
            return self._safe_run_snapshot(selected)
        if normalized == "interrupt":
            if selected["status"] in {"SUCCEEDED", "FAILED", "INTERRUPTED", "CANCELLED"}:
                return {
                    **self._safe_run_snapshot(selected),
                    "interrupt_requested": False,
                    "already_terminal": True,
                }
            with _REGISTRY.lock:
                live_record = next((
                    item for item in _REGISTRY.records.values()
                    if item.owner_session_id == owner and item.worker_id == worker_id
                    and item.run_id == selected["run_id"] and item.agent is not None
                    and item.state not in {
                        SubagentState.SUCCEEDED, SubagentState.FAILED,
                        SubagentState.INTERRUPTED, SubagentState.CANCELLED,
                    }
                ), None)
            if selected["status"] == "PENDING":
                cancelled = store.cancel_pending_run(selected["run_id"], owner)
                if live_record is not None:
                    with _REGISTRY.lock:
                        live_record.state = SubagentState.INTERRUPTED
                        live_record.result = SubagentResult(
                            live_record.handle,
                            SubagentState.INTERRUPTED,
                            True,
                            completed_at=time.time(),
                            error_classification="INTERRUPTED",
                            error_message="Worker run interrupted before it started.",
                        )
                        live_record.updated_at = time.time()
                    with contextlib.suppress(Exception):
                        live_record.agent.close()
                    live_record.agent = None
                current = cancelled or store.get_run(selected["run_id"], owner)
                return {
                    **self._safe_run_snapshot(current),
                    "interrupt_requested": bool(cancelled),
                    "scope": "run",
                }
            if live_record is None:
                return {
                    **self._safe_run_snapshot(selected),
                    "interrupt_requested": False,
                    "unsupported": True,
                    "reason": "The running worker is not attached to this process; tree cancellation remains separate.",
                }
            accepted = request_hard_interrupt(
                live_record.agent,
                "Worker run interruption requested by its owner.",
                tool_reason="worker run interruption requested",
            )
            if accepted:
                with _REGISTRY.lock:
                    live_record.updated_at = time.time()
            return {
                **self._safe_run_snapshot(selected),
                "interrupt_requested": bool(accepted),
                "unsupported": not accepted,
                "scope": "run",
            }
        if normalized == "cancel":
            subtree = self._subtree_ids(store, owner, worker_id)
            cancelled_pending = store.cancel_pending_runs(subtree, owner)
            accepted = 0
            queued_records = []
            with _REGISTRY.lock:
                live_records = [
                    item for item in _REGISTRY.records.values()
                    if item.owner_session_id == owner and item.worker_id in subtree
                    and item.agent is not None and item.state not in {
                        SubagentState.SUCCEEDED, SubagentState.FAILED,
                        SubagentState.INTERRUPTED, SubagentState.CANCELLED,
                    }
                ]
                for item in live_records:
                    if item.lease_token is None:
                        queued_records.append(item)
                        item.state = SubagentState.CANCELLED
                        item.result = SubagentResult(
                            item.handle,
                            SubagentState.CANCELLED,
                            True,
                            completed_at=time.time(),
                            error_classification="CANCELLED",
                            error_message="Worker tree cancelled before this run started.",
                        )
                    else:
                        item.state = SubagentState.CANCEL_REQUESTED
                    item.updated_at = time.time()
            for item in queued_records:
                with contextlib.suppress(Exception):
                    item.agent.close()
                item.agent = None
            queued_record_ids = {id(item) for item in queued_records}
            for item in (record for record in live_records if id(record) not in queued_record_ids):
                if request_hard_interrupt(
                    item.agent,
                    "Worker tree cancellation requested by its owner.",
                    tool_reason="worker cancellation requested",
                ):
                    accepted += 1
            return {
                **self._safe_run_snapshot(selected),
                "cancel_requested": bool(accepted or cancelled_pending),
                "workers_targeted": subtree,
                "live_interrupts": accepted,
                "queued_runs_cancelled": len(cancelled_pending),
            }
        if normalized == "reconcile":
            if selected["run_id"] != runs[-1]["run_id"]:
                raise SubagentLifecycleError("Reconcile must target the worker's latest run.")
            if not worker.get("uncertain_side_effect"):
                raise SubagentLifecycleError("The selected worker has no uncertain tool effect to reconcile.")
            store.reconcile_run(
                selected["run_id"],
                owner,
                disposition=reconciliation_disposition or "",
                note=message or "",
            )
            return {
                **self._safe_run_snapshot(store.get_run(selected["run_id"], owner)),
                "reconciled": True,
                "disposition": reconciliation_disposition,
            }
        if normalized == "resume":
            if not message or not message.strip():
                raise SubagentLifecycleError("action='resume' requires message as the next worker turn.")
            if selected["run_id"] != runs[-1]["run_id"]:
                raise SubagentLifecycleError("Resume must link from the worker's latest run.")
            if selected["status"] == "CANCELLED":
                raise SubagentLifecycleError(
                    "Cancelled workers stay cancelled and cannot accept another turn."
                )
            route = dict((worker.get("policy") or {}).get("route") or {})
            prior = SubagentHandle(
                PUBLIC_CONTRACT_VERSION, "durable-snapshot", owner, None, worker["created_at"],
                route.get("resolved_provider"), route.get("resolved_model"),
                str((worker.get("policy") or {}).get("role") or "leaf"),
                int(worker["depth"]), "", worker_id, selected["run_id"],
            )
            handle = self._launch_existing_worker(
                SubagentLaunchRequest(
                    goal=message,
                    profile=worker["profile"],
                    role=str((worker.get("policy") or {}).get("role") or "leaf"),
                ),
                prior,
                worker,
                selected,
            )
            return {"worker_id": handle.worker_id, "run_id": handle.run_id, "status": "PENDING"}
        if normalized == "ack":
            store.ack_completion(selected["run_id"], owner)
            return {**self._safe_run_snapshot(store.get_run(selected["run_id"], owner)), "acknowledged": True}
        raise SubagentLifecycleError(f"Unsupported durable worker action '{normalized}'.")

    @staticmethod
    def _safe_run_snapshot(run: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            key: run.get(key) for key in (
                "run_id", "worker_id", "previous_run_id", "status", "uncertain_side_effect",
                "completion_ack", "created_at", "updated_at",
            )
        }

    @classmethod
    def _completion_snapshot(cls, run: Mapping[str, Any]) -> Mapping[str, Any]:
        result = dict(run.get("result") or {})
        return {
            **cls._safe_run_snapshot(run),
            "summary": result.get("summary"),
            "termination": result.get("termination"),
            "messages_to_parent": list(result.get("messages_to_parent") or []),
        }

    @classmethod
    def _inspect_run_snapshot(cls, run: Mapping[str, Any]) -> Mapping[str, Any]:
        result = dict(run.get("result") or {})
        return {
            **cls._safe_run_snapshot(run),
            "result": {
                key: result.get(key) for key in (
                    "summary", "error_classification", "error_message", "result_hash",
                    "route", "lineage", "termination", "usage", "cost",
                    "effective_tools", "reconciliation",
                )
            } if result else None,
        }

    @staticmethod
    def _inspect_conversation(history: list) -> list[Mapping[str, Any]]:
        visible = []
        for item in history:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "")
            if role not in {"user", "assistant", "tool"}:
                continue
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for block in content:
                    if not isinstance(block, Mapping):
                        continue
                    if block.get("type") not in {"text", "input_text", "output_text"}:
                        continue
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                content = "\n".join(parts)
            if not isinstance(content, str):
                content = ""
            entry = {"role": role, "content": _clip(content)}
            if role == "tool":
                for key in ("name", "tool_call_id"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        entry[key] = value
            visible.append(entry)
        return visible

    @classmethod
    def _safe_worker_snapshot(cls, store: Any, worker: Mapping[str, Any], owner: str) -> Mapping[str, Any]:
        runs = store.list_runs(worker["worker_id"], owner)
        return {
            "worker_id": worker["worker_id"],
            "parent_worker_id": worker["parent_worker_id"],
            "root_worker_id": worker["root_worker_id"],
            "depth": worker["depth"],
            "profile": worker["profile"],
            "config_revision": worker["config_revision"],
            "frozen_prompt_hash": worker["frozen_prompt_hash"],
            "uncertain_side_effect": worker["uncertain_side_effect"],
            "latest_run": cls._safe_run_snapshot(runs[-1]) if runs else None,
        }

    def resume(
        self,
        handle: SubagentHandle,
        message: str,
        *,
        reconcile_uncertain: bool = False,
        reconciliation_disposition: Optional[str] = None,
        reconciliation_note: Optional[str] = None,
    ) -> SubagentHandle:
        """Start a linked run on the stable worker after revalidating current profile policy."""
        parent = self._parent_agent_resolver()
        owner = _owner_session_id_of(parent)
        durable = self._durable_snapshot(handle)
        if durable is None or not owner or not handle.worker_id or not handle.run_id:
            raise SubagentLifecycleError("Durable resume is unavailable for this handle.")
        store, worker, previous = durable
        runs = store.list_runs(handle.worker_id, owner)
        if not runs or runs[-1]["run_id"] != previous["run_id"]:
            raise SubagentLifecycleError("Resume must link from the worker's latest run.")
        if worker["uncertain_side_effect"]:
            if not reconcile_uncertain:
                raise SubagentLifecycleError(
                    "The prior run stopped with an uncertain tool side effect; reconcile it before resume.")
            store.reconcile_run(
                handle.run_id,
                owner,
                disposition=reconciliation_disposition or "accepted_unknown_no_replay",
                note=reconciliation_note or "",
            )
        request = SubagentLaunchRequest(
            goal=message,
            profile=worker["profile"],
            role=handle.role,
            parent_session_id=owner,
            correlation_id=f"resume-{uuid.uuid4().hex}",
        )
        return self._launch_existing_worker(request, handle, worker, previous)

    def _launch_existing_worker(
        self, request: SubagentLaunchRequest, prior: SubagentHandle, worker: Mapping[str, Any], previous: Mapping[str, Any],
    ) -> SubagentHandle:
        """Validate and queue one retained turn behind the exact durable FIFO."""
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS
        parent = self._parent_agent_resolver()
        owner = _owner_session_id_of(parent)
        store = _persistent_store(parent)
        if store is None or not owner:
            raise SubagentLifecycleError("Durable resume is unavailable for this session.")
        store.recover_expired_runs(owner, [worker["worker_id"]])
        worker = store.get_worker(worker["worker_id"], owner)
        if worker["uncertain_side_effect"]:
            raise SubagentLifecycleError(
                "The prior run stopped with an uncertain tool side effect; reconcile it before resume.")
        runs = store.list_runs(worker["worker_id"], owner)
        if not runs or runs[-1]["run_id"] != previous["run_id"]:
            raise SubagentLifecycleError("Followup must link from the worker's latest run.")
        # A cold public handle may point at a queued predecessor.  Rehydrate
        # that exact run before appending another turn; no unrelated control
        # action is required to start the older FIFO item.
        if previous["status"] == "PENDING":
            self._schedule_owner(
                store, owner, parent,
                worker_id=worker["worker_id"], run_id=previous["run_id"],
            )
        authority = self._retained_parent_authority(store, owner, worker, parent)
        if authority is None:
            raise SubagentLifecycleError(
                "The retained worker's parent authority is unavailable; its queued work remains pending.")
        validation_service = type(self)(lambda: authority)
        try:
            child, current_creds, cfg, stored_policy = validation_service._build_revalidated_child(
                worker, goal=request.goal, role=request.role)
        except (RuntimeError, ValueError) as exc:
            raise SubagentLifecycleError(f"Worker resume admission failed: {exc}") from exc
        with contextlib.suppress(Exception):
            child.close()
        followup_limit = ((stored_policy.get("profile_contract") or {}).get("execution_limits") or {}).get("max_followups")
        if isinstance(followup_limit, int) and len(runs) - 1 >= followup_limit:
            raise SubagentLifecycleError("Worker followup limit reached.")
        created = time.time()
        subagent_id = f"queued-{uuid.uuid4().hex}"
        capability = self._capability(subagent_id, owner, created)
        queued = store.enqueue_run(
            worker["worker_id"], owner, goal=request.goal, previous_run_id=previous["run_id"],
            capability_digest=hashlib.sha256(capability.encode()).hexdigest(),
            budget_epoch_id=(previous.get("budget_epoch_id") if int(worker["depth"]) > 1 else None),
            budget_limits=_tree_budget_limits(cfg, current_creds, DEFAULT_MAX_ITERATIONS),
        )
        max_concurrent = _concurrency_limit(cfg)
        result_handle = dataclasses.replace(
            prior,
            subagent_id=subagent_id,
            created_at=created,
            capability=capability,
            run_id=queued["run_id"],
        )
        record = _Record(
            result_handle, SubagentState.PENDING, created, agent=None, store=store,
            owner_session_id=owner, worker_id=worker["worker_id"], run_id=queued["run_id"],
            conversation_history=list(worker.get("history") or []),
            max_concurrent=max_concurrent, goal=request.goal, parent_agent=parent,
            budget_epoch_id=queued.get("budget_epoch_id"),
        )
        with _REGISTRY.lock:
            _REGISTRY.records[subagent_id] = record
        self._schedule_owner(
            store, owner, parent, worker_id=worker["worker_id"], run_id=queued["run_id"])
        return result_handle

    def _build_revalidated_child(
        self, worker: Mapping[str, Any], *, goal: str, role: str,
    ) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        """Build one retained turn only after current route/tool policy revalidation."""
        parent = self._parent_agent_resolver()
        from tools.delegate_tool import DEFAULT_MAX_ITERATIONS, _build_child_preserving_parent_tools, _profile_task_overrides
        from tools.delegate_tool_config import _load_config, _resolve_delegation_credentials
        cfg = _load_config()
        stored_policy = dict(worker.get("policy") or {})
        prior_route = dict(stored_policy.get("route") or {})
        route_kwargs = {}
        routing_mode = str((cfg or {}).get("routing_mode") or "profile_only").strip().lower()
        # A retained route is not a fresh model-authored override.  Reapply
        # only the original explicit request in dynamic mode, then compare the
        # newly resolved route with the stored effective route below.
        profile = worker.get("profile")
        if profile and routing_mode == "dynamic" and any(
            prior_route.get(key) is not None
            for key in ("requested_provider", "requested_model", "requested_reasoning_effort")
        ):
            route_kwargs = {
                "requested_provider": prior_route.get("requested_provider"),
                "requested_model": prior_route.get("requested_model"),
                "requested_reasoning_effort": prior_route.get("requested_reasoning_effort"),
            }
        creds = _resolve_delegation_credentials(cfg, parent, profile, **route_kwargs)
        max_iterations = _iteration_limit(cfg, creds, DEFAULT_MAX_ITERATIONS)
        overrides = _profile_task_overrides(creds) if profile else {
            "override_provider": creds["provider"], "override_base_url": creds["base_url"],
            "override_api_key": creds["api_key"], "override_api_mode": creds["api_mode"],
            "override_request_overrides": creds.get("request_overrides"),
            "override_acp_command": creds.get("command"), "override_acp_args": creds.get("args"),
            "routing_cfg": cfg,
        }
        launch_toolsets = stored_policy.get("launch_allowed_toolsets")
        launch_blocked_tools = stored_policy.get("launch_blocked_tools")
        from agent.worker_interfaces import frozen_worker_interface_contract

        retained_interface = (
            stored_policy["worker_interface"]
            if "worker_interface" in stored_policy
            else frozen_worker_interface_contract(None)
        )
        child = _build_child_preserving_parent_tools(
            task_index=0, goal=goal, context=None,
            toolsets=list(launch_toolsets) if isinstance(launch_toolsets, list) else None,
            model=creds.get("model"),
            max_iterations=max_iterations, task_count=1, parent_agent=parent, role=role,
            frozen_system_prompt=worker["frozen_prompt"],
            retained_child_depth=int(worker["depth"]),
            retained_parent_worker_id=worker.get("parent_worker_id"),
            request_blocked_tools=(
                list(launch_blocked_tools)
                if isinstance(launch_blocked_tools, list) else None
            ),
            worker_interface_contract=retained_interface,
            **overrides,
        )
        _revision, current_policy = _profile_policy_snapshot(cfg, profile, creds, child=child)
        route_keys = ("resolved_provider", "resolved_model", "resolved_reasoning_effort")
        changed_route = any(
            (current_policy.get("route") or {}).get(key) != prior_route.get(key) for key in route_keys
        )
        changed_contract = current_policy.get("profile_contract") != stored_policy.get("profile_contract")
        changed_tools = current_policy.get("effective_tools") != stored_policy.get("effective_tools")
        if changed_route or changed_contract or changed_tools:
            with contextlib.suppress(Exception):
                child.close()
            details = []
            if changed_route:
                details.append("resolved route")
            if changed_contract:
                details.append("profile instructions or policy")
            if changed_tools:
                details.append("effective tools or parent permissions")
            raise SubagentLifecycleError(
                f"Worker {'/'.join(details)} changed since launch; start a new worker to adopt the new surface.")
        return child, creds, cfg, stored_policy

    @classmethod
    def _start_record(cls, record: _Record) -> None:
        worker = record.store.get_worker(record.worker_id, record.owner_session_id)
        record.conversation_history = list(worker.get("history") or [])
        record.agent._worker_resume_history = list(record.conversation_history)
        record.agent._worker_run_id = record.run_id
        record.state = SubagentState.PENDING
        cls._bind_tool_boundary(record)
        _attach_tree_budget(record)
        service = cls(lambda: record.parent_agent)
        record.future = _EXECUTOR.submit(service._run, record, record.goal, record.parent_agent)

    @classmethod
    def _schedule_pending(cls, completed: _Record) -> None:
        if completed.store is None or not completed.owner_session_id:
            return
        cls._schedule_owner(completed.store, completed.owner_session_id, completed.parent_agent)

    @classmethod
    def _retained_parent_authority(
        cls, store: Any, owner_session_id: str,
        worker: Mapping[str, Any], triggering_agent: Any = None,
    ) -> Any:
        """Resolve the worker's immediate retained parent, never a sibling actor."""
        expected = worker.get("parent_worker_id")

        def matches(agent: Any) -> bool:
            if agent is None or _owner_session_id_of(agent) != owner_session_id:
                return False
            actor = getattr(agent, "_worker_id", None)
            return actor == expected if expected else not actor

        if matches(triggering_agent):
            return triggering_agent
        with _REGISTRY.lock:
            records = list(_REGISTRY.records.values())
        # A warm record remembers the authority that created it.
        for record in records:
            if record.owner_session_id == owner_session_id and record.worker_id == worker["worker_id"]:
                if matches(record.parent_agent):
                    return record.parent_agent
        # For a nested worker the immediate parent worker's live agent is its
        # authority.  Its root owner identity remains separate in storage.
        if expected:
            for record in records:
                if record.owner_session_id == owner_session_id and record.worker_id == expected:
                    if matches(record.agent):
                        return record.agent
        return None

    @classmethod
    def _schedule_owner(
        cls, store: Any, owner_session_id: str, parent_agent: Any = None,
        *, worker_id: Optional[str] = None, run_id: Optional[str] = None,
    ) -> None:
        """Just-in-time admission for the exact durable FIFO run(s)."""
        from tools.delegate_tool_config import _load_config

        max_concurrent = _concurrency_limit(_load_config())
        target_sequence = None
        if run_id is not None:
            target = store.get_run(run_id, owner_session_id)
            if worker_id is not None and target["worker_id"] != worker_id:
                raise SubagentLifecycleError("Requested run does not belong to the target worker.")
            worker_id = target["worker_id"]
            target_sequence = int(target["sequence"])
        for pending in store.pending_runs(owner_session_id):
            if worker_id is not None and pending["worker_id"] != worker_id:
                continue
            if target_sequence is not None and int(pending["sequence"]) > target_sequence:
                continue
            worker = store.get_worker(pending["worker_id"], owner_session_id)
            authority = cls._retained_parent_authority(
                store, owner_session_id, worker, parent_agent)
            if authority is None:
                continue
            with _REGISTRY.lock:
                record = next((
                    item for item in _REGISTRY.records.values()
                    if item.owner_session_id == owner_session_id
                    and item.run_id == pending["run_id"]
                ), None)
            # A live lease already owns execution.  Every unleased record,
            # including an old warm/prebuilt record, is rebuilt below.
            if record is not None and record.lease_token is not None:
                continue
            role = str((worker.get("policy") or {}).get("role") or "leaf")
            service = cls(lambda authority=authority: authority)
            try:
                child, creds, cfg, _policy = service._build_revalidated_child(
                    worker, goal=pending["goal"], role=role)
            except Exception as exc:
                claimed = store.claim_run(
                    pending["run_id"], owner_session_id, max_concurrent=max_concurrent)
                if claimed is not None:
                    failed = store.finish_run(
                        pending["run_id"], owner_session_id, claimed["lease_token"],
                        status="FAILED",
                        result={
                            "summary": None,
                            "error_classification": "AUTHORITY_REVALIDATION_FAILED",
                            "error_message": _clip(exc),
                            "termination": {
                                "status": "FAILED", "reason": "authority_revalidation_failed"},
                        },
                        history=list(worker.get("history") or []),
                    )
                if record is not None and claimed is not None:
                    with _REGISTRY.lock:
                        _REGISTRY.records.pop(record.handle.subagent_id, None)
                        record.agent = None
                        record.state = SubagentState.FAILED
                        record.completed_at = record.updated_at = failed["updated_at"]
                        record.result = SubagentResult(
                            record.handle, SubagentState.FAILED, True,
                            completed_at=record.completed_at,
                            error_classification="AUTHORITY_REVALIDATION_FAILED",
                            error_message=_clip(exc),
                        )
                continue
            max_concurrent = _concurrency_limit(cfg)
            claimed = store.claim_run(
                pending["run_id"], owner_session_id, max_concurrent=max_concurrent)
            if claimed is None:
                with contextlib.suppress(Exception):
                    child.close()
                continue
            child._worker_id = pending["worker_id"]
            child._worker_run_id = pending["run_id"]
            child._worker_owner_session_id = owner_session_id
            created = time.time()
            if record is None:
                subagent_id = str(
                    getattr(child, "_subagent_id", "") or f"recovered-{pending['run_id']}")
                handle = SubagentHandle(
                    PUBLIC_CONTRACT_VERSION, subagent_id, owner_session_id, None, created,
                    _text_or_none(getattr(child, "provider", None)),
                    _text_or_none(getattr(child, "model", None)), role, int(worker["depth"]), "",
                    pending["worker_id"], pending["run_id"],
                )
                record = _Record(
                    handle, SubagentState.PENDING, created, store=store,
                    owner_session_id=owner_session_id, worker_id=pending["worker_id"],
                    run_id=pending["run_id"], conversation_history=list(worker.get("history") or []),
                    goal=pending["goal"], budget_epoch_id=pending.get("budget_epoch_id"),
                )
                with _REGISTRY.lock:
                    _REGISTRY.records[subagent_id] = record
            else:
                old_agent = record.agent
                if old_agent is not None and old_agent is not child:
                    with contextlib.suppress(Exception):
                        old_agent.close()
                child._subagent_id = record.handle.subagent_id
            record.agent = child
            record.parent_agent = authority
            record.lease_token = claimed["lease_token"]
            record.max_tool_calls = getattr(creds.get("execution_limits"), "max_tool_calls", None)
            record.max_concurrent = max_concurrent
            record.goal = pending["goal"]
            record.budget_epoch_id = pending.get("budget_epoch_id")
            cls._start_record(record)

    @staticmethod
    def _bind_tool_boundary(record: _Record) -> None:
        if record.store is None or record.agent is None or not record.run_id or not record.lease_token:
            return
        record.agent._worker_lifecycle_record = record

    def _record(self, handle: SubagentHandle) -> Optional[_Record]:
        """Registry record for a well-formed, capability-verified handle owned by the active parent."""
        if not _handle_is_well_formed(handle):
            return None
        if _owner_session_id_of(self._parent_agent_resolver()) != handle.parent_session_id:
            return None
        with _REGISTRY.lock:
            record = _REGISTRY.records.get(handle.subagent_id)
        if record is None:
            return None
        # The bearer is valid only for the immutable target it was issued for;
        # changing worker/run/provider metadata must not retarget authority.
        if handle != record.handle:
            return None
        return record if hmac.compare_digest(handle.capability, record.handle.capability) else None

    def _durable_snapshot(self, handle: SubagentHandle) -> Optional[tuple[Any, Mapping[str, Any], Mapping[str, Any]]]:
        if not _handle_is_well_formed(handle) or not handle.worker_id or not handle.run_id:
            return None
        parent = self._parent_agent_resolver()
        owner = _owner_session_id_of(parent)
        if not owner or owner != handle.parent_session_id:
            return None
        try:
            store = _persistent_store(parent)
            if store is None:
                return None
            worker = store.get_worker(handle.worker_id, owner)
            run = store.get_run(handle.run_id, owner)
        except (PermissionError, ValueError):
            return None
        stored_digest = str(run.get("capability_digest") or "")
        if not stored_digest:
            # Compatibility for a pre-migration worker with exactly one run.
            # A worker-level digest cannot authorize a substituted run once
            # more than one immutable run target exists.
            runs = store.list_runs(worker["worker_id"], owner)
            if len(runs) == 1:
                stored_digest = str((worker.get("policy") or {}).get("capability_digest") or "")
        supplied_digest = hashlib.sha256(handle.capability.encode()).hexdigest()
        if not stored_digest or not hmac.compare_digest(stored_digest, supplied_digest):
            return None
        if run.get("worker_id") != worker.get("worker_id"):
            return None
        return store, worker, run

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
        lease_stop = threading.Event()
        lease_thread = None
        delivered_ids: list[str] = []
        if record.store is not None and record.run_id and record.lease_token:
            try:
                pending = record.store.claim_messages(
                    record.run_id, record.owner_session_id, record.lease_token)
                delivered_ids = [item["message_id"] for item in pending]
                record.delivered_message_ids = list(delivered_ids)
                if pending:
                    queued = "\n\n".join(item["content"] for item in pending)
                    goal = f"{goal}\n\nQueued followups (FIFO):\n{queued}"
                record.agent._worker_resume_history = list(record.conversation_history)
            except Exception as exc:
                state = SubagentState.FAILED
                fields = dict(error_classification="PERSISTENCE_FAILURE", error_message=_clip(exc))
                self._publish_result(record, state, fields)
                return

            def renew_lease():
                while not lease_stop.wait(15.0):
                    try:
                        record.store.heartbeat_run(
                            record.run_id, record.owner_session_id, record.lease_token, lease_seconds=60)
                    except Exception:
                        request_hard_interrupt(
                            record.agent,
                            "Durable worker execution lease was lost.",
                            tool_reason="worker lease lost",
                        )
                        return

            lease_thread = threading.Thread(target=renew_lease, name="hermes-worker-lease", daemon=True)
            lease_thread.start()
        raw: Any = {}
        try:
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
                summary=_clip(raw.get("summary")), error_message=_clip(raw.get("error") or None),
                error_classification=None if state == SubagentState.SUCCEEDED else status.upper(),
                usage_metadata={"api_calls": raw.get("api_calls", 0)} if is_dict else {},
                tool_execution_summary={"duration_seconds": raw.get("duration_seconds", 0)} if is_dict else {},
            )
        except Exception as exc:
            state = SubagentState.FAILED
            fields = dict(error_classification=type(exc).__name__, error_message=_clip(exc))
        finally:
            lease_stop.set()
            if lease_thread is not None:
                lease_thread.join(timeout=1.0)
        self._publish_result(
            record,
            state,
            fields,
            delivered_ids=delivered_ids,
            evidence={
                "exit_reason": raw.get("exit_reason") if isinstance(raw, dict) else None,
                "tokens": raw.get("tokens") if isinstance(raw, dict) else {},
                "cost_usd": raw.get("cost_usd") if isinstance(raw, dict) else None,
                "cost_status": raw.get("cost_status") if isinstance(raw, dict) else "unknown",
            },
        )

    @staticmethod
    def _publish_result(
        record: _Record,
        state: SubagentState,
        fields: Mapping[str, Any],
        *,
        delivered_ids: tuple[str, ...] | list[str] = (),
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if record.lease_stop is not None:
            record.lease_stop.set()
        if record.lease_thread is not None:
            record.lease_thread.join(timeout=1.0)
        result = SubagentResult(record.handle, state, True, started_at=record.started_at, completed_at=time.time(), **fields)
        # ``dataclasses.asdict`` recursively deep-copies values.  Runtime
        # metadata may contain provider objects (and test doubles) which own
        # thread locks, so hashing the public result must stay at the JSON
        # boundary instead of attempting to clone those objects.
        payload = {
            "handle": result.handle.to_dict(),
            "terminal_state": result.terminal_state.value,
            "ready": result.ready,
            "summary": result.summary,
            "structured_payload": result.structured_payload,
            "started_at": result.started_at,
            "completed_at": result.completed_at,
            "error_classification": result.error_classification,
            "error_message": result.error_message,
            "usage_metadata": dict(result.usage_metadata or {}),
            "tool_execution_summary": dict(result.tool_execution_summary or {}),
        }
        result = dataclasses.replace(result, result_hash=hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest())
        if record.store is not None and record.run_id and record.lease_token:
            try:
                worker = record.store.get_worker(record.worker_id, record.owner_session_id)
                final_history = getattr(record.agent, "_worker_last_history", None)
                # A failure path may not publish _worker_last_history even
                # though one or more tool outcomes were already checkpointed.
                # Prefer that durable history over the launch-time snapshot.
                durable_history = list(worker.get("history") or record.conversation_history)
                history = (
                    list(final_history)
                    if isinstance(final_history, list) and len(final_history) >= len(durable_history)
                    else durable_history
                )
                route = dict(getattr(record.agent, "_worker_route_receipt", None) or {})
                budget = record.store.budget_snapshot(record.run_id, record.owner_session_id)
                budget_reason = getattr(record.agent, "_worker_budget_termination_reason", None)
                durable_result = {
                    "summary": result.summary,
                    "error_classification": result.error_classification,
                    "error_message": result.error_message,
                    "result_hash": result.result_hash,
                    "route": route or None,
                    "lineage": {
                        "worker_id": record.worker_id,
                        "run_id": record.run_id,
                        "parent_worker_id": worker.get("parent_worker_id"),
                        "root_worker_id": worker.get("root_worker_id"),
                    },
                    "termination": {
                        "status": _terminal_status(state),
                        "reason": budget_reason or (evidence or {}).get("exit_reason") or result.error_classification,
                    },
                    "tree_budget": budget,
                    "usage": {
                        **dict(result.usage_metadata or {}),
                        "tokens": dict((evidence or {}).get("tokens") or {}),
                    },
                    "cost": {
                        "status": (evidence or {}).get("cost_status") or "unknown",
                        "usd": (
                            (evidence or {}).get("cost_usd")
                            if (evidence or {}).get("cost_status") not in (None, "unknown")
                            else None
                        ),
                    },
                    "effective_tools": list((worker.get("policy") or {}).get("effective_tools") or []),
                }
                finished = record.store.finish_run(
                    record.run_id,
                    record.owner_session_id,
                    record.lease_token,
                    status=_terminal_status(state),
                    result=durable_result,
                    history=history,
                    delivered_message_ids=tuple(delivered_ids),
                )
                published = {
                    item["message_id"]: item["status"]
                    for item in ((finished.get("result") or {}).get("messages_to_parent") or [])
                }
                for item in list(getattr(record.agent, "_delegate_outbound_messages", None) or []):
                    if item.get("message_id") in published:
                        item["status"] = published[item["message_id"]]
            except Exception as exc:
                result = dataclasses.replace(
                    result,
                    terminal_state=SubagentState.FAILED,
                    error_classification="PERSISTENCE_FAILURE",
                    error_message=_clip(exc),
                )
        with _REGISTRY.lock:
            record.agent, record.result, record.state = None, result, result.terminal_state
            record.completed_at = record.updated_at = result.completed_at
        SubagentLifecycleService._schedule_pending(record)

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
