"""Host-owned dispatch for built-in and plugin whole-turn runtimes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, replace
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from agent.runtime_api import (
    AgentRuntime,
    CompactionOwnership,
    RuntimeApprovalRequestEvent,
    RuntimeBackgroundOutcome,
    RuntimeBackgroundResult,
    RuntimeCancelledEvent,
    RuntimeCompactionEvent,
    RuntimeCompletedEvent,
    RuntimeContentEvent,
    RuntimeEvent,
    RuntimeFailedEvent,
    RuntimeFailure,
    RuntimeFailurePhase,
    RuntimeHostServices,
    RuntimeMCPServerInventoryEntry,
    RuntimeDescriptor,
    RuntimeRegistration,
    RuntimeSelection,
    RuntimeStateEvent,
    RuntimeStateEnvelope,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeToolInventory,
    RuntimeToolInventoryEntry,
    RuntimeToolInventorySurface,
    RuntimeTurnRequest,
    RuntimeUsageEvent,
    RuntimeUsageReceipt,
)


_RUNTIME_EVENT_TYPES = (
    RuntimeContentEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeApprovalRequestEvent,
    RuntimeCompactionEvent,
    RuntimeStateEvent,
    RuntimeUsageEvent,
    RuntimeCompletedEvent,
    RuntimeCancelledEvent,
    RuntimeFailedEvent,
)

_RUNTIME_TOOL_REQUEST_ID_MAX_LENGTH = 256
_RUNTIME_TOOL_TURN_NAMESPACE_LENGTH = 16

_RUNTIME_SIDE_EFFECT_EVENT_TYPES = (
    RuntimeCompactionEvent,
    RuntimeStateEvent,
    RuntimeUsageEvent,
)

_RUNTIME_VISIBLE_EVENT_TYPES = (
    RuntimeContentEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeApprovalRequestEvent,
)


class RuntimeExecutionError(RuntimeError):
    """A runtime failed its preflight or terminal event contract."""

    def __init__(self, message: str, *, failure: RuntimeFailure | None = None):
        super().__init__(message)
        self.failure = failure


@dataclass(frozen=True)
class RuntimeDispatchResult:
    response: Mapping[str, Any]
    events: tuple[RuntimeEvent, ...]
    terminal: RuntimeCompletedEvent | RuntimeCancelledEvent | RuntimeFailedEvent | None = None
    failure: RuntimeFailure | None = None
    cancelled: bool = False

    @property
    def completed(self) -> bool:
        """Whether the runtime produced a successful terminal event."""
        return isinstance(self.terminal, RuntimeCompletedEvent)

    @property
    def replay_safe(self) -> bool:
        """Expose the runtime's explicit replay classification to the host."""
        return bool(self.failure is not None and self.failure.replay_safe)


def _unclassified_failure_phase(events: Sequence[RuntimeEvent]) -> RuntimeFailurePhase:
    """Return the most conservative phase proved by already-emitted events."""

    if any(isinstance(event, _RUNTIME_SIDE_EFFECT_EVENT_TYPES) for event in events):
        return RuntimeFailurePhase.AFTER_SIDE_EFFECTS
    if any(isinstance(event, _RUNTIME_VISIBLE_EVENT_TYPES) for event in events):
        return RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    return RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT


def _constrain_failure_to_host_evidence(
    failure: RuntimeFailure,
    events: Sequence[RuntimeEvent],
    host: RuntimeHostServices,
    *,
    side_effects_at_turn_start: int,
) -> RuntimeFailure:
    """Prevent runtime replay claims from contradicting host-observed effects."""

    host_side_effects = getattr(host, "_side_effect_count", 0)
    side_effect_observed = host_side_effects > side_effects_at_turn_start or any(
        isinstance(event, _RUNTIME_SIDE_EFFECT_EVENT_TYPES) for event in events
    )
    if side_effect_observed:
        if (
            failure.phase is RuntimeFailurePhase.AFTER_SIDE_EFFECTS
            and not failure.replay_safe
        ):
            return failure
        return replace(
            failure,
            phase=RuntimeFailurePhase.AFTER_SIDE_EFFECTS,
            replay_safe=False,
        )
    visible_output_observed = any(
        isinstance(event, _RUNTIME_VISIBLE_EVENT_TYPES) for event in events
    )
    if visible_output_observed:
        # A runtime cannot make a later replay-safe claim after content,
        # status, tool, or approval output has crossed the host boundary.
        # Preserve an explicit stronger runtime phase, but always clear its
        # replay claim.
        if failure.phase is RuntimeFailurePhase.AFTER_SIDE_EFFECTS:
            return replace(failure, replay_safe=False)
        if (
            failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
            and not failure.replay_safe
        ):
            return failure
        return replace(
            failure,
            phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
            replay_safe=False,
        )
    return failure


def _freeze_value(value: Any) -> Any:
    """Deep-copy host-owned input into immutable public-contract values."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in copy.deepcopy(dict(value)).items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in copy.deepcopy(value))
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in copy.deepcopy(value))
    return copy.deepcopy(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy host-owned turn input before exposing it to a plugin."""
    return _freeze_value(value)


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeExecutionError(
            "runtime tool inventory schema is not canonical JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _runtime_tool_declared_by(name: str) -> str:
    """Classify one delivered tool through the active profile registry."""
    from tools.registry import registry
    from tools.tool_search import BRIDGE_TOOL_NAMES

    origin = registry.get_registration_origin(name)
    if origin is not None:
        return origin
    if name in BRIDGE_TOOL_NAMES:
        return "host"
    # The only remaining post-build schemas at this boundary are supplied by
    # external memory providers or non-default context engines. Built-in
    # ContextCompressor contributes no schemas.
    return "plugin"


def build_runtime_tool_inventory(
    tool_schemas: Sequence[Mapping[str, Any]],
    *,
    declared_by_by_name: Mapping[str, str] | None = None,
) -> RuntimeToolInventory:
    """Snapshot the exact delivered request surface without another discovery pass.

    Per-tool hashes cover the normalized input schema only. MCP server hashes
    cover the sorted delivered tool-name/schema-hash projection for that
    sanitized server name. Omitted tools and zero-tool servers are outside the
    ``delivered_request`` surface; every represented entry is therefore enabled.
    """
    entries: list[RuntimeToolInventoryEntry] = []
    seen: set[str] = set()
    server_tools: dict[str, list[dict[str, Any]]] = {}
    explicit_origins = declared_by_by_name or {}

    for raw_schema in tool_schemas:
        if not isinstance(raw_schema, Mapping):
            raise RuntimeExecutionError("runtime tool inventory schema must be a mapping")
        function = raw_schema.get("function")
        if raw_schema.get("type") == "function" and isinstance(function, Mapping):
            schema = function
        else:
            schema = raw_schema
        name = schema.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeExecutionError("runtime tool inventory schema has no name")
        if name in seen:
            raise RuntimeExecutionError(
                f"runtime tool inventory contains duplicate tool name: {name}"
            )
        seen.add(name)
        parameters = schema.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise RuntimeExecutionError(
                f"runtime tool inventory schema for {name} has invalid parameters"
            )
        schema_sha256 = _canonical_sha256(dict(parameters))
        declared_by = explicit_origins.get(name)
        if declared_by is None:
            declared_by = _runtime_tool_declared_by(name)
        entry = RuntimeToolInventoryEntry(
            name=name,
            schema_sha256=schema_sha256,
            declared_by=declared_by,
            enabled=True,
        )
        entries.append(entry)

        if name.startswith("mcp__"):
            server_name, separator, tool_name = name[5:].partition("__")
            if server_name and separator and tool_name:
                server_tools.setdefault(server_name, []).append(
                    {
                        "name": name,
                        "schema_sha256": schema_sha256,
                        "enabled": True,
                    }
                )

    sorted_entries = tuple(sorted(entries, key=lambda item: item.name))
    mcp_servers = tuple(
        RuntimeMCPServerInventoryEntry(
            name=server_name,
            schema_sha256=_canonical_sha256(
                sorted(server_tools[server_name], key=lambda item: item["name"])
            ),
            enabled=True,
        )
        for server_name in sorted(server_tools)
    )
    return RuntimeToolInventory(
        tools=sorted_entries,
        mcp_servers=mcp_servers,
        surface=RuntimeToolInventorySurface.DELIVERED_REQUEST,
        schema_version=1,
    )


def build_runtime_turn_request(
    *,
    provider: str,
    model: str,
    api_mode: str,
    messages: Sequence[Mapping[str, Any]],
    prompt_snapshot: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    tool_inventory: RuntimeToolInventory | None = None,
    session_state: RuntimeStateEnvelope | None = None,
    attachments: Sequence[Mapping[str, Any]] = (),
    correlation_id: str | None = None,
) -> RuntimeTurnRequest:
    try:
        from agent.turn_context import effective_prompt_sha256

        prompt_hash = effective_prompt_sha256(str(prompt_snapshot), messages)
    except (TypeError, ValueError) as exc:
        raise RuntimeExecutionError("runtime prompt is not canonical JSON") from exc
    canonical_tool_schemas = json.dumps(
        list(tool_schemas),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    frozen_session_state = None
    if session_state is not None:
        frozen_session_state = RuntimeStateEnvelope(
            runtime_id=str(session_state.runtime_id),
            schema_version=int(session_state.schema_version),
            state=_freeze_mapping(session_state.state),
        )
    frozen_tool_inventory = None
    if tool_inventory is not None:
        inventory_origins = {
            item.name: item.declared_by for item in tool_inventory.tools
        }
        expected_inventory = build_runtime_tool_inventory(
            tool_schemas,
            declared_by_by_name=inventory_origins,
        )
        if expected_inventory != tool_inventory:
            raise RuntimeExecutionError(
                "runtime tool inventory does not match the delivered tool schemas"
            )
        frozen_tool_inventory = expected_inventory
    return RuntimeTurnRequest(
        selection=RuntimeSelection(
            provider=provider,
            model=model,
            api_mode=api_mode,
        ),
        messages=tuple(_freeze_mapping(item) for item in messages),
        prompt_snapshot=str(prompt_snapshot),
        tool_schemas=tuple(_freeze_mapping(item) for item in tool_schemas),
        tool_schema_hash=hashlib.sha256(canonical_tool_schemas).hexdigest(),
        prompt_hash=prompt_hash,
        tool_inventory=frozen_tool_inventory,
        session_state=frozen_session_state,
        attachments=tuple(_freeze_mapping(item) for item in attachments),
        correlation_id=correlation_id,
    )


async def _collect_runtime_turn(
    runtime: AgentRuntime,
    request: RuntimeTurnRequest,
    host: RuntimeHostServices,
    descriptor: RuntimeDescriptor | None = None,
) -> RuntimeDispatchResult:
    events: list[RuntimeEvent] = []
    side_effects_at_turn_start = getattr(host, "_side_effect_count", 0)
    terminal: RuntimeCompletedEvent | RuntimeCancelledEvent | RuntimeFailedEvent | None = None
    try:
        failure = runtime.preflight(request)
        if failure is not None:
            terminal = RuntimeFailedEvent(failure=failure)
            events.append(terminal)
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                failure=failure,
            )

        async for event in runtime.run_turn(request, host):
            if not isinstance(event, _RUNTIME_EVENT_TYPES):
                raise RuntimeExecutionError(
                    f"runtime emitted unsupported event type: {type(event).__name__}"
                )
            if terminal is not None:
                raise RuntimeExecutionError(
                    "runtime emitted an event after its terminal event"
                )
            if isinstance(event, RuntimeFailedEvent):
                constrained_failure = _constrain_failure_to_host_evidence(
                    event.failure,
                    events,
                    host,
                    side_effects_at_turn_start=side_effects_at_turn_start,
                )
                if constrained_failure is not event.failure:
                    event = RuntimeFailedEvent(failure=constrained_failure)
            events.append(event)
            if isinstance(event, RuntimeToolRequestEvent):
                # Runtime events are requests crossing into host-owned
                # services.  Keep the runtime out of Hermes' executor and
                # preserve the request event in the lifecycle record; the
                # host service owns validation, policy, and result shaping.
                await host.execute_tool(
                    event.name,
                    event.arguments,
                    request_id=event.request_id,
                )
            elif isinstance(event, RuntimeApprovalRequestEvent):
                # Approval is fail-closed.  A runtime that emits a denied (or
                # malformed) decision must not be allowed to continue to its
                # own completion event.
                approved = await host.request_approval(event.action, event.details)
                if approved is not True:
                    terminal = RuntimeFailedEvent(
                        failure=RuntimeFailure(
                            code="runtime_approval_denied",
                            message="runtime approval was denied",
                            phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                            replay_safe=False,
                            retryable=False,
                        )
                    )
                    events.append(terminal)
                    break
            elif isinstance(event, RuntimeStatusEvent):
                await host.emit_status(event.message)
            elif isinstance(event, RuntimeContentEvent):
                # Runtime content is projected through the same sanitized
                # stream funnel as built-in provider deltas. Older test and
                # host doubles may not implement the optional method yet;
                # they remain valid no-op consumers of the typed event.
                emit_content = getattr(host, "emit_content", None)
                if callable(emit_content):
                    await emit_content(event.text)
            elif isinstance(event, RuntimeCompactionEvent):
                # Compaction is an observable lifecycle event, not a signal to
                # invoke the host compressor. Runtime-native implementations
                # own the actual operation; the host only projects the event.
                if (
                    descriptor is not None
                    and descriptor.compaction_ownership
                    is not CompactionOwnership.RUNTIME_NATIVE
                ):
                    raise RuntimeExecutionError(
                        "runtime emitted compaction event while host owns compaction"
                    )
                projector = getattr(host, "emit_compaction", None)
                if callable(projector):
                    await projector(event)
            elif isinstance(event, RuntimeStateEvent):
                await host.persist_state(event.state)
            elif isinstance(event, RuntimeUsageEvent):
                await host.persist_usage(event.receipt)
            if isinstance(
                event,
                (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent),
            ):
                flush_content = getattr(host, "flush_content", None)
                if callable(flush_content):
                    await flush_content()
                terminal = event

        if terminal is None:
            raise RuntimeExecutionError("runtime ended without a terminal event")
        if isinstance(terminal, RuntimeFailedEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                failure=terminal.failure,
            )
        if isinstance(terminal, RuntimeCancelledEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                cancelled=True,
            )
        return RuntimeDispatchResult(
            response=terminal.result or {},
            events=tuple(events),
            terminal=terminal,
        )
    except asyncio.CancelledError:
        # A task cancellation is a terminal runtime outcome. It never implies
        # replay safety: no provider/runtime exception is used to authorize a
        # host fallback.
        # If the runtime already emitted its terminal event, preserve it:
        # cancellation while the async generator unwinds must not create a
        # second terminal outcome for the same turn.
        if isinstance(terminal, RuntimeCompletedEvent):
            return RuntimeDispatchResult(
                response=terminal.result or {},
                events=tuple(events),
                terminal=terminal,
            )
        if isinstance(terminal, RuntimeFailedEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                failure=terminal.failure,
            )
        if isinstance(terminal, RuntimeCancelledEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                cancelled=True,
            )
        terminal = RuntimeCancelledEvent(reason="runtime task cancelled")
        events.append(terminal)
        return RuntimeDispatchResult(
            response={},
            events=tuple(events),
            terminal=terminal,
            cancelled=True,
        )
    except RuntimeExecutionError:
        # Contract violations (unknown event, duplicate terminal, or missing
        # terminal) remain hard errors. They are not classified as replayable
        # runtime failures and therefore cannot silently enter fallback.
        raise
    except Exception:
        # An unclassified runtime exception is fail-closed. Do not infer
        # replay safety from its type or message; expose only a bounded,
        # conservative result for host policy.
        # If the runtime already emitted a terminal event, preserve it:
        # an exception while the async generator unwinds must not create a
        # second terminal outcome for the same turn.
        if isinstance(terminal, RuntimeCompletedEvent):
            return RuntimeDispatchResult(
                response=terminal.result or {},
                events=tuple(events),
                terminal=terminal,
            )
        if isinstance(terminal, RuntimeFailedEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                failure=terminal.failure,
            )
        if isinstance(terminal, RuntimeCancelledEvent):
            return RuntimeDispatchResult(
                response={},
                events=tuple(events),
                terminal=terminal,
                cancelled=True,
            )
        failure = _constrain_failure_to_host_evidence(
            RuntimeFailure(
                code="runtime_exception",
                message="runtime execution failed",
                phase=_unclassified_failure_phase(events),
                replay_safe=False,
                retryable=False,
            ),
            events,
            host,
            side_effects_at_turn_start=side_effects_at_turn_start,
        )
        terminal = RuntimeFailedEvent(failure=failure)
        events.append(terminal)
        return RuntimeDispatchResult(
            response={},
            events=tuple(events),
            terminal=terminal,
            failure=failure,
        )


def run_runtime_sync(
    runtime: AgentRuntime,
    request: RuntimeTurnRequest,
    host: RuntimeHostServices,
    *,
    descriptor: RuntimeDescriptor | None = None,
) -> RuntimeDispatchResult:
    """Run the async contract from Hermes' existing synchronous turn loop."""
    from model_tools import _run_async

    return _run_async(_collect_runtime_turn(runtime, request, host, descriptor))


class BuiltInCodexRuntime:
    """Codex whole-turn consumer of AgentRuntime v1.

    The callback is host-owned and captures the legacy Codex adapter while it
    is incrementally migrated.  Third-party runtimes never receive this
    callback or the private AIAgent object it closes over.
    """

    def __init__(self, runner: Callable[[], Mapping[str, Any]]):
        self._runner = runner

    def preflight(self, request: RuntimeTurnRequest) -> RuntimeFailure | None:
        return None

    async def run_turn(self, request, host):
        if host.cancellation_requested():
            yield RuntimeCancelledEvent(reason="cancelled before runtime start")
            return
        yield RuntimeCompletedEvent(result=self._runner())

    async def close(self) -> None:
        return None

    def refresh_runner(self, runner: Callable[[], Mapping[str, Any]]) -> None:
        """Refresh the per-turn host callback without replacing the session runtime."""
        self._runner = runner


def make_builtin_codex_registration(
    runner: Callable[[], Mapping[str, Any]],
) -> RuntimeRegistration:
    """Return the host-owned Codex consumer for the shared runtime resolver."""
    return RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="hermes-codex-app-server",
            plugin_version="builtin",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset({"cancellation_v1"}),
            provider_ids=frozenset(),
            api_modes=frozenset({"codex_app_server"}),
            session_state_schema_version=1,
            compaction_ownership=CompactionOwnership.RUNTIME_NATIVE,
        ),
        factory=lambda: BuiltInCodexRuntime(runner),
        plugin_id="hermes-core",
    )


class RuntimeToolPersistenceError(RuntimeError):
    """The canonical tool-call row could not be flushed before execution."""


class HermesRuntimeHostServices:
    """The only stateful Hermes surface available to runtime plugins."""

    def __init__(
        self,
        agent: Any,
        *,
        task_id: str,
        runtime_id: str,
        turn_messages: list[dict[str, Any]] | None = None,
        correlation_id: str | None = None,
    ):
        self._agent = agent
        self._task_id = str(task_id)
        self._runtime_id = str(runtime_id)
        self._parent_session_id = str(getattr(agent, "session_id", None) or "")
        self._closed = False
        self._delivery_lock = threading.Lock()
        self._delivery_counter = 0
        self._route: dict[str, str] = {}
        self._turn_messages = turn_messages
        self._turn_namespace = ""
        self._tool_results_by_id: dict[str, Any] = {}
        self._tool_call_specs_by_id: dict[str, tuple[str, str]] = {}
        self._tool_call_ids_seen: set[str] = set()
        self._tool_calls_in_flight: set[str] = set()
        self._tool_call_count = 0
        self._side_effect_count = 0
        self._compaction_events: list[dict[str, Any]] = []
        from agent.agent_runtime_helpers import StreamingToolCallMarkupScrubber

        self._content_scrubber = StreamingToolCallMarkupScrubber()
        self.refresh_turn(
            task_id,
            turn_messages=turn_messages,
            correlation_id=correlation_id,
        )
        try:
            # A fresh host is created for each whole turn, so the lifecycle
            # projection cannot accidentally bleed into a later turn.
            self._agent._runtime_compaction_events = self._compaction_events
        except Exception:
            pass

    def _ensure_open_parent_locked(self) -> None:
        """Fail closed unless this host still owns its captured Hermes parent."""
        if self._closed:
            raise RuntimeExecutionError("runtime host binding is closed")
        current_parent = str(getattr(self._agent, "session_id", None) or "")
        if current_parent != self._parent_session_id:
            raise RuntimeExecutionError(
                "runtime host binding cannot move to a different Hermes session"
            )

    def _ensure_open_parent(self) -> None:
        """Linearize a host operation before it touches state or side effects."""
        with self._delivery_lock:
            self._ensure_open_parent_locked()

    def _emit_status_locked(self, message: str) -> None:
        touch = getattr(self._agent, "_touch_activity", None)
        if callable(touch):
            touch(message)

    def refresh_turn(
        self,
        task_id: str,
        turn_messages: list[dict[str, Any]] | None = None,
        *,
        correlation_id: str | None = None,
    ) -> None:
        """Refresh per-turn correlation and route data for the bound parent only."""
        with self._delivery_lock:
            self._ensure_open_parent_locked()
            self._task_id = str(task_id)
            self._turn_messages = turn_messages
            # The gateway-issued correlation ID is the stable identity of a
            # whole turn.  Task IDs intentionally may be reused by a cached
            # gateway agent, so they cannot namespace fallback tool IDs across
            # turns.  Keep the task fallback for direct/manual host callers
            # that do not have a request identity yet.
            turn_key = str(correlation_id or self._task_id)
            turn_identity = json.dumps(
                [self._runtime_id, turn_key],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            self._turn_namespace = hashlib.sha256(turn_identity).hexdigest()[
                :_RUNTIME_TOOL_TURN_NAMESPACE_LENGTH
            ]
            self._tool_results_by_id = {}
            self._tool_call_specs_by_id = {}
            self._tool_call_ids_seen = set()
            self._tool_calls_in_flight = set()
            self._tool_call_count = 0
            self._content_scrubber.reset()
            try:
                from gateway.session_context import get_session_env

                self._route = {
                    event_key: get_session_env(env_name, "")
                    for event_key, env_name in (
                        ("session_key", "HERMES_SESSION_KEY"),
                        ("origin_ui_session_id", "HERMES_UI_SESSION_ID"),
                        ("platform", "HERMES_SESSION_PLATFORM"),
                        ("chat_type", "HERMES_SESSION_CHAT_TYPE"),
                        ("chat_id", "HERMES_SESSION_CHAT_ID"),
                        ("thread_id", "HERMES_SESSION_THREAD_ID"),
                        ("user_id", "HERMES_SESSION_USER_ID"),
                        ("scope_id", "HERMES_SESSION_SCOPE_ID"),
                    )
                }
            except Exception:
                self._route = {}
            allowed = set(getattr(self._agent, "valid_tool_names", ()) or ())
            for schema in getattr(self._agent, "tools", ()) or ():
                if not isinstance(schema, Mapping):
                    continue
                function = schema.get("function")
                if isinstance(function, Mapping) and function.get("name"):
                    allowed.add(str(function["name"]))
            self._allowed_tool_names = frozenset(allowed)

    @staticmethod
    def _normalize_tool_request_id(request_id: str | None) -> str | None:
        if request_id is None:
            return None
        if not isinstance(request_id, str):
            raise RuntimeExecutionError("runtime tool request id must be a string")
        normalized = request_id.strip()
        if not normalized:
            raise RuntimeExecutionError("runtime tool request id must not be empty")
        if len(normalized) > _RUNTIME_TOOL_REQUEST_ID_MAX_LENGTH:
            raise RuntimeExecutionError(
                "runtime tool request id exceeds the 256 character bound"
            )
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in normalized):
            raise RuntimeExecutionError(
                "runtime tool request id contains a control character"
            )
        return normalized

    async def execute_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> Any:
        """Execute one runtime-requested tool through Hermes' canonical funnel.

        The synthetic assistant/tool-call objects are host-private adapters for
        the existing single-call executor.  That executor owns scope checks,
        plugin middleware and approval, guardrails, progress, persistence, and
        terminal result normalization.  The plugin receives only the canonical
        tool result content.
        """
        self._ensure_open_parent()
        normalized_name = str(name or "").strip()
        if not normalized_name or normalized_name not in self._allowed_tool_names:
            raise RuntimeExecutionError(
                f"tool '{normalized_name or '<empty>'}' is not available in this session"
            )
        if not isinstance(arguments, Mapping):
            raise RuntimeExecutionError("runtime tool arguments must be a mapping")
        normalized_request_id = self._normalize_tool_request_id(request_id)

        executor = getattr(self._agent, "_execute_tool_calls", None)
        if not callable(executor):
            raise RuntimeExecutionError("Hermes tool executor is unavailable")

        try:
            arguments_json = json.dumps(
                dict(arguments),
                sort_keys=True,
                ensure_ascii=False,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeExecutionError(
                "runtime tool arguments must be JSON serializable"
            ) from exc

        # A public request ID is the transcript pairing key.  Cache the exact
        # name/arguments tuple too, so a duplicate ID cannot silently replay a
        # result for a different call payload.
        with self._delivery_lock:
            tool_call_id = normalized_request_id
            if tool_call_id is None:
                self._tool_call_count += 1
                tool_call_id = (
                    f"runtime-tool-{self._turn_namespace}-"
                    f"{self._tool_call_count:04d}"
                )
            call_spec = (normalized_name, arguments_json)
            prior_result = self._tool_results_by_id.get(tool_call_id)
            if tool_call_id in self._tool_results_by_id:
                if self._tool_call_specs_by_id.get(tool_call_id) != call_spec:
                    raise RuntimeExecutionError(
                        "duplicate runtime tool call id with different payload: "
                        f"{tool_call_id}"
                    )
                return prior_result
            if (
                tool_call_id in self._tool_call_ids_seen
                or tool_call_id in self._tool_calls_in_flight
            ):
                raise RuntimeExecutionError(
                    f"duplicate runtime tool call id: {tool_call_id}"
                )
            self._tool_call_specs_by_id[tool_call_id] = call_spec
            self._tool_call_ids_seen.add(tool_call_id)
            self._tool_calls_in_flight.add(tool_call_id)

        # Build the same paired assistant row the ordinary loop persists
        # before invoking Hermes' canonical executor. This row is deliberately
        # appended to the live turn list, not a throwaway adapter list, so the
        # finalizer and the runtime host share one transcript authority.
        tool_call = SimpleNamespace(
            id=tool_call_id,
            type="function",
            function=SimpleNamespace(
                name=normalized_name,
                arguments=arguments_json,
            ),
        )
        assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
        assistant_row = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": normalized_name,
                        "arguments": arguments_json,
                    },
                }
            ],
            "finish_reason": "tool_calls",
        }
        turn_messages = self._turn_messages
        if turn_messages is None:
            turn_messages = []
        assistant_index = len(turn_messages)
        turn_messages.append(assistant_row)
        flush = getattr(self._agent, "_flush_messages_to_session_db", None)
        if callable(flush):
            try:
                persisted = flush(turn_messages)
            except Exception as exc:
                turn_messages.pop()
                self._agent._incremental_persistence_failed = True
                self._agent._last_persistence_error_cause = "runtime_tool_call"
                with self._delivery_lock:
                    self._tool_calls_in_flight.discard(tool_call_id)
                    self._tool_call_ids_seen.discard(tool_call_id)
                    self._tool_call_specs_by_id.pop(tool_call_id, None)
                raise RuntimeToolPersistenceError(
                    "runtime tool-call persistence failed before execution"
                ) from exc
            if persisted is False:
                turn_messages.pop()
                self._agent._incremental_persistence_failed = True
                self._agent._last_persistence_error_cause = "runtime_tool_call"
                with self._delivery_lock:
                    self._tool_calls_in_flight.discard(tool_call_id)
                    self._tool_call_ids_seen.discard(tool_call_id)
                    self._tool_call_specs_by_id.pop(tool_call_id, None)
                raise RuntimeToolPersistenceError(
                    "runtime tool-call persistence failed before execution"
                )

        # Match the ordinary loop's post-flush projection ordering: UI sees a
        # tool-call card only after its canonical assistant row is durable.
        interim = getattr(self._agent, "_emit_interim_assistant_message", None)
        if callable(interim):
            try:
                interim(assistant_row)
            except Exception:
                pass
        display_callback = getattr(self._agent, "stream_delta_callback", None)
        if callable(display_callback):
            try:
                display_callback(None)
            except Exception:
                pass

        # Entering the canonical executor is host-observed effect evidence.
        # Count it before invocation so an executor failure cannot make a
        # potentially executed tool replayable.
        self._side_effect_count += 1
        try:
            executor(assistant_message, turn_messages, self._task_id)
            if getattr(self._agent, "_incremental_persistence_failed", False):
                self._agent._last_persistence_error_cause = (
                    getattr(
                        self._agent,
                        "_last_persistence_error_cause",
                        None,
                    )
                    or "runtime_tool_result"
                )
                raise RuntimeToolPersistenceError(
                    "runtime tool result persistence failed"
                )
            matches = [
                message
                for message in turn_messages[assistant_index + 1 :]
                if isinstance(message, Mapping)
                and message.get("role") == "tool"
                and message.get("tool_call_id") == tool_call_id
            ]
            if len(matches) != 1:
                raise RuntimeExecutionError(
                    "Hermes tool executor did not produce exactly one canonical result"
                )
            result = matches[0].get("content")
            with self._delivery_lock:
                self._tool_results_by_id[tool_call_id] = result
            return result
        finally:
            with self._delivery_lock:
                self._tool_calls_in_flight.discard(tool_call_id)

    async def request_approval(
        self,
        action: str,
        details: Mapping[str, Any],
    ) -> bool:
        self._ensure_open_parent()
        from tools.approval import request_tool_approval

        try:
            from tools.terminal_tool import _get_approval_callback

            callback = _get_approval_callback()
        except Exception:
            callback = None
        decision = request_tool_approval(
            action,
            str(details.get("reason") or f"Runtime requested approval for {action}"),
            rule_key=str(details.get("rule_key") or ""),
            approval_callback=callback,
        )
        return bool(decision.get("approved"))

    async def emit_status(self, message: str) -> None:
        self._ensure_open_parent()
        self._emit_status_locked(message)

    async def emit_content(self, text: str) -> None:
        """Project visible runtime content through Hermes' sanitizing funnel."""
        self._ensure_open_parent()
        if not isinstance(text, str):
            raise RuntimeExecutionError("runtime content must be text")
        if not text:
            return
        # Remove provider-neutral tool-call markup before handing the delta to
        # AIAgent. Its stateful scrubber keeps split tool markers buffered;
        # reasoning tags remain intact for the separate reasoning scrubber.
        text = self._content_scrubber.feed(text)
        if not text:
            return
        self._deliver_content(text)

    def _deliver_content(self, text: str) -> None:
        """Send already-sanitized content through the existing callback seam."""
        if not text:
            return
        fire_delta = getattr(self._agent, "_fire_stream_delta", None)
        if callable(fire_delta):
            fire_delta(text)
            return
        # Minimal host doubles may not expose the full sanitizer. Preserve
        # their existing callback seam without inventing a second stream API.
        callbacks = []
        for callback in (
            getattr(self._agent, "stream_delta_callback", None),
            getattr(self._agent, "_stream_callback", None),
        ):
            if callable(callback) and callback not in callbacks:
                callbacks.append(callback)
        for callback in callbacks:
            try:
                callback(text)
            except Exception:
                pass

    async def flush_content(self) -> None:
        """Flush benign buffered content before a runtime terminal event."""
        self._ensure_open_parent()
        self._deliver_content(self._content_scrubber.flush())

    async def persist_state(self, state: RuntimeStateEnvelope) -> None:
        self._ensure_open_parent()
        if state.runtime_id != self._runtime_id:
            raise RuntimeExecutionError(
                "runtime state identity does not match the selected runtime"
            )
        database = getattr(self._agent, "_session_db", None)
        if database is None or not self._parent_session_id:
            raise RuntimeExecutionError(
                "runtime state persistence requires an active Hermes session"
            )
        database.update_runtime_state(self._parent_session_id, state)

    async def persist_usage(self, receipt: RuntimeUsageReceipt) -> None:
        self._ensure_open_parent()
        if receipt.runtime_id != self._runtime_id:
            raise RuntimeExecutionError(
                "runtime usage identity does not match the selected runtime"
            )
        database = getattr(self._agent, "_session_db", None)
        if database is None or not self._parent_session_id:
            raise RuntimeExecutionError(
                "runtime usage persistence requires an active Hermes session"
            )
        inserted = database.record_runtime_usage_receipt(
            self._parent_session_id,
            receipt,
        )
        if not inserted:
            return
        database.queue_token_counts(
            self._parent_session_id,
            input_tokens=receipt.input_tokens,
            output_tokens=receipt.output_tokens,
            cache_read_tokens=receipt.cache_read_tokens,
            cache_write_tokens=receipt.cache_write_tokens,
            reasoning_tokens=receipt.reasoning_tokens,
            billing_provider=receipt.provider,
            billing_mode=receipt.billing_mode,
            cost_status=receipt.cost_status,
            model=receipt.model,
            api_call_count=1,
        )

    async def emit_compaction(self, event: RuntimeCompactionEvent) -> None:
        """Project runtime-native compaction into the host lifecycle stream.

        The event remains runtime-owned: this method never invokes Hermes'
        compressor.  Only its typed phase and a bounded set of scalar details
        are retained on the agent for lifecycle observers; arbitrary runtime
        payloads are intentionally not persisted or surfaced.
        """
        with self._delivery_lock:
            self._ensure_open_parent_locked()
            if not isinstance(event, RuntimeCompactionEvent):
                raise RuntimeExecutionError(
                    "runtime compaction event has an unsupported type"
                )
            details = {
                key: value
                for key, value in event.details.items()
                if key in {"watchdog_seconds"}
                and isinstance(value, (str, int, float, bool))
            }
            record = {
                "runtime_id": self._runtime_id,
                "phase": event.phase.value,
                "details": details,
            }
            self._compaction_events.append(record)

            # Keep existing host status delivery as the lifecycle projection. The
            # message is derived solely from the typed phase and carries no runtime
            # details or provider-specific policy.
            self._emit_status_locked(f"Runtime compaction {event.phase.value}")

    async def emit_background_result(
        self,
        result: RuntimeBackgroundResult,
    ) -> None:
        """Queue a detached result on Hermes' existing host delivery rail."""
        if not isinstance(result, RuntimeBackgroundResult):
            raise RuntimeExecutionError("background result has an unsupported type")
        if not self._parent_session_id:
            raise RuntimeExecutionError(
                "background delivery requires an active Hermes parent session"
            )
        with self._delivery_lock:
            self._ensure_open_parent_locked()
            from tools.process_registry import process_registry

            self._delivery_counter += 1
            delivery_id = (
                f"runtime-background-{self._delivery_counter:04d}-{uuid.uuid4().hex}"
            )
            now = time.time()
            completed = result.outcome is RuntimeBackgroundOutcome.COMPLETED
            event = {
                # Reuse the existing async-completion consumer. Legacy events
                # without a durable delegation row are already supported: the
                # gateway/TUI claim, exact-parent target preflight, busy-session
                # requeue, transcript turn, and adapter retry paths stay host-owned.
                "type": "async_delegation",
                "delegation_id": delivery_id,
                "dispatched_at": now,
                "completed_at": now,
                "parent_session_id": self._parent_session_id,
                "goal": "runtime background work",
                "role": "runtime",
                "model": "host-routed",
                "status": "completed" if completed else "failed",
                "summary": result.content if completed else None,
                "error": None if completed else result.content,
                "api_calls": 0,
                "duration_seconds": 0,
                **{key: value for key, value in self._route.items() if value},
            }
            process_registry.completion_queue.put(event)
            self._side_effect_count += 1

    async def close(self) -> None:
        """Reject future background emissions from the retired parent binding."""
        with self._delivery_lock:
            self._closed = True

    def cancellation_requested(self) -> bool:
        with self._delivery_lock:
            current_parent = str(getattr(self._agent, "session_id", None) or "")
            return bool(
                self._closed
                or current_parent != self._parent_session_id
                or getattr(self._agent, "_interrupt_requested", False)
            )


class RuntimeSessionBinding:
    """One runtime and host-services binding owned by one Hermes parent session."""

    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        host: HermesRuntimeHostServices,
        descriptor: RuntimeDescriptor,
        plugin_id: str,
        parent_session_id: str,
    ):
        self.runtime = runtime
        self.host = host
        self.descriptor = descriptor
        self.plugin_id = plugin_id
        self.parent_session_id = parent_session_id
        self._closed = False
        self._closing = False
        self._close_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name=f"hermes-runtime-{runtime.__class__.__name__}",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=5.0):
            raise RuntimeExecutionError("runtime session loop did not start")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    def _run(self, coroutine: Any, *, allow_closing: bool = False) -> Any:
        with self._close_lock:
            unavailable = self._closed or (
                self._closing and not allow_closing
            )
        if (
            unavailable
            or self._loop.is_closed()
            or not self._loop_thread.is_alive()
            or threading.current_thread() is self._loop_thread
        ):
            close = getattr(coroutine, "close", None)
            if callable(close):
                close()
            raise RuntimeExecutionError("runtime session loop is closed")
        # run_coroutine_threadsafe carries the caller's ContextVars. The
        # wrapper supplies the thread-local approval/sudo callbacks for the
        # duration of this turn on the lifecycle-stable runtime loop.
        from tools.thread_context import propagate_callbacks_to_async_thread

        future = asyncio.run_coroutine_threadsafe(
            propagate_callbacks_to_async_thread(coroutine),
            self._loop,
        )
        return future.result()

    def run_turn(self, request: RuntimeTurnRequest) -> RuntimeDispatchResult:
        """Run every turn for this binding on one lifecycle-stable event loop."""
        return self._run(
            _collect_runtime_turn(
                self.runtime,
                request,
                self.host,
                self.descriptor,
            )
        )

    def close(self) -> None:
        """Close the host gate and runtime exactly once."""
        with self._close_lock:
            if self._closed or self._closing:
                return
            self._closing = True

        async def _close() -> None:
            await self.host.close()
            try:
                await self.runtime.close()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        try:
            self._run(_close(), allow_closing=True)
        finally:
            with self._close_lock:
                self._closed = True
                self._closing = False
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5.0)
            if self._loop_thread.is_alive():
                raise RuntimeExecutionError("runtime session loop did not stop")


def get_runtime_session(
    agent: Any,
    registration: RuntimeRegistration,
    *,
    task_id: str,
    turn_messages: list[dict[str, Any]] | None = None,
    correlation_id: str | None = None,
) -> RuntimeSessionBinding:
    """Return the exact-parent runtime binding, creating it once per session."""
    parent_session_id = str(getattr(agent, "session_id", None) or "")
    existing = getattr(agent, "_runtime_session_binding", None)
    if isinstance(existing, RuntimeSessionBinding):
        same_identity = (
            existing.parent_session_id == parent_session_id
            and existing.descriptor == registration.descriptor
            and existing.plugin_id == registration.plugin_id
        )
        if same_identity:
            if isinstance(existing.runtime, BuiltInCodexRuntime):
                candidate = registration.factory()
                if isinstance(candidate, BuiltInCodexRuntime):
                    existing.runtime.refresh_runner(candidate._runner)
            existing.host.refresh_turn(
                task_id,
                turn_messages,
                correlation_id=correlation_id,
            )
            return existing
        close_runtime_session(agent)

    runtime = registration.factory()
    host = HermesRuntimeHostServices(
        agent,
        task_id=task_id,
        runtime_id=registration.descriptor.runtime_id,
        turn_messages=turn_messages,
        correlation_id=correlation_id,
    )
    binding = RuntimeSessionBinding(
        runtime=runtime,
        host=host,
        descriptor=registration.descriptor,
        plugin_id=registration.plugin_id,
        parent_session_id=parent_session_id,
    )
    agent._runtime_session_binding = binding
    return binding


def close_runtime_session(agent: Any) -> None:
    """Detach and close an agent's cached runtime binding exactly once."""
    binding = getattr(agent, "_runtime_session_binding", None)
    if binding is None:
        return
    agent._runtime_session_binding = None
    close = getattr(binding, "close", None)
    if callable(close):
        close()
