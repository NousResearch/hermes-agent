"""Whole-turn runtime dispatch behavior contracts."""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import threading
from collections.abc import AsyncIterator

import pytest

from agent.runtime_api import (
    RuntimeBackgroundOutcome,
    RuntimeBackgroundResult,
    RuntimeApprovalRequestEvent,
    RuntimeCancelledEvent,
    RuntimeCompactionEvent,
    RuntimeCompactionPhase,
    CompactionOwnership,
    RuntimeCompletedEvent,
    RuntimeContentEvent,
    RuntimeEventKind,
    RuntimeFailedEvent,
    RuntimeFailure,
    RuntimeFailurePhase,
    RuntimeDescriptor,
    RuntimeSelection,
    RuntimeStateEnvelope,
    RuntimeStateEvent,
    RuntimeStatusEvent,
    RuntimeToolRequestEvent,
    RuntimeToolInventory,
    RuntimeToolInventoryEntry,
    RuntimeToolInventorySurface,
    RuntimeUsageEvent,
    RuntimeUsageReceipt,
    RuntimeRegistration,
    runtime_api_manifest,
)
from agent.runtime_dispatch import (
    HermesRuntimeHostServices,
    RuntimeExecutionError,
    RuntimeToolPersistenceError,
    build_runtime_tool_inventory,
    build_runtime_turn_request,
    close_runtime_session,
    get_runtime_session,
    make_builtin_codex_registration,
    run_runtime_sync,
)
from agent.turn_context import (
    build_effective_prompt_messages,
    compose_effective_system_prompt,
    effective_prompt_sha256,
)
from model_tools import _run_async


class _HostServices:
    def __init__(self):
        self.statuses = []
        self.states = []
        self.receipts = []
        self.compactions = []

    async def execute_tool(self, name, arguments):
        raise AssertionError("not used")

    async def request_approval(self, action, details):
        raise AssertionError("not used")

    async def emit_status(self, message):
        self.statuses.append(message)

    async def persist_state(self, state):
        self.states.append(state)

    async def persist_usage(self, receipt):
        self.receipts.append(receipt)

    async def emit_compaction(self, event):
        self.compactions.append(event)

    def cancellation_requested(self):
        return False


class _RuntimeRequestHost(_HostServices):
    def __init__(self, *, approval: bool = True):
        super().__init__()
        self.approval = approval
        self.calls = []
        self.request_ids = []

    async def execute_tool(self, name, arguments, *, request_id=None):
        self.calls.append(("tool", name, dict(arguments)))
        self.request_ids.append(request_id)
        return {"ok": True, "name": name}

    async def request_approval(self, action, details):
        self.calls.append(("approval", action, dict(details)))
        return self.approval


class _ContentHost(_HostServices):
    def __init__(self):
        super().__init__()
        self.contents = []

    async def emit_content(self, text):
        self.contents.append(text)


def _request():
    return build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=({"role": "user", "content": "hello"},),
        prompt_snapshot="stable prompt",
        tool_schemas=(),
    )


def test_runtime_turn_request_deep_freezes_state_and_host_inputs():
    messages = [{"role": "user", "content": {"parts": ["hello"]}}]
    tools = [{"type": "function", "function": {"name": "pwd"}}]
    state_data = {"resume": {"external": "synthetic"}}

    request = build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=messages,
        prompt_snapshot="stable prompt",
        tool_schemas=tools,
        session_state=RuntimeStateEnvelope(
            runtime_id="example-runtime",
            schema_version=1,
            state=state_data,
        ),
    )
    messages[0]["content"]["parts"].append("late mutation")
    tools[0]["function"]["name"] = "terminal"
    state_data["resume"]["external"] = "late mutation"

    assert request.messages[0]["content"]["parts"] == ("hello",)
    assert request.tool_schemas[0]["function"]["name"] == "pwd"
    assert request.session_state.state["resume"]["external"] == "synthetic"
    with pytest.raises(TypeError):
        request.session_state.state["resume"]["external"] = "blocked"


def test_effective_prompt_projection_is_shared_by_runtime_request_and_hash():
    messages = [
        {
            "role": "user",
            "content": "hello",
            "api_content": "hello\n\n<synthetic-context>",
            "display_kind": "user",
            "_db_persisted": True,
        },
        {
            "role": "assistant",
            "content": "answer",
            "api_content": "answer",
            "display_metadata": {"synthetic": True},
        },
    ]
    effective_messages = build_effective_prompt_messages(
        messages,
        current_turn_user_idx=0,
        ext_prefetch_cache="unused",
        plugin_user_context="unused",
    )
    effective_system = compose_effective_system_prompt(
        "base system",
        "ephemeral system",
    )
    request = build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=effective_messages,
        prompt_snapshot=effective_system,
        tool_schemas=(),
    )

    assert list(request.messages) == effective_messages
    assert request.prompt_hash == effective_prompt_sha256(
        effective_system,
        effective_messages,
    )
    assert request.effective_prompt_hash == request.prompt_hash
    assert effective_messages[0]["content"] == "hello\n\n<synthetic-context>"
    assert "api_content" not in effective_messages[0]
    assert "display_kind" not in effective_messages[0]


def test_runtime_tool_inventory_is_stable_and_hashes_input_schemas():
    filesystem_parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    plugin_parameters = {"type": "object", "properties": {}}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "plugin_tool",
                "description": "Synthetic plugin tool",
                "parameters": plugin_parameters,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mcp__filesystem__read_file",
                "description": "Synthetic MCP tool",
                "parameters": filesystem_parameters,
            },
        },
    ]

    inventory = build_runtime_tool_inventory(
        tools,
        declared_by_by_name={
            "mcp__filesystem__read_file": "host",
            "plugin_tool": "plugin",
        },
    )
    reordered = build_runtime_tool_inventory(
        list(reversed(tools)),
        declared_by_by_name={
            "mcp__filesystem__read_file": "host",
            "plugin_tool": "plugin",
        },
    )

    assert inventory == reordered
    assert inventory.surface is RuntimeToolInventorySurface.DELIVERED_REQUEST
    assert [entry.name for entry in inventory.tools] == [
        "mcp__filesystem__read_file",
        "plugin_tool",
    ]
    assert [entry.declared_by for entry in inventory.tools] == ["host", "plugin"]
    assert all(entry.enabled for entry in inventory.tools)
    expected_schema_hash = hashlib.sha256(
        json.dumps(
            filesystem_parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert inventory.tools[0].schema_sha256 == expected_schema_hash
    assert [server.name for server in inventory.mcp_servers] == ["filesystem"]
    assert inventory.mcp_servers[0].enabled is True
    assert len(inventory.mcp_servers[0].schema_sha256) == 64


def test_runtime_turn_request_copies_the_tool_inventory_contract():
    tool_schema = {
        "type": "function",
        "function": {
            "name": "pwd",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    inventory = build_runtime_tool_inventory(
        (tool_schema,),
        declared_by_by_name={"pwd": "host"},
    )
    request = build_runtime_turn_request(
        provider="example",
        model="example-large",
        api_mode="example_runtime",
        messages=(),
        prompt_snapshot="stable prompt",
        tool_schemas=(tool_schema,),
        tool_inventory=inventory,
    )

    assert request.tool_inventory == inventory
    assert request.tool_inventory is not inventory
    with pytest.raises(Exception):
        request.tool_inventory.tools[0].name = "mutated"


def test_runtime_tool_inventory_normalizes_direct_sequences_to_tuples():
    source_entries = [
        RuntimeToolInventoryEntry(
            name="pwd",
            schema_sha256="0" * 64,
            declared_by="host",
        )
    ]

    inventory = RuntimeToolInventory(tools=source_entries)
    source_entries.clear()

    assert isinstance(inventory.tools, tuple)
    assert [entry.name for entry in inventory.tools] == ["pwd"]


def test_runtime_turn_request_rejects_inventory_schema_mismatch():
    inventory = build_runtime_tool_inventory(
        (
            {
                "type": "function",
                "function": {
                    "name": "pwd",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ),
        declared_by_by_name={"pwd": "host"},
    )

    with pytest.raises(RuntimeExecutionError, match="does not match"):
        build_runtime_turn_request(
            provider="example",
            model="example-large",
            api_mode="example_runtime",
            messages=(),
            prompt_snapshot="stable prompt",
            tool_schemas=(),
            tool_inventory=inventory,
        )


def test_runtime_tool_inventory_marks_host_tool_search_bridges_as_host():
    inventory = build_runtime_tool_inventory(
        (
            {
                "type": "function",
                "function": {
                    "name": "tool_search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        )
    )

    assert inventory.tools[0].declared_by == "host"


def test_runtime_tool_inventory_rejects_duplicate_delivered_names():
    duplicate = {
        "type": "function",
        "function": {
            "name": "pwd",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    with pytest.raises(RuntimeExecutionError, match="duplicate tool name"):
        build_runtime_tool_inventory((duplicate, duplicate))


def test_public_runtime_request_events_are_typed_and_frozen():
    tool = RuntimeToolRequestEvent(
        request_id="tool-1",
        name="pwd",
        arguments={"path": "."},
    )
    approval = RuntimeApprovalRequestEvent(
        request_id="approval-1",
        action="terminal",
        details={"reason": "synthetic test"},
    )
    compaction = RuntimeCompactionEvent(
        phase=RuntimeCompactionPhase.STARTED,
        details={"watchdog_seconds": 60},
    )

    assert tool.kind is RuntimeEventKind.TOOL_REQUEST
    assert approval.kind is RuntimeEventKind.APPROVAL_REQUEST
    assert compaction.kind is RuntimeEventKind.COMPACTION
    with pytest.raises(Exception):
        tool.name = "terminal"


class _UnknownEventRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield object()
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_calls += 1


def test_dispatch_rejects_unknown_event_types_without_closing_session_runtime():
    runtime = _UnknownEventRuntime()

    with pytest.raises(RuntimeExecutionError, match="unsupported event type"):
        run_runtime_sync(runtime, _request(), _HostServices())

    assert runtime.close_calls == 0


class _RequestEventsRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeContentEvent(text="visible before requests")
        yield RuntimeToolRequestEvent(
            request_id="tool-request-1",
            name="synthetic_tool",
            arguments={"value": "one"},
        )
        yield RuntimeApprovalRequestEvent(
            request_id="approval-request-1",
            action="synthetic_action",
            details={"reason": "synthetic approval"},
        )
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        return None


def test_dispatch_routes_typed_tool_and_approval_events_through_host_services():
    host = _RuntimeRequestHost()

    result = run_runtime_sync(_RequestEventsRuntime(), _request(), host)

    assert result.completed is True
    assert host.calls == [
        ("tool", "synthetic_tool", {"value": "one"}),
        ("approval", "synthetic_action", {"reason": "synthetic approval"}),
    ]
    assert host.request_ids == ["tool-request-1"]
    assert [event.request_id for event in result.events if isinstance(
        event, (RuntimeToolRequestEvent, RuntimeApprovalRequestEvent)
    )] == ["tool-request-1", "approval-request-1"]


def test_dispatch_projects_runtime_content_before_terminal_completion():
    class _ContentRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            yield RuntimeContentEvent(text="visible runtime delta")
            yield RuntimeCompletedEvent(result={"final_response": "done"})

        async def close(self):
            return None

    host = _ContentHost()
    result = run_runtime_sync(_ContentRuntime(), _request(), host)

    assert host.contents == ["visible runtime delta"]
    assert [type(event) for event in result.events] == [
        RuntimeContentEvent,
        RuntimeCompletedEvent,
    ]
    assert result.events.index(result.terminal) == 1


def test_host_content_uses_the_agent_stream_sanitization_funnel():
    agent = _RuntimeAgent()
    streamed = []
    agent._fire_stream_delta = streamed.append
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(host.emit_content("visible runtime content"))

    assert streamed == ["visible runtime content"]


def test_host_content_drops_cut_tool_markup_before_stream_delivery():
    agent = _RuntimeAgent()
    streamed = []
    agent._fire_stream_delta = streamed.append
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(
        host.emit_content(
            "Visible runtime prefix.\n"
            "<arg_key>session_id</arg_key>\n"
            "<arg_value>synthetic-session"
        )
    )

    assert streamed == ["Visible runtime prefix."]


def test_host_content_keeps_cut_markup_hidden_across_runtime_deltas():
    agent = _RuntimeAgent()
    streamed = []
    agent._fire_stream_delta = streamed.append
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(host.emit_content("Visible runtime prefix.\n<tool_"))
    _run_async(host.emit_content("call>hidden tool payload"))
    _run_async(host.emit_content("</tool_call>Visible tail."))

    assert streamed == ["Visible runtime prefix.", "Visible tail."]


def test_host_content_keeps_cut_argument_tail_hidden_across_runtime_deltas():
    agent = _RuntimeAgent()
    streamed = []
    agent._fire_stream_delta = streamed.append
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(host.emit_content("Visible runtime prefix.\n<arg_"))
    _run_async(host.emit_content("key>session_id</arg_key>"))
    _run_async(host.emit_content("<arg_value>synthetic-session"))

    assert streamed == ["Visible runtime prefix."]


def test_dispatch_denied_typed_approval_fails_closed_before_completion():
    host = _RuntimeRequestHost(approval=False)

    result = run_runtime_sync(_RequestEventsRuntime(), _request(), host)

    assert result.failure is not None
    assert result.failure.code == "runtime_approval_denied"
    assert result.failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False
    assert isinstance(result.terminal, RuntimeFailedEvent)
    assert not result.completed


class _RuntimeDatabase:
    def __init__(self):
        self.states = []
        self.receipts = []
        self.aggregate_receipts = []
        self.inserted = True

    def update_runtime_state(self, session_id, state):
        self.states.append((session_id, state))

    def record_runtime_usage_receipt(self, session_id, receipt):
        self.receipts.append((session_id, receipt))
        return self.inserted

    def queue_token_counts(self, session_id, **kwargs):
        self.aggregate_receipts.append((session_id, kwargs))


class _RuntimeAgent:
    valid_tool_names = frozenset()
    tools = ()
    session_id = "synthetic-session"
    _interrupt_requested = False

    def __init__(self):
        self._session_db = _RuntimeDatabase()


class _GuardedRuntimeAgent(_RuntimeAgent):
    valid_tool_names = frozenset({"synthetic_tool"})

    def __init__(self):
        super().__init__()
        self.tool_calls = []
        self.statuses = []

    def _execute_tool_calls(self, assistant_message, messages, task_id):
        tool_call = assistant_message.tool_calls[0]
        self.tool_calls.append((tool_call.function.name, task_id))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "tool completed",
            }
        )

    def _touch_activity(self, message):
        self.statuses.append(message)


class _PersistingRuntimeToolAgent(_GuardedRuntimeAgent):
    """Small fake that preserves the host's pre/post-flush ordering contract."""

    def __init__(self, *, fail_flush: bool = False):
        super().__init__()
        self.fail_flush = fail_flush
        self.flush_calls = []
        self.persisted_rows = []
        self.execution_calls = 0
        self._persisted_message_ids = set()

    def _flush_messages_to_session_db(self, messages):
        self.flush_calls.append(
            [
                (
                    message.get("role"),
                    message.get("tool_call_id"),
                    tuple(
                        call.get("id")
                        for call in message.get("tool_calls", ())
                        if isinstance(call, dict)
                    ),
                )
                for message in messages
            ]
        )
        if self.fail_flush:
            return False
        for message in messages:
            marker = id(message)
            if marker not in self._persisted_message_ids:
                self._persisted_message_ids.add(marker)
                self.persisted_rows.append(
                    (message.get("role"), message.get("tool_call_id"))
                )
        return True

    def _execute_tool_calls(self, assistant_message, messages, task_id):
        self.execution_calls += 1
        tool_call = assistant_message.tool_calls[0]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": "tool completed",
            }
        )
        assert self._flush_messages_to_session_db(messages) is True


def test_runtime_tool_call_persists_pair_before_effect_and_is_idempotent():
    agent = _PersistingRuntimeToolAgent()
    messages = [{"role": "user", "content": "synthetic request"}]
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
        turn_messages=messages,
    )

    first = _run_async(
        host.execute_tool(
            "synthetic_tool",
            {"value": "one"},
            request_id="synthetic-call-1",
        )
    )
    second = _run_async(
        host.execute_tool(
            "synthetic_tool",
            {"value": "one"},
            request_id="synthetic-call-1",
        )
    )

    assert first == second == "tool completed"
    assert agent.execution_calls == 1
    assert len(agent.flush_calls) == 2
    assert agent.flush_calls[0][-1] == (
        "assistant",
        None,
        ("synthetic-call-1",),
    )
    assert agent.flush_calls[1][-1] == ("tool", "synthetic-call-1", ())
    assert agent.persisted_rows.count(("assistant", None)) == 1
    assert agent.persisted_rows.count(("tool", "synthetic-call-1")) == 1
    assert [message.get("tool_call_id") for message in messages if message.get("role") == "tool"] == [
        "synthetic-call-1"
    ]


def test_runtime_tool_request_id_conflict_is_rejected_within_turn():
    agent = _PersistingRuntimeToolAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
        turn_messages=[],
    )

    _run_async(
        host.execute_tool(
            "synthetic_tool",
            {"value": "one"},
            request_id="synthetic-call-1",
        )
    )

    with pytest.raises(RuntimeExecutionError, match="different payload"):
        _run_async(
            host.execute_tool(
                "synthetic_tool",
                {"value": "two"},
                request_id="synthetic-call-1",
            )
        )


def test_runtime_tool_fallback_id_is_namespaced_per_turn():
    agent = _PersistingRuntimeToolAgent()
    messages = [{"role": "user", "content": "synthetic request"}]
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-turn-1",
        runtime_id="example-runtime",
        turn_messages=messages,
    )

    _run_async(host.execute_tool("synthetic_tool", {"value": "one"}))
    first_id = next(
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    )

    host.refresh_turn("synthetic-turn-2", turn_messages=messages)
    _run_async(host.execute_tool("synthetic_tool", {"value": "two"}))
    second_id = [
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    ][-1]

    assert first_id != second_id
    assert first_id.startswith("runtime-tool-")
    assert second_id.startswith("runtime-tool-")
    assert first_id.endswith("-0001")
    assert second_id.endswith("-0001")


def test_runtime_tool_fallback_id_uses_host_turn_correlation_id():
    registration = RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="example-runtime",
            plugin_version="0.1.0",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset(),
            provider_ids=frozenset({"example"}),
            api_modes=frozenset({"example_runtime"}),
            session_state_schema_version=1,
        ),
        factory=_SuccessfulRuntime,
        plugin_id="synthetic-plugin",
    )
    agent = _PersistingRuntimeToolAgent()
    messages = [{"role": "user", "content": "synthetic request"}]

    first = get_runtime_session(
        agent,
        registration,
        task_id="same-task",
        turn_messages=messages,
        correlation_id="synthetic-session:same-task:turn-1",
    )
    first_result = _run_async(
        first.host.execute_tool("synthetic_tool", {"value": "one"})
    )
    first_id = next(
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    )

    second = get_runtime_session(
        agent,
        registration,
        task_id="same-task",
        turn_messages=messages,
        correlation_id="synthetic-session:same-task:turn-2",
    )
    second_result = _run_async(
        second.host.execute_tool("synthetic_tool", {"value": "two"})
    )
    second_id = [
        message["tool_call_id"]
        for message in messages
        if message.get("role") == "tool"
    ][-1]
    close_runtime_session(agent)

    assert first_result == "tool completed"
    assert second_result == "tool completed"
    assert agent.execution_calls == 2
    assert first_id != second_id
    assert first_id.startswith("runtime-tool-")
    assert second_id.startswith("runtime-tool-")
    assert first_id.endswith("-0001")
    assert second_id.endswith("-0001")


def test_runtime_tool_fallback_id_is_deterministic_for_replayed_turn_identity():
    def run_turn(task_id):
        agent = _PersistingRuntimeToolAgent()
        messages = [{"role": "user", "content": "synthetic request"}]
        host = HermesRuntimeHostServices(
            agent,
            task_id=task_id,
            runtime_id="example-runtime",
            turn_messages=messages,
        )
        _run_async(host.execute_tool("synthetic_tool", {"value": "one"}))
        return next(
            message["tool_call_id"]
            for message in messages
            if message.get("role") == "tool"
        )

    first_id = run_turn("synthetic-turn-replay")
    replay_id = run_turn("synthetic-turn-replay")
    next_turn_id = run_turn("synthetic-turn-next")

    assert first_id == replay_id
    assert first_id != next_turn_id
    assert len(first_id) <= 256


def test_runtime_tool_request_id_is_bounded():
    agent = _PersistingRuntimeToolAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
        turn_messages=[],
    )

    with pytest.raises(RuntimeExecutionError, match="request id"):
        _run_async(
            host.execute_tool(
                "synthetic_tool",
                {"value": "one"},
                request_id="x" * 257,
            )
        )


def test_runtime_tool_persistence_failure_prevents_executor_side_effect():
    agent = _PersistingRuntimeToolAgent(fail_flush=True)
    messages = [{"role": "user", "content": "synthetic request"}]
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
        turn_messages=messages,
    )

    with pytest.raises(RuntimeToolPersistenceError, match="before execution"):
        _run_async(host.execute_tool("synthetic_tool", {"value": "one"}))

    assert agent.execution_calls == 0
    assert messages == [{"role": "user", "content": "synthetic request"}]


def _stateful_host_operations(host):
    state = RuntimeStateEnvelope(
        runtime_id="example-runtime",
        schema_version=1,
        state={"external": "synthetic"},
    )
    receipt = RuntimeUsageReceipt(
        runtime_id="example-runtime",
        provider="example",
        model="example-large",
        billing_mode="subscription_included",
        cost_status="included",
    )
    compaction = RuntimeCompactionEvent(
        phase=RuntimeCompactionPhase.STARTED,
        details={"watchdog_seconds": 60},
    )
    return (
        ("refresh", lambda: host.refresh_turn("late-task")),
        (
            "tool",
            lambda: _run_async(host.execute_tool("synthetic_tool", {"value": "one"})),
        ),
        (
            "approval",
            lambda: _run_async(
                host.request_approval("terminal", {"reason": "synthetic"})
            ),
        ),
        ("status", lambda: _run_async(host.emit_status("late status"))),
        ("state", lambda: _run_async(host.persist_state(state))),
        ("usage", lambda: _run_async(host.persist_usage(receipt))),
        ("compaction", lambda: _run_async(host.emit_compaction(compaction))),
        (
            "background",
            lambda: _run_async(
                host.emit_background_result(
                    RuntimeBackgroundResult(content="late background result")
                )
            ),
        ),
    )


def _assert_guarded_agent_has_no_late_effects(agent):
    assert agent.tool_calls == []
    assert agent.statuses == []
    assert agent._session_db.states == []
    assert agent._session_db.receipts == []
    assert agent._session_db.aggregate_receipts == []
    assert agent._runtime_compaction_events == []


def test_runtime_host_rejects_every_stateful_operation_after_close_and_rebind():
    agent = _GuardedRuntimeAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(host.close())
    agent.session_id = "different-session"

    for _operation, invoke in _stateful_host_operations(host):
        with pytest.raises(RuntimeExecutionError, match="closed"):
            invoke()

    assert host.cancellation_requested() is True
    _assert_guarded_agent_has_no_late_effects(agent)


def test_runtime_host_rejects_parent_rebind_before_any_stateful_effect():
    agent = _GuardedRuntimeAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )
    agent.session_id = "different-session"

    for _operation, invoke in _stateful_host_operations(host):
        with pytest.raises(
            RuntimeExecutionError,
            match="different Hermes session",
        ):
            invoke()

    assert host.cancellation_requested() is True
    _assert_guarded_agent_has_no_late_effects(agent)


def test_host_persists_runtime_state_and_idempotent_usage_for_selected_runtime():
    agent = _RuntimeAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )
    state = RuntimeStateEnvelope(
        runtime_id="example-runtime",
        schema_version=1,
        state={"external": "synthetic"},
    )
    receipt = RuntimeUsageReceipt(
        runtime_id="example-runtime",
        provider="example",
        model="example-large",
        billing_mode="subscription_included",
        cost_status="included",
        correlation_id="synthetic-turn",
        fallback_used=True,
        failure_phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
        selected_model="example-requested",
        effective_model="example-effective",
        canonical_model="example-large",
        model_resolution="canonicalized",
    )

    _run_async(host.persist_state(state))
    _run_async(host.persist_usage(receipt))
    assert agent._session_db.states == [("synthetic-session", state)]
    assert agent._session_db.receipts == [("synthetic-session", receipt)]
    assert agent._session_db.receipts[0][1].fallback_used is True
    assert (
        agent._session_db.receipts[0][1].failure_phase
        is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    )
    assert agent._session_db.receipts[0][1].selected_model == "example-requested"
    assert agent._session_db.receipts[0][1].effective_model == "example-effective"
    assert agent._session_db.receipts[0][1].canonical_model == "example-large"
    assert agent._session_db.receipts[0][1].model_resolution == "canonicalized"
    assert len(agent._session_db.aggregate_receipts) == 1
    assert agent._session_db.aggregate_receipts[0][1]["model"] == "example-large"

    agent._session_db.inserted = False
    _run_async(host.persist_usage(receipt))
    assert len(agent._session_db.receipts) == 2
    assert len(agent._session_db.aggregate_receipts) == 1


def test_host_rejects_state_and_usage_for_a_different_runtime():
    host = HermesRuntimeHostServices(
        _RuntimeAgent(),
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )
    wrong_state = RuntimeStateEnvelope(
        runtime_id="other-runtime",
        schema_version=1,
        state={},
    )
    wrong_receipt = RuntimeUsageReceipt(
        runtime_id="other-runtime",
        provider="example",
        model="example-large",
        billing_mode="subscription_included",
        cost_status="included",
    )

    with pytest.raises(RuntimeExecutionError, match="identity does not match"):
        _run_async(host.persist_state(wrong_state))
    with pytest.raises(RuntimeExecutionError, match="identity does not match"):
        _run_async(host.persist_usage(wrong_receipt))


class _PostTerminalRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCompletedEvent(result={"final_response": "done"})
        yield RuntimeStatusEvent(message="too late")

    async def close(self):
        self.close_calls += 1


def test_dispatch_rejects_events_after_terminal_without_closing_session_runtime():
    runtime = _PostTerminalRuntime()

    with pytest.raises(RuntimeExecutionError, match="after its terminal event"):
        run_runtime_sync(runtime, _request(), _HostServices())

    assert runtime.close_calls == 0


class _SuccessfulRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeStatusEvent(message="working")
        yield RuntimeStateEvent(
            state=RuntimeStateEnvelope(
                runtime_id="example-runtime",
                schema_version=1,
                state={"external_session": "synthetic"},
            )
        )
        yield RuntimeUsageEvent(
            receipt=RuntimeUsageReceipt(
                runtime_id="example-runtime",
                provider="example",
                model="example-large",
                billing_mode="subscription_included",
                cost_status="included",
            )
        )
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_calls += 1


def test_dispatch_keeps_runtime_open_after_success():
    runtime = _SuccessfulRuntime()
    host = _HostServices()

    result = run_runtime_sync(runtime, _request(), host)

    assert result.response == {"final_response": "done"}
    assert host.statuses == ["working"]
    assert [state.runtime_id for state in host.states] == ["example-runtime"]
    assert [receipt.billing_mode for receipt in host.receipts] == [
        "subscription_included"
    ]
    assert runtime.close_calls == 0


class _FailedRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeFailedEvent(
            failure=RuntimeFailure(
                code="synthetic_failure",
                message="synthetic failure",
                phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                replay_safe=False,
                retryable=True,
            )
        )

    async def close(self):
        self.close_calls += 1


def test_dispatch_returns_classified_failure_without_authorizing_fallback():
    runtime = _FailedRuntime()

    result = run_runtime_sync(runtime, _request(), _HostServices())

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False
    assert result.replay_safe is False
    assert isinstance(result.terminal, RuntimeFailedEvent)
    assert runtime.close_calls == 0


def test_host_tool_execution_overrides_runtime_replay_safe_claim():
    class _ToolAgent(_RuntimeAgent):
        valid_tool_names = frozenset({"synthetic_tool"})

        def _execute_tool_calls(self, assistant_message, messages, task_id):
            tool_call = assistant_message.tool_calls[0]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "tool completed",
                }
            )

    class _ReplayClaimingRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            await host.execute_tool("synthetic_tool", {"value": "one"})
            yield RuntimeFailedEvent(
                failure=RuntimeFailure(
                    code="synthetic_failure",
                    message="synthetic failure",
                    phase=RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT,
                    replay_safe=True,
                    retryable=True,
                )
            )

        async def close(self):
            return None

    host = HermesRuntimeHostServices(
        _ToolAgent(),
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    result = run_runtime_sync(_ReplayClaimingRuntime(), _request(), host)

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_SIDE_EFFECTS
    assert result.failure.replay_safe is False
    assert result.failure.retryable is True
    assert result.replay_safe is False
    assert isinstance(result.terminal, RuntimeFailedEvent)
    assert result.terminal.failure is result.failure
    assert result.events[-1] is result.terminal


def test_refresh_turn_rebuilds_allowed_tools_from_current_agent_inventory():
    class _ToolAgent(_RuntimeAgent):
        valid_tool_names = frozenset({"old_tool"})
        tools = (
            {
                "type": "function",
                "function": {"name": "old_tool"},
            },
        )

        def _execute_tool_calls(self, assistant_message, messages, task_id):
            tool_call = assistant_message.tool_calls[0]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "tool completed",
                }
            )

    agent = _ToolAgent()
    host = HermesRuntimeHostServices(
        agent,
        task_id="synthetic-turn-1",
        runtime_id="example-runtime",
    )

    agent.valid_tool_names = frozenset({"new_tool"})
    agent.tools = (
        {
            "type": "function",
            "function": {"name": "new_tool"},
        },
    )
    host.refresh_turn("synthetic-turn-2")

    _run_async(host.execute_tool("new_tool", {"value": "one"}))
    with pytest.raises(RuntimeExecutionError, match="old_tool.*not available"):
        _run_async(host.execute_tool("old_tool", {"value": "two"}))


def test_background_delivery_overrides_runtime_replay_safe_claim(monkeypatch):
    from queue import SimpleQueue

    from tools.process_registry import process_registry

    class _ReplayClaimingRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            await host.emit_background_result(
                RuntimeBackgroundResult(content="synthetic background result")
            )
            yield RuntimeFailedEvent(
                failure=RuntimeFailure(
                    code="synthetic_failure",
                    message="synthetic failure",
                    phase=RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT,
                    replay_safe=True,
                    retryable=True,
                )
            )

        async def close(self):
            return None

    queue = SimpleQueue()
    monkeypatch.setattr(process_registry, "completion_queue", queue)
    host = HermesRuntimeHostServices(
        _RuntimeAgent(),
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    result = run_runtime_sync(_ReplayClaimingRuntime(), _request(), host)

    assert queue.get_nowait()["parent_session_id"] == "synthetic-session"
    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_SIDE_EFFECTS
    assert result.failure.replay_safe is False
    assert result.failure.retryable is True
    assert result.replay_safe is False
    assert isinstance(result.terminal, RuntimeFailedEvent)
    assert result.terminal.failure is result.failure
    assert result.events[-1] is result.terminal


def test_runtime_replay_safe_claim_survives_without_host_side_effects():
    class _ReplayClaimingRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            yield RuntimeFailedEvent(
                failure=RuntimeFailure(
                    code="synthetic_failure",
                    message="synthetic failure",
                    phase=RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT,
                    replay_safe=True,
                    retryable=True,
                )
            )

        async def close(self):
            return None

    result = run_runtime_sync(_ReplayClaimingRuntime(), _request(), _HostServices())

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT
    assert result.failure.replay_safe is True
    assert result.replay_safe is True


def test_prior_turn_tool_execution_does_not_override_current_replay_safe_claim():
    class _ToolAgent(_RuntimeAgent):
        valid_tool_names = frozenset({"synthetic_tool"})

        def _execute_tool_calls(self, assistant_message, messages, task_id):
            tool_call = assistant_message.tool_calls[0]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "tool completed",
                }
            )

    class _ToolThenCompleteRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            await host.execute_tool("synthetic_tool", {"value": "one"})
            yield RuntimeCompletedEvent(result={"final_response": "done"})

        async def close(self):
            return None

    class _ReplayClaimingRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            yield RuntimeFailedEvent(
                failure=RuntimeFailure(
                    code="synthetic_failure",
                    message="synthetic failure",
                    phase=RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT,
                    replay_safe=True,
                    retryable=True,
                )
            )

        async def close(self):
            return None

    host = HermesRuntimeHostServices(
        _ToolAgent(),
        task_id="synthetic-turn-1",
        runtime_id="example-runtime",
    )
    first = run_runtime_sync(_ToolThenCompleteRuntime(), _request(), host)
    assert first.completed is True

    host.refresh_turn("synthetic-turn-2")
    second = run_runtime_sync(_ReplayClaimingRuntime(), _request(), host)

    assert second.failure is not None
    assert second.failure.phase is RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT
    assert second.failure.replay_safe is True
    assert second.replay_safe is True


class _ExplodingRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        raise RuntimeError("synthetic transport failure")
        yield  # pragma: no cover

    async def close(self):
        self.close_calls += 1


def test_unclassified_runtime_exception_is_fail_closed_without_per_turn_close():
    runtime = _ExplodingRuntime()

    result = run_runtime_sync(runtime, _request(), _HostServices())

    assert result.failure is not None
    assert result.failure.code == "runtime_exception"
    assert result.failure.phase is RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False
    assert runtime.close_calls == 0


class _ExplodingAfterStateRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeStateEvent(
            state=RuntimeStateEnvelope(
                runtime_id="example-runtime",
                schema_version=1,
                state={"external_session": "synthetic"},
            )
        )
        raise RuntimeError("synthetic failure after persistence")

    async def close(self):
        return None


def test_unclassified_exception_after_persistence_is_classified_after_side_effects():
    result = run_runtime_sync(
        _ExplodingAfterStateRuntime(),
        _request(),
        _HostServices(),
    )

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_SIDE_EFFECTS
    assert result.failure.replay_safe is False


class _ExplodingAfterStatusRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeStatusEvent(message="working")
        raise RuntimeError("synthetic failure after visible status")

    async def close(self):
        return None


def test_unclassified_exception_after_visible_status_is_not_preflight_safe():
    result = run_runtime_sync(
        _ExplodingAfterStatusRuntime(),
        _request(),
        _HostServices(),
    )

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False


@pytest.mark.parametrize(
    "visible_event",
    (
        RuntimeContentEvent(text="visible content"),
        RuntimeStatusEvent(message="visible status"),
        RuntimeToolRequestEvent(
            request_id="tool-request-visible",
            name="synthetic_tool",
            arguments={},
        ),
        RuntimeApprovalRequestEvent(
            request_id="approval-request-visible",
            action="synthetic_action",
            details={},
        ),
    ),
    ids=("content", "status", "tool", "approval"),
)
def test_visible_runtime_events_constrain_later_replay_claims(visible_event):
    class _ReplayClaimingRuntime:
        def preflight(self, request):
            return None

        async def run_turn(self, request, host) -> AsyncIterator[object]:
            yield visible_event
            yield RuntimeFailedEvent(
                failure=RuntimeFailure(
                    code="synthetic_failure",
                    message="synthetic failure",
                    phase=RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT,
                    replay_safe=True,
                    retryable=True,
                )
            )

        async def close(self):
            return None

    result = run_runtime_sync(
        _ReplayClaimingRuntime(),
        _request(),
        _RuntimeRequestHost(),
    )

    assert result.failure is not None
    assert result.failure.phase is RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT
    assert result.failure.replay_safe is False


class _CompactingRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCompactionEvent(
            phase=RuntimeCompactionPhase.STARTED,
            details={"watchdog_seconds": 30},
        )
        yield RuntimeCompactionEvent(phase=RuntimeCompactionPhase.COMPLETED)
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_calls += 1


def test_runtime_compaction_events_are_projected_and_recorded_before_completion():
    runtime = _CompactingRuntime()
    host = _HostServices()

    result = run_runtime_sync(runtime, _request(), host)

    assert [event.phase for event in host.compactions] == [
        RuntimeCompactionPhase.STARTED,
        RuntimeCompactionPhase.COMPLETED,
    ]
    assert [event.phase for event in result.events if isinstance(event, RuntimeCompactionEvent)] == [
        RuntimeCompactionPhase.STARTED,
        RuntimeCompactionPhase.COMPLETED,
    ]
    assert runtime.close_calls == 0


def test_host_owned_compaction_rejects_runtime_compaction_event():
    runtime = _CompactingRuntime()
    descriptor = RuntimeDescriptor(
        runtime_id="host-owned-runtime",
        plugin_version="0.1.0",
        runtime_api_min=1,
        runtime_api_max=1,
        required_host_capabilities=frozenset(),
        provider_ids=frozenset({"example"}),
        api_modes=frozenset({"example_runtime"}),
        session_state_schema_version=1,
        compaction_ownership=CompactionOwnership.HOST,
    )

    with pytest.raises(RuntimeExecutionError, match="host owns compaction"):
        run_runtime_sync(
            runtime,
            _request(),
            _HostServices(),
            descriptor=descriptor,
        )

    assert runtime.close_calls == 0


class _CancelledRuntime:
    def __init__(self):
        self.close_calls = 0

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCancelledEvent(reason="synthetic cancellation")

    async def close(self):
        self.close_calls += 1


def test_runtime_cancellation_is_one_terminal_outcome_without_per_turn_close():
    runtime = _CancelledRuntime()

    result = run_runtime_sync(runtime, _request(), _HostServices())

    assert result.cancelled is True
    assert isinstance(result.terminal, RuntimeCancelledEvent)
    assert sum(
        isinstance(event, (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent))
        for event in result.events
    ) == 1
    assert runtime.close_calls == 0


class _CancelledAfterTerminalRuntime:
    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield RuntimeCompletedEvent(result={"final_response": "done"})
        raise asyncio.CancelledError

    async def close(self):
        return None


def test_cancellation_after_terminal_preserves_exactly_one_terminal_event():
    result = run_runtime_sync(
        _CancelledAfterTerminalRuntime(),
        _request(),
        _HostServices(),
    )

    assert result.completed is True
    assert result.response == {"final_response": "done"}
    assert sum(
        isinstance(event, (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent))
        for event in result.events
    ) == 1


class _ExceptionAfterTerminalRuntime:
    def __init__(self, terminal):
        self.terminal = terminal

    def preflight(self, request):
        return None

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        yield self.terminal
        raise RuntimeError("synthetic generator cleanup failure")

    async def close(self):
        return None


@pytest.mark.parametrize(
    "terminal",
    (
        RuntimeCompletedEvent(result={"final_response": "done"}),
        RuntimeCancelledEvent(reason="synthetic cancellation"),
        RuntimeFailedEvent(
            failure=RuntimeFailure(
                code="synthetic_failure",
                message="synthetic failure",
                phase=RuntimeFailurePhase.AFTER_VISIBLE_OUTPUT,
                replay_safe=False,
            )
        ),
    ),
    ids=("completed", "cancelled", "failed"),
)
def test_unclassified_exception_after_terminal_preserves_exactly_one_terminal_event(
    terminal,
):
    result = run_runtime_sync(
        _ExceptionAfterTerminalRuntime(terminal),
        _request(),
        _HostServices(),
    )

    assert result.terminal is terminal
    assert sum(
        isinstance(event, (RuntimeCompletedEvent, RuntimeCancelledEvent, RuntimeFailedEvent))
        for event in result.events
    ) == 1
    if isinstance(terminal, RuntimeCompletedEvent):
        assert result.completed is True
        assert result.response == {"final_response": "done"}
    elif isinstance(terminal, RuntimeCancelledEvent):
        assert result.cancelled is True
    else:
        assert result.failure is terminal.failure


def test_background_result_is_bounded_provider_neutral_and_immutable():
    result = RuntimeBackgroundResult(
        content="synthetic background result",
        outcome=RuntimeBackgroundOutcome.COMPLETED,
    )

    assert result.content == "synthetic background result"
    assert set(result.__dataclass_fields__) == {"content", "outcome"}
    with pytest.raises(Exception):
        result.content = "late mutation"
    with pytest.raises(ValueError, match="content exceeds"):
        RuntimeBackgroundResult(content="x" * 16_385)


def test_background_delivery_capability_has_a_host_consumer():
    assert "background_delivery_v1" in runtime_api_manifest()["host_capabilities"]
    assert callable(HermesRuntimeHostServices.emit_background_result)


def test_host_queues_background_result_for_exact_bound_parent_and_rejects_after_close(
    monkeypatch,
):
    from queue import SimpleQueue

    from gateway import session_context
    from tools.process_registry import format_process_notification, process_registry

    route = {
        "HERMES_SESSION_KEY": "telegram:direct:synthetic-chat",
        "HERMES_UI_SESSION_ID": "synthetic-ui",
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_TYPE": "direct",
        "HERMES_SESSION_CHAT_ID": "synthetic-chat",
        "HERMES_SESSION_THREAD_ID": "synthetic-thread",
        "HERMES_SESSION_USER_ID": "synthetic-user",
        "HERMES_SESSION_SCOPE_ID": "synthetic-scope",
    }
    monkeypatch.setattr(
        session_context,
        "get_session_env",
        lambda name, default="": route.get(name, default),
    )
    queue = SimpleQueue()
    monkeypatch.setattr(process_registry, "completion_queue", queue)
    host = HermesRuntimeHostServices(
        _RuntimeAgent(),
        task_id="synthetic-task",
        runtime_id="example-runtime",
    )

    _run_async(
        host.emit_background_result(
            RuntimeBackgroundResult(content="background complete")
        )
    )

    event = queue.get_nowait()
    assert event["parent_session_id"] == "synthetic-session"
    assert event["session_key"] == route["HERMES_SESSION_KEY"]
    assert event["origin_ui_session_id"] == route["HERMES_UI_SESSION_ID"]
    assert event["chat_id"] == route["HERMES_SESSION_CHAT_ID"]
    assert event["summary"] == "background complete"
    assert event["type"] == "async_delegation"
    assert "background complete" in format_process_notification(event)

    _run_async(host.close())
    with pytest.raises(RuntimeExecutionError, match="closed"):
        _run_async(
            host.emit_background_result(RuntimeBackgroundResult(content="too late"))
        )


def test_runtime_and_host_binding_are_reused_until_session_close():
    instances = []

    def factory():
        runtime = _SuccessfulRuntime()
        instances.append(runtime)
        return runtime

    registration = RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="example-runtime",
            plugin_version="0.1.0",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset({"background_delivery_v1"}),
            provider_ids=frozenset({"example"}),
            api_modes=frozenset({"example_runtime"}),
            session_state_schema_version=1,
        ),
        factory=factory,
        plugin_id="synthetic-plugin",
    )
    agent = _RuntimeAgent()

    first = get_runtime_session(
        agent,
        registration,
        task_id="synthetic-turn-1",
    )
    second = get_runtime_session(
        agent,
        registration,
        task_id="synthetic-turn-2",
    )

    assert first is second
    assert first.runtime is instances[0]
    assert len(instances) == 1
    close_runtime_session(agent)
    close_runtime_session(agent)
    assert instances[0].close_calls == 1


_RUNTIME_TURN_CONTEXT = contextvars.ContextVar(
    "runtime_turn_context",
    default="missing",
)


class _LoopAffineRuntime:
    """Runtime double whose long-lived reader must stay on one event loop."""

    def __init__(self):
        self.loop_ids = []
        self.context_values = []
        self.approval_callbacks = []
        self.reader_task = None
        self.reader_wakeup = None
        self.reader_observed = threading.Event()
        self.close_loop_id = None

    def preflight(self, request):
        return None

    async def _reader(self):
        self.reader_wakeup = asyncio.Event()
        await self.reader_wakeup.wait()
        self.reader_observed.set()
        await asyncio.Event().wait()

    async def run_turn(self, request, host) -> AsyncIterator[object]:
        loop = asyncio.get_running_loop()
        self.loop_ids.append(id(loop))
        self.context_values.append(_RUNTIME_TURN_CONTEXT.get())
        from tools.terminal_tool import _get_approval_callback

        self.approval_callbacks.append(_get_approval_callback())
        if self.reader_task is None:
            self.reader_task = asyncio.create_task(self._reader())
        elif loop is not self.reader_task.get_loop() or self.reader_task.done():
            yield RuntimeFailedEvent(
                failure=RuntimeFailure(
                    code="runtime_loop_affinity_lost",
                    message="runtime loop affinity was lost",
                    phase=RuntimeFailurePhase.BEFORE_VISIBLE_OUTPUT,
                    replay_safe=False,
                )
            )
            return
        yield RuntimeCompletedEvent(result={"final_response": "done"})

    async def close(self):
        self.close_loop_id = id(asyncio.get_running_loop())
        if self.reader_task is not None:
            self.reader_task.cancel()
            await asyncio.gather(self.reader_task, return_exceptions=True)


def test_runtime_session_keeps_loop_affinity_across_async_gateway_turns():
    from tools.terminal_tool import set_approval_callback

    runtime = _LoopAffineRuntime()
    agent = _RuntimeAgent()
    registration = RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="example-runtime",
            plugin_version="0.1.0",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset(),
            provider_ids=frozenset({"example"}),
            api_modes=frozenset({"example_runtime"}),
            session_state_schema_version=1,
        ),
        factory=lambda: runtime,
        plugin_id="synthetic-plugin",
    )
    binding = get_runtime_session(agent, registration, task_id="turn-1")

    def first_approval_callback(*_args, **_kwargs):
        return True

    def second_approval_callback(*_args, **_kwargs):
        return True

    async def gateway_turns():
        try:
            _RUNTIME_TURN_CONTEXT.set("first-turn")
            set_approval_callback(first_approval_callback)
            first = binding.run_turn(_request())
            runtime.reader_task.get_loop().call_soon_threadsafe(
                runtime.reader_wakeup.set
            )
            assert runtime.reader_observed.wait(timeout=1.0)
            binding.host.refresh_turn("turn-2")
            _RUNTIME_TURN_CONTEXT.set("second-turn")
            set_approval_callback(second_approval_callback)
            second = binding.run_turn(_request())
            return first, second
        finally:
            set_approval_callback(None)

    first, second = asyncio.run(gateway_turns())
    close_runtime_session(agent)

    assert first.response == {"final_response": "done"}
    assert second.response == {"final_response": "done"}
    assert len(set(runtime.loop_ids)) == 1
    assert runtime.context_values == ["first-turn", "second-turn"]
    assert runtime.approval_callbacks == [
        first_approval_callback,
        second_approval_callback,
    ]
    assert runtime.close_loop_id == runtime.loop_ids[0]


def test_session_change_closes_old_runtime_before_rebinding():
    instances = []

    def factory():
        runtime = _SuccessfulRuntime()
        instances.append(runtime)
        return runtime

    registration = RuntimeRegistration(
        descriptor=RuntimeDescriptor(
            runtime_id="example-runtime",
            plugin_version="0.1.0",
            runtime_api_min=1,
            runtime_api_max=1,
            required_host_capabilities=frozenset({"background_delivery_v1"}),
            provider_ids=frozenset({"example"}),
            api_modes=frozenset({"example_runtime"}),
            session_state_schema_version=1,
        ),
        factory=factory,
        plugin_id="synthetic-plugin",
    )
    agent = _RuntimeAgent()
    first = get_runtime_session(agent, registration, task_id="turn-1")

    agent.session_id = "synthetic-session-2"
    second = get_runtime_session(agent, registration, task_id="turn-2")

    assert first is not second
    assert instances[0].close_calls == 1
    assert instances[1].close_calls == 0
    close_runtime_session(agent)


def test_builtin_codex_session_refreshes_its_per_turn_runner():
    agent = _RuntimeAgent()
    first = get_runtime_session(
        agent,
        make_builtin_codex_registration(lambda: {"final_response": "first"}),
        task_id="turn-1",
    )
    first_result = first.run_turn(_request())
    second = get_runtime_session(
        agent,
        make_builtin_codex_registration(lambda: {"final_response": "second"}),
        task_id="turn-2",
    )
    second_result = second.run_turn(_request())

    assert first is second
    assert first_result.response == {"final_response": "first"}
    assert second_result.response == {"final_response": "second"}
    close_runtime_session(agent)
