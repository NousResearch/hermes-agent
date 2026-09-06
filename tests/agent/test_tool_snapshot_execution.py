"""Request, handler-start, mixed-batch and durable stale-snapshot regressions."""

import copy
import json
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import run_agent
import model_tools
from run_agent import AIAgent
from agent.iteration_budget import IterationBudget
from tests.run_agent.test_run_agent import agent, _make_tool_defs, _mock_tool_call, _mock_assistant_msg, _mock_response  # noqa: F401

def _make_tool_search_agent(enabled_toolsets):
    from tools.tool_search import ToolSearchConfig

    with (
        patch(
            "tools.tool_search.load_config",
            return_value=ToolSearchConfig.from_raw({"enabled": "on"}),
        ),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            enabled_toolsets=enabled_toolsets,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


class TestExecuteToolCalls:
    def test_refreshed_context_engine_schema_matches_sequential_dispatch(
        self,
        agent,
        monkeypatch,
    ):
        registry_schema = {
            "type": "function",
            "function": {
                "name": "shared_tool",
                "description": "registry contract",
                "parameters": {
                    "type": "object",
                    "properties": {"registry_arg": {"type": "integer"}},
                    "required": ["registry_arg"],
                },
            },
        }
        context_schema = {
            "name": "shared_tool",
            "description": "context contract",
            "parameters": {
                "type": "object",
                "properties": {"context_arg": {"type": "string"}},
                "required": ["context_arg"],
            },
        }
        handler_calls = []
        agent.tools = [registry_schema]
        agent.valid_tool_names = {"shared_tool"}
        agent.enabled_toolsets = None
        agent.disabled_toolsets = None
        agent._memory_provider_tool_names = set()
        agent._context_engine_tool_names = set()
        agent.context_compressor = SimpleNamespace(
            get_tool_schemas=lambda: [context_schema],
            handle_tool_call=lambda name, args, **_kw: (
                handler_calls.append((name, args)) or "context-dispatch"
            ),
        )

        import model_tools
        from tools import mcp_tool_agent as mcp_tool

        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **_kw: [])
        mcp_tool.refresh_agent_mcp_tools(agent)

        assert agent.tools == [{"type": "function", "function": context_schema}]
        assert agent._context_engine_tool_names == {"shared_tool"}

        tool_call = _mock_tool_call(
            name="shared_tool",
            arguments='{"context_arg":"needle"}',
            call_id="context-1",
        )
        assistant_message = _mock_assistant_msg(
            content="",
            tool_calls=[tool_call],
        )
        messages = []

        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "task-1",
        )

        assert handler_calls == [
            ("shared_tool", {"context_arg": "needle"}),
        ]
        assert messages[-1]["content"] == "context-dispatch"


    @pytest.mark.parametrize(
        "transition",
        [
            "registry_to_context",
            "context_to_registry",
            "registry_to_memory",
            "memory_to_registry",
            "context_schema_update",
        ],
    )
    def test_inflight_response_cannot_cross_tool_snapshot(
        self,
        agent,
        monkeypatch,
        transition,
    ):
        old_schema = {
            "name": "shared_tool",
            "description": "old contract",
            "parameters": {
                "type": "object",
                "properties": {"old_arg": {"type": "integer"}},
                "required": ["old_arg"],
            },
        }
        new_schema = {
            "name": "shared_tool",
            "description": "new contract",
            "parameters": {
                "type": "object",
                "properties": {"new_arg": {"type": "string"}},
                "required": ["new_arg"],
            },
        }
        registry_defs = []
        context_schemas = []
        memory_schemas = []
        handler_calls = []

        agent.enabled_toolsets = None
        agent.disabled_toolsets = None
        agent._tool_snapshot_epoch = 0
        agent.context_compressor = SimpleNamespace(
            get_tool_schemas=lambda: list(context_schemas),
            handle_tool_call=lambda name, args, **_kw: (
                handler_calls.append(("context", name, args))
                or "context-dispatch"
            ),
        )
        agent._memory_manager = SimpleNamespace(
            get_all_tool_schemas=lambda: list(memory_schemas),
            handle_tool_call=lambda name, args: (
                handler_calls.append(("memory", name, args))
                or "memory-dispatch"
            ),
        )

        if transition in {"context_to_registry", "context_schema_update"}:
            agent.tools = [{"type": "function", "function": old_schema}]
            agent._context_engine_tool_names = {"shared_tool"}
            agent._memory_provider_tool_names = set()
            context_schemas.append(old_schema)
        elif transition == "memory_to_registry":
            agent.tools = [{"type": "function", "function": old_schema}]
            agent._context_engine_tool_names = set()
            agent._memory_provider_tool_names = {"shared_tool"}
            memory_schemas.append(old_schema)
        else:
            agent.tools = [{"type": "function", "function": old_schema}]
            agent._context_engine_tool_names = set()
            agent._memory_provider_tool_names = set()
            registry_defs.append(
                {"type": "function", "function": old_schema}
            )
        agent.valid_tool_names = {"shared_tool"}

        request_kwargs = agent._build_api_kwargs(
            [{"role": "user", "content": "use shared_tool"}]
        )
        request_epoch = request_kwargs._hermes_tool_snapshot_epoch
        assert request_kwargs["tools"] == [
            {"type": "function", "function": old_schema}
        ]

        registry_defs.clear()
        context_schemas.clear()
        memory_schemas.clear()
        if transition in {"context_to_registry", "memory_to_registry"}:
            registry_defs.append(
                {"type": "function", "function": new_schema}
            )
        elif transition in {"registry_to_context", "context_schema_update"}:
            context_schemas.append(new_schema)
        else:
            memory_schemas.append(new_schema)

        import model_tools
        from agent.tool_snapshot import ToolSnapshotChangedError
        from tools import mcp_tool_agent as mcp_tool

        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda **_kw: list(registry_defs),
        )
        monkeypatch.setattr(
            model_tools,
            "handle_function_call",
            lambda name, args, *_a, **_kw: (
                handler_calls.append(("registry", name, args))
                or "registry-dispatch"
            ),
        )
        mcp_tool.refresh_agent_mcp_tools(agent)

        assert agent._tool_snapshot_epoch != request_epoch
        tool_call = _mock_tool_call(
            name="shared_tool",
            arguments='{"old_arg":7}',
            call_id=f"{transition}-1",
        )
        assistant_message = _mock_assistant_msg(
            content="",
            tool_calls=[tool_call],
        )
        assistant_message._hermes_tool_snapshot_epoch = request_epoch

        with pytest.raises(ToolSnapshotChangedError):
            agent._execute_tool_calls_sequential(
                assistant_message,
                [],
                "task-1",
            )

        assert handler_calls == []


    def test_current_tool_snapshot_response_dispatches_normally(
        self,
        agent,
        monkeypatch,
    ):
        schema = {
            "name": "shared_tool",
            "description": "current contract",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        }
        handler_calls = []
        agent.tools = [{"type": "function", "function": schema}]
        agent.valid_tool_names = {"shared_tool"}
        agent._context_engine_tool_names = {"shared_tool"}
        agent._memory_provider_tool_names = set()
        agent._tool_snapshot_epoch = 4
        agent.context_compressor = SimpleNamespace(
            handle_tool_call=lambda name, args, **_kw: (
                handler_calls.append((name, args)) or "context-dispatch"
            ),
        )

        request_kwargs = agent._build_api_kwargs(
            [{"role": "user", "content": "use shared_tool"}]
        )
        tool_call = _mock_tool_call(
            name="shared_tool",
            arguments='{"value":3}',
            call_id="current-1",
        )
        assistant_message = _mock_assistant_msg(
            content="",
            tool_calls=[tool_call],
        )
        assistant_message._hermes_tool_snapshot_epoch = (
            request_kwargs._hermes_tool_snapshot_epoch
        )
        messages = []

        agent._execute_tool_calls_sequential(
            assistant_message,
            messages,
            "task-1",
        )

        assert handler_calls == [("shared_tool", {"value": 3})]
        assert messages[-1]["content"] == "context-dispatch"


    def test_sequential_refresh_during_request_middleware_rejects_new_owner(
        self,
        agent,
        monkeypatch,
    ):
        """Route publication after the loop check must precede no real handler."""
        from agent.tool_snapshot import ToolSnapshotChangedError
        from tools import mcp_tool_agent as mcp_tool

        handler_calls = []
        agent._tool_snapshot_epoch = 0
        agent._context_engine_tool_names = set()
        agent._memory_provider_tool_names = set()
        agent.context_compressor = SimpleNamespace(
            handle_tool_call=lambda name, args, **_kw: (
                handler_calls.append(("context", name, args))
                or "context-dispatch"
            ),
        )

        def _refreshing_request_middleware(*_args, **_kwargs):
            with mcp_tool._agent_tools_lock:
                agent._context_engine_tool_names = {"shared_tool"}
                agent._tool_snapshot_epoch = 1
            return SimpleNamespace(payload={"old_arg": 7}, trace=[])

        monkeypatch.setattr(
            "hermes_cli.middleware.apply_tool_request_middleware",
            _refreshing_request_middleware,
        )
        monkeypatch.setattr(
            "hermes_cli.plugins.resolve_pre_tool_block",
            lambda *_args, **_kwargs: None,
        )

        assistant_message = _mock_assistant_msg(
            content="",
            tool_calls=[
                _mock_tool_call(
                    name="shared_tool",
                    arguments='{"old_arg":7}',
                    call_id="sequential-race",
                ),
            ],
        )
        assistant_message._hermes_tool_snapshot_epoch = 0
        messages = []

        with pytest.raises(ToolSnapshotChangedError):
            agent._execute_tool_calls_sequential(
                assistant_message,
                messages,
                "task-1",
            )

        assert handler_calls == []
        assert {m["tool_call_id"] for m in messages} == {tc.id for tc in assistant_message.tool_calls}
        assert all(m["effect_disposition"] == "none" for m in messages)


    @pytest.mark.parametrize(
        "replacement_toolset",
        ["mcp-origin", "mcp-replacement"],
        ids=["tools-list-changed", "mcp-to-mcp"],
    )
    def test_request_rejects_registry_entry_replaced_before_agent_refresh(
        self,
        agent,
        monkeypatch,
        replacement_toolset,
    ):
        """A live registry replacement cannot inherit an old advertised schema."""
        from agent.tool_snapshot import ToolSnapshotChangedError
        from tools.registry import registry

        tool_name = f"mcp__snapshot_test__{uuid.uuid4().hex}"
        old_schema = {
            "name": tool_name,
            "description": "old contract",
            "parameters": {
                "type": "object",
                "properties": {"old_arg": {"type": "integer"}},
                "required": ["old_arg"],
            },
        }
        new_schema = {
            "name": tool_name,
            "description": "new contract",
            "parameters": {
                "type": "object",
                "properties": {"new_arg": {"type": "string"}},
                "required": ["new_arg"],
            },
        }
        handler_calls = []

        def _old_handler(args, **_kwargs):
            handler_calls.append(("old", args))
            return "old"

        def _new_handler(args, **_kwargs):
            handler_calls.append(("new", args))
            return "new"

        registry.register(
            name=tool_name,
            toolset="mcp-origin",
            schema=old_schema,
            handler=_old_handler,
        )
        try:
            old_entry = registry.get_entry(tool_name)
            agent.tools = [{"type": "function", "function": old_schema}]
            agent.valid_tool_names = {tool_name}
            agent._context_engine_tool_names = set()
            agent._memory_provider_tool_names = set()
            agent._tool_snapshot_generation = registry._generation
            agent._tool_snapshot_epoch = 0
            agent._tool_registry_routes = {tool_name: old_entry}

            request_kwargs = agent._build_api_kwargs(
                [{"role": "user", "content": f"use {tool_name}"}]
            )

            # Same toolset models MCP tools/list_changed. A different mcp-*
            # toolset models the registry's explicitly permitted MCP-to-MCP
            # collision replacement. Neither path publishes an agent refresh.
            registry.register(
                name=tool_name,
                toolset=replacement_toolset,
                schema=new_schema,
                handler=_new_handler,
                override=True,
            )
            assert registry.get_entry(tool_name) is not old_entry
            assert agent._tool_snapshot_epoch == 0

            assistant_message = _mock_assistant_msg(
                content="",
                tool_calls=[
                    _mock_tool_call(
                        name=tool_name,
                        arguments='{"old_arg":7}',
                        call_id=f"{replacement_toolset}-race",
                    ),
                ],
            )
            assistant_message._hermes_tool_snapshot_epoch = (
                request_kwargs._hermes_tool_snapshot_epoch
            )
            messages = []

            with pytest.raises(ToolSnapshotChangedError):
                agent._execute_tool_calls_sequential(
                    assistant_message,
                    messages,
                    "task-1",
                )

            assert handler_calls == []
            assert {m["tool_call_id"] for m in messages} == {tc.id for tc in assistant_message.tool_calls}
            assert all(m["effect_disposition"] == "none" for m in messages)
        finally:
            registry.deregister(tool_name)


    def test_sequential_dispatch_retains_entry_captured_at_handler_start(
        self,
        agent,
        monkeypatch,
    ):
        """A replacement after route capture cannot steal the advertised call."""
        from tools import mcp_tool_agent
        from tools.registry import registry

        tool_name = f"mcp__captured_route__{uuid.uuid4().hex}"
        schema = {
            "name": tool_name,
            "description": "captured contract",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        }
        handler_calls = []

        def _register(label):
            registry.register(
                name=tool_name,
                toolset="mcp-captured-route",
                schema=schema,
                handler=lambda args, **_kw: (
                    handler_calls.append((label, args)) or label
                ),
            )

        _register("captured")
        try:
            captured_entry = registry.get_entry(tool_name)
            agent.tools = [{"type": "function", "function": schema}]
            agent.valid_tool_names = {tool_name}
            agent._context_engine_tool_names = set()
            agent._memory_provider_tool_names = set()
            agent._tool_snapshot_epoch = 0
            agent._tool_registry_routes = {tool_name: captured_entry}

            original_capture = mcp_tool_agent.capture_agent_tool_execution_route
            replaced = False

            def _capture_then_replace(*args, **kwargs):
                nonlocal replaced
                entry = original_capture(*args, **kwargs)
                if not replaced:
                    replaced = True
                    _register("replacement")
                return entry

            monkeypatch.setattr(
                mcp_tool_agent,
                "capture_agent_tool_execution_route",
                _capture_then_replace,
            )
            assistant = _mock_assistant_msg(
                content="",
                tool_calls=[
                    _mock_tool_call(
                        name=tool_name,
                        arguments='{"value":7}',
                        call_id="captured-route",
                    ),
                ],
            )
            assistant._hermes_tool_snapshot_epoch = 0
            messages = []

            agent._execute_tool_calls_sequential(
                assistant,
                messages,
                "task-1",
            )

            assert registry.get_entry(tool_name) is not captured_entry
            assert handler_calls == [("captured", {"value": 7})]
            assert messages[-1]["content"] == "captured"
        finally:
            registry.deregister(tool_name)


    @pytest.mark.parametrize(
        "toolset",
        ["mcp-tool-search-init", "tool-search-init-plugin"],
        ids=["mcp", "plugin"],
    )
    def test_tool_search_init_retains_deferred_route_for_metadata_call(
        self,
        toolset,
    ):
        """Initial collapse hides schemas without dropping authorized routes."""
        from model_tools import handle_function_call
        from tools.registry import registry
        from tools.tool_search import BRIDGE_TOOL_NAMES

        tool_name = f"{toolset.replace('-', '_')}__{uuid.uuid4().hex}"
        handler_calls = []
        registry.register(
            name=tool_name,
            toolset=toolset,
            schema=_make_tool_defs(tool_name)[0]["function"],
            handler=lambda args, **_kw: (
                handler_calls.append(args) or json.dumps({"ok": True})
            ),
        )
        try:
            initialized = _make_tool_search_agent([toolset])
            assert initialized.valid_tool_names == set(BRIDGE_TOOL_NAMES)
            assert tool_name not in initialized.valid_tool_names
            request = initialized._build_api_kwargs(
                [{"role": "user", "content": "use the deferred tool"}]
            )

            result = json.loads(
                handle_function_call(
                    "tool_call",
                    {"name": tool_name, "arguments": {"value": 7}},
                    enabled_toolsets=[toolset],
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    tool_snapshot_agent=initialized,
                    expected_tool_snapshot_epoch=(
                        request._hermes_tool_snapshot_epoch
                    ),
                )
            )

            assert result == {"ok": True}
            assert handler_calls == [{"value": 7}]
        finally:
            registry.deregister(tool_name)


    @pytest.mark.parametrize(
        "toolset",
        ["mcp-tool-search-refresh", "tool-search-refresh-plugin"],
        ids=["mcp", "plugin"],
    )
    @pytest.mark.parametrize(
        "execution_mode",
        ["sequential", "concurrent"],
    )
    def test_tool_search_refresh_retains_new_deferred_route(
        self,
        toolset,
        execution_mode,
    ):
        """Refresh publishes newly authorized hidden routes with the catalog."""
        from tools import mcp_tool_agent as mcp_tool
        from tools.registry import registry
        from tools.tool_search import ToolSearchConfig

        first_name = f"{toolset.replace('-', '_')}__first_{uuid.uuid4().hex}"
        second_name = f"{toolset.replace('-', '_')}__second_{uuid.uuid4().hex}"
        calls = []

        def _register(name):
            registry.register(
                name=name,
                toolset=toolset,
                schema=_make_tool_defs(name)[0]["function"],
                handler=lambda args, **_kw: (
                    calls.append((name, args)) or json.dumps({"tool": name})
                ),
            )

        _register(first_name)
        try:
            initialized = _make_tool_search_agent([toolset])
            _register(second_name)
            with patch(
                "tools.tool_search.load_config",
                return_value=ToolSearchConfig.from_raw({"enabled": "on"}),
            ):
                mcp_tool.refresh_agent_mcp_tools(initialized)
            request = initialized._build_api_kwargs(
                [{"role": "user", "content": "use the new deferred tool"}]
            )
            assistant = _mock_assistant_msg(
                content="",
                tool_calls=[
                    _mock_tool_call(
                        name="tool_call",
                        arguments=json.dumps(
                            {"name": second_name, "arguments": {"value": 9}}
                        ),
                        call_id="deferred-refresh",
                    )
                ],
            )
            assistant._hermes_tool_snapshot_epoch = (
                request._hermes_tool_snapshot_epoch
            )
            messages = []

            execute = getattr(
                initialized,
                f"_execute_tool_calls_{execution_mode}",
            )
            execute(assistant, messages, "task-1")

            assert calls == [(second_name, {"value": 9})]
            assert second_name in messages[-1]["content"]
            assert second_name not in initialized.valid_tool_names
            assert second_name in initialized._tool_registry_routes
        finally:
            registry.deregister(first_name)
            registry.deregister(second_name)


    def test_tool_search_metadata_scope_and_live_identity_remain_enforced(self):
        """Hidden routes do not widen scope or survive live replacement."""
        from agent.tool_snapshot import ToolSnapshotChangedError
        from model_tools import handle_function_call
        from tools.registry import registry

        allowed_toolset = "mcp-tool-search-identity"
        denied_toolset = "tool-search-denied-plugin"
        allowed_name = f"mcp_tool_search_identity__{uuid.uuid4().hex}"
        denied_name = f"tool_search_denied_plugin__{uuid.uuid4().hex}"
        calls = []

        def _register(name, toolset, label):
            registry.register(
                name=name,
                toolset=toolset,
                schema=_make_tool_defs(name)[0]["function"],
                handler=lambda args, **_kw: (
                    calls.append((label, args)) or json.dumps({"label": label})
                ),
            )

        _register(allowed_name, allowed_toolset, "old")
        _register(denied_name, denied_toolset, "denied")
        try:
            initialized = _make_tool_search_agent([allowed_toolset])
            request = initialized._build_api_kwargs(
                [{"role": "user", "content": "use deferred tools"}]
            )
            metadata = {
                "tool_snapshot_agent": initialized,
                "expected_tool_snapshot_epoch": (
                    request._hermes_tool_snapshot_epoch
                ),
            }

            denied = json.loads(
                handle_function_call(
                    "tool_call",
                    {"name": denied_name, "arguments": {}},
                    enabled_toolsets=[allowed_toolset],
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    **metadata,
                )
            )
            assert "not available in this session" in denied["error"]

            _register(allowed_name, allowed_toolset, "new")
            with pytest.raises(
                ToolSnapshotChangedError,
                match="before advertised handler start",
            ):
                handle_function_call(
                    "tool_call",
                    {"name": allowed_name, "arguments": {"old": True}},
                    enabled_toolsets=[allowed_toolset],
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                    **metadata,
                )
            assert calls == []
        finally:
            registry.deregister(allowed_name)
            registry.deregister(denied_name)


class TestConcurrentToolExecution:
    def test_concurrent_workers_reject_provider_route_after_epoch_changes(
        self,
        agent,
        monkeypatch,
    ):
        """Queued workers validate after execution middleware, before handlers."""
        from tools import mcp_tool_agent as mcp_tool

        provider_calls = []
        registry_calls = []
        workers_at_dispatch = 0
        workers_lock = threading.Lock()
        all_workers_at_dispatch = threading.Event()
        refresh_complete = threading.Event()

        agent._tool_snapshot_epoch = 0
        agent._memory_provider_tool_names = {"shared_tool"}
        agent._context_engine_tool_names = set()
        agent._memory_manager = SimpleNamespace(
            handle_tool_call=lambda name, args: (
                provider_calls.append((name, args)) or "provider-dispatch"
            ),
        )

        def _barrier_execution_middleware(
            _name,
            args,
            execute,
            **_kwargs,
        ):
            nonlocal workers_at_dispatch
            with workers_lock:
                workers_at_dispatch += 1
                if workers_at_dispatch == 2:
                    all_workers_at_dispatch.set()
            assert refresh_complete.wait(5)
            return execute(args)

        def _publish_registry_owner():
            assert all_workers_at_dispatch.wait(5)
            with mcp_tool._agent_tools_lock:
                agent._memory_provider_tool_names = set()
                agent._tool_snapshot_epoch = 1
            refresh_complete.set()

        monkeypatch.setattr(
            "hermes_cli.middleware.run_tool_execution_middleware",
            _barrier_execution_middleware,
        )
        monkeypatch.setattr(
            model_tools,
            "handle_function_call",
            lambda name, args, *_a, **_kw: (
                registry_calls.append((name, args)) or "registry-dispatch"
            ),
        )
        refresher = threading.Thread(target=_publish_registry_owner)
        refresher.start()

        assistant_message = _mock_assistant_msg(
            content="",
            tool_calls=[
                _mock_tool_call(
                    name="shared_tool",
                    arguments='{"old_arg":1}',
                    call_id="concurrent-race-1",
                ),
                _mock_tool_call(
                    name="shared_tool",
                    arguments='{"old_arg":2}',
                    call_id="concurrent-race-2",
                ),
            ],
        )
        assistant_message._hermes_tool_snapshot_epoch = 0
        messages = []

        from agent.tool_snapshot import ToolSnapshotChangedError

        with pytest.raises(ToolSnapshotChangedError):
            agent._execute_tool_calls_concurrent(
                assistant_message,
                messages,
                "task-1",
            )
        refresher.join(5)

        assert not refresher.is_alive()
        assert provider_calls == []
        assert registry_calls == []
        assert {m["tool_call_id"] for m in messages} == {tc.id for tc in assistant_message.tool_calls}
        assert all(m["effect_disposition"] == "none" for m in messages)


    def test_segmented_batch_preserves_completed_result_and_rejects_remainder(
        self,
        agent,
    ):
        """A completed context call survives a context-to-provider refresh."""
        from agent.tool_executor import execute_tool_calls_segmented
        from tools import mcp_tool_agent as mcp_tool

        context_calls = []
        provider_calls = []
        agent._tool_snapshot_epoch = 0
        agent._context_engine_tool_names = {"shared_tool"}
        agent._memory_provider_tool_names = set()
        agent._memory_manager = SimpleNamespace(
            handle_tool_call=lambda name, args: (
                provider_calls.append((name, args)) or "provider-dispatch"
            ),
        )

        def _context_handler(name, args, **_kwargs):
            context_calls.append((name, args))
            with mcp_tool._agent_tools_lock:
                agent._context_engine_tool_names = set()
                agent._memory_provider_tool_names = {"shared_tool"}
                agent._tool_snapshot_epoch = 1
            return "context-complete"

        agent.context_compressor = SimpleNamespace(
            handle_tool_call=_context_handler,
        )
        calls = [
            _mock_tool_call(
                name="shared_tool",
                arguments='{"old_arg":1}',
                call_id="segment-complete",
            ),
            _mock_tool_call(
                name="shared_tool",
                arguments='{"old_arg":2}',
                call_id="segment-stale-1",
            ),
            _mock_tool_call(
                name="shared_tool",
                arguments='{"old_arg":3}',
                call_id="segment-stale-2",
            ),
        ]
        assistant_message = _mock_assistant_msg(content="", tool_calls=calls)
        assistant_message._hermes_tool_snapshot_epoch = 0
        messages = []

        execute_tool_calls_segmented(
            agent,
            assistant_message,
            messages,
            "task-1",
            segments=[
                ("sequential", [calls[0]]),
                ("sequential", calls[1:]),
            ],
        )

        assert context_calls == [("shared_tool", {"old_arg": 1})]
        assert provider_calls == []
        assert [message["tool_call_id"] for message in messages] == [
            "segment-complete",
            "segment-stale-1",
            "segment-stale-2",
        ]
        assert messages[0]["content"] == "context-complete"
        assert all(
            "tool snapshot changed" in message["content"].lower()
            for message in messages[1:]
        )



class TestRunConversation:
    def _setup_agent(self, agent):
        """Common setup for run_conversation tests."""
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.save_trajectories = False


    def test_inflight_stale_tool_response_is_retried_before_validation(
        self,
        agent,
        monkeypatch,
    ):
        self._setup_agent(agent)
        agent.max_iterations = 1
        agent.iteration_budget = IterationBudget(1)
        old_schema = {
            "name": "web_search",
            "description": "registry contract",
            "parameters": {
                "type": "object",
                "properties": {"registry_arg": {"type": "integer"}},
                "required": ["registry_arg"],
            },
        }
        new_schema = {
            "name": "web_search",
            "description": "context contract",
            "parameters": {
                "type": "object",
                "properties": {"context_arg": {"type": "string"}},
                "required": ["context_arg"],
            },
        }
        agent.tools = [{"type": "function", "function": old_schema}]
        agent.valid_tool_names = {"web_search"}
        agent._context_engine_tool_names = set()
        agent._memory_provider_tool_names = set()
        agent._tool_snapshot_epoch = 0
        monkeypatch.setattr(
            agent.context_compressor,
            "get_tool_schemas",
            lambda: [new_schema],
        )
        context_calls = []
        monkeypatch.setattr(
            agent.context_compressor,
            "handle_tool_call",
            lambda name, args, **_kw: (
                context_calls.append((name, args)) or "context-dispatch"
            ),
        )

        import model_tools
        from tools import mcp_tool_agent as mcp_tool

        monkeypatch.setattr(
            model_tools,
            "get_tool_definitions",
            lambda **_kw: [],
        )
        responses = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="web_search",
                        arguments='{"registry_arg":7}',
                        call_id="stale-1",
                    )
                ],
            ),
            _mock_response(content="Retried safely", finish_reason="stop"),
        ]

        def _respond(**kwargs):
            assert "_hermes_tool_snapshot_epoch" not in kwargs
            response = responses.pop(0)
            if response.choices[0].message.tool_calls:
                assert kwargs["tools"] == [
                    {"type": "function", "function": old_schema}
                ]
                mcp_tool.refresh_agent_mcp_tools(agent)
            return response

        agent.client.chat.completions.create.side_effect = _respond
        with (
            patch(
                "model_tools.handle_function_call",
                side_effect=AssertionError("stale registry call executed"),
            ),
            patch.object(
                agent,
                "_handle_max_iterations",
                return_value="unexpected budget fallback",
            ) as mock_max_iterations,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("use web_search")

        assert result["final_response"] == "Retried safely"
        assert agent.client.chat.completions.create.call_count == 2
        assert context_calls == []
        assert responses == []
        assert result["api_calls"] == 1
        assert agent._api_call_count == 1
        assert agent.iteration_budget.used == 1
        mock_max_iterations.assert_not_called()


    def test_inflight_stale_tool_response_cap_stays_charged(
        self,
        agent,
    ):
        self._setup_agent(agent)
        from tools import mcp_tool_agent as mcp_tool

        agent.max_iterations = 1
        agent.iteration_budget = IterationBudget(1)
        agent._tool_snapshot_epoch = 0
        responses = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="web_search",
                        arguments="{}",
                        call_id=f"stale-{attempt}",
                    )
                ],
            )
            for attempt in range(1, 4)
        ]

        def _respond(**_kwargs):
            response = responses.pop(0)
            with mcp_tool._agent_tools_lock:
                agent._tool_snapshot_epoch += 1
            return response

        agent.client.chat.completions.create.side_effect = _respond
        with (
            patch(
                "model_tools.handle_function_call",
                side_effect=AssertionError("stale tool call executed"),
            ),
            patch.object(
                agent,
                "_handle_max_iterations",
                return_value="unexpected budget fallback",
            ) as mock_max_iterations,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("use web_search")

        assert agent.client.chat.completions.create.call_count == 3
        assert responses == []
        assert result["failed"] is True
        assert result["completed"] is False
        assert result["turn_exit_reason"] == "stale_tool_snapshot_exhausted"
        assert result["api_calls"] == 1
        assert agent._api_call_count == 1
        assert agent.iteration_budget.used == 1
        mock_max_iterations.assert_not_called()


    def test_late_stale_executor_retries_with_persisted_pairing(
        self,
        agent,
    ):
        self._setup_agent(agent)
        from tools import mcp_tool_agent as mcp_tool

        agent.max_iterations = 1
        agent.iteration_budget = IterationBudget(1)
        agent._tool_snapshot_epoch = 0
        responses = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="web_search",
                        arguments=json.dumps({"query": suffix}),
                        call_id=call_id,
                    )
                    for suffix, call_id in (
                        ("a", "late-stale-a"),
                        ("b", "late-stale-b"),
                    )
                ],
            ),
            _mock_response(content="Recovered after late stale", finish_reason="stop"),
        ]
        persisted_snapshots = []

        def _respond(**_kwargs):
            return responses.pop(0)

        def _flush(current_messages, *_args, **_kwargs):
            persisted_snapshots.append(copy.deepcopy(current_messages))
            last = current_messages[-1] if current_messages else None
            if (
                isinstance(last, dict)
                and last.get("role") == "assistant"
                and last.get("tool_calls")
            ):
                with mcp_tool._agent_tools_lock:
                    agent._tool_snapshot_epoch += 1

        agent.client.chat.completions.create.side_effect = _respond
        with (
            patch(
                "model_tools.handle_function_call",
                side_effect=AssertionError("late-stale handler executed"),
            ),
            patch.object(
                agent,
                "_flush_messages_to_session_db",
                side_effect=_flush,
            ),
            patch.object(
                agent,
                "_handle_max_iterations",
                return_value="unexpected budget fallback",
            ) as mock_max_iterations,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("use web_search")

        assert result["final_response"] == "Recovered after late stale"
        assert result["completed"] is True
        assert result["api_calls"] == 1
        assert agent._api_call_count == 1
        assert agent.iteration_budget.used == 1
        assert agent.client.chat.completions.create.call_count == 2
        assert responses == []
        stale_results = [
            message
            for message in result["messages"]
            if message.get("role") == "tool"
            and message.get("tool_call_id") in {"late-stale-a", "late-stale-b"}
        ]
        assert [message["tool_call_id"] for message in stale_results] == [
            "late-stale-a",
            "late-stale-b",
        ]
        assert all(
            message["effect_disposition"] == "none"
            and '"error_type": "tool_snapshot_changed"' in message["content"]
            for message in stale_results
        )
        assert any(
            any(
                message.get("role") == "assistant"
                and {
                    call.get("id")
                    for call in message.get("tool_calls", [])
                }
                >= {"late-stale-a", "late-stale-b"}
                for message in snapshot
            )
            and {
                message.get("tool_call_id")
                for message in snapshot
                if message.get("role") == "tool"
            }
            >= {"late-stale-a", "late-stale-b"}
            for snapshot in persisted_snapshots
        )
        mock_max_iterations.assert_not_called()


    def test_late_stale_registry_route_refreshes_before_retry(
        self,
        agent,
    ):
        """Retry must publish the replacement route instead of resending stale tools."""
        self._setup_agent(agent)
        from tools.registry import registry
        from tools.tool_search import ToolSearchConfig

        tool_name = f"stale_retry__{uuid.uuid4().hex}"
        old_schema = {
            "name": tool_name,
            "description": "old contract",
            "parameters": {
                "type": "object",
                "properties": {"old_arg": {"type": "integer"}},
                "required": ["old_arg"],
            },
        }
        new_schema = {
            "name": tool_name,
            "description": "new contract",
            "parameters": {
                "type": "object",
                "properties": {"new_arg": {"type": "string"}},
                "required": ["new_arg"],
            },
        }
        handler_calls = []

        registry.register(
            name=tool_name,
            toolset="web",
            schema=old_schema,
            handler=lambda args, **_kw: (
                handler_calls.append(("old", args)) or "old"
            ),
        )
        try:
            old_entry = registry.get_entry(tool_name)
            agent.max_iterations = 1
            agent.iteration_budget = IterationBudget(1)
            agent.tools = [{"type": "function", "function": old_schema}]
            agent.valid_tool_names = {tool_name}
            agent.enabled_toolsets = ["web"]
            agent.disabled_toolsets = None
            agent._context_engine_tool_names = set()
            agent._memory_provider_tool_names = set()
            agent._tool_snapshot_generation = registry._generation
            agent._tool_snapshot_epoch = 0
            agent._tool_registry_routes = {tool_name: old_entry}
            request_schemas = []

            def _respond(**kwargs):
                schema = next(
                    tool["function"]
                    for tool in kwargs["tools"]
                    if tool["function"]["name"] == tool_name
                )
                request_schemas.append(schema)
                if schema == new_schema:
                    return _mock_response(
                        content="Recovered with replacement",
                        finish_reason="stop",
                    )
                attempt = len(request_schemas)
                return _mock_response(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        _mock_tool_call(
                            name=tool_name,
                            arguments='{"old_arg":7}',
                            call_id=f"stale-route-{attempt}",
                        ),
                    ],
                )

            replaced = False

            def _flush(current_messages, *_args, **_kwargs):
                nonlocal replaced
                last = current_messages[-1] if current_messages else None
                if (
                    not replaced
                    and isinstance(last, dict)
                    and last.get("role") == "assistant"
                    and last.get("tool_calls")
                ):
                    replaced = True
                    registry.register(
                        name=tool_name,
                        toolset="web",
                        schema=new_schema,
                        handler=lambda args, **_kw: (
                            handler_calls.append(("new", args)) or "new"
                        ),
                    )

            agent.client.chat.completions.create.side_effect = _respond
            with (
                patch.object(
                    agent,
                    "_flush_messages_to_session_db",
                    side_effect=_flush,
                ),
                patch.object(
                    agent,
                    "_handle_max_iterations",
                    return_value="unexpected budget fallback",
                ) as mock_max_iterations,
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
                patch(
                    "tools.tool_search.load_config",
                    return_value=ToolSearchConfig.from_raw({"enabled": "off"}),
                ),
            ):
                result = agent.run_conversation(f"use {tool_name}")

            assert result["final_response"] == "Recovered with replacement"
            assert result["completed"] is True
            assert agent.client.chat.completions.create.call_count == 2
            assert request_schemas == [old_schema, new_schema]
            assert handler_calls == []
            mock_max_iterations.assert_not_called()
        finally:
            registry.deregister(tool_name)


    def test_late_stale_refresh_failure_does_not_resend_snapshot(
        self,
        agent,
    ):
        """A failed coherent refresh terminates without another provider call."""
        self._setup_agent(agent)
        from agent.tool_snapshot import ToolSnapshotChangedError

        agent.max_iterations = 1
        agent.iteration_budget = IterationBudget(1)
        agent._tool_snapshot_epoch = 0
        response = _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call(
                    name="web_search",
                    arguments="{}",
                    call_id="refresh-failed",
                ),
            ],
        )
        agent.client.chat.completions.create.return_value = response

        with (
            patch.object(
                agent,
                "_execute_tool_calls",
                side_effect=ToolSnapshotChangedError(
                    "registry route replaced before handler start"
                ),
            ),
            patch(
                "tools.mcp_tool_agent.refresh_agent_mcp_tools",
                side_effect=RuntimeError("coherent refresh unavailable"),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("use web_search")

        assert result["failed"] is True
        assert result["completed"] is False
        assert result["turn_exit_reason"] == "tool_snapshot_refresh_failed"
        assert "could not be refreshed safely" in result["final_response"]
        assert agent.client.chat.completions.create.call_count == 1
        stale_results = [
            message
            for message in result["messages"]
            if message.get("role") == "tool"
            and message.get("tool_call_id") == "refresh-failed"
        ]
        assert len(stale_results) == 1
        assert (
            '"error_type": "tool_snapshot_changed"'
            in stale_results[0]["content"]
        )
        assert stale_results[0]["effect_disposition"] == "none"


    def test_all_stale_parallel_workers_retry_before_effects(
        self,
        agent,
    ):
        """An all-stale parallel batch reaches coherent bounded recovery."""
        self._setup_agent(agent)
        from agent.tool_snapshot import ToolSnapshotChangedError

        agent.max_iterations = 1
        agent.iteration_budget = IterationBudget(1)
        responses = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="web_search",
                        arguments=json.dumps({"query": query}),
                        call_id=call_id,
                    )
                    for query, call_id in (
                        ("first", "parallel-stale-1"),
                        ("second", "parallel-stale-2"),
                    )
                ],
            ),
            _mock_response(
                content="Recovered after parallel stale",
                finish_reason="stop",
            ),
        ]
        agent.client.chat.completions.create.side_effect = responses

        with (
            patch.object(
                agent,
                "_invoke_tool",
                side_effect=ToolSnapshotChangedError(
                    "parallel route changed before handler start"
                ),
            ) as invoke_tool,
            patch(
                "tools.mcp_tool_agent.refresh_agent_mcp_tools",
                return_value=set(),
            ) as refresh_snapshot,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search twice")

        assert result["final_response"] == "Recovered after parallel stale"
        assert result["completed"] is True
        assert result["api_calls"] == 1
        assert agent.client.chat.completions.create.call_count == 2
        assert invoke_tool.call_count == 2
        refresh_snapshot.assert_called_once_with(agent, quiet_mode=True, preserve_prefix=True)
        stale_results = [
            message
            for message in result["messages"]
            if message.get("tool_call_id", "").startswith("parallel-stale-")
        ]
        assert [message["tool_call_id"] for message in stale_results] == [
            "parallel-stale-1",
            "parallel-stale-2",
        ]
        assert all(
            message["effect_disposition"] == "none"
            and '"error_type": "tool_snapshot_changed"' in message["content"]
            for message in stale_results
        )


    def test_mixed_parallel_stale_preserves_completed_effect_without_retry(
        self,
        agent,
    ):
        """A completed sibling is never replayed when another worker is stale."""
        self._setup_agent(agent)
        from agent.tool_snapshot import ToolSnapshotChangedError

        agent.max_iterations = 2
        agent.iteration_budget = IterationBudget(2)
        responses = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="web_search",
                        arguments=json.dumps({"query": query}),
                        call_id=call_id,
                    )
                    for query, call_id in (
                        ("completed", "parallel-completed"),
                        ("stale", "parallel-mixed-stale"),
                    )
                ],
            ),
            _mock_response(
                content="Continued without replay",
                finish_reason="stop",
            ),
        ]
        agent.client.chat.completions.create.side_effect = responses
        completed_calls = []

        def _invoke(function_name, function_args, *_args, **_kwargs):
            if function_args["query"] == "completed":
                completed_calls.append(function_name)
                return "completed-result"
            raise ToolSnapshotChangedError(
                "parallel route changed before handler start"
            )

        with (
            patch.object(
                agent,
                "_invoke_tool",
                side_effect=_invoke,
            ) as invoke_tool,
            patch(
                "tools.mcp_tool_agent.refresh_agent_mcp_tools",
                return_value=set(),
            ) as refresh_snapshot,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search with mixed outcomes")

        assert result["final_response"] == "Continued without replay"
        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert agent.iteration_budget.used == 2
        assert agent.client.chat.completions.create.call_count == 2
        assert invoke_tool.call_count == 2
        assert completed_calls == ["web_search"]
        refresh_snapshot.assert_called_once_with(agent, quiet_mode=True, preserve_prefix=True)
        by_id = {
            message.get("tool_call_id"): message
            for message in result["messages"]
            if message.get("role") == "tool"
        }
        assert by_id["parallel-completed"]["content"] == "completed-result"
        stale = by_id["parallel-mixed-stale"]
        assert stale["effect_disposition"] == "none"
        assert '"error_type": "tool_snapshot_changed"' in stale["content"]


    def test_late_stale_executor_exhaustion_is_paired_and_charged(
        self,
        agent,
    ):
        self._setup_agent(agent)
        from tools import mcp_tool_agent as mcp_tool

        agent.max_iterations = 1
        agent.iteration_budget = IterationBudget(1)
        agent._tool_snapshot_epoch = 0
        responses = [
            _mock_response(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    _mock_tool_call(
                        name="web_search",
                        arguments="{}",
                        call_id=f"late-stale-{attempt}",
                    )
                ],
            )
            for attempt in range(1, 4)
        ]
        persisted_snapshots = []

        def _respond(**_kwargs):
            return responses.pop(0)

        def _flush(current_messages, *_args, **_kwargs):
            persisted_snapshots.append(copy.deepcopy(current_messages))
            last = current_messages[-1] if current_messages else None
            if (
                isinstance(last, dict)
                and last.get("role") == "assistant"
                and last.get("tool_calls")
            ):
                with mcp_tool._agent_tools_lock:
                    agent._tool_snapshot_epoch += 1

        agent.client.chat.completions.create.side_effect = _respond
        with (
            patch(
                "model_tools.handle_function_call",
                side_effect=AssertionError("late-stale handler executed"),
            ),
            patch.object(
                agent,
                "_flush_messages_to_session_db",
                side_effect=_flush,
            ),
            patch.object(
                agent,
                "_handle_max_iterations",
                return_value="unexpected budget fallback",
            ) as mock_max_iterations,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("use web_search")

        assert result["failed"] is True
        assert result["completed"] is False
        assert result["turn_exit_reason"] == "stale_tool_snapshot_exhausted"
        assert result["api_calls"] == 1
        assert agent._api_call_count == 1
        assert agent.iteration_budget.used == 1
        assert agent.client.chat.completions.create.call_count == 3
        assert responses == []
        stale_results = [
            message
            for message in result["messages"]
            if message.get("role") == "tool"
            and message.get("tool_call_id", "").startswith("late-stale-")
        ]
        assert [message["tool_call_id"] for message in stale_results] == [
            "late-stale-1",
            "late-stale-2",
            "late-stale-3",
        ]
        assert all(
            message["effect_disposition"] == "none"
            and '"error_type": "tool_snapshot_changed"' in message["content"]
            for message in stale_results
        )
        final_persisted = persisted_snapshots[-1]
        for attempt in range(1, 4):
            call_id = f"late-stale-{attempt}"
            assert any(
                message.get("role") == "assistant"
                and any(
                    call.get("id") == call_id
                    for call in message.get("tool_calls", [])
                )
                for message in final_persisted
            )
            assert any(
                message.get("role") == "tool"
                and message.get("tool_call_id") == call_id
                for message in final_persisted
            )
        mock_max_iterations.assert_not_called()


    def test_late_stale_retry_counter_resets_after_successful_execution(
        self,
        agent,
    ):
        self._setup_agent(agent)
        from tools import mcp_tool_agent as mcp_tool

        agent.max_iterations = 2
        agent.iteration_budget = IterationBudget(2)
        agent._tool_snapshot_epoch = 0
        stale_ids = {"before-1", "before-2", "after-1", "after-2"}
        responses = [
            *[
                _mock_response(
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        _mock_tool_call(
                            name="web_search",
                            arguments="{}",
                            call_id=call_id,
                        )
                    ],
                )
                for call_id in ("before-1", "before-2", "success", "after-1", "after-2")
            ],
            _mock_response(content="Completed after reset", finish_reason="stop"),
        ]
        handler_calls = []

        def _respond(**_kwargs):
            return responses.pop(0)

        def _flush(current_messages, *_args, **_kwargs):
            last = current_messages[-1] if current_messages else None
            if not (
                isinstance(last, dict)
                and last.get("role") == "assistant"
                and last.get("tool_calls")
            ):
                return
            call_id = last["tool_calls"][0]["id"]
            if call_id in stale_ids:
                with mcp_tool._agent_tools_lock:
                    agent._tool_snapshot_epoch += 1

        agent.client.chat.completions.create.side_effect = _respond
        with (
            patch(
                "model_tools.handle_function_call",
                side_effect=lambda *_args, **_kwargs: (
                    handler_calls.append("success") or "executed"
                ),
            ),
            patch.object(
                agent,
                "_flush_messages_to_session_db",
                side_effect=_flush,
            ),
            patch.object(
                agent,
                "_handle_max_iterations",
                return_value="unexpected budget fallback",
            ) as mock_max_iterations,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("use web_search")

        assert result["final_response"] == "Completed after reset"
        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert agent._api_call_count == 2
        assert agent.iteration_budget.used == 2
        assert agent.client.chat.completions.create.call_count == 6
        assert handler_calls == ["success"]
        assert responses == []
        mock_max_iterations.assert_not_called()


    def test_tool_call_none_args_verbose_logging_does_not_crash(self, agent):
        self._setup_agent(agent)
        agent.verbose_logging = True
        tc = _mock_tool_call(name="web_search", arguments=None, call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        with (
            patch("model_tools.handle_function_call", return_value="search result") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")

        assert result["final_response"] == "Done searching"
        assert mock_handle_function_call.call_args.args[:2] == ("web_search", {})


    def _run_wrapped_429_output_cap(self, agent, *, fallback_chain):
        self._setup_agent(agent)
        agent.api_mode = "chat_completions"
        agent.provider = "custom"
        agent.base_url = "http://192.168.1.254:20128/v1"
        agent.model = "deepseekv4flash"
        agent.max_tokens = 98_304
        agent.compression_enabled = True
        agent._fallback_chain = fallback_chain
        agent._fallback_index = 0
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.should_compress = MagicMock(return_value=False)

        error_msg = (
            "Error code: 429 - {'error': {'message': \"[400]: max_tokens "
            "(98304) exceeds model's maximum output tokens (65536) for model "
            "deepseek-v4-flash (ref: 37bde60f-44e7-44e2-b995-4af17fba6d6b)\", "
            "'type': 'rate_limit_error', 'code': 'rate_limit_exceeded'}}"
        )
        exc = Exception(error_msg)
        exc.status_code = 429
        exc.code = "rate_limit_exceeded"

        ok_resp = _mock_response(content="done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [exc, ok_resp]

        mock_compress = MagicMock(return_value=(
            [{"role": "user", "content": "hello"}],
            "You are helpful.",
        ))
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent.context_compressor, "update_model"),
            patch.object(agent, "_compress_context", mock_compress),
        ):
            result = agent.run_conversation("hello")

        assert len(agent.client.chat.completions.create.call_args_list) == 2
        second_call = agent.client.chat.completions.create.call_args_list[1].kwargs
        assert result["completed"] is True
        assert second_call["max_tokens"] <= 65_472
        assert agent.context_compressor.context_length == 200_000
        # The clamp, not provider failover, must have recovered: no fallback
        # slot consumed and the model unchanged.
        assert agent._fallback_index == 0
        assert agent.model == "deepseekv4flash"
