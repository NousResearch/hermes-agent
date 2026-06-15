"""E2E integration tests for the DeepAgents runtime path.

Tests the full call chain from the agent facade through DeepAgents components
with minimal mocking — just enough to make internal function calls succeed.

Test categories:
  - TestRunConversation: DeepAgentsAIAgent.run_conversation end-to-end
  - TestStreamingEndToEnd: Full streaming event routing
  - TestToolDispatcherEndToEnd: Tool execution integration
  - TestMessageRoundTrip: Hermes ↔ LangChain message conversion
  - TestDeepAgentsAIAgentIntegration: Full facade with agent attributes
"""

import json
import os
import sys
import types
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Reusable named tuple for ToolResult used in tests
ToolResult = namedtuple("ToolResult", ["value", "image_url", "is_error"])


def _ensure_deepagents_mocks():
    """Register minimal deepagents SDK mocks in sys.modules so the
    deep_agents_runtime module can import its SDK constants."""
    if "deepagents" not in sys.modules:
        mock_mod = MagicMock()
        mock_mod.ToolResult = ToolResult
        mock_mod.Thread = MagicMock()
        mock_mod.Runner = MagicMock()
        sys.modules["deepagents"] = mock_mod

    if "deepagents.runtime" not in sys.modules:
        rt_mod = types.ModuleType("deepagents.runtime")
        rt_mod.ToolResult = ToolResult
        rt_mod.Thread = MagicMock()
        rt_mod.Runner = MagicMock()
        sys.modules["deepagents.runtime"] = rt_mod


def _mock_langchain_for_tests():
    """Ensure langchain_core messages are importable for tests that
    use them directly (convert tests)."""
    try:
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        return {
            "AIMessage": AIMessage,
            "HumanMessage": HumanMessage,
            "SystemMessage": SystemMessage,
            "ToolMessage": ToolMessage,
        }
    except ImportError:
        return None


def _make_langchain_message(msg_type, **kwargs):
    """Create a LangChain message of the given type."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    content = kwargs.get("content", "")
    additional_kwargs = kwargs.get("additional_kwargs", {})
    if msg_type == "system":
        return SystemMessage(content=content)
    elif msg_type == "user":
        return HumanMessage(content=content)
    elif msg_type == "assistant":
        msg = AIMessage(content=content)
        if kwargs.get("tool_calls"):
            msg.tool_calls = kwargs["tool_calls"]
        if kwargs.get("reasoning"):
            msg.reasoning = kwargs["reasoning"]
        if kwargs.get("additional_kwargs"):
            msg.additional_kwargs = kwargs["additional_kwargs"]
        return msg
    elif msg_type == "tool":
        return ToolMessage(
            content=content, tool_call_id=kwargs.get("tool_call_id", "call_1")
        )
    else:
        raise ValueError(f"Unknown msg_type: {msg_type}")


def _make_hermes_message(role, content="", tool_calls=None, tool_call_id=None):
    """Create a Hermes-format message dict."""
    msg = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


# ---------------------------------------------------------------------------
# Autouse fixture: ensure deepagents mocks for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_deepagents_modules():
    """Ensure a clean deepagents mock before each test."""
    _ensure_deepagents_mocks()
    # Remove deep_agents_runtime from cache so functions re-evaluate imports
    for mod_name in list(sys.modules.keys()):
        if mod_name == "agent.deep_agents_runtime":
            del sys.modules[mod_name]
        elif mod_name.startswith("agent.deep_agents_"):
            del sys.modules[mod_name]
    yield
    # Cleanup: remove our mocks so other tests aren't affected
    sys.modules.pop("deepagents", None)
    sys.modules.pop("deepagents.runtime", None)


# ---------------------------------------------------------------------------
# TestRunConversation — DeepAgentsAIAgent facade integration
# ---------------------------------------------------------------------------


class TestRunConversation:
    """E2E: DeepAgentsAIAgent.run_conversation calls through the full stack."""

    def _make_sync_agent(self, **overrides):
        """Build an agent configured to take the sync (non-streaming) path."""
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        defaults = dict(
            mode="deepagents",
            _quiet_mode=False,
            _skip_memory=True,
            _platform=None,
            _session_id="test-session",
            _max_iterations=90,
            provider="",
            _api_key=None,
            _base_url=None,
            _callbacks={},
            _agent=MagicMock(),
        )
        defaults.update(overrides)
        agent = object.__new__(DeepAgentsAIAgent)
        for k, v in defaults.items():
            object.__setattr__(agent, k, v)
        return agent

    def test_run_conversation_sync_path(self):
        """Sync run_conversation calls self._agent.invoke and returns parsed result."""
        agent = self._make_sync_agent(_max_iterations=30)
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="Hello!")],
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            result = agent.run_conversation(user_message="Hi")

        assert result["final_response"] == "Hello!"
        assert result["completed"] is True
        agent._agent.invoke.assert_called_once()
        call_kwargs = agent._agent.invoke.call_args.kwargs
        assert call_kwargs["config"]["recursion_limit"] == 30

    def test_run_conversation_sync_error_handling(self):
        """Errors during agent.invoke are caught and returned as error dict."""
        agent = self._make_sync_agent()
        agent._agent.invoke.side_effect = RuntimeError("model unavailable")

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            result = agent.run_conversation(user_message="test")

        assert result["failed"] is True
        assert result["completed"] is False
        assert "model unavailable" in result["final_response"]

    def test_run_conversation_with_conversation_history(self):
        """run_conversation appends history messages to the user message."""
        agent = self._make_sync_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="continuing...")],
        }

        history = [
            _make_hermes_message("system", "You are helpful."),
            _make_hermes_message("user", "first question"),
            _make_hermes_message("assistant", "first answer"),
        ]

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(
                user_message="second question",
                conversation_history=history,
            )

        call_args = agent._agent.invoke.call_args
        messages_arg = call_args[0][0]["messages"]  # first positional arg is state dict
        assert len(messages_arg) == 4  # 3 history + 1 new user message

    def test_run_conversation_system_message_prepended(self):
        """System message is prepended to conversation history."""
        agent = self._make_sync_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="ok")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(
                user_message="hello",
                system_message="You are a cat.",
                conversation_history=[],
            )

        call_args = agent._agent.invoke.call_args
        messages_arg = call_args[0][0]["messages"]
        from langchain_core.messages import SystemMessage, HumanMessage

        assert isinstance(messages_arg[0], SystemMessage)
        assert messages_arg[0].content == "You are a cat."
        assert isinstance(messages_arg[1], HumanMessage)
        assert messages_arg[1].content == "hello"

    def test_run_conversation_chat_returns_final_response(self):
        """chat() is a thin wrapper that returns final_response string."""
        agent = self._make_sync_agent()
        agent._agent.invoke.return_value = {
            "messages": [
                _make_langchain_message("assistant", content="the answer is 42")
            ]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            response = agent.chat("What is 6 * 7?")

        assert response == "the answer is 42"

    def test_run_conversation_passes_task_id_in_config(self):
        """task_id ends up in the config's 'configurable' section."""
        agent = self._make_sync_agent(
            _session_id="main-sess",
            provider="",
            _api_key=None,
            _base_url=None,
            _max_iterations=30,
        )
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="ok")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(
                user_message="test",
                task_id="unique-task-id-123",
            )

        cfg = agent._agent.invoke.call_args.kwargs["config"]
        assert cfg["configurable"]["task_id"] == "unique-task-id-123"
        assert cfg["configurable"]["session_id"] == "main-sess"

    def test_run_conversation_session_id_fallback(self):
        """When session_id is empty, it defaults to 'default'."""
        agent = self._make_sync_agent()
        object.__setattr__(agent, "_session_id", "")
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="ok")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            agent.run_conversation(user_message="hi")

        cfg = agent._agent.invoke.call_args.kwargs["config"]
        assert cfg["configurable"]["session_id"] == "default"


# ---------------------------------------------------------------------------
# TestStreamingEndToEnd — streaming event routing
# ---------------------------------------------------------------------------


class TestStreamingEndToEnd:
    """E2E: streaming path — bridge processes events and callbacks fire."""

    def _make_deepagents_agent(self, **kwargs):
        """Create a DeepAgentsAIAgent via __new__ with a mock _agent."""
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = object.__new__(DeepAgentsAIAgent)
        object.__setattr__(agent, "mode", "deepagents")
        object.__setattr__(agent, "_quiet_mode", kwargs.get("quiet_mode", False))
        object.__setattr__(agent, "_skip_memory", kwargs.get("skip_memory", True))
        object.__setattr__(agent, "_platform", kwargs.get("platform"))
        object.__setattr__(agent, "_session_id", kwargs.get("session_id", "test"))
        object.__setattr__(agent, "_max_iterations", kwargs.get("max_iterations", 90))
        object.__setattr__(agent, "provider", kwargs.get("provider", ""))
        object.__setattr__(agent, "_api_key", kwargs.get("api_key"))
        object.__setattr__(agent, "_base_url", kwargs.get("base_url"))
        object.__setattr__(agent, "_callbacks", kwargs.get("_callbacks", {}))
        return agent

    def test_streaming_path_fires_callbacks(self):
        """When callbacks are set, run_conversation enters _run_streamed."""
        deltas = []
        steps = []

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()

        def fake_stream(**kwargs):
            yield {"events": [{"type": "AIMessageChunk", "data": {"content": "hi"}}]}
            yield {"output": "final answer"}
            return []

        agent._agent.stream.return_value = fake_stream()
        agent._agent.get_state.return_value = type(
            "State",
            (),
            {
                "values": {
                    "messages": [
                        _make_langchain_message("assistant", content="final answer")
                    ]
                }
            },
        )

        # Set up callbacks via the _CAPTURED_NAMES mechanism
        agent.stream_delta_callback = lambda x: deltas.append(x)
        agent.step_callback = lambda n, s: steps.append((n, s))

        response = agent.run_conversation(user_message="hello")

        assert response["final_response"] == "final answer"
        assert "hi" in deltas
        assert (1, []) in steps  # step callback fires on AIMessageChunk

    def test_streaming_path_no_callbacks_runs_sync(self):
        """Without callbacks, run_conversation degrades to sync path."""
        agent = self._make_deepagents_agent(_callbacks={})
        agent._agent = MagicMock()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="answer")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            response = agent.run_conversation(user_message="hello")

        assert response["final_response"] == "answer"
        agent._agent.invoke.assert_called_once()
        agent._agent.stream.assert_not_called()

    def test_streaming_bridge_event_routing(self):
        """Bridge routes AIMessageChunk to stream_delta, ToolCall to tool_progress."""
        from agent.deep_agents_runtime import _HermesStreamingBridge

        deltas = []
        tools = []

        bridge = _HermesStreamingBridge(
            agent=MagicMock(),
            stream_delta=lambda x: deltas.append(x),
            tool_progress=lambda a, **kw: tools.append((a, kw)),
        )

        bridge.process_event({
            "events": [
                {"type": "AIMessageChunk", "data": {"content": "stream "}},
                {
                    "type": "ToolCall",
                    "data": {"name": "read_file", "args": {"path": "x.txt"}},
                },
            ]
        })

        assert "stream " in deltas
        assert len(tools) == 1
        assert tools[0][0] == "tool.started"
        assert tools[0][1]["tool_name"] == "read_file"

    def test_streaming_bridge_output_event_yields_final_response(self):
        """Bridge processes output events and extracts final_response text."""
        from agent.deep_agents_runtime import _HermesStreamingBridge

        deltas = []
        bridge = _HermesStreamingBridge(
            agent=MagicMock(),
            stream_delta=lambda x: deltas.append(x),
        )

        bridge.process_event({"output": {"final_response": "done!", "api_calls": 2}})
        bridge.process_event({"output": "plain text fallback"})

        assert "done!" in deltas
        assert "plain text fallback" in deltas

    def test_streaming_bridge_handles_non_dicts_gracefully(self):
        """Non-dict events in the stream don't crash the bridge."""
        from agent.deep_agents_runtime import _HermesStreamingBridge

        bridge = _HermesStreamingBridge(agent=MagicMock())

        bridge.process_event("string event")
        bridge.process_event(None)
        bridge.process_event(123)
        bridge.process_event(["list", "of", "items"])

        # Should not raise

    def test_streaming_bridge_empty_stream(self):
        """An empty stream yields no callbacks but _run_streamed doesn't crash."""
        from agent.deep_agents_runtime import _HermesStreamingBridge

        agent = self._make_deepagents_agent()
        agent._agent = MagicMock()
        agent._agent.stream.return_value = iter([])
        agent._agent.get_state.return_value = type(
            "State",
            (),
            {
                "values": {
                    "messages": [_make_langchain_message("assistant", content="")]
                }
            },
        )
        object.__setattr__(agent, "_quiet_mode", False)
        agent.stream_delta_callback = lambda x: None

        response = agent.run_conversation(user_message="hello")

        assert response["final_response"] == ""
        assert response["completed"] is False


# ---------------------------------------------------------------------------
# TestToolDispatcherEndToEnd — tool execution integration
# ---------------------------------------------------------------------------


class TestToolDispatcherEndToEnd:
    """E2E: tool dispatcher integration with agent context."""

    def test_hermes_tool_adapter_creates_structured_tool(self):
        """_HermesToolAdapter produces a LangChain StructuredTool."""
        from langchain_core.tools import StructuredTool
        from agent.deep_agents_runtime import _HermesToolAdapter

        entry = types.SimpleNamespace(
            name="wc",
            toolset="search",
            schema={
                "name": "wc",
                "description": "word count",
                "parameters": {"type": "object"},
            },
            description="wc tool",
        )
        adapter = _HermesToolAdapter(entry)
        assert isinstance(adapter.langchain_tool, StructuredTool)
        assert adapter.name == "wc"
        assert adapter.toolset == "search"

    def test_tool_adapter_invokes_handle_function_call(self):
        """Calling the adapter's tool executes handle_function_call."""
        with patch("model_tools.handle_function_call") as mock_hfc:
            mock_hfc.return_value = json.dumps({"count": 42})

            from agent.deep_agents_runtime import _HermesToolAdapter

            entry = types.SimpleNamespace(
                name="test_tool",
                toolset="test",
                schema={
                    "name": "test_tool",
                    "description": "test",
                    "parameters": {"type": "object"},
                },
                description="test",
            )
            adapter = _HermesToolAdapter(entry)

            result = adapter.langchain_tool.func(myarg="value")
            result_obj = json.loads(result)

            assert result_obj == {"count": 42}
            mock_hfc.assert_called_once_with(
                function_name="test_tool",
                function_args={"myarg": "value"},
            )

    def test_tool_adapter_wraps_errors_as_json(self):
        """Exceptions in handle_function_call are returned as JSON error."""
        with patch("model_tools.handle_function_call") as mock_hfc:
            mock_hfc.side_effect = FileNotFoundError("no such file")

            from agent.deep_agents_runtime import _HermesToolAdapter

            entry = types.SimpleNamespace(
                name="read_file",
                toolset="file",
                schema={
                    "name": "read_file",
                    "description": "read",
                    "parameters": {"type": "object"},
                },
                description="read file",
            )
            adapter = _HermesToolAdapter(entry)

            result = adapter.langchain_tool.func(path="/nonexistent")
            result_obj = json.loads(result)

            assert "error" in result_obj
            assert "no such file" in result_obj["error"]

    def test_build_hermes_tools_creates_tools_from_registry(self):
        """build_hermes_tools queries the registry and returns adapters."""
        from agent.deep_agents_runtime import build_hermes_tools

        mock_entry = types.SimpleNamespace(
            name="list_files",
            toolset="terminal",
            schema={
                "name": "list_files",
                "description": "list files",
                "parameters": {},
            },
            description="list files",
        )
        mock_reg = MagicMock()
        mock_reg.get_definitions.return_value = [{"name": "list_files"}]
        mock_reg.registry = mock_reg  # Make MagicMock return itself for attr access
        mock_reg.get_entry.return_value = mock_entry

        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            tools = build_hermes_tools(
                enabled_toolsets=["terminal"], disabled_toolsets=[]
            )
            assert len(tools) == 1
            assert hasattr(tools[0], "func")  # StructuredTool has .func
        finally:
            if saved is not None:
                sys.modules["tools.registry"] = saved
            else:
                sys.modules.pop("tools.registry", None)

    def test_build_hermes_tools_empty_on_missing_entry(self):
        """Tools with None registry entry are skipped."""
        from agent.deep_agents_runtime import build_hermes_tools

        mock_reg = MagicMock()
        mock_reg.get_definitions.return_value = [{"name": "missing"}]
        mock_reg.registry = mock_reg
        mock_reg.get_entry.return_value = None

        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            tools = build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[])
            assert tools == []
        finally:
            sys.modules.pop("tools.registry", None)

    def test_build_hermes_tools_handles_registry_exception(self):
        """If registry.get_definitions raises, an empty list is returned."""
        from agent.deep_agents_runtime import build_hermes_tools

        mock_reg = MagicMock()
        mock_reg.get_definitions.side_effect = ImportError("registry gone")
        mock_reg.registry = mock_reg

        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            tools = build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[])
            assert tools == []
        finally:
            sys.modules.pop("tools.registry", None)


# ---------------------------------------------------------------------------
# TestMessageRoundTrip — Hermes ↔ LangChain conversion
# ---------------------------------------------------------------------------


class TestMessageRoundTrip:
    """E2E: Hermes message dict ↔ LangChain message round-trip."""

    def test_hermes_to_langchain_to_hermes_roundtrip_system_user(self):
        """System → User roundtrip preserves content and roles."""
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        original = [
            _make_hermes_message("system", "You are a cat."),
            _make_hermes_message("user", "Meow?"),
        ]
        lc = _convert_messages_to_langchain(original)
        back = _convert_langchain_to_hermes(lc)

        assert back == original

    def test_hermes_to_langchain_to_hermes_roundtrip_tool_call(self):
        """Assistant with tool_calls → LangChain AIMessage → back."""
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        tool_calls = [
            {"id": "call_abc", "function": {"name": "cmd", "arguments": '{"n": 1}'}}
        ]
        original = [
            _make_hermes_message("assistant", "checking", tool_calls=tool_calls),
            _make_hermes_message("tool", "result", tool_call_id="call_abc"),
        ]
        lc = _convert_messages_to_langchain(original)
        back = _convert_langchain_to_hermes(lc)

        assert back == original

    def test_full_sequence_roundtrip(self):
        """Full conversation sequence: system → user → assistant → tool → assistant."""
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        original = [
            _make_hermes_message("system", "Be brief."),
            _make_hermes_message("user", "What is 2+2?"),
            _make_hermes_message(
                "assistant",
                "4",
                tool_calls=[
                    {"id": "call_1", "function": {"name": "verify", "arguments": "{}"}},
                ],
            ),
            _make_hermes_message("tool", "verified", tool_call_id="call_1"),
            _make_hermes_message("assistant", "2+2=4"),
        ]

        lc = _convert_messages_to_langchain(original)
        back = _convert_langchain_to_hermes(lc)

        assert back == original
        roles = [m["role"] for m in back]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    def test_langchain_to_hermes_none_input_raises(self):
        """_convert_langchain_to_hermes raises TypeError on None."""
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        with pytest.raises(TypeError):
            _convert_langchain_to_hermes(None)

    def test_hermes_to_langchain_empty_messages(self):
        """Empty list returns empty list."""
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert _convert_messages_to_langchain([]) == []

    def test_hermes_to_langchain_with_tool_calls_preserves_args(self):
        """Tool calls on assistant messages preserve id and args."""
        from agent.deep_agents_runtime import _convert_messages_to_langchain
        from langchain_core.messages import AIMessage

        hermes = [
            _make_hermes_message(
                "assistant",
                "let me check",
                tool_calls=[
                    {
                        "id": "tc-123",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"q": "test query"}',
                        },
                    }
                ],
            ),
        ]
        lc = _convert_messages_to_langchain(hermes)
        ai = lc[0]
        assert isinstance(ai, AIMessage)
        assert len(ai.tool_calls) == 1
        assert ai.tool_calls[0]["id"] == "tc-123"
        assert ai.tool_calls[0]["function"]["name"] == "web_search"
        assert ai.tool_calls[0]["function"]["arguments"] == '{"q": "test query"}'

    def test_parse_langgraph_result_full_sequence(self):
        """Full conversation: tool_calls → AI message with text → parse."""
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai1 = _make_langchain_message(
            "assistant",
            content="checking tools",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}},
                {
                    "id": "call_2",
                    "function": {"name": "run_command", "arguments": "{}"},
                },
            ],
        )
        ai2 = _make_langchain_message(
            "assistant", content="All done, here's the answer."
        )
        ai2.reasoning = "reasoned thinking"

        result = _parse_langgraph_result({"messages": [ai1, ai2]})

        assert result["final_response"] == "All done, here's the answer."
        assert result["api_calls"] == 1  # only ai2 counts (it has no tool_calls)
        assert result["completed"] is True
        assert result["last_reasoning"] == "reasoned thinking"
        assert result["failed"] is False
        assert result["turn_exit_reason"] == "completed"

    def test_parse_error_result_full_shape(self):
        """Error result dict has all the keys that run_conversation expects."""
        from agent.deep_agents_runtime import _parse_error_result

        e = Exception("deepagents failed because X")
        result = _parse_error_result(e)

        expected_keys = {
            "final_response",
            "messages",
            "api_calls",
            "completed",
            "failed",
            "interrupted",
            "partial",
            "turn_exit_reason",
        }
        assert expected_keys.issubset(set(result.keys()))
        assert result["failed"] is True
        assert "X" in result["final_response"]

    def test_parse_langgraph_result_multimodal_content(self):
        """Multimodal content (list of dicts) extracts text parts."""
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content=[
                {"type": "text", "text": "Here's the image analysis."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,..."},
                },
            ],
        )

        result = _parse_langgraph_result({"messages": [ai]})
        assert result["final_response"] == "Here's the image analysis."

    def test_inject_provider_env_isolation(self):
        """_inject_provider_env sets the right env vars, no cross-contamination."""
        from agent.deep_agents_runtime import _inject_provider_env

        old_api = os.environ.pop("OPENAI_API_KEY", None)
        old_base = os.environ.pop("OPENAI_API_BASE", None)
        try:
            _inject_provider_env("openai", "https://my-api.com", "sk-key")
            assert os.environ["OPENAI_API_KEY"] == "sk-key"
            assert os.environ["OPENAI_API_BASE"] == "https://my-api.com"
        finally:
            if old_api is not None:
                os.environ["OPENAI_API_KEY"] = old_api
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if old_base is not None:
                os.environ["OPENAI_API_BASE"] = old_base
            else:
                os.environ.pop("OPENAI_API_BASE", None)


# ---------------------------------------------------------------------------
# TestDeepAgentsAIAgentIntegration — full facade with agent attributes
# ---------------------------------------------------------------------------


class TestDeepAgentsAIAgentIntegration:
    """E2E: test the DeepAgentsAIAgent facade as the agent loop would use it."""

    def _build_agent(self, **overrides):
        """Build a DeepAgentsAIAgent with minimal state via __new__."""
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = object.__new__(DeepAgentsAIAgent)
        defaults = dict(
            mode="deepagents",
            _quiet_mode=False,
            _skip_memory=True,
            _platform=None,
            _session_id="test-session",
            _max_iterations=90,
            provider="",
            _api_key=None,
            _base_url=None,
            _callbacks={},
            _agent=MagicMock(),
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            object.__setattr__(agent, k, v)
        return agent

    def test_agent_has_correct_mode(self):
        agent = self._build_agent()
        assert agent.mode == "deepagents"

    def test_agent_compatibility_properties(self):
        """Properties that the agent loop reads are accessible."""
        from agent.deep_agents_runtime import DeepAgentsAIAgent

        agent = self._build_agent()
        assert isinstance(agent.iteration_budget, DeepAgentsAIAgent)  # returns self
        assert agent.model == ""
        assert agent.tools == []
        assert agent.quiet_mode is False
        assert isinstance(agent.skip_memory, bool)

    def test_callback_forwarding_via_setattr(self):
        """Setting callback attrs stores them in _callbacks dict."""
        agent = self._build_agent()

        def my_delta(text):
            pass

        agent.tool_progress_callback = my_delta
        assert agent._get_cap("tool_progress_callback") is my_delta

    def test_callback_get_cap_returns_default_missing(self):
        agent = self._build_agent()
        assert agent._get_cap("nonexistent_callback", "fallback") == "fallback"

    def test_callback_get_cap_returns_none_missing_with_no_default(self):
        agent = self._build_agent()
        assert agent._get_cap("missing") is None
        agent2 = self._build_agent()
        object.__setattr__(agent2, "_callbacks", {})
        assert agent2._get_cap("missing") is None

    def test_interrupt_is_noop(self):
        agent = self._build_agent()
        agent.interrupt()  # should not raise

    def test_get_memory_context_returns_none(self):
        agent = self._build_agent()
        assert agent.get_memory_context() is None

    def test_save_memory_is_noop(self):
        agent = self._build_agent()
        agent.save_memory([])  # should not raise

    def test_chat_calls_run_conversation(self):
        agent = self._build_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="chat response")]
        }

        with patch("agent.deep_agents_runtime._HermesStreamingBridge") as mock_br:
            mock_br.return_value.any_callbacks_set.return_value = False
            resp = agent.chat("hello")

        assert resp == "chat response"
        assert agent._agent.invoke.called
