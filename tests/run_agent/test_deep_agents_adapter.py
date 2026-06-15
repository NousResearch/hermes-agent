"""Unit tests for deep_agents_runtime.py adapter layer.

Tests _HermesToolAdapter, build_hermes_tools,
_convert_messages_to_langchain, _convert_langchain_to_hermes,
_parse_langgraph_result, _parse_error_result, _inject_provider_env,
and _HermesStreamingBridge - all without live LLM calls.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _make_tool_entry(name="test_tool", toolset="test", **overrides):
    schema = overrides.pop(
        "schema",
        {
            "name": name,
            "description": f"{name} tool",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    )
    desc = overrides.pop("description", schema.get("description", f"{name} tool"))
    return types.SimpleNamespace(
        name=name,
        toolset=toolset,
        schema=schema,
        description=desc,
        check_fn=None,
        handler=MagicMock(),
        requires_env=[],
        is_async=False,
        emoji="",
        max_result_size_chars=None,
        dynamic_schema_overrides=None,
    )


def _make_langchain_message(msg_type, **kwargs):
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
        if additional_kwargs:
            msg.additional_kwargs = additional_kwargs
        return msg
    elif msg_type == "tool":
        return ToolMessage(
            content=content, tool_call_id=kwargs.get("tool_call_id", "call_1")
        )
    else:
        raise ValueError(f"Unknown msg_type: {msg_type}")


def _make_hermes_message(role, content="", tool_calls=None, tool_call_id=None):
    msg = {"role": role, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


class TestConvertMessagesToLangchain:
    def test_none_input_returns_empty_list(self):
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert _convert_messages_to_langchain(None) == []

    def test_empty_list_returns_empty_list(self):
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert _convert_messages_to_langchain([]) == []

    def test_system_message(self):
        from langchain_core.messages import SystemMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("system", "You are helpful.")
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], SystemMessage)
            and msgs[0].content == "You are helpful."
        )

    def test_user_message(self):
        from langchain_core.messages import HumanMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([_make_hermes_message("user", "Hello!")])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], HumanMessage)
            and msgs[0].content == "Hello!"
        )

    def test_assistant_message_without_tool_calls(self):
        from langchain_core.messages import AIMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("assistant", "Sure thing!")
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], AIMessage)
            and msgs[0].content == "Sure thing!"
        )

    def test_assistant_message_with_tool_calls(self):
        from langchain_core.messages import AIMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        tc = [
            {
                "id": "call_1",
                "function": {"name": "web_search", "arguments": '{"q": "test"}'},
            }
        ]
        msgs = _convert_messages_to_langchain([
            _make_hermes_message("assistant", "Let me check.", tool_calls=tc)
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], AIMessage)
            and msgs[0].tool_calls == tc
        )

    def test_tool_message(self):
        from langchain_core.messages import ToolMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("tool", "Result", tool_call_id="call_1")
        ])
        assert (
            len(msgs) == 1
            and isinstance(msgs[0], ToolMessage)
            and msgs[0].content == "Result"
        )

    def test_mixed_sequence(self):
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        hermes = [
            _make_hermes_message("system", "You are helpful."),
            _make_hermes_message("user", "Hi"),
            _make_hermes_message("assistant", "Hello!"),
            _make_hermes_message("tool", "tool result", tool_call_id="call_x"),
        ]
        msgs = _convert_messages_to_langchain(hermes)
        assert len(msgs) == 4
        assert isinstance(msgs[0], SystemMessage) and isinstance(msgs[1], HumanMessage)
        assert isinstance(msgs[2], AIMessage) and isinstance(msgs[3], ToolMessage)

    def test_missing_role_defaults_to_no_conversion(self):
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        assert len(_convert_messages_to_langchain([{"content": "no role field"}])) == 0

    def test_empty_content_converted(self):
        from langchain_core.messages import SystemMessage, HumanMessage
        from agent.deep_agents_runtime import _convert_messages_to_langchain

        msgs = _convert_messages_to_langchain([
            _make_hermes_message("system", ""),
            _make_hermes_message("user", ""),
        ])
        assert len(msgs) == 2 and msgs[0].content == "" and msgs[1].content == ""


class TestConvertLangchainToHermes:
    def test_none_input_raises_typeerror(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        with pytest.raises(TypeError):
            _convert_langchain_to_hermes(None)

    def test_empty_list(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        assert _convert_langchain_to_hermes([]) == []

    def test_system_message(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("system", content="You are helpful.")
        ])
        assert len(msgs) == 1 and msgs[0] == {
            "role": "system",
            "content": "You are helpful.",
        }

    def test_user_message(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("user", content="Hello?")
        ])
        assert len(msgs) == 1 and msgs[0] == {"role": "user", "content": "Hello?"}

    def test_assistant_message_without_tool_calls(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("assistant", content="Yes.")
        ])
        assert len(msgs) == 1 and msgs[0] == {"role": "assistant", "content": "Yes."}

    def test_assistant_message_with_empty_content(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("assistant", content="")
        ])
        assert (
            len(msgs) == 1
            and msgs[0]["role"] == "assistant"
            and msgs[0]["content"] == ""
        )

    def test_assistant_message_with_tool_calls(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        tc = [
            {
                "id": "call_1",
                "function": {"name": "read_file", "arguments": '{"path": "foo.txt"}'},
            }
        ]
        msgs = _convert_langchain_to_hermes([
            _make_langchain_message("assistant", content="", tool_calls=tc)
        ])
        assert len(msgs) == 1 and msgs[0]["tool_calls"] == tc

    def test_tool_message(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        msgs = _convert_langchain_to_hermes([
            _make_langchain_message(
                "tool", content="file contents", tool_call_id="call_abc"
            )
        ])
        assert msgs[0] == {
            "role": "tool",
            "content": "file contents",
            "tool_call_id": "call_abc",
        }

    def test_mixed_sequence(self):
        from agent.deep_agents_runtime import _convert_langchain_to_hermes

        langchain = [
            _make_langchain_message("system", content="Be brief."),
            _make_langchain_message("user", content="What's 2+2?"),
            _make_langchain_message("assistant", content="4"),
            _make_langchain_message("tool", content="ok", tool_call_id="call_10"),
        ]
        hermes = _convert_langchain_to_hermes(langchain)
        assert len(hermes) == 4 and hermes[3]["tool_call_id"] == "call_10"


class TestParseLanggraphResult:
    def test_simple_text_response(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message("assistant", content="Hello there!")
        result = _parse_langgraph_result({"messages": [ai]})
        assert (
            result["final_response"] == "Hello there!" and result["completed"] is True
        )

    def test_tool_call_then_text_response(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai1 = _make_langchain_message(
            "assistant",
            content="let me search",
            tool_calls=[
                {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}
            ],
        )
        ai2 = _make_langchain_message("assistant", content="Final answer!")
        r = _parse_langgraph_result({"messages": [ai1, ai2]})
        assert r["final_response"] == "Final answer!" and r["api_calls"] == 1

    def test_only_tool_calls_no_final_text(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content="",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        )
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["completed"] is False and r["turn_exit_reason"] == "no_response"

    def test_multimodal_content_text_type(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content=[
                {"type": "text", "text": "Multimodal response."},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        )
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["final_response"] == "Multimodal response."

    def test_multimodal_content_plain_strings(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [
                _make_langchain_message("assistant", content=["plain string part"])
            ]
        })
        assert r["final_response"] == "plain string part"

    def test_multimodal_content_string_part_in_list(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content=["part1"])]
        })
        assert r["final_response"] == "part1"

    def test_reasoning_from_msg_reasoning_attr(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message("assistant", content="answer")
        ai.reasoning = "I thought about it..."
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["last_reasoning"] == "I thought about it..."

    def test_reasoning_from_additional_kwargs(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant", content="answer", additional_kwargs={"reasoning": "inferred"}
        )
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["last_reasoning"] == "inferred"

    def test_reasoning_attr_takes_precedence_over_kwargs(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai = _make_langchain_message(
            "assistant",
            content="answer",
            additional_kwargs={"reasoning": "kwarg reasoning"},
        )
        ai.reasoning = "direct reasoning"
        r = _parse_langgraph_result({"messages": [ai]})
        assert r["last_reasoning"] == "direct reasoning"

    def test_multiple_tool_call_aimessages_counted(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        t1 = _make_langchain_message(
            "assistant",
            content="checking...",
            tool_calls=[
                {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        )
        t2 = _make_langchain_message(
            "assistant",
            content="done...",
            tool_calls=[
                {"id": "call_2", "function": {"name": "write_file", "arguments": "{}"}}
            ],
        )
        f = _make_langchain_message("assistant", content="All done.")
        r = _parse_langgraph_result({"messages": [t1, t2, f]})
        assert r["api_calls"] == 2 and r["final_response"] == "All done."

    def test_empty_messages_result(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({"messages": []})
        assert (
            r["final_response"] == ""
            and r["completed"] is False
            and r["api_calls"] == 0
        )

    def test_non_dict_result(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({"messages": "not a list"})
        assert r["final_response"] == "" and r["completed"] is False

    def test_hermes_messages_converted_in_result(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content="answer")]
        })
        assert r["messages"][0] == {"role": "assistant", "content": "answer"}

    def test_model_field_is_empty_string(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        assert _parse_langgraph_result({"messages": []})["model"] == ""

    def test_turn_completed_with_final_response(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content="Final")]
        })
        assert r["completed"] is True and r["turn_exit_reason"] == "completed"

    def test_turn_no_response_without_ai_message(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [
                _make_langchain_message("system", content="Be helpful"),
                _make_langchain_message("user", content="Hi"),
            ]
        })
        assert r["completed"] is False and r["turn_exit_reason"] == "no_response"


class TestParseErrorResult:
    def test_error_includes_exception_message(self):
        from agent.deep_agents_runtime import _parse_error_result

        result = _parse_error_result(RuntimeError("connection refused"))
        assert (
            result["final_response"] == "Error: connection refused"
            and result["failed"] is True
        )

    def test_error_is_empty_for_empty_exception(self):
        from agent.deep_agents_runtime import _parse_error_result

        assert _parse_error_result(Exception(""))["final_response"] == "Error: "

    def test_error_result_dict_shape(self):
        from agent.deep_agents_runtime import _parse_error_result

        r = _parse_error_result(Exception("test"))
        assert set(r.keys()) >= {
            "final_response",
            "messages",
            "api_calls",
            "completed",
            "failed",
            "interrupted",
            "partial",
            "turn_exit_reason",
        }


class TestInjectProviderEnv:
    def test_empty_provider_maps_to_openai(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("", None, "sk-123")
        assert os.environ.get("OPENAI_API_KEY") == "sk-123"

    def test_openai_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("openai", "https://custom.openai.com", "sk-key")
        assert (
            os.environ["OPENAI_API_KEY"] == "sk-key"
            and os.environ["OPENAI_API_BASE"] == "https://custom.openai.com"
        )

    def test_anthropic_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("anthropic", None, "sk-ant-123")
        assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-123"

    def test_google_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("google", "https://google.custom", "gkey")
        assert (
            os.environ["GOOGLE_API_KEY"] == "gkey"
            and os.environ["GOOGLE_API_BASE"] == "https://google.custom"
        )

    def test_xai_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("xai", None, "xai-key")
        assert os.environ["XAI_API_KEY"] == "xai-key"

    def test_cohere_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("cohere", None, "co-key")
        assert os.environ["COHERE_API_KEY"] == "co-key"

    def test_groq_provider(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("groq", None, "groq-key")
        assert os.environ["GROQ_API_KEY"] == "groq-key"

    def test_unknown_provider_uses_generic_names(self):
        from agent.deep_agents_runtime import _inject_provider_env

        _inject_provider_env("unknown-provider", "https://unknown.com", "uk-key")
        assert (
            os.environ["API_KEY"] == "uk-key"
            and os.environ["BASE_URL"] == "https://unknown.com"
        )

    def test_base_url_only(self):
        from agent.deep_agents_runtime import _inject_provider_env

        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            _inject_provider_env("openai", "https://base.com", None)
            assert os.environ["OPENAI_API_BASE"] == "https://base.com"
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


class TestHermesToolAdapter:
    def test_adapter_creates_structured_tool(self):
        from langchain_core.tools import StructuredTool
        from agent.deep_agents_runtime import _HermesToolAdapter

        assert isinstance(
            _HermesToolAdapter(
                _make_tool_entry("read_file", toolset="file")
            ).langchain_tool,
            StructuredTool,
        )

    def test_adapter_preserves_name(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        assert (
            _HermesToolAdapter(_make_tool_entry("my_tool", toolset="custom")).name
            == "my_tool"
        )

    def test_adapter_preserves_schema(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        s = {"name": "x", "description": "desc", "parameters": {}}
        assert (
            _HermesToolAdapter(_make_tool_entry("x", toolset="s", schema=s)).schema == s
        )

    def test_adapter_preserves_toolset(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        assert (
            _HermesToolAdapter(_make_tool_entry("tool_x", toolset="my_toolset")).toolset
            == "my_toolset"
        )

    def test_adapter_calls_handle_function_call(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        import model_tools

        saved = model_tools.handle_function_call
        mock = MagicMock()
        mock.return_value = json.dumps({"ok": True})
        try:
            model_tools.handle_function_call = mock
            adapter = _HermesToolAdapter(_make_tool_entry("read_file", toolset="file"))
            # Call __func directly - LangChain's .invoke() passes arguments
            # through a Pydantic schema that may rename / drop keys, so
            # .func() is the reliable way to test the closure.
            adapter.langchain_tool.func(path="foo.txt")
            assert mock.called, "handle_function_call was not called"
            call_kwargs = mock.call_args
            assert call_kwargs.kwargs["function_name"] == "read_file"
            assert call_kwargs.kwargs["function_args"] == {"path": "foo.txt"}
        finally:
            model_tools.handle_function_call = saved

    @patch("model_tools.handle_function_call")
    def test_adapter_wraps_exceptions_as_json_error(self, mock_handle):
        from agent.deep_agents_runtime import _HermesToolAdapter

        mock_handle.side_effect = IOError("disk full")
        parsed = json.loads(
            _HermesToolAdapter(
                _make_tool_entry("write_file", toolset="file")
            ).langchain_tool.invoke({"path": "bar.txt"})
        )
        assert "error" in parsed and "disk full" in parsed["error"]

    def test_adapter_uses_entry_description_over_schema(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        adapter = _HermesToolAdapter(
            _make_tool_entry(
                "x", toolset="s", schema={"name": "x"}, description="custom desc"
            )
        )
        assert adapter._tool.description == "custom desc"

    def test_adapter_falls_back_schema_description(self):
        from agent.deep_agents_runtime import _HermesToolAdapter

        adapter = _HermesToolAdapter(
            _make_tool_entry(
                "x",
                toolset="s",
                schema={"name": "x", "description": "schema desc"},
                description=None,
            )
        )
        assert adapter._tool.description == "schema desc"


class TestBuildHermesTools:
    def _mock_reg(self, definitions, entry_fn):
        """Create mock registry that works with `from tools.registry import registry`."""
        m = MagicMock()
        m.get_definitions.return_value = definitions
        if entry_fn is not None:
            m.get_entry.side_effect = entry_fn
        # `from tools.registry import registry` reads the *registry* attribute
        # from the module. Without this assignment, MagicMock auto-creates a
        # child MagicMock, so we make registry return the mock itself.
        m.registry = m
        return m

    def test_builds_structured_tools_from_definitions(self):
        from langchain_core.tools import StructuredTool
        from agent.deep_agents_runtime import build_hermes_tools

        ea = _make_tool_entry("tool_a", toolset="ts1")
        eb = _make_tool_entry("tool_b", toolset="ts2")
        mock_reg = self._mock_reg(
            [{"name": "tool_a"}, {"name": "tool_b"}],
            lambda n: {"tool_a": ea, "tool_b": eb}.get(n),
        )
        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            tools = build_hermes_tools(
                enabled_toolsets=["ts1", "ts2"], disabled_toolsets=[]
            )
            assert len(tools) == 2 and all(isinstance(t, StructuredTool) for t in tools)
        finally:
            if saved is not None:
                sys.modules["tools.registry"] = saved
            else:
                sys.modules.pop("tools.registry", None)

    def test_handles_missing_tool_entries(self):
        from agent.deep_agents_runtime import build_hermes_tools

        ep = _make_tool_entry("present", toolset="ts")
        mock_reg = self._mock_reg(
            [{"name": "present"}, {"name": "missing"}],
            lambda n: ep if n == "present" else None,
        )
        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            assert (
                len(build_hermes_tools(enabled_toolsets=["ts"], disabled_toolsets=[]))
                == 1
            )
        finally:
            if saved is not None:
                sys.modules["tools.registry"] = saved
            else:
                sys.modules.pop("tools.registry", None)

    def test_handles_empty_definitions(self):
        from agent.deep_agents_runtime import build_hermes_tools

        mock_reg = self._mock_reg([], None)
        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            assert build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[]) == []
        finally:
            if saved is not None:
                sys.modules["tools.registry"] = saved
            else:
                sys.modules.pop("tools.registry", None)

    def test_returns_empty_list_on_registry_error(self):
        from agent.deep_agents_runtime import build_hermes_tools

        mock_reg = self._mock_reg([], None)
        mock_reg.get_definitions.side_effect = ImportError("registry unavailable")
        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            assert (
                build_hermes_tools(enabled_toolsets=["ts"], disabled_toolsets=[]) == []
            )
        finally:
            if saved is not None:
                sys.modules["tools.registry"] = saved
            else:
                sys.modules.pop("tools.registry", None)

    def test_empty_name_tool_skipped(self):
        from agent.deep_agents_runtime import build_hermes_tools

        mock_reg = self._mock_reg([{"name": ""}], None)
        mock_reg.get_entry.return_value = None
        saved = sys.modules.get("tools.registry")
        try:
            sys.modules["tools.registry"] = mock_reg
            assert build_hermes_tools(enabled_toolsets=[], disabled_toolsets=[]) == []
        finally:
            if saved is not None:
                sys.modules["tools.registry"] = saved
            else:
                sys.modules.pop("tools.registry", None)


class TestHermesStreamingBridge:
    def _make_bridge(self, **kwargs):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        agent_mock = MagicMock()
        for k, v in kwargs.items():
            setattr(agent_mock, k, v)
        return _HermesStreamingBridge(
            agent=agent_mock,
            stream_delta=kwargs.get("stream_delta"),
            tool_progress=kwargs.get("tool_progress"),
            thinking=kwargs.get("thinking"),
            step=kwargs.get("step"),
        )

    def test_any_callbacks_set_with_stream_delta(self):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        bridge = _HermesStreamingBridge(agent=MagicMock(), stream_delta=lambda x: None)
        assert bridge.any_callbacks_set() is True

    def test_any_callbacks_set_with_tool_progress(self):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        bridge = _HermesStreamingBridge(
            agent=MagicMock(), tool_progress=lambda *a, **kw: None
        )
        assert bridge.any_callbacks_set() is True

    def test_any_callbacks_set_always_true_with_noop_fallback(self):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        assert _HermesStreamingBridge(agent=MagicMock()).any_callbacks_set() is True

    def test_process_event_non_dict_ignored(self):
        result = self._make_bridge().process_event("not a dict")
        assert result is None

    def test_process_event_with_events_key(self):
        from agent.deep_agents_runtime import _noop_cb

        cals = []
        tp = lambda action, **kw: cals.append((action, kw))
        bridge = self._make_bridge(stream_delta=_noop_cb("sd"), tool_progress=tp)
        bridge.process_event({
            "events": [
                {"type": "AIMessageChunk", "data": {"content": "hello"}},
                {
                    "type": "ToolCall",
                    "data": {"name": "read_file", "args": {"path": "a.txt"}},
                },
            ]
        })
        assert (
            len(cals) == 1
            and cals[0][0] == "tool.started"
            and "read_file" in cals[0][1].get("tool_name", "")
        )

    def test_process_event_with_output_key_text(self):
        deltas = []
        self._make_bridge(stream_delta=lambda x: deltas.append(x)).process_event({
            "output": "Final response text"
        })
        assert deltas == ["Final response text"]

    def test_process_event_with_output_key_dict(self):
        deltas = []
        self._make_bridge(stream_delta=lambda x: deltas.append(x)).process_event({
            "output": {"final_response": "Done!"}
        })
        assert deltas == ["Done!"]

    def test_process_event_with_output_key_non_dict(self):
        deltas = []
        self._make_bridge(stream_delta=lambda x: deltas.append(x)).process_event({
            "output": [1, 2, 3]
        })
        assert deltas[0] == "[1, 2, 3]"

    def test_process_event_with_missing_keys_ignored(self):
        from agent.deep_agents_runtime import _noop_cb

        sd = MagicMock()
        self._make_bridge(stream_delta=sd).process_event({"something_else": "value"})
        sd.assert_not_called()

    def test_ai_message_chunk_uses_stream_delta(self):
        deltas = []
        self._make_bridge(stream_delta=lambda x: deltas.append(x))._process_sub_event({
            "type": "AIMessageChunk",
            "data": {"content": "streaming text"},
        })
        assert deltas == ["streaming text"]

    def test_ai_message_chunk_empty_content_no_emit(self):
        deltas = []
        self._make_bridge(stream_delta=lambda x: deltas.append(x))._process_sub_event({
            "type": "AIMessageChunk",
            "data": {"content": ""},
        })
        assert deltas == []

    def test_ai_message_chunk_with_step_callback(self):
        steps = []
        bridge = self._make_bridge(
            stream_delta=lambda x: None, step=lambda n, s: steps.append((n, s))
        )
        bridge._process_sub_event({
            "type": "AIMessageChunk",
            "data": {"content": "test"},
        })
        assert steps == [(1, [])]

    def test_tool_call_uses_tool_progress(self):
        cals = []
        self._make_bridge(
            tool_progress=lambda a, **kw: cals.append((a, kw))
        )._process_sub_event({
            "type": "ToolCall",
            "data": {
                "name": "write_file",
                "args": {"path": "big.txt", "content": "x" * 300},
            },
        })
        assert (
            len(cals) == 1
            and cals[0][0] == "tool.started"
            and len(cals[0][1]["preview"]) <= 200
        )

    def test_unknown_event_type_ignored(self):
        self._make_bridge()._process_sub_event({"type": "UnknownType", "data": {}})

    def test_sub_event_non_dict_ignored(self):
        b = self._make_bridge()
        b._process_sub_event("not a dict")
        b._process_sub_event(None)

    def test_output_dict_missing_final_response_uses_str(self):
        deltas = []
        self._make_bridge(stream_delta=lambda x: deltas.append(x))._process_output({
            "some_key": "value",
            "other": 123,
        })
        assert deltas[0] == str({"some_key": "value", "other": 123})

    def test_tool_call_no_args_fires_progress(self):
        cals = []
        self._make_bridge(
            tool_progress=lambda a, **kw: cals.append((a, kw))
        )._process_sub_event({"type": "ToolCall", "data": {"name": "ping"}})
        assert len(cals) == 1

    def test_tool_call_no_name_defaults_empty(self):
        cals = []
        self._make_bridge(
            tool_progress=lambda a, **kw: cals.append((a, kw))
        )._process_sub_event({"type": "ToolCall", "data": {"args": "no-name"}})
        assert cals[0][1]["tool_name"] == ""

    def test_noop_callback_does_not_raise(self):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        bridge = _HermesStreamingBridge(agent=MagicMock())
        bridge.process_event({
            "events": [{"type": "AIMessageChunk", "data": {"content": "x"}}]
        })
        bridge.process_event({"output": "y"})

    def test_bridge_fetches_callbacks_from_agent(self):
        from agent.deep_agents_runtime import _HermesStreamingBridge

        agent_mock = MagicMock()
        agent_mock.stream_delta_callback = lambda x: None
        agent_mock.tool_progress_callback = lambda *a, **kw: None
        agent_mock.step_callback = None
        bridge = _HermesStreamingBridge(agent=agent_mock)
        assert bridge.any_callbacks_set() is True
        assert bridge._stream_delta == agent_mock.stream_delta_callback


class TestNoopCallback:
    def test_noop_accepts_args(self):
        from agent.deep_agents_runtime import _noop_cb

        cb = _noop_cb("test")
        cb(1, 2, 3, "keyword")

    def test_noop_return_none(self):
        from agent.deep_agents_runtime import _noop_cb

        assert _noop_cb("test")() is None

    def test_noop_has_name(self):
        from agent.deep_agents_runtime import _noop_cb

        assert _noop_cb("my_name").__name__ == "noop"


class TestDeepAgentsAIAgentAttributes:
    def _make_agent(self, **kwargs):
        """Create a DeepAgentsAIAgent with all SDK deps mocked via __new__."""
        from agent import deep_agents_runtime as dar

        agent = object.__new__(dar.DeepAgentsAIAgent)
        # Set internal state as __init__ would (without calling __init__)
        object.__setattr__(agent, "mode", "deepagents")
        object.__setattr__(agent, "_quiet_mode", kwargs.get("quiet_mode", False))
        object.__setattr__(agent, "_skip_memory", kwargs.get("skip_memory", False))
        object.__setattr__(agent, "_platform", kwargs.get("platform"))
        object.__setattr__(agent, "_session_id", kwargs.get("session_id", ""))
        object.__setattr__(agent, "_max_iterations", kwargs.get("max_iterations", 90))
        object.__setattr__(agent, "provider", kwargs.get("provider", ""))
        object.__setattr__(agent, "_api_key", kwargs.get("api_key", None))
        object.__setattr__(agent, "_base_url", kwargs.get("base_url"))
        object.__setattr__(agent, "_callbacks", {})
        object.__setattr__(agent, "_agent", kwargs.get("_agent_mock", MagicMock()))
        return agent

    def test_basic_attribute_access(self):
        agent = self._make_agent(quiet_mode=True, skip_memory=True, platform=None)
        assert agent.mode == "deepagents" and agent.quiet_mode is True
        assert agent.skip_memory is True and agent.platform is None

    def test_iteration_budget_returns_self(self):
        agent = self._make_agent()
        assert agent.iteration_budget is agent

    def test_model_returns_empty_string(self):
        assert self._make_agent().model == ""

    def test_tools_returns_empty_list(self):
        assert self._make_agent().tools == []

    def test_max_iterations(self):
        assert self._make_agent(max_iterations=42).max_iterations == 42

    def test_run_conversation_returns_dict(self):
        agent = self._make_agent()
        agent._agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="response")]
        }
        result = agent.run_conversation("hello")
        assert isinstance(result, dict) and "final_response" in result

    def test_chat_returns_final_response(self):
        agent = self._make_agent()
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [_make_langchain_message("assistant", content="chat answer")]
        }
        mock_agent.stream.return_value = iter([])
        mock_agent.get_state.return_value = type(
            "State",
            (),
            {
                "values": {
                    "messages": [
                        _make_langchain_message("assistant", content="chat answer")
                    ]
                }
            },
        )
        object.__setattr__(agent, "_agent", mock_agent)
        assert agent.chat("what's up") == "chat answer"

    def test_interrupt_is_noop(self):
        self._make_agent().interrupt()

    def test_get_memory_context_returns_none(self):
        assert self._make_agent().get_memory_context() is None

    def test_save_memory_is_noop(self):
        self._make_agent().save_memory([])

    def test_reasoning_callback_forwarding(self):
        agent = self._make_agent()
        cb = lambda *a, **kw: None
        agent.reasoning_callback = cb
        assert agent._get_cap("reasoning_callback") is cb

    def test_step_callback_forwarding(self):
        agent = self._make_agent()
        step_cb = lambda *a, **kw: None
        agent.step_callback = step_cb
        _ = agent.reasoning_callback
        assert agent._get_cap("step_callback") is step_cb

    def test_attribute_not_in_captured_raises_attributeerror(self):
        agent = self._make_agent()
        with pytest.raises(AttributeError, match="nonexistent_attr"):
            _ = agent.nonexistent_attr

    def test_resolve_model_strips_provider_prefix(self):
        agent = self._make_agent(api_key="k", model="openai/gpt-4o")
        assert agent.mode == "deepagents"

    def test_default_model_for_anthropic(self):
        agent = self._make_agent(api_key="k", provider="anthropic")
        assert agent.provider == "anthropic"

    def test_default_model_for_google(self):
        agent = self._make_agent(api_key="k", provider="google")
        assert agent.provider == "google"


class TestReasoningExtraction:
    def test_reasoning_on_last_ai_message_extracted(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        ai1 = _make_langchain_message(
            "assistant",
            content="tool result",
            tool_calls=[{"id": "call_1", "function": {"name": "x", "arguments": "{}"}}],
        )
        ai2 = _make_langchain_message("assistant", content="final answer")
        ai2.reasoning = "last reasoning"
        r = _parse_langgraph_result({"messages": [ai1, ai2]})
        assert r["last_reasoning"] == "last reasoning"

    def test_no_reasoning_set(self):
        from agent.deep_agents_runtime import _parse_langgraph_result

        r = _parse_langgraph_result({
            "messages": [_make_langchain_message("assistant", content="plain response")]
        })
        assert r["last_reasoning"] is None


class TestRoundTripConversion:
    def test_hermes_to_langchain_to_hermes_roundtrip(self):
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        original = [
            _make_hermes_message("system", "Be helpful."),
            _make_hermes_message("user", "What is 2+2?"),
        ]
        assert (
            _convert_langchain_to_hermes(_convert_messages_to_langchain(original))
            == original
        )

    def test_assistant_with_tool_calls_roundtrip(self):
        from agent.deep_agents_runtime import (
            _convert_messages_to_langchain,
            _convert_langchain_to_hermes,
        )

        tool_calls = [{"id": "c1", "function": {"name": "x", "arguments": "{}"}}]
        original = [
            _make_hermes_message("assistant", "checking", tool_calls=tool_calls)
        ]
        step1 = _convert_messages_to_langchain(original)
        step2 = _convert_langchain_to_hermes(step1)
        assert step2[0]["role"] == "assistant" and step2[0]["tool_calls"] == tool_calls
