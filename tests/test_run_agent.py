"""Unit tests for run_agent.py (AIAgent).

Tests cover pure functions, state/structure methods, and conversation loop
pieces. The OpenAI client and tool loading are mocked so no network calls
are made.
"""

import json
import re
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from run_agent import AIAgent
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked OpenAI client and tool loading."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


@pytest.fixture()
def agent_with_memory_tool():
    """Agent whose valid_tool_names includes 'memory'."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search", "memory")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


# ---------------------------------------------------------------------------
# Helper to build mock assistant messages (API response objects)
# ---------------------------------------------------------------------------

def _mock_assistant_msg(
    content="Hello",
    tool_calls=None,
    reasoning=None,
    reasoning_content=None,
    reasoning_details=None,
):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletionMessage."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        msg.reasoning = reasoning
    if reasoning_content is not None:
        msg.reasoning_content = reasoning_content
    if reasoning_details is not None:
        msg.reasoning_details = reasoning_details
    return msg


def _mock_tool_call(name="web_search", arguments='{}', call_id=None):
    """Return a SimpleNamespace mimicking a tool call object."""
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None,
                    reasoning=None, usage=None):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletion response."""
    msg = _mock_assistant_msg(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    if usage:
        resp.usage = SimpleNamespace(**usage)
    else:
        resp.usage = None
    return resp


# ===================================================================
# Grup 1: Pure Functions
# ===================================================================


class TestHasContentAfterThinkBlock:
    def test_none_returns_false(self, agent):
        assert agent._has_content_after_think_block(None) is False

    def test_empty_returns_false(self, agent):
        assert agent._has_content_after_think_block("") is False

    def test_only_think_block_returns_false(self, agent):
        assert agent._has_content_after_think_block("<think>reasoning</think>") is False

    def test_content_after_think_returns_true(self, agent):
        assert agent._has_content_after_think_block("<think>r</think> actual answer") is True

    def test_no_think_block_returns_true(self, agent):
        assert agent._has_content_after_think_block("just normal content") is True


class TestStripThinkBlocks:
    def test_none_returns_empty(self, agent):
        assert agent._strip_think_blocks(None) == ""

    def test_no_blocks_unchanged(self, agent):
        assert agent._strip_think_blocks("hello world") == "hello world"

    def test_single_block_removed(self, agent):
        result = agent._strip_think_blocks("<think>reasoning</think> answer")
        assert "reasoning" not in result
        assert "answer" in result

    def test_multiline_block_removed(self, agent):
        text = "<think>\nline1\nline2\n</think>\nvisible"
        result = agent._strip_think_blocks(text)
        assert "line1" not in result
        assert "visible" in result


class TestExtractReasoning:
    def test_reasoning_field(self, agent):
        msg = _mock_assistant_msg(reasoning="thinking hard")
        assert agent._extract_reasoning(msg) == "thinking hard"

    def test_reasoning_content_field(self, agent):
        msg = _mock_assistant_msg(reasoning_content="deep thought")
        assert agent._extract_reasoning(msg) == "deep thought"

    def test_reasoning_details_array(self, agent):
        msg = _mock_assistant_msg(
            reasoning_details=[{"summary": "step-by-step analysis"}],
        )
        assert "step-by-step analysis" in agent._extract_reasoning(msg)

    def test_no_reasoning_returns_none(self, agent):
        msg = _mock_assistant_msg()
        assert agent._extract_reasoning(msg) is None

    def test_combined_reasoning(self, agent):
        msg = _mock_assistant_msg(
            reasoning="part1",
            reasoning_content="part2",
        )
        result = agent._extract_reasoning(msg)
        assert "part1" in result
        assert "part2" in result

    def test_deduplication(self, agent):
        msg = _mock_assistant_msg(
            reasoning="same text",
            reasoning_content="same text",
        )
        result = agent._extract_reasoning(msg)
        assert result == "same text"


class TestCleanSessionContent:
    def test_none_passthrough(self):
        assert AIAgent._clean_session_content(None) is None

    def test_scratchpad_converted(self):
        text = "<REASONING_SCRATCHPAD>think</REASONING_SCRATCHPAD> answer"
        result = AIAgent._clean_session_content(text)
        assert "<REASONING_SCRATCHPAD>" not in result
        assert "<think>" in result

    def test_extra_newlines_cleaned(self):
        text = "\n\n\n<think>x</think>\n\n\nafter"
        result = AIAgent._clean_session_content(text)
        # Should not have excessive newlines around think block
        assert "\n\n\n" not in result


class TestGetMessagesUpToLastAssistant:
    def test_empty_list(self, agent):
        assert agent._get_messages_up_to_last_assistant([]) == []

    def test_no_assistant_returns_copy(self, agent):
        msgs = [{"role": "user", "content": "hi"}]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert result == msgs
        assert result is not msgs  # should be a copy

    def test_single_assistant(self, agent):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_multiple_assistants_returns_up_to_last(self, agent):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert len(result) == 3
        assert result[-1]["content"] == "q2"

    def test_assistant_then_tool_messages(self, agent):
        msgs = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1"}]},
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ]
        # Last assistant is at index 1, so result = msgs[:1]
        result = agent._get_messages_up_to_last_assistant(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"


class TestMaskApiKey:
    def test_none_returns_none(self, agent):
        assert agent._mask_api_key_for_logs(None) is None

    def test_short_key_returns_stars(self, agent):
        assert agent._mask_api_key_for_logs("short") == "***"

    def test_long_key_masked(self, agent):
        key = "sk-or-v1-abcdefghijklmnop"
        result = agent._mask_api_key_for_logs(key)
        assert result.startswith("sk-or-v1")
        assert result.endswith("mnop")
        assert "..." in result


# ===================================================================
# Grup 2: State / Structure Methods
# ===================================================================


class TestInit:
    def test_prompt_caching_claude_openrouter(self):
        """Claude model via OpenRouter should enable prompt caching."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                model="anthropic/claude-sonnet-4-20250514",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a._use_prompt_caching is True

    def test_prompt_caching_non_claude(self):
        """Non-Claude model should disable prompt caching."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                model="openai/gpt-4o",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a._use_prompt_caching is False

    def test_prompt_caching_non_openrouter(self):
        """Custom base_url (not OpenRouter) should disable prompt caching."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                model="anthropic/claude-sonnet-4-20250514",
                base_url="http://localhost:8080/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a._use_prompt_caching is False

    def test_valid_tool_names_populated(self):
        """valid_tool_names should contain names from loaded tools."""
        tools = _make_tool_defs("web_search", "terminal")
        with (
            patch("run_agent.get_tool_definitions", return_value=tools),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a.valid_tool_names == {"web_search", "terminal"}

    def test_session_id_auto_generated(self):
        """Session ID should be auto-generated when not provided."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a.session_id is not None
            assert len(a.session_id) > 0


class TestInterrupt:
    def test_interrupt_sets_flag(self, agent):
        with patch("run_agent._set_interrupt"):
            agent.interrupt()
            assert agent._interrupt_requested is True

    def test_interrupt_with_message(self, agent):
        with patch("run_agent._set_interrupt"):
            agent.interrupt("new question")
            assert agent._interrupt_message == "new question"

    def test_clear_interrupt(self, agent):
        with patch("run_agent._set_interrupt"):
            agent.interrupt("msg")
            agent.clear_interrupt()
            assert agent._interrupt_requested is False
            assert agent._interrupt_message is None

    def test_is_interrupted_property(self, agent):
        assert agent.is_interrupted is False
        with patch("run_agent._set_interrupt"):
            agent.interrupt()
            assert agent.is_interrupted is True


class TestHydrateTodoStore:
    def test_no_todo_in_history(self, agent):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()

    def test_recovers_from_history(self, agent):
        todos = [{"id": "1", "content": "do thing", "status": "pending"}]
        history = [
            {"role": "user", "content": "plan"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": json.dumps({"todos": todos}), "tool_call_id": "c1"},
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert agent._todo_store.has_items()

    def test_skips_non_todo_tools(self, agent):
        history = [
            {"role": "tool", "content": '{"result": "search done"}', "tool_call_id": "c1"},
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()

    def test_invalid_json_skipped(self, agent):
        history = [
            {"role": "tool", "content": 'not valid json "todos" oops', "tool_call_id": "c1"},
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()


class TestBuildSystemPrompt:
    def test_always_has_identity(self, agent):
        prompt = agent._build_system_prompt()
        assert DEFAULT_AGENT_IDENTITY in prompt

    def test_includes_system_message(self, agent):
        prompt = agent._build_system_prompt(system_message="Custom instruction")
        assert "Custom instruction" in prompt

    def test_memory_guidance_when_memory_tool_loaded(self, agent_with_memory_tool):
        from agent.prompt_builder import MEMORY_GUIDANCE
        prompt = agent_with_memory_tool._build_system_prompt()
        assert MEMORY_GUIDANCE in prompt

    def test_no_memory_guidance_without_tool(self, agent):
        from agent.prompt_builder import MEMORY_GUIDANCE
        prompt = agent._build_system_prompt()
        assert MEMORY_GUIDANCE not in prompt

    def test_includes_datetime(self, agent):
        prompt = agent._build_system_prompt()
        # Should contain current date info like "Conversation started:"
        assert "Conversation started:" in prompt


class TestInvalidateSystemPrompt:
    def test_clears_cache(self, agent):
        agent._cached_system_prompt = "cached value"
        agent._invalidate_system_prompt()
        assert agent._cached_system_prompt is None

    def test_reloads_memory_store(self, agent):
        mock_store = MagicMock()
        agent._memory_store = mock_store
        agent._cached_system_prompt = "cached"
        agent._invalidate_system_prompt()
        mock_store.load_from_disk.assert_called_once()


class TestBuildApiKwargs:
    def test_basic_kwargs(self, agent):
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["model"] == agent.model
        assert kwargs["messages"] is messages
        assert kwargs["timeout"] == 600.0

    def test_provider_preferences_injected(self, agent):
        agent.providers_allowed = ["Anthropic"]
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["provider"]["only"] == ["Anthropic"]

    def test_reasoning_config_default_openrouter(self, agent):
        """Default reasoning config for OpenRouter should be xhigh."""
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        reasoning = kwargs["extra_body"]["reasoning"]
        assert reasoning["enabled"] is True
        assert reasoning["effort"] == "xhigh"

    def test_reasoning_config_custom(self, agent):
        agent.reasoning_config = {"enabled": False}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}

    def test_max_tokens_injected(self, agent):
        agent.max_tokens = 4096
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["max_tokens"] == 4096


class TestBuildAssistantMessage:
    def test_basic_message(self, agent):
        msg = _mock_assistant_msg(content="Hello!")
        result = agent._build_assistant_message(msg, "stop")
        assert result["role"] == "assistant"
        assert result["content"] == "Hello!"
        assert result["finish_reason"] == "stop"

    def test_with_reasoning(self, agent):
        msg = _mock_assistant_msg(content="answer", reasoning="thinking")
        result = agent._build_assistant_message(msg, "stop")
        assert result["reasoning"] == "thinking"

    def test_with_tool_calls(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "web_search"

    def test_with_reasoning_details(self, agent):
        details = [{"type": "reasoning.summary", "text": "step1", "signature": "sig1"}]
        msg = _mock_assistant_msg(content="ans", reasoning_details=details)
        result = agent._build_assistant_message(msg, "stop")
        assert "reasoning_details" in result
        assert result["reasoning_details"][0]["text"] == "step1"

    def test_empty_content(self, agent):
        msg = _mock_assistant_msg(content=None)
        result = agent._build_assistant_message(msg, "stop")
        assert result["content"] == ""


class TestFormatToolsForSystemMessage:
    def test_no_tools_returns_empty_array(self, agent):
        agent.tools = []
        assert agent._format_tools_for_system_message() == "[]"

    def test_formats_single_tool(self, agent):
        agent.tools = _make_tool_defs("web_search")
        result = agent._format_tools_for_system_message()
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "web_search"

    def test_formats_multiple_tools(self, agent):
        agent.tools = _make_tool_defs("web_search", "terminal", "read_file")
        result = agent._format_tools_for_system_message()
        parsed = json.loads(result)
        assert len(parsed) == 3
        names = {t["name"] for t in parsed}
        assert names == {"web_search", "terminal", "read_file"}


# ===================================================================
# Grup 3: Conversation Loop Pieces (OpenAI mock)
# ===================================================================


class TestExecuteToolCalls:
    def test_single_tool_executed(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch("run_agent.handle_function_call", return_value="search result") as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            mock_hfc.assert_called_once_with("web_search", {"q": "test"}, "task-1")
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "search result" in messages[0]["content"]

    def test_interrupt_skips_remaining(self, agent):
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("run_agent._set_interrupt"):
            agent.interrupt()

        agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Both calls should be skipped with cancellation messages
        assert len(messages) == 2
        assert "cancelled" in messages[0]["content"].lower() or "interrupted" in messages[0]["content"].lower()

    def test_invalid_json_args_defaults_empty(self, agent):
        tc = _mock_tool_call(name="web_search", arguments="not valid json", call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch("run_agent.handle_function_call", return_value="ok"):
            agent._execute_tool_calls(mock_msg, messages, "task-1")
        assert len(messages) == 1

    def test_result_truncation_over_100k(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        big_result = "x" * 150_000
        with patch("run_agent.handle_function_call", return_value=big_result):
            agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Content should be truncated
        assert len(messages[0]["content"]) < 150_000
        assert "Truncated" in messages[0]["content"]


class TestHandleMaxIterations:
    def test_returns_summary(self, agent):
        resp = _mock_response(content="Here is a summary of what I did.")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        result = agent._handle_max_iterations(messages, 60)
        assert "summary" in result.lower()

    def test_api_failure_returns_error(self, agent):
        agent.client.chat.completions.create.side_effect = Exception("API down")
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        result = agent._handle_max_iterations(messages, 60)
        assert "Error" in result or "error" in result


class TestRunConversation:
    """Tests for the main run_conversation method.

    Each test mocks client.chat.completions.create to return controlled
    responses, exercising different code paths without real API calls.
    """

    def _setup_agent(self, agent):
        """Common setup for run_conversation tests."""
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False

    def test_stop_finish_reason_returns_response(self, agent):
        self._setup_agent(agent)
        resp = _mock_response(content="Final answer", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")
        assert result["final_response"] == "Final answer"
        assert result["completed"] is True

    def test_tool_calls_then_stop(self, agent):
        self._setup_agent(agent)
        tc = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]
        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")
        assert result["final_response"] == "Done searching"
        assert result["api_calls"] == 2

    def test_interrupt_breaks_loop(self, agent):
        self._setup_agent(agent)

        def interrupt_side_effect(api_kwargs):
            agent._interrupt_requested = True
            raise InterruptedError("Agent interrupted during API call")

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent._set_interrupt"),
            patch.object(agent, "_interruptible_api_call", side_effect=interrupt_side_effect),
        ):
            result = agent.run_conversation("hello")
        assert result["interrupted"] is True

    def test_invalid_tool_name_retry(self, agent):
        """Model hallucinates an invalid tool name, agent retries and succeeds."""
        self._setup_agent(agent)
        bad_tc = _mock_tool_call(name="nonexistent_tool", arguments='{}', call_id="c1")
        resp_bad = _mock_response(content="", finish_reason="tool_calls", tool_calls=[bad_tc])
        resp_good = _mock_response(content="Got it", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp_bad, resp_good]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("do something")
        assert result["final_response"] == "Got it"

    def test_empty_content_retry_and_fallback(self, agent):
        """Empty content (only think block) retries, then falls back to partial."""
        self._setup_agent(agent)
        empty_resp = _mock_response(
            content="<think>internal reasoning</think>",
            finish_reason="stop",
        )
        # Return empty 3 times to exhaust retries
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp,
        ]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        # After 3 retries with no real content, should return partial
        assert result["completed"] is False
        assert result.get("partial") is True

    def test_context_compression_triggered(self, agent):
        """When compressor says should_compress, compression runs."""
        self._setup_agent(agent)
        agent.compression_enabled = True

        tc = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="All done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        with (
            patch("run_agent.handle_function_call", return_value="result"),
            patch.object(agent.context_compressor, "should_compress", return_value=True),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # _compress_context should return (messages, system_prompt)
            mock_compress.return_value = (
                [{"role": "user", "content": "search something"}],
                "compressed system prompt",
            )
            result = agent.run_conversation("search something")
        mock_compress.assert_called_once()
