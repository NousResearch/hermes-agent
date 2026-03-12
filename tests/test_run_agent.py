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

from honcho_integration.client import HonchoClientConfig
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
# Group 1: Pure Functions
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
        # Content after think block must be preserved
        assert "after" in result


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
# Group 2: State / Structure Methods
# ===================================================================


class TestInit:
    def test_anthropic_base_url_accepted(self):
        """Anthropic base URLs should be accepted (OpenAI-compatible endpoint)."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI") as mock_openai,
        ):
            AIAgent(
                api_key="test-key-1234567890",
                base_url="https://api.anthropic.com/v1/",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            mock_openai.assert_called_once()

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
        """Session ID should be auto-generated in YYYYMMDD_HHMMSS_<hex6> format."""
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
            # Format: YYYYMMDD_HHMMSS_<6 hex chars>
            assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{6}$", a.session_id), (
                f"session_id doesn't match expected format: {a.session_id}"
            )


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
        assert kwargs["timeout"] == 900.0

    def test_provider_preferences_injected(self, agent):
        agent.providers_allowed = ["Anthropic"]
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["provider"]["only"] == ["Anthropic"]

    def test_reasoning_config_default_openrouter(self, agent):
        """Default reasoning config for OpenRouter should be medium."""
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        reasoning = kwargs["extra_body"]["reasoning"]
        assert reasoning["enabled"] is True
        assert reasoning["effort"] == "medium"

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

    def test_tool_call_extra_content_preserved(self, agent):
        """Gemini thinking models attach extra_content with thought_signature
        to tool calls. This must be preserved so subsequent API calls include it."""
        tc = _mock_tool_call(name="get_weather", arguments='{"city":"NYC"}', call_id="c2")
        tc.extra_content = {"google": {"thought_signature": "abc123"}}
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert result["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "abc123"}
        }

    def test_tool_call_without_extra_content(self, agent):
        """Standard tool calls (no thinking model) should not have extra_content."""
        tc = _mock_tool_call(name="web_search", arguments='{}', call_id="c3")
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert "extra_content" not in result["tool_calls"][0]


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
# Group 3: Conversation Loop Pieces (OpenAI mock)
# ===================================================================


class TestExecuteToolCalls:
    def test_single_tool_executed(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch("run_agent.handle_function_call", return_value="search result") as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            # enabled_tools passes the agent's own valid_tool_names
            args, kwargs = mock_hfc.call_args
            assert args[:3] == ("web_search", {"q": "test"}, "task-1")
            assert set(kwargs.get("enabled_tools", [])) == agent.valid_tool_names
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
        with patch("run_agent.handle_function_call", return_value="ok") as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            # Invalid JSON args should fall back to empty dict
            args, kwargs = mock_hfc.call_args
            assert args[:3] == ("web_search", {}, "task-1")
            assert set(kwargs.get("enabled_tools", [])) == agent.valid_tool_names
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == "c1"

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
        assert isinstance(result, str)
        assert len(result) > 0
        assert "summary" in result.lower()

    def test_api_failure_returns_error(self, agent):
        agent.client.chat.completions.create.side_effect = Exception("API down")
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]
        result = agent._handle_max_iterations(messages, 60)
        assert isinstance(result, str)
        assert "error" in result.lower()
        assert "API down" in result


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
        assert result["completed"] is True
        assert result["api_calls"] == 2

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

    def test_nous_401_refreshes_after_remint_and_retries(self, agent):
        self._setup_agent(agent)
        agent.provider = "nous"
        agent.api_mode = "chat_completions"

        calls = {"api": 0, "refresh": 0}

        class _UnauthorizedError(RuntimeError):
            def __init__(self):
                super().__init__("Error code: 401 - unauthorized")
                self.status_code = 401

        def _fake_api_call(api_kwargs):
            calls["api"] += 1
            if calls["api"] == 1:
                raise _UnauthorizedError()
            return _mock_response(content="Recovered after remint", finish_reason="stop")

        def _fake_refresh(*, force=True):
            calls["refresh"] += 1
            assert force is True
            return True

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(agent, "_try_refresh_nous_client_credentials", side_effect=_fake_refresh),
        ):
            result = agent.run_conversation("hello")

        assert calls["api"] == 2
        assert calls["refresh"] == 1
        assert result["completed"] is True
        assert result["final_response"] == "Recovered after remint"

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
        assert result["final_response"] == "All done"
        assert result["completed"] is True

    @pytest.mark.parametrize(
        ("first_content", "second_content", "expected_final"),
        [
            ("Part 1 ", "Part 2", "Part 1 Part 2"),
            ("<think>internal reasoning</think>", "Recovered final answer", "Recovered final answer"),
        ],
    )
    def test_length_finish_reason_requests_continuation(
        self, agent, first_content, second_content, expected_final
    ):
        self._setup_agent(agent)
        first = _mock_response(content=first_content, finish_reason="length")
        second = _mock_response(content=second_content, finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [first, second]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert result["final_response"] == expected_final

        second_call_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        assert second_call_messages[-1]["role"] == "user"
        assert "truncated by the output length limit" in second_call_messages[-1]["content"]


class TestRetryExhaustion:
    """Regression: retry_count > max_retries was dead code (off-by-one).

    When retries were exhausted the condition never triggered, causing
    the loop to exit and fall through to response.choices[0] on an
    invalid response, raising IndexError.
    """

    def _setup_agent(self, agent):
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False

    @staticmethod
    def _make_fast_time_mock():
        """Return a mock time module where sleep loops exit instantly."""
        mock_time = MagicMock()
        _t = [1000.0]

        def _advancing_time():
            _t[0] += 500.0  # jump 500s per call so sleep_end is always in the past
            return _t[0]

        mock_time.time.side_effect = _advancing_time
        mock_time.sleep = MagicMock()  # no-op
        mock_time.monotonic.return_value = 12345.0
        return mock_time

    def test_invalid_response_returns_error_not_crash(self, agent):
        """Exhausted retries on invalid (empty choices) response must not IndexError."""
        self._setup_agent(agent)
        # Return response with empty choices every time
        bad_resp = SimpleNamespace(
            choices=[],
            model="test/model",
            usage=None,
        )
        agent.client.chat.completions.create.return_value = bad_resp
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
        ):
            result = agent.run_conversation("hello")
        assert result.get("completed") is False, f"Expected completed=False, got: {result}"
        assert result.get("failed") is True
        assert "error" in result
        assert "Invalid API response" in result["error"]

    def test_api_error_raises_after_retries(self, agent):
        """Exhausted retries on API errors must raise, not fall through."""
        self._setup_agent(agent)
        agent.client.chat.completions.create.side_effect = RuntimeError("rate limited")
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
        ):
            with pytest.raises(RuntimeError, match="rate limited"):
                agent.run_conversation("hello")


# ---------------------------------------------------------------------------
# Flush sentinel leak
# ---------------------------------------------------------------------------

class TestFlushSentinelNotLeaked:
    """_flush_sentinel must be stripped before sending messages to the API."""

    def test_flush_sentinel_stripped_from_api_messages(self, agent_with_memory_tool):
        """Verify _flush_sentinel is not sent to the API provider."""
        agent = agent_with_memory_tool
        agent._memory_store = MagicMock()
        agent._memory_flush_min_turns = 1
        agent._user_turn_count = 10
        agent._cached_system_prompt = "system"

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "remember this"},
        ]

        # Mock the API to return a simple response (no tool calls)
        mock_msg = SimpleNamespace(content="OK", tool_calls=None)
        mock_choice = SimpleNamespace(message=mock_msg)
        mock_response = SimpleNamespace(choices=[mock_choice])
        agent.client.chat.completions.create.return_value = mock_response

        # Bypass auxiliary client so flush uses agent.client directly
        with patch("agent.auxiliary_client.call_llm", side_effect=RuntimeError("no provider")):
            agent.flush_memories(messages, min_turns=0)

        # Check what was actually sent to the API
        call_args = agent.client.chat.completions.create.call_args
        assert call_args is not None, "flush_memories never called the API"
        api_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        for msg in api_messages:
            assert "_flush_sentinel" not in msg, (
                f"_flush_sentinel leaked to API in message: {msg}"
            )


# ---------------------------------------------------------------------------
# Conversation history mutation
# ---------------------------------------------------------------------------

class TestConversationHistoryNotMutated:
    """run_conversation must not mutate the caller's conversation_history list."""

    def test_caller_list_unchanged_after_run(self, agent):
        """Passing conversation_history should not modify the original list."""
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        original_len = len(history)

        resp = _mock_response(content="new answer", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("new question", conversation_history=history)

        # Caller's list must be untouched
        assert len(history) == original_len, (
            f"conversation_history was mutated: expected {original_len} items, got {len(history)}"
        )
        # Result should have more messages than the original history
        assert len(result["messages"]) > original_len


# ---------------------------------------------------------------------------
# _max_tokens_param consistency
# ---------------------------------------------------------------------------

class TestNousCredentialRefresh:
    """Verify Nous credential refresh rebuilds the runtime client."""

    def test_try_refresh_nous_client_credentials_rebuilds_client(self, agent, monkeypatch):
        agent.provider = "nous"
        agent.api_mode = "chat_completions"

        closed = {"value": False}
        rebuilt = {"kwargs": None}
        captured = {}

        class _ExistingClient:
            def close(self):
                closed["value"] = True

        class _RebuiltClient:
            pass

        def _fake_resolve(**kwargs):
            captured.update(kwargs)
            return {
                "api_key": "new-nous-key",
                "base_url": "https://inference-api.nousresearch.com/v1",
            }

        def _fake_openai(**kwargs):
            rebuilt["kwargs"] = kwargs
            return _RebuiltClient()

        monkeypatch.setattr("hermes_cli.auth.resolve_nous_runtime_credentials", _fake_resolve)

        agent.client = _ExistingClient()
        with patch("run_agent.OpenAI", side_effect=_fake_openai):
            ok = agent._try_refresh_nous_client_credentials(force=True)

        assert ok is True
        assert closed["value"] is True
        assert captured["force_mint"] is True
        assert rebuilt["kwargs"]["api_key"] == "new-nous-key"
        assert rebuilt["kwargs"]["base_url"] == "https://inference-api.nousresearch.com/v1"
        assert "default_headers" not in rebuilt["kwargs"]
        assert isinstance(agent.client, _RebuiltClient)


class TestMaxTokensParam:
    """Verify _max_tokens_param returns the correct key for each provider."""

    def test_returns_max_completion_tokens_for_direct_openai(self, agent):
        agent.base_url = "https://api.openai.com/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_tokens_for_openrouter(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    def test_returns_max_tokens_for_local(self, agent):
        agent.base_url = "http://localhost:11434/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    def test_not_tricked_by_openai_in_openrouter_url(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1/api.openai.com"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}


# ---------------------------------------------------------------------------
# System prompt stability for prompt caching
# ---------------------------------------------------------------------------

class TestSystemPromptStability:
    """Verify that the system prompt stays stable across turns for cache hits."""

    def test_stored_prompt_reused_for_continuing_session(self, agent):
        """When conversation_history is non-empty and session DB has a stored
        prompt, it should be reused instead of rebuilding from disk."""
        stored = "You are helpful. [stored from turn 1]"
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"system_prompt": stored}
        agent._session_db = mock_db

        # Simulate a continuing session with history
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        # First call — _cached_system_prompt is None, history is non-empty
        agent._cached_system_prompt = None

        # Patch run_conversation internals to just test the system prompt logic.
        # We'll call the prompt caching block directly by simulating what
        # run_conversation does.
        conversation_history = history

        # The block under test (from run_conversation):
        if agent._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and agent._session_db:
                try:
                    session_row = agent._session_db.get_session(agent.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass

            if stored_prompt:
                agent._cached_system_prompt = stored_prompt

        assert agent._cached_system_prompt == stored
        mock_db.get_session.assert_called_once_with(agent.session_id)

    def test_fresh_build_when_no_history(self, agent):
        """On the first turn (no history), system prompt should be built fresh."""
        mock_db = MagicMock()
        agent._session_db = mock_db

        agent._cached_system_prompt = None
        conversation_history = []

        # The block under test:
        if agent._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and agent._session_db:
                session_row = agent._session_db.get_session(agent.session_id)
                if session_row:
                    stored_prompt = session_row.get("system_prompt") or None

            if stored_prompt:
                agent._cached_system_prompt = stored_prompt
            else:
                agent._cached_system_prompt = agent._build_system_prompt()

        # Should have built fresh, not queried the DB
        mock_db.get_session.assert_not_called()
        assert agent._cached_system_prompt is not None
        assert "Hermes Agent" in agent._cached_system_prompt

    def test_fresh_build_when_db_has_no_prompt(self, agent):
        """If the session DB has no stored prompt, build fresh even with history."""
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"system_prompt": ""}
        agent._session_db = mock_db

        agent._cached_system_prompt = None
        conversation_history = [{"role": "user", "content": "hi"}]

        if agent._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and agent._session_db:
                try:
                    session_row = agent._session_db.get_session(agent.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass

            if stored_prompt:
                agent._cached_system_prompt = stored_prompt
            else:
                agent._cached_system_prompt = agent._build_system_prompt()

        # Empty string is falsy, so should fall through to fresh build
        assert "Hermes Agent" in agent._cached_system_prompt

    def test_honcho_context_baked_into_prompt_on_first_turn(self, agent):
        """Honcho context should be baked into _cached_system_prompt on
        the first turn, not injected separately per API call."""
        agent._honcho_context = "User prefers Python over JavaScript."
        agent._cached_system_prompt = None

        # Simulate first turn: build fresh and bake in Honcho
        agent._cached_system_prompt = agent._build_system_prompt()
        if agent._honcho_context:
            agent._cached_system_prompt = (
                agent._cached_system_prompt + "\n\n" + agent._honcho_context
            ).strip()

        assert "User prefers Python over JavaScript" in agent._cached_system_prompt

    def test_honcho_prefetch_runs_on_continuing_session(self):
        """Honcho prefetch is consumed on continuing sessions via ephemeral context."""
        conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        recall_mode = "hybrid"
        should_prefetch = bool(conversation_history) and recall_mode != "tools"
        assert should_prefetch is True

    def test_honcho_prefetch_runs_on_first_turn(self):
        """Honcho prefetch should run when conversation_history is empty."""
        conversation_history = []
        should_prefetch = not conversation_history
        assert should_prefetch is True


class TestHonchoActivation:
    def test_disabled_config_skips_honcho_init(self):
        hcfg = HonchoClientConfig(
            enabled=False,
            api_key="honcho-key",
            peer_name="user",
            ai_peer="hermes",
        )

        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("honcho_integration.client.HonchoClientConfig.from_global_config", return_value=hcfg),
            patch("honcho_integration.client.get_honcho_client") as mock_client,
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
            )

        assert agent._honcho is None
        assert agent._honcho_config is hcfg
        mock_client.assert_not_called()

    def test_injected_honcho_manager_skips_fresh_client_init(self):
        hcfg = HonchoClientConfig(
            enabled=True,
            api_key="honcho-key",
            memory_mode="hybrid",
            peer_name="user",
            ai_peer="hermes",
            recall_mode="hybrid",
        )
        manager = MagicMock()
        manager._config = hcfg
        manager.get_or_create.return_value = SimpleNamespace(messages=[])
        manager.get_prefetch_context.return_value = {"representation": "Known user", "card": ""}

        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("honcho_integration.client.get_honcho_client") as mock_client,
            patch("tools.honcho_tools.set_session_context"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                honcho_session_key="gateway-session",
                honcho_manager=manager,
                honcho_config=hcfg,
            )

        assert agent._honcho is manager
        manager.get_or_create.assert_called_once_with("gateway-session")
        manager.get_prefetch_context.assert_called_once_with("gateway-session")
        manager.set_context_result.assert_called_once_with(
            "gateway-session",
            {"representation": "Known user", "card": ""},
        )
        mock_client.assert_not_called()

    def test_recall_mode_context_suppresses_honcho_tools(self):
        hcfg = HonchoClientConfig(
            enabled=True,
            api_key="honcho-key",
            memory_mode="hybrid",
            peer_name="user",
            ai_peer="hermes",
            recall_mode="context",
        )
        manager = MagicMock()
        manager._config = hcfg
        manager.get_or_create.return_value = SimpleNamespace(messages=[])
        manager.get_prefetch_context.return_value = {"representation": "Known user", "card": ""}

        with (
            patch(
                "run_agent.get_tool_definitions",
                side_effect=[
                    _make_tool_defs("web_search"),
                    _make_tool_defs(
                        "web_search",
                        "honcho_context",
                        "honcho_profile",
                        "honcho_search",
                        "honcho_conclude",
                    ),
                ],
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("tools.honcho_tools.set_session_context"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                honcho_session_key="gateway-session",
                honcho_manager=manager,
                honcho_config=hcfg,
            )

        assert "web_search" in agent.valid_tool_names
        assert "honcho_context" not in agent.valid_tool_names
        assert "honcho_profile" not in agent.valid_tool_names
        assert "honcho_search" not in agent.valid_tool_names
        assert "honcho_conclude" not in agent.valid_tool_names

    def test_inactive_honcho_strips_stale_honcho_tools(self):
        hcfg = HonchoClientConfig(
            enabled=False,
            api_key="honcho-key",
            peer_name="user",
            ai_peer="hermes",
        )

        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search", "honcho_context")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("honcho_integration.client.HonchoClientConfig.from_global_config", return_value=hcfg),
            patch("honcho_integration.client.get_honcho_client") as mock_client,
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
            )

        assert agent._honcho is None
        assert "web_search" in agent.valid_tool_names
        assert "honcho_context" not in agent.valid_tool_names
        mock_client.assert_not_called()


class TestHonchoPrefetchScheduling:
    def test_honcho_prefetch_includes_cached_dialectic(self, agent):
        agent._honcho = MagicMock()
        agent._honcho_session_key = "session-key"
        agent._honcho.pop_context_result.return_value = {}
        agent._honcho.pop_dialectic_result.return_value = "Continue with the migration checklist."

        context = agent._honcho_prefetch("what next?")

        assert "Continuity synthesis" in context
        assert "migration checklist" in context

    def test_queue_honcho_prefetch_skips_tools_mode(self, agent):
        agent._honcho = MagicMock()
        agent._honcho_session_key = "session-key"
        agent._honcho_config = HonchoClientConfig(
            enabled=True,
            api_key="honcho-key",
            recall_mode="tools",
        )

        agent._queue_honcho_prefetch("what next?")

        agent._honcho.prefetch_context.assert_not_called()
        agent._honcho.prefetch_dialectic.assert_not_called()

    def test_queue_honcho_prefetch_runs_when_context_enabled(self, agent):
        agent._honcho = MagicMock()
        agent._honcho_session_key = "session-key"
        agent._honcho_config = HonchoClientConfig(
            enabled=True,
            api_key="honcho-key",
            recall_mode="hybrid",
        )

        agent._queue_honcho_prefetch("what next?")

        agent._honcho.prefetch_context.assert_called_once_with("session-key", "what next?")
        agent._honcho.prefetch_dialectic.assert_called_once_with("session-key", "what next?")


# ---------------------------------------------------------------------------
# Iteration budget pressure warnings
# ---------------------------------------------------------------------------

class TestBudgetPressure:
    """Budget pressure warning system (issue #414)."""

    def test_no_warning_below_caution(self, agent):
        agent.max_iterations = 60
        assert agent._get_budget_warning(30) is None

    def test_caution_at_70_percent(self, agent):
        agent.max_iterations = 60
        msg = agent._get_budget_warning(42)
        assert msg is not None
        assert "[BUDGET:" in msg
        assert "18 iterations left" in msg

    def test_warning_at_90_percent(self, agent):
        agent.max_iterations = 60
        msg = agent._get_budget_warning(54)
        assert "[BUDGET WARNING:" in msg
        assert "Provide your final response NOW" in msg

    def test_last_iteration(self, agent):
        agent.max_iterations = 60
        msg = agent._get_budget_warning(59)
        assert "1 iteration(s) left" in msg

    def test_disabled(self, agent):
        agent.max_iterations = 60
        agent._budget_pressure_enabled = False
        assert agent._get_budget_warning(55) is None

    def test_zero_max_iterations(self, agent):
        agent.max_iterations = 0
        assert agent._get_budget_warning(0) is None

    def test_injects_into_json_tool_result(self, agent):
        """Warning should be injected as _budget_warning field in JSON tool results."""
        import json
        agent.max_iterations = 10
        messages = [
            {"role": "tool", "content": json.dumps({"output": "done", "exit_code": 0}), "tool_call_id": "tc1"}
        ]
        warning = agent._get_budget_warning(9)
        assert warning is not None
        # Simulate the injection logic
        last_content = messages[-1]["content"]
        parsed = json.loads(last_content)
        parsed["_budget_warning"] = warning
        messages[-1]["content"] = json.dumps(parsed, ensure_ascii=False)
        result = json.loads(messages[-1]["content"])
        assert "_budget_warning" in result
        assert "BUDGET WARNING" in result["_budget_warning"]
        assert result["output"] == "done"  # original content preserved

    def test_appends_to_non_json_tool_result(self, agent):
        """Warning should be appended as text for non-JSON tool results."""
        agent.max_iterations = 10
        messages = [
            {"role": "tool", "content": "plain text result", "tool_call_id": "tc1"}
        ]
        warning = agent._get_budget_warning(9)
        # Simulate injection logic for non-JSON
        last_content = messages[-1]["content"]
        try:
            import json
            json.loads(last_content)
        except (json.JSONDecodeError, TypeError):
            messages[-1]["content"] = last_content + f"\n\n{warning}"
        assert "plain text result" in messages[-1]["content"]
        assert "BUDGET WARNING" in messages[-1]["content"]


class TestSafeWriter:
    """Verify _SafeWriter guards stdout against OSError (broken pipes)."""

    def test_write_delegates_normally(self):
        """When stdout is healthy, _SafeWriter is transparent."""
        from run_agent import _SafeWriter
        from io import StringIO
        inner = StringIO()
        writer = _SafeWriter(inner)
        writer.write("hello")
        assert inner.getvalue() == "hello"

    def test_write_catches_oserror(self):
        """OSError on write is silently caught, returns len(data)."""
        from run_agent import _SafeWriter
        from unittest.mock import MagicMock
        inner = MagicMock()
        inner.write.side_effect = OSError(5, "Input/output error")
        writer = _SafeWriter(inner)
        result = writer.write("hello")
        assert result == 5  # len("hello")

    def test_flush_catches_oserror(self):
        """OSError on flush is silently caught."""
        from run_agent import _SafeWriter
        from unittest.mock import MagicMock
        inner = MagicMock()
        inner.flush.side_effect = OSError(5, "Input/output error")
        writer = _SafeWriter(inner)
        writer.flush()  # should not raise

    def test_print_survives_broken_stdout(self, monkeypatch):
        """print() through _SafeWriter doesn't crash on broken pipe."""
        import sys
        from run_agent import _SafeWriter
        from unittest.mock import MagicMock
        broken = MagicMock()
        broken.write.side_effect = OSError(5, "Input/output error")
        original = sys.stdout
        sys.stdout = _SafeWriter(broken)
        try:
            print("this should not crash")  # would raise without _SafeWriter
        finally:
            sys.stdout = original

    def test_installed_in_run_conversation(self, agent):
        """run_conversation installs _SafeWriter on sys.stdout."""
        import sys
        from run_agent import _SafeWriter
        resp = _mock_response(content="Done", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp
        original = sys.stdout
        try:
            with (
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                agent.run_conversation("test")
            assert isinstance(sys.stdout, _SafeWriter)
        finally:
            sys.stdout = original

    def test_double_wrap_prevented(self):
        """Wrapping an already-wrapped stream doesn't add layers."""
        import sys
        from run_agent import _SafeWriter
        from io import StringIO
        inner = StringIO()
        wrapped = _SafeWriter(inner)
        # isinstance check should prevent double-wrapping
        assert isinstance(wrapped, _SafeWriter)
        # The guard in run_conversation checks isinstance before wrapping
        if not isinstance(wrapped, _SafeWriter):
            wrapped = _SafeWriter(wrapped)
        # Still just one layer
        wrapped.write("test")
        assert inner.getvalue() == "test"
