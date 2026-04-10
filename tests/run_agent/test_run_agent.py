"""Unit tests for run_agent.py (AIAgent).

Tests cover pure functions, state/structure methods, and conversation loop
pieces. The OpenAI client and tool loading are mocked so no network calls
are made.
"""

import io
import json
import logging
import re
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import run_agent
from run_agent import AIAgent
from agent.error_classifier import FailoverReason
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY


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
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
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
        patch(
            "run_agent.get_tool_definitions",
            return_value=_make_tool_defs("web_search", "memory"),
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-k...7890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def test_aiagent_reuses_existing_errors_log_handler():
    """Repeated AIAgent init should not accumulate duplicate errors.log handlers."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    error_log_path = (run_agent._hermes_home / "logs" / "errors.log").resolve()

    try:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        preexisting_handler = RotatingFileHandler(
            error_log_path,
            maxBytes=2 * 1024 * 1024,
            backupCount=2,
        )
        root_logger.addHandler(preexisting_handler)

        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            AIAgent(
                api_key="test-k...7890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            AIAgent(
                api_key="test-k...7890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        matching_handlers = [
            handler for handler in root_logger.handlers
            if isinstance(handler, RotatingFileHandler)
            and error_log_path == Path(handler.baseFilename).resolve()
        ]
        assert len(matching_handlers) == 1
    finally:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            if handler not in original_handlers:
                handler.close()
        for handler in original_handlers:
            root_logger.addHandler(handler)


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


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    """Return a SimpleNamespace mimicking a tool call object."""
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(
    content="Hello",
    finish_reason="stop",
    tool_calls=None,
    reasoning=None,
    reasoning_content=None,
    reasoning_details=None,
    usage=None,
):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletion response."""
    msg = _mock_assistant_msg(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
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
        assert (
            agent._has_content_after_think_block("<think>r</think> actual answer")
            is True
        )

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

    def test_orphaned_closing_think_tag(self, agent):
        result = agent._strip_think_blocks("some reasoning</think>actual answer")
        assert "</think>" not in result
        assert "actual answer" in result

    def test_orphaned_closing_thinking_tag(self, agent):
        result = agent._strip_think_blocks("reasoning</thinking>answer")
        assert "</thinking>" not in result
        assert "answer" in result

    def test_orphaned_opening_think_tag(self, agent):
        result = agent._strip_think_blocks("<think>orphaned reasoning without close")
        assert "<think>" not in result

    def test_mixed_orphaned_and_paired_tags(self, agent):
        text = "stray</think><think>paired reasoning</think> visible"
        result = agent._strip_think_blocks(text)
        assert "</think>" not in result
        assert "<think>" not in result
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

    @pytest.mark.parametrize(
        ("content", "expected"),
        [
            ("<think>thinking hard</think>", "thinking hard"),
            ("<thinking>step by step</thinking>", "step by step"),
            (
                "<REASONING_SCRATCHPAD>scratch analysis</REASONING_SCRATCHPAD>",
                "scratch analysis",
            ),
        ],
    )
    def test_inline_reasoning_blocks_fallback(self, agent, content, expected):
        msg = _mock_assistant_msg(content=content)
        assert agent._extract_reasoning(msg) == expected


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
        """Anthropic base URLs should route to native Anthropic client."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter._anthropic_sdk") as mock_anthropic,
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://api.anthropic.com/v1/",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert agent.api_mode == "anthropic_messages"
            mock_anthropic.Anthropic.assert_called_once()

    def test_prompt_caching_claude_openrouter(self):
        """Claude model via OpenRouter should enable prompt caching."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            a = AIAgent(
                api_key="test-k...7890",
                model="anthropic/claude-sonnet-4-20250514",
                base_url="https://openrouter.ai/api/v1",
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

    def test_prompt_caching_native_anthropic(self):
        """Native Anthropic provider should enable prompt caching."""
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter._anthropic_sdk"),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://api.anthropic.com/v1/",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            assert a.api_mode == "anthropic_messages"
            assert a._use_prompt_caching is True

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
            {
                "role": "tool",
                "content": json.dumps({"todos": todos}),
                "tool_call_id": "c1",
            },
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert agent._todo_store.has_items()

    def test_skips_non_todo_tools(self, agent):
        history = [
            {
                "role": "tool",
                "content": '{"result": "search done"}',
                "tool_call_id": "c1",
            },
        ]
        with patch("run_agent._set_interrupt"):
            agent._hydrate_todo_store(history)
        assert not agent._todo_store.has_items()

    def test_invalid_json_skipped(self, agent):
        history = [
            {
                "role": "tool",
                "content": 'not valid json "todos" oops',
                "tool_call_id": "c1",
            },
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

    def test_includes_nous_subscription_prompt(self, agent, monkeypatch):
        monkeypatch.setattr(run_agent, "build_nous_subscription_prompt", lambda tool_names: "NOUS SUBSCRIPTION BLOCK")
        prompt = agent._build_system_prompt()
        assert "NOUS SUBSCRIPTION BLOCK" in prompt

    def test_skills_prompt_derives_available_toolsets_from_loaded_tools(self):
        tools = _make_tool_defs("web_search", "skills_list", "skill_view", "skill_manage")
        toolset_map = {
            "web_search": "web",
            "skills_list": "skills",
            "skill_view": "skills",
            "skill_manage": "skills",
        }

        with (
            patch("run_agent.get_tool_definitions", return_value=tools),
            patch(
                "run_agent.check_toolset_requirements",
                side_effect=AssertionError("should not re-check toolset requirements"),
            ),
            patch("run_agent.get_toolset_for_tool", create=True, side_effect=toolset_map.get),
            patch("run_agent.build_skills_system_prompt", return_value="SKILLS_PROMPT") as mock_skills,
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-k...7890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

            prompt = agent._build_system_prompt()

        assert "SKILLS_PROMPT" in prompt
        assert mock_skills.call_args.kwargs["available_tools"] == set(toolset_map)
        assert mock_skills.call_args.kwargs["available_toolsets"] == {"web", "skills"}


class TestToolUseEnforcementConfig:
    """Tests for the agent.tool_use_enforcement config option."""

    def _make_agent(self, model="openai/gpt-4.1", tool_use_enforcement="auto"):
        """Create an agent with tools and a specific enforcement config."""
        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("terminal", "web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"tool_use_enforcement": tool_use_enforcement}},
            ),
        ):
            a = AIAgent(
                model=model,
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            a.client = MagicMock()
            return a

    def test_auto_injects_for_gpt(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/gpt-4.1", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_injects_for_codex(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/codex-mini", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_skips_for_claude(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-sonnet-4", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_true_forces_for_all_models(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-sonnet-4", tool_use_enforcement=True)
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_string_true_forces_for_all_models(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-sonnet-4", tool_use_enforcement="true")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_always_forces_for_all_models(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="deepseek/deepseek-r1", tool_use_enforcement="always")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_false_disables_for_gpt(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/gpt-4.1", tool_use_enforcement=False)
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_string_false_disables(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/gpt-4.1", tool_use_enforcement="off")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_custom_list_matches(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(
            model="deepseek/deepseek-r1",
            tool_use_enforcement=["deepseek", "gemini"],
        )
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_custom_list_no_match(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(
            model="anthropic/claude-sonnet-4",
            tool_use_enforcement=["deepseek", "gemini"],
        )
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_custom_list_case_insensitive(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(
            model="openai/GPT-4.1",
            tool_use_enforcement=["GPT", "Codex"],
        )
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_no_tools_never_injects(self):
        """Even with enforcement=true, no injection when agent has no tools."""
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"tool_use_enforcement": True}},
            ),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=[],
            )
            a.client = MagicMock()
            prompt = a._build_system_prompt()
            assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt


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
        assert kwargs["timeout"] == 1800.0

    def test_provider_preferences_injected(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.providers_allowed = ["Anthropic"]
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["provider"]["only"] == ["Anthropic"]

    def test_reasoning_config_default_openrouter(self, agent):
        """Default reasoning config for OpenRouter should be medium."""
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "anthropic/claude-sonnet-4-20250514"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        reasoning = kwargs["extra_body"]["reasoning"]
        assert reasoning["enabled"] is True
        assert reasoning["effort"] == "medium"

    def test_reasoning_config_custom(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "anthropic/claude-sonnet-4-20250514"
        agent.reasoning_config = {"enabled": False}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}

    def test_reasoning_not_sent_for_unsupported_openrouter_model(self, agent):
        agent.model = "minimax/minimax-m2.5"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning" not in kwargs.get("extra_body", {})

    def test_reasoning_sent_for_supported_openrouter_model(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "qwen/qwen3.5-plus-02-15"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"

    def test_reasoning_sent_for_nous_route(self, agent):
        agent.base_url = "https://inference-api.nousresearch.com/v1"
        agent.model = "minimax/minimax-m2.5"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"

    def test_reasoning_sent_for_copilot_gpt5(self, agent):
        agent.base_url = "https://api.githubcopilot.com"
        agent.model = "gpt-5.4"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"] == {"effort": "medium"}

    def test_reasoning_xhigh_normalized_for_copilot(self, agent):
        agent.base_url = "https://api.githubcopilot.com"
        agent.model = "gpt-5.4"
        agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}

    def test_reasoning_omitted_for_non_reasoning_copilot_model(self, agent):
        agent.base_url = "https://api.githubcopilot.com"
        agent.model = "gpt-4.1"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning" not in kwargs.get("extra_body", {})

    def test_max_tokens_injected(self, agent):
        agent.max_tokens = 4096
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["max_tokens"] == 4096

    def test_qwen_portal_formats_messages_and_metadata(self, agent):
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.session_id = "sess-123"
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Got it"},
            {"role": "user", "content": "hi"},
        ]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["metadata"]["sessionId"] == "sess-123"
        assert kwargs["extra_body"]["vl_high_resolution_images"] is True
        assert isinstance(kwargs["messages"][0]["content"], list)
        assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["messages"][2]["content"][0]["text"] == "hi"

    def test_qwen_portal_normalizes_bare_string_content_parts(self, agent):
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": ["hello", {"type": "text", "text": "world"}]},
        ]
        kwargs = agent._build_api_kwargs(messages)
        user_content = kwargs["messages"][1]["content"]
        assert user_content[0] == {"type": "text", "text": "hello"}
        assert user_content[1] == {"type": "text", "text": "world"}

    def test_qwen_portal_no_system_message(self, agent):
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        # Should not crash even without a system message
        assert kwargs["messages"][0]["content"][0]["text"] == "hi"
        assert "cache_control" not in kwargs["messages"][0]["content"][0]

    def test_qwen_portal_omits_max_tokens(self, agent):
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.max_tokens = 4096
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs


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
        tc = _mock_tool_call(
            name="get_weather", arguments='{"city":"NYC"}', call_id="c2"
        )
        tc.extra_content = {"google": {"thought_signature": "abc123"}}
        msg = _mock_assistant_msg(content="", tool_calls=[tc])
        result = agent._build_assistant_message(msg, "tool_calls")
        assert result["tool_calls"][0]["extra_content"] == {
            "google": {"thought_signature": "abc123"}
        }

    def test_tool_call_without_extra_content(self, agent):
        """Standard tool calls (no thinking model) should not have extra_content."""
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c3")
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
        with patch(
            "run_agent.handle_function_call", return_value="search result"
        ) as mock_hfc:
            agent._execute_tool_calls(mock_msg, messages, "task-1")
            # enabled_tools passes the agent's own valid_tool_names
            args, kwargs = mock_hfc.call_args
            assert args[:3] == ("web_search", {"q": "test"}, "task-1")
            assert set(kwargs.get("enabled_tools", [])) == agent.valid_tool_names
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert "search result" in messages[0]["content"]

    def test_interrupt_skips_remaining(self, agent):
        tc1 = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments="{}", call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("run_agent._set_interrupt"):
            agent.interrupt()

        agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Both calls should be skipped with cancellation messages
        assert len(messages) == 2
        assert (
            "cancelled" in messages[0]["content"].lower()
            or "interrupted" in messages[0]["content"].lower()
        )

    def test_invalid_json_args_defaults_empty(self, agent):
        tc = _mock_tool_call(
            name="web_search", arguments="not valid json", call_id="c1"
        )
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

    def test_result_truncation_over_100k(self, agent, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        big_result = "x" * 150_000
        with patch("run_agent.handle_function_call", return_value=big_result):
            agent._execute_tool_calls(mock_msg, messages, "task-1")
        # Content should be replaced with persisted-output or truncation
        assert len(messages[0]["content"]) < 150_000
        assert ("Truncated" in messages[0]["content"] or "<persisted-output>" in messages[0]["content"])

    def test_quiet_tool_output_suppressed_when_progress_callback_present(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        agent.tool_progress_callback = lambda *args, **kwargs: None

        with patch("run_agent.handle_function_call", return_value="search result"), \
             patch.object(agent, "_safe_print") as mock_print:
            agent._execute_tool_calls(mock_msg, messages, "task-1")

        mock_print.assert_not_called()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"

    def test_quiet_tool_output_prints_without_progress_callback(self, agent):
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        agent.tool_progress_callback = None

        with patch("run_agent.handle_function_call", return_value="search result"), \
             patch.object(agent, "_safe_print") as mock_print:
            agent._execute_tool_calls(mock_msg, messages, "task-1")

        mock_print.assert_called_once()
        assert "search" in str(mock_print.call_args.args[0]).lower()
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"

    def test_vprint_suppressed_in_parseable_quiet_mode(self, agent):
        agent.suppress_status_output = True

        with patch.object(agent, "_safe_print") as mock_print:
            agent._vprint("status line", force=True)
            agent._vprint("normal line")

        mock_print.assert_not_called()

    def test_run_conversation_suppresses_retry_noise_in_parseable_quiet_mode(self, agent):
        class _RateLimitError(Exception):
            status_code = 429

            def __str__(self):
                return "Error code: 429 - Rate limit exceeded."

        responses = [_RateLimitError(), _mock_response(content="Recovered")]

        def _fake_api_call(api_kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        agent.suppress_status_output = True
        agent._interruptible_api_call = _fake_api_call
        agent._persist_session = lambda *args, **kwargs: None
        agent._save_trajectory = lambda *args, **kwargs: None
        agent._save_session_log = lambda *args, **kwargs: None

        captured = io.StringIO()
        agent._print_fn = lambda *args, **kw: print(*args, file=captured, **kw)

        with patch("run_agent.time.sleep", return_value=None):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered"
        output = captured.getvalue()
        assert "API call failed" not in output
        assert "Rate limit reached" not in output


class TestConcurrentToolExecution:
    """Tests for _execute_tool_calls_concurrent and dispatch logic."""

    def test_single_tool_uses_sequential_path(self, agent):
        """Single tool call should use sequential path, not concurrent."""
        tc = _mock_tool_call(name="web_search", arguments='{"q":"test"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_clarify_forces_sequential(self, agent):
        """Batch containing clarify should use sequential path."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="clarify", arguments='{"question":"ok?"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_multiple_tools_uses_concurrent_path(self, agent):
        """Multiple read-only tools should use concurrent path."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="read_file", arguments='{"path":"x.py"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_con.assert_called_once()
                mock_seq.assert_not_called()

    def test_terminal_batch_forces_sequential(self, agent):
        """Stateful tools should not share the concurrent execution path."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="terminal", arguments='{"command":"pwd"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_write_batch_forces_sequential(self, agent):
        """File mutations should stay ordered within a turn."""
        tc1 = _mock_tool_call(name="read_file", arguments='{"path":"x.py"}', call_id="c1")
        tc2 = _mock_tool_call(name="write_file", arguments='{"path":"x.py","content":"print(1)"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_disjoint_write_batch_uses_concurrent_path(self, agent):
        """Independent file writes should still run concurrently."""
        tc1 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/a.py","content":"print(1)"}',
            call_id="c1",
        )
        tc2 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/b.py","content":"print(2)"}',
            call_id="c2",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_con.assert_called_once()
                mock_seq.assert_not_called()

    def test_overlapping_write_batch_forces_sequential(self, agent):
        """Writes to the same file must stay ordered."""
        tc1 = _mock_tool_call(
            name="write_file",
            arguments='{"path":"src/a.py","content":"print(1)"}',
            call_id="c1",
        )
        tc2 = _mock_tool_call(
            name="patch",
            arguments='{"path":"src/a.py","old_string":"1","new_string":"2"}',
            call_id="c2",
        )
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_malformed_json_args_forces_sequential(self, agent):
        """Unparseable tool arguments should fall back to sequential."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments="NOT JSON {{{", call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_non_dict_args_forces_sequential(self, agent):
        """Tool arguments that parse to a non-dict type should fall back to sequential."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='"just a string"', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        with patch.object(agent, "_execute_tool_calls_sequential") as mock_seq:
            with patch.object(agent, "_execute_tool_calls_concurrent") as mock_con:
                agent._execute_tool_calls(mock_msg, messages, "task-1")
                mock_seq.assert_called_once()
                mock_con.assert_not_called()

    def test_concurrent_executes_all_tools(self, agent):
        """Concurrent path should execute all tools and append results in order."""
        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"alpha"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"beta"}', call_id="c2")
        tc3 = _mock_tool_call(name="web_search", arguments='{"q":"gamma"}', call_id="c3")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2, tc3])
        messages = []

        call_log = []

        def fake_handle(name, args, task_id, **kwargs):
            call_log.append(name)
            return json.dumps({"result": args.get("q", "")})

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 3
        # Results must be in original order
        assert messages[0]["tool_call_id"] == "c1"
        assert messages[1]["tool_call_id"] == "c2"
        assert messages[2]["tool_call_id"] == "c3"
        # All should be tool messages
        assert all(m["role"] == "tool" for m in messages)
        # Content should contain the query results
        assert "alpha" in messages[0]["content"]
        assert "beta" in messages[1]["content"]
        assert "gamma" in messages[2]["content"]

    def test_concurrent_preserves_order_despite_timing(self, agent):
        """Even if tools finish in different order, messages should be in original order."""
        import time as _time

        tc1 = _mock_tool_call(name="web_search", arguments='{"q":"slow"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"q":"fast"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        def fake_handle(name, args, task_id, **kwargs):
            q = args.get("q", "")
            if q == "slow":
                _time.sleep(0.1)  # Slow tool
            return f"result_{q}"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert messages[0]["tool_call_id"] == "c1"
        assert "result_slow" in messages[0]["content"]
        assert messages[1]["tool_call_id"] == "c2"
        assert "result_fast" in messages[1]["content"]

    def test_concurrent_handles_tool_error(self, agent):
        """If one tool raises, others should still complete."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        call_count = [0]
        def fake_handle(name, args, task_id, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            return "success"

        with patch("run_agent.handle_function_call", side_effect=fake_handle):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 2
        # First tool should have error
        assert "Error" in messages[0]["content"] or "boom" in messages[0]["content"]
        # Second tool should succeed
        assert "success" in messages[1]["content"]

    def test_concurrent_interrupt_before_start(self, agent):
        """If interrupt is requested before concurrent execution, all tools are skipped."""
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="read_file", arguments='{}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []

        with patch("run_agent._set_interrupt"):
            agent.interrupt()

        agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")
        assert len(messages) == 2
        assert "cancelled" in messages[0]["content"].lower() or "skipped" in messages[0]["content"].lower()
        assert "cancelled" in messages[1]["content"].lower() or "skipped" in messages[1]["content"].lower()

    def test_concurrent_truncates_large_results(self, agent, tmp_path, monkeypatch):
        """Concurrent path should save oversized results to file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        (tmp_path / ".hermes").mkdir()
        tc1 = _mock_tool_call(name="web_search", arguments='{}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        big_result = "x" * 150_000

        with patch("run_agent.handle_function_call", return_value=big_result):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert len(messages) == 2
        for m in messages:
            assert len(m["content"]) < 150_000
            assert ("Truncated" in m["content"] or "<persisted-output>" in m["content"])

    def test_invoke_tool_dispatches_to_handle_function_call(self, agent):
        """_invoke_tool should route regular tools through handle_function_call."""
        with patch("run_agent.handle_function_call", return_value="result") as mock_hfc:
            result = agent._invoke_tool("web_search", {"q": "test"}, "task-1")
            mock_hfc.assert_called_once_with(
                "web_search", {"q": "test"}, "task-1",
                tool_call_id=None,
                session_id=agent.session_id,
                enabled_tools=list(agent.valid_tool_names),

            )
            assert result == "result"

    def test_sequential_tool_callbacks_fire_in_order(self, agent):
        tool_call = _mock_tool_call(name="web_search", arguments='{"query":"hello"}', call_id="c1")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tool_call])
        messages = []
        starts = []
        completes = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))

        with patch("run_agent.handle_function_call", return_value='{"success": true}'):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

        assert starts == [("c1", "web_search", {"query": "hello"})]
        assert completes == [("c1", "web_search", {"query": "hello"}, '{"success": true}')]

    def test_concurrent_tool_callbacks_fire_for_each_tool(self, agent):
        tc1 = _mock_tool_call(name="web_search", arguments='{"query":"one"}', call_id="c1")
        tc2 = _mock_tool_call(name="web_search", arguments='{"query":"two"}', call_id="c2")
        mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
        messages = []
        starts = []
        completes = []
        agent.tool_start_callback = lambda tool_call_id, function_name, function_args: starts.append((tool_call_id, function_name, function_args))
        agent.tool_complete_callback = lambda tool_call_id, function_name, function_args, function_result: completes.append((tool_call_id, function_name, function_args, function_result))

        with patch("run_agent.handle_function_call", side_effect=['{"id":1}', '{"id":2}']):
            agent._execute_tool_calls_concurrent(mock_msg, messages, "task-1")

        assert starts == [
            ("c1", "web_search", {"query": "one"}),
            ("c2", "web_search", {"query": "two"}),
        ]
        assert len(completes) == 2
        assert {entry[0] for entry in completes} == {"c1", "c2"}
        assert {entry[3] for entry in completes} == {'{"id":1}', '{"id":2}'}

    def test_invoke_tool_handles_agent_level_tools(self, agent):
        """_invoke_tool should handle todo tool directly."""
        with patch("tools.todo_tool.todo_tool", return_value='{"ok":true}') as mock_todo:
            result = agent._invoke_tool("todo", {"todos": []}, "task-1")
            mock_todo.assert_called_once()
        assert "ok" in result


class TestPathsOverlap:
    """Unit tests for the _paths_overlap helper."""

    def test_same_path_overlaps(self):
        from run_agent import _paths_overlap
        assert _paths_overlap(Path("src/a.py"), Path("src/a.py"))

    def test_siblings_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path("src/a.py"), Path("src/b.py"))

    def test_parent_child_overlap(self):
        from run_agent import _paths_overlap
        assert _paths_overlap(Path("src"), Path("src/sub/a.py"))

    def test_different_roots_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path("src/a.py"), Path("other/a.py"))

    def test_nested_vs_flat_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path("src/sub/a.py"), Path("src/a.py"))

    def test_empty_paths_do_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path(""), Path(""))

    def test_one_empty_path_does_not_overlap(self):
        from run_agent import _paths_overlap
        assert not _paths_overlap(Path(""), Path("src/a.py"))
        assert not _paths_overlap(Path("src/a.py"), Path(""))


class TestParallelScopePathNormalization:
    def test_extract_parallel_scope_path_normalizes_relative_to_cwd(self, tmp_path, monkeypatch):
        from run_agent import _extract_parallel_scope_path

        monkeypatch.chdir(tmp_path)

        scoped = _extract_parallel_scope_path("write_file", {"path": "./notes.txt"})

        assert scoped == tmp_path / "notes.txt"

    def test_extract_parallel_scope_path_treats_relative_and_absolute_same_file_as_same_scope(self, tmp_path, monkeypatch):
        from run_agent import _extract_parallel_scope_path, _paths_overlap

        monkeypatch.chdir(tmp_path)
        abs_path = tmp_path / "notes.txt"

        rel_scoped = _extract_parallel_scope_path("write_file", {"path": "notes.txt"})
        abs_scoped = _extract_parallel_scope_path("write_file", {"path": str(abs_path)})

        assert rel_scoped == abs_scoped
        assert _paths_overlap(rel_scoped, abs_scoped)

    def test_should_parallelize_tool_batch_rejects_same_file_with_mixed_path_spellings(self, tmp_path, monkeypatch):
        from run_agent import _should_parallelize_tool_batch

        monkeypatch.chdir(tmp_path)
        tc1 = _mock_tool_call(name="write_file", arguments='{"path":"notes.txt","content":"one"}', call_id="c1")
        tc2 = _mock_tool_call(name="write_file", arguments=f'{{"path":"{tmp_path / "notes.txt"}","content":"two"}}', call_id="c2")

        assert not _should_parallelize_tool_batch([tc1, tc2])


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

    def test_summary_skips_reasoning_for_unsupported_openrouter_model(self, agent):
        agent.model = "minimax/minimax-m2.5"
        resp = _mock_response(content="Summary")
        agent.client.chat.completions.create.return_value = resp
        agent._cached_system_prompt = "You are helpful."
        messages = [{"role": "user", "content": "do stuff"}]

        result = agent._handle_max_iterations(messages, 60)

        assert result == "Summary"
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert "reasoning" not in kwargs.get("extra_body", {})


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
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]
        with (
            patch("run_agent.handle_function_call", return_value="search result") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")
        assert result["final_response"] == "Done searching"
        assert result["api_calls"] == 2
        assert mock_handle_function_call.call_args.kwargs["tool_call_id"] == "c1"
        assert mock_handle_function_call.call_args.kwargs["session_id"] == agent.session_id

    def test_request_scoped_api_hooks_fire_for_each_api_call(self, agent):
        self._setup_agent(agent)
        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="Done searching", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        hook_calls = []

        def _record_hook(name, **kwargs):
            hook_calls.append((name, kwargs))
            return []

        with (
            patch("run_agent.handle_function_call", return_value="search result"),
            patch("hermes_cli.plugins.invoke_hook", side_effect=_record_hook),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("search something")

        assert result["final_response"] == "Done searching"
        pre_request_calls = [kw for name, kw in hook_calls if name == "pre_api_request"]
        post_request_calls = [kw for name, kw in hook_calls if name == "post_api_request"]
        assert len(pre_request_calls) == 2
        assert len(post_request_calls) == 2
        assert [call["api_call_count"] for call in pre_request_calls] == [1, 2]
        assert [call["api_call_count"] for call in post_request_calls] == [1, 2]
        assert all(call["session_id"] == agent.session_id for call in pre_request_calls)
        assert all("message_count" in c and "messages" not in c for c in pre_request_calls)
        assert all("usage" in c and "response" not in c for c in post_request_calls)

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
            patch.object(
                agent, "_interruptible_api_call", side_effect=interrupt_side_effect
            ),
        ):
            result = agent.run_conversation("hello")
        assert result["interrupted"] is True

    def test_invalid_tool_name_retry(self, agent):
        """Model hallucinates an invalid tool name, agent retries and succeeds."""
        self._setup_agent(agent)
        bad_tc = _mock_tool_call(name="nonexistent_tool", arguments="{}", call_id="c1")
        resp_bad = _mock_response(
            content="", finish_reason="tool_calls", tool_calls=[bad_tc]
        )
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

    def test_inline_think_blocks_reasoning_only_accepted(self, agent):
        """Inline <think> reasoning-only responses accepted with (empty) content, no retries."""
        self._setup_agent(agent)
        empty_resp = _mock_response(
            content="<think>internal reasoning</think>",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [empty_resp]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "(empty)"
        assert result["api_calls"] == 1  # no retries
        # Reasoning should be preserved in the assistant message
        assistant_msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
        assert any(m.get("reasoning") for m in assistant_msgs)

    def test_reasoning_only_local_resumed_no_compression_triggered(self, agent):
        """Reasoning-only responses no longer trigger compression — prefill then accepted."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        agent.compression_enabled = True
        empty_resp = _mock_response(
            content=None,
            finish_reason="stop",
            reasoning_content="reasoning only",
        )
        prefill = [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
        ]

        # 3 responses: original + 2 prefill continuations (structured reasoning triggers prefill)
        with (
            patch.object(agent, "_interruptible_api_call", side_effect=[empty_resp, empty_resp, empty_resp]),
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_not_called()  # no compression triggered
        assert result["completed"] is True
        assert result["final_response"] == "(empty)"
        assert result["api_calls"] == 3  # 1 original + 2 prefill continuations

    def test_reasoning_only_response_prefill_then_empty(self, agent):
        """Structured reasoning-only triggers prefill continuation (up to 2), then falls through to (empty)."""
        self._setup_agent(agent)
        empty_resp = _mock_response(
            content=None,
            finish_reason="stop",
            reasoning_content="structured reasoning answer",
        )
        # 3 responses: original + 2 prefill continuations, all reasoning-only
        agent.client.chat.completions.create.side_effect = [empty_resp, empty_resp, empty_resp]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "(empty)"
        assert result["api_calls"] == 3  # 1 original + 2 prefill continuations

    def test_reasoning_only_prefill_succeeds_on_continuation(self, agent):
        """When prefill continuation produces content, it becomes the final response."""
        self._setup_agent(agent)
        empty_resp = _mock_response(
            content=None,
            finish_reason="stop",
            reasoning_content="structured reasoning answer",
        )
        content_resp = _mock_response(
            content="Here is the actual answer.",
            finish_reason="stop",
        )
        agent.client.chat.completions.create.side_effect = [empty_resp, content_resp]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "Here is the actual answer."
        assert result["api_calls"] == 2  # 1 original + 1 prefill continuation
        # Prefill message should be cleaned up — no consecutive assistant messages
        roles = [m.get("role") for m in result["messages"]]
        for i in range(len(roles) - 1):
            if roles[i] == "assistant" and roles[i + 1] == "assistant":
                raise AssertionError("Consecutive assistant messages found in history")

    def test_truly_empty_response_retries_3_times_then_empty(self, agent):
        """Truly empty response (no content, no reasoning) retries 3 times then falls through to (empty)."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        empty_resp = _mock_response(content=None, finish_reason="stop")
        # 4 responses: 1 original + 3 nudge retries, all empty
        agent.client.chat.completions.create.side_effect = [
            empty_resp, empty_resp, empty_resp, empty_resp,
        ]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "(empty)"
        assert result["api_calls"] == 4  # 1 original + 3 retries

    def test_truly_empty_response_succeeds_on_nudge(self, agent):
        """Model produces content after being nudged for empty response."""
        self._setup_agent(agent)
        agent.base_url = "http://127.0.0.1:1234/v1"
        empty_resp = _mock_response(content=None, finish_reason="stop")
        content_resp = _mock_response(
            content="Here is the actual answer.",
            finish_reason="stop",
        )
        # 1 empty response, then model produces content on nudge
        agent.client.chat.completions.create.side_effect = [empty_resp, content_resp]
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("answer me")
        assert result["completed"] is True
        assert result["final_response"] == "Here is the actual answer."
        assert result["api_calls"] == 2  # 1 original + 1 nudge retry

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
            return _mock_response(
                content="Recovered after remint", finish_reason="stop"
            )

        def _fake_refresh(*, force=True):
            calls["refresh"] += 1
            assert force is True
            return True

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call),
            patch.object(
                agent, "_try_refresh_nous_client_credentials", side_effect=_fake_refresh
            ),
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

        tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
        resp1 = _mock_response(content="", finish_reason="tool_calls", tool_calls=[tc])
        resp2 = _mock_response(content="All done", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [resp1, resp2]

        with (
            patch("run_agent.handle_function_call", return_value="result"),
            patch.object(
                agent.context_compressor, "should_compress", return_value=True
            ),
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

    def test_glm_prompt_exceeds_max_length_triggers_compression(self, agent):
        """GLM/Z.AI uses 'Prompt exceeds max length' for context overflow."""
        self._setup_agent(agent)
        err_400 = Exception(
            "Error code: 400 - {'error': {'code': '1261', 'message': 'Prompt exceeds max length'}}"
        )
        err_400.status_code = 400
        ok_resp = _mock_response(content="Recovered after compression", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [err_400, ok_resp]
        prefill = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]

        with (
            patch.object(agent, "_compress_context") as mock_compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            mock_compress.return_value = (
                [{"role": "user", "content": "hello"}],
                "compressed system prompt",
            )
            result = agent.run_conversation("hello", conversation_history=prefill)

        mock_compress.assert_called_once()
        assert result["final_response"] == "Recovered after compression"
        assert result["completed"] is True

    def test_length_finish_reason_requests_continuation(self, agent):
        """Normal truncation (partial real content) triggers continuation."""
        self._setup_agent(agent)
        first = _mock_response(content="Part 1 ", finish_reason="length")
        second = _mock_response(content="Part 2", finish_reason="stop")
        agent.client.chat.completions.create.side_effect = [first, second]

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert result["final_response"] == "Part 1 Part 2"

        second_call_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
        assert second_call_messages[-1]["role"] == "user"
        assert "truncated by the output length limit" in second_call_messages[-1]["content"]

    def test_length_thinking_exhausted_skips_continuation(self, agent):
        """When finish_reason='length' but content is only thinking, skip retries."""
        self._setup_agent(agent)
        resp = _mock_response(
            content="<think>internal reasoning</think>",
            finish_reason="length",
        )
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        # Should return immediately — no continuation, only 1 API call
        assert result["completed"] is False
        assert result["api_calls"] == 1
        assert "reasoning" in result["error"].lower()
        assert "output tokens" in result["error"].lower()
        # Should have a user-friendly response (not None)
        assert result["final_response"] is not None
        assert "Thinking Budget Exhausted" in result["final_response"]
        assert "/thinkon" in result["final_response"]

    def test_length_empty_content_detected_as_thinking_exhausted(self, agent):
        """When finish_reason='length' and content is None/empty, detect exhaustion."""
        self._setup_agent(agent)
        resp = _mock_response(content=None, finish_reason="length")
        agent.client.chat.completions.create.return_value = resp

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is False
        assert result["api_calls"] == 1
        assert "reasoning" in result["error"].lower()
        # User-friendly message is returned
        assert result["final_response"] is not None
        assert "Thinking Budget Exhausted" in result["final_response"]

    def test_length_with_tool_calls_returns_partial_without_executing_tools(self, agent):
        self._setup_agent(agent)
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        resp = _mock_response(content="", finish_reason="length", tool_calls=[bad_tc])
        agent.client.chat.completions.create.return_value = resp

        with (
            patch("run_agent.handle_function_call") as mock_handle_function_call,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("write the report")

        assert result["completed"] is False
        assert result["partial"] is True
        assert "truncated due to output length limit" in result["error"]
        mock_handle_function_call.assert_not_called()

    def test_truncated_tool_call_retries_once_before_refusing(self, agent):
        """When tool call args are truncated, the agent retries the API call
        once. If the retry succeeds (valid JSON args), tool execution proceeds."""
        self._setup_agent(agent)
        agent.valid_tool_names.add("write_file")
        bad_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"partial',
            call_id="c1",
        )
        truncated_resp = _mock_response(
            content="", finish_reason="length", tool_calls=[bad_tc],
        )
        good_tc = _mock_tool_call(
            name="write_file",
            arguments='{"path":"report.md","content":"full content"}',
            call_id="c2",
        )
        good_resp = _mock_response(
            content="", finish_reason="stop", tool_calls=[good_tc],
        )
        with (
            patch("run_agent.handle_function_call", return_value='{"success":true}') as mock_hfc,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            # First call: truncated → retry. Second: valid → execute tool.
            # Third: final text response.
            final_resp = _mock_response(content="Done!", finish_reason="stop")
            agent.client.chat.completions.create.side_effect = [
                truncated_resp, good_resp, final_resp,
            ]
            result = agent.run_conversation("write the report")

        # Tool was executed on the retry (good_resp)
        mock_hfc.assert_called_once()
        assert result["final_response"] == "Done!"


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
        assert result.get("completed") is False, (
            f"Expected completed=False, got: {result}"
        )
        assert result.get("failed") is True
        assert "error" in result
        assert "Invalid API response" in result["error"]

    def test_api_error_returns_gracefully_after_retries(self, agent):
        """Exhausted retries on API errors must return error result, not crash."""
        self._setup_agent(agent)
        agent.client.chat.completions.create.side_effect = RuntimeError("rate limited")
        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.time", self._make_fast_time_mock()),
        ):
            result = agent.run_conversation("hello")
        assert result.get("completed") is False
        assert result.get("failed") is True
        assert "error" in result
        assert "rate limited" in result["error"]


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
            result = agent.run_conversation(
                "new question", conversation_history=history
            )

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

    def test_try_refresh_nous_client_credentials_rebuilds_client(
        self, agent, monkeypatch
    ):
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

        monkeypatch.setattr(
            "hermes_cli.auth.resolve_nous_runtime_credentials", _fake_resolve
        )

        agent.client = _ExistingClient()
        with patch("run_agent.OpenAI", side_effect=_fake_openai):
            ok = agent._try_refresh_nous_client_credentials(force=True)

        assert ok is True
        assert closed["value"] is True
        assert captured["force_mint"] is True
        assert rebuilt["kwargs"]["api_key"] == "new-nous-key"
        assert (
            rebuilt["kwargs"]["base_url"] == "https://inference-api.nousresearch.com/v1"
        )
        assert "default_headers" not in rebuilt["kwargs"]
        assert isinstance(agent.client, _RebuiltClient)


class TestCredentialPoolRecovery:
    def test_recover_with_pool_rotates_on_402(self, agent):
        current = SimpleNamespace(label="primary")
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            def current(self):
                return current

            def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
                assert status_code == 402
                assert error_context is None
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=402,
            has_retried_429=False,
        )

        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_recover_with_pool_rotates_on_billing_reason_even_with_http_400(self, agent):
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
                assert status_code == 400
                assert error_context == {"reason": "out_of_extra_usage"}
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=400,
            has_retried_429=False,
            classified_reason=FailoverReason.billing,
            error_context={"reason": "out_of_extra_usage"},
        )

        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_recover_with_pool_retries_first_429_then_rotates(self, agent):
        next_entry = SimpleNamespace(label="secondary")

        class _Pool:
            def current(self):
                return SimpleNamespace(label="primary")

            def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
                assert status_code == 429
                assert error_context is None
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=False,
        )
        assert recovered is False
        assert retry_same is True
        agent._swap_credential.assert_not_called()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=True,
        )
        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)


    def test_recover_with_pool_refreshes_on_401(self, agent):
        """401 with successful refresh should swap to refreshed credential."""
        refreshed_entry = SimpleNamespace(label="refreshed-primary", id="abc")

        class _Pool:
            def try_refresh_current(self):
                return refreshed_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
        )

        assert recovered is True
        agent._swap_credential.assert_called_once_with(refreshed_entry)

    def test_recover_with_pool_rotates_on_401_when_refresh_fails(self, agent):
        """401 with failed refresh should rotate to next credential."""
        next_entry = SimpleNamespace(label="secondary", id="def")

        class _Pool:
            def try_refresh_current(self):
                return None  # refresh failed

            def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
                assert status_code == 401
                assert error_context is None
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
        )

        assert recovered is True
        assert retry_same is False
        agent._swap_credential.assert_called_once_with(next_entry)

    def test_recover_with_pool_401_refresh_fails_no_more_credentials(self, agent):
        """401 with failed refresh and no other credentials returns not recovered."""

        class _Pool:
            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
                assert error_context is None
                return None  # no more credentials

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=401,
            has_retried_429=False,
        )

        assert recovered is False
        agent._swap_credential.assert_not_called()

    def test_extract_api_error_context_uses_reset_timestamp_and_reason(self, agent):
        response = SimpleNamespace(headers={})
        error = SimpleNamespace(
            body={
                "error": {
                    "code": "device_code_exhausted",
                    "message": "Weekly credits exhausted.",
                    "resets_at": "2026-04-12T10:30:00Z",
                }
            },
            response=response,
        )

        context = agent._extract_api_error_context(error)

        assert context["reason"] == "device_code_exhausted"
        assert context["message"] == "Weekly credits exhausted."
        assert context["reset_at"] == "2026-04-12T10:30:00Z"

    def test_recover_with_pool_passes_error_context_on_rotated_429(self, agent):
        next_entry = SimpleNamespace(label="secondary")
        captured = {}

        class _Pool:
            def current(self):
                return SimpleNamespace(label="primary")

            def mark_exhausted_and_rotate(self, *, status_code, error_context=None):
                captured["status_code"] = status_code
                captured["error_context"] = error_context
                return next_entry

        agent._credential_pool = _Pool()
        agent._swap_credential = MagicMock()

        recovered, retry_same = agent._recover_with_credential_pool(
            status_code=429,
            has_retried_429=True,
            error_context={"reason": "device_code_exhausted", "reset_at": "2026-04-12T10:30:00Z"},
        )

        assert recovered is True
        assert retry_same is False
        assert captured["status_code"] == 429
        assert captured["error_context"]["reason"] == "device_code_exhausted"


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
        """run_conversation installs _SafeWriter on stdio."""
        import sys
        from run_agent import _SafeWriter
        resp = _mock_response(content="Done", finish_reason="stop")
        agent.client.chat.completions.create.return_value = resp
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with (
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
            ):
                agent.run_conversation("test")
            assert isinstance(sys.stdout, _SafeWriter)
            assert isinstance(sys.stderr, _SafeWriter)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    # test_installed_before_init_time_honcho_error_prints removed —
    # Honcho integration extracted to plugin (PR #4154).

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


class TestSaveSessionLogAtomicWrite:
    def test_uses_shared_atomic_json_helper(self, agent, tmp_path):
        agent.session_log_file = tmp_path / "session.json"
        messages = [{"role": "user", "content": "hello"}]

        with patch("run_agent.atomic_json_write", create=True) as mock_atomic_write:
            agent._save_session_log(messages)

        mock_atomic_write.assert_called_once()
        call_args = mock_atomic_write.call_args
        assert call_args.args[0] == agent.session_log_file
        payload = call_args.args[1]
        assert payload["session_id"] == agent.session_id
        assert payload["messages"] == messages
        assert call_args.kwargs["indent"] == 2
        assert call_args.kwargs["default"] is str


# ===================================================================
# Anthropic adapter integration fixes
# ===================================================================


class TestBuildApiKwargsAnthropicMaxTokens:
    """Bug fix: max_tokens was always None for Anthropic mode, ignoring user config."""

    def test_max_tokens_passed_to_anthropic(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.max_tokens = 4096
        agent.reasoning_config = None

        with patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build:
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs([{"role": "user", "content": "test"}])
            _, kwargs = mock_build.call_args
            if not kwargs:
                kwargs = dict(zip(
                    ["model", "messages", "tools", "max_tokens", "reasoning_config"],
                    mock_build.call_args[0],
                ))
            assert kwargs.get("max_tokens") == 4096 or mock_build.call_args[1].get("max_tokens") == 4096

    def test_max_tokens_none_when_unset(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.max_tokens = None
        agent.reasoning_config = None

        with patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build:
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 16384}
            agent._build_api_kwargs([{"role": "user", "content": "test"}])
            call_args = mock_build.call_args
            # max_tokens should be None (let adapter use its default)
            if call_args[1]:
                assert call_args[1].get("max_tokens") is None
            else:
                assert call_args[0][3] is None


class TestAnthropicImageFallback:
    def test_build_api_kwargs_converts_multimodal_user_image_to_text(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.reasoning_config = None

        api_messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Can you see this now?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }]

        with (
            patch("tools.vision_tools.vision_analyze_tool", new=AsyncMock(return_value=json.dumps({"success": True, "analysis": "A cat sitting on a chair."}))),
            patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build,
        ):
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs(api_messages)

        kwargs = mock_build.call_args.kwargs or dict(zip(
            ["model", "messages", "tools", "max_tokens", "reasoning_config"],
            mock_build.call_args.args,
        ))
        transformed = kwargs["messages"]
        assert isinstance(transformed[0]["content"], str)
        assert "A cat sitting on a chair." in transformed[0]["content"]
        assert "Can you see this now?" in transformed[0]["content"]
        assert "vision_analyze with image_url: https://example.com/cat.png" in transformed[0]["content"]

    def test_build_api_kwargs_reuses_cached_image_analysis_for_duplicate_images(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.reasoning_config = None
        data_url = "data:image/png;base64,QUFBQQ=="

        api_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "input_image", "image_url": data_url},
                ],
            },
        ]

        mock_vision = AsyncMock(return_value=json.dumps({"success": True, "analysis": "A small test image."}))
        with (
            patch("tools.vision_tools.vision_analyze_tool", new=mock_vision),
            patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build,
        ):
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs(api_messages)

        assert mock_vision.await_count == 1


class TestFallbackAnthropicProvider:
    """Bug fix: _try_activate_fallback had no case for anthropic provider."""

    def test_fallback_to_anthropic_sets_api_mode(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-test"

        with (
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            mock_build.return_value = MagicMock()
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "anthropic_messages"
        assert agent._anthropic_client is not None
        assert agent.client is None

    def test_fallback_to_anthropic_enables_prompt_caching(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-test"

        with (
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=None),
        ):
            agent._try_activate_fallback()

        assert agent._use_prompt_caching is True

    def test_fallback_to_openrouter_uses_openai_client(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://openrouter.ai/api/v1"
        mock_client.api_key = "sk-or-test"

        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(mock_client, None)):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent.api_mode == "chat_completions"
        assert agent.client is mock_client


def test_aiagent_uses_copilot_acp_client():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI") as mock_openai,
        patch("agent.copilot_acp_client.CopilotACPClient") as mock_acp_client,
    ):
        acp_client = MagicMock()
        mock_acp_client.return_value = acp_client

        agent = AIAgent(
            api_key="copilot-acp",
            base_url="acp://copilot",
            provider="copilot-acp",
            acp_command="/usr/local/bin/copilot",
            acp_args=["--acp", "--stdio"],
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent.client is acp_client
    mock_openai.assert_not_called()
    mock_acp_client.assert_called_once()
    assert mock_acp_client.call_args.kwargs["base_url"] == "acp://copilot"
    assert mock_acp_client.call_args.kwargs["api_key"] == "copilot-acp"
    assert mock_acp_client.call_args.kwargs["command"] == "/usr/local/bin/copilot"
    assert mock_acp_client.call_args.kwargs["args"] == ["--acp", "--stdio"]


def test_quiet_spinner_allowed_with_explicit_print_fn(agent):
    agent._print_fn = lambda *_a, **_kw: None
    with patch.object(run_agent.sys.stdout, "isatty", return_value=False):
        assert agent._should_start_quiet_spinner() is True


def test_quiet_spinner_allowed_on_real_tty(agent):
    agent._print_fn = None
    with patch.object(run_agent.sys.stdout, "isatty", return_value=True):
        assert agent._should_start_quiet_spinner() is True


def test_quiet_spinner_suppressed_on_non_tty_without_print_fn(agent):
    agent._print_fn = None
    with patch.object(run_agent.sys.stdout, "isatty", return_value=False):
        assert agent._should_start_quiet_spinner() is False


def test_is_openai_client_closed_honors_custom_client_flag():
    assert AIAgent._is_openai_client_closed(SimpleNamespace(is_closed=True)) is True
    assert AIAgent._is_openai_client_closed(SimpleNamespace(is_closed=False)) is False


def test_is_openai_client_closed_handles_method_form():
    """Fix for issue #4377: is_closed as method (openai SDK) vs property (httpx).

    The openai SDK's is_closed is a method, not a property. Prior to this fix,
    getattr(client, "is_closed", False) returned the bound method object, which
    is always truthy, causing the function to incorrectly report all clients as
    closed and triggering unnecessary client recreation on every API call.
    """

    class MethodFormClient:
        """Mimics openai.OpenAI where is_closed() is a method."""

        def __init__(self, closed: bool):
            self._closed = closed

        def is_closed(self) -> bool:
            return self._closed

    # Method returning False - client is open
    open_client = MethodFormClient(closed=False)
    assert AIAgent._is_openai_client_closed(open_client) is False

    # Method returning True - client is closed
    closed_client = MethodFormClient(closed=True)
    assert AIAgent._is_openai_client_closed(closed_client) is True


def test_is_openai_client_closed_falls_back_to_http_client():
    """Verify fallback to _client.is_closed when top-level is_closed is None."""

    class ClientWithHttpClient:
        is_closed = None  # No top-level is_closed

        def __init__(self, http_closed: bool):
            self._client = SimpleNamespace(is_closed=http_closed)

    assert AIAgent._is_openai_client_closed(ClientWithHttpClient(http_closed=False)) is False
    assert AIAgent._is_openai_client_closed(ClientWithHttpClient(http_closed=True)) is True


class TestAnthropicBaseUrlPassthrough:
    """Bug fix: base_url was filtered with 'anthropic in base_url', blocking proxies."""

    def test_custom_proxy_base_url_passed_through(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            a = AIAgent(
                api_key="sk-ant-api03-test1234567890",
                base_url="https://llm-proxy.company.com/v1",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            call_args = mock_build.call_args
            # base_url should be passed through, not filtered out
            assert call_args[0][1] == "https://llm-proxy.company.com/v1"

    def test_none_base_url_passed_as_none(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
        ):
            mock_build.return_value = MagicMock()
            a = AIAgent(
                api_key="sk-ant-api03-test1234567890",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            call_args = mock_build.call_args
            # No base_url provided, should be default empty string or None
            passed_url = call_args[0][1]
            assert not passed_url or passed_url is None


class TestAnthropicCredentialRefresh:
    def test_try_refresh_anthropic_client_credentials_rebuilds_client(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client") as mock_build,
        ):
            old_client = MagicMock()
            new_client = MagicMock()
            mock_build.side_effect = [old_client, new_client]
            agent = AIAgent(
                api_key="sk-ant-oat01-stale-token",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        agent._anthropic_client = old_client
        agent._anthropic_api_key = "sk-ant-oat01-stale-token"
        agent._anthropic_base_url = "https://api.anthropic.com"
        agent.provider = "anthropic"

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat01-fresh-token"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=new_client) as rebuild,
        ):
            assert agent._try_refresh_anthropic_client_credentials() is True

        old_client.close.assert_called_once()
        rebuild.assert_called_once_with("sk-ant-oat01-fresh-token", "https://api.anthropic.com")
        assert agent._anthropic_client is new_client
        assert agent._anthropic_api_key == "sk-ant-oat01-fresh-token"

    def test_try_refresh_anthropic_client_credentials_returns_false_when_token_unchanged(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-same-token",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        old_client = MagicMock()
        agent._anthropic_client = old_client
        agent._anthropic_api_key = "sk-ant-oat01-same-token"

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat01-same-token"),
            patch("agent.anthropic_adapter.build_anthropic_client") as rebuild,
        ):
            assert agent._try_refresh_anthropic_client_credentials() is False

        old_client.close.assert_not_called()
        rebuild.assert_not_called()

    def test_anthropic_messages_create_preflights_refresh(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        ):
            agent = AIAgent(
                api_key="sk-ant-oat01-current-token",
                api_mode="anthropic_messages",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        response = SimpleNamespace(content=[])
        agent._anthropic_client = MagicMock()
        agent._anthropic_client.messages.create.return_value = response

        with patch.object(agent, "_try_refresh_anthropic_client_credentials", return_value=True) as refresh:
            result = agent._anthropic_messages_create({"model": "claude-sonnet-4-20250514"})

        refresh.assert_called_once_with()
        agent._anthropic_client.messages.create.assert_called_once_with(model="claude-sonnet-4-20250514")
        assert result is response


# ===================================================================
# _streaming_api_call tests
# ===================================================================

def _make_chunk(content=None, tool_calls=None, finish_reason=None, model="test/model"):
    """Build a SimpleNamespace mimicking an OpenAI streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(model=model, choices=[choice])


def _make_tc_delta(index=0, tc_id=None, name=None, arguments=None):
    """Build a SimpleNamespace mimicking a streaming tool_call delta."""
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)


class TestStreamingApiCall:
    """Tests for _streaming_api_call — voice TTS streaming pipeline."""

    def test_content_assembly(self, agent):
        chunks = [
            _make_chunk(content="Hel"),
            _make_chunk(content="lo "),
            _make_chunk(content="World"),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        callback = MagicMock()
        agent.stream_delta_callback = callback

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello World"
        assert resp.choices[0].finish_reason == "stop"
        assert callback.call_count == 3
        callback.assert_any_call("Hel")
        callback.assert_any_call("lo ")
        callback.assert_any_call("World")

    def test_tool_call_accumulation(self, agent):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "web_", '{"q":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, None, "search", '"test"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].function.name == "web_search"
        assert tc[0].function.arguments == '{"q":"test"}'
        assert tc[0].id == "call_1"

    def test_multiple_tool_calls(self, agent):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{}')]),
            _make_chunk(tool_calls=[_make_tc_delta(1, "call_b", "read", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "search"
        assert tc[1].function.name == "read"

    def test_truncated_tool_call_args_upgrade_finish_reason_to_length(self, agent):
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "write_file", '{"path":"x.txt","content":"hel')]),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 1
        assert tc[0].function.name == "write_file"
        assert tc[0].function.arguments == '{"path":"x.txt","content":"hel'
        assert resp.choices[0].finish_reason == "length"

    def test_ollama_reused_index_separate_tool_calls(self, agent):
        """Ollama sends every tool call at index 0 with different ids.

        Without the fix, names and arguments get concatenated into one slot.
        """
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{"q":"hello"}')]),
            # Second tool call at the SAME index 0, but different id
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_b", "read_file", '{"path":"x.py"}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2, f"Expected 2 tool calls, got {len(tc)}: {[t.function.name for t in tc]}"
        assert tc[0].function.name == "search"
        assert tc[0].function.arguments == '{"q":"hello"}'
        assert tc[0].id == "call_a"
        assert tc[1].function.name == "read_file"
        assert tc[1].function.arguments == '{"path":"x.py"}'
        assert tc[1].id == "call_b"

    def test_ollama_reused_index_streamed_args(self, agent):
        """Ollama with streamed arguments across multiple chunks at same index."""
        chunks = [
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_a", "search", '{"q":')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, None, None, '"hello"}')]),
            # New tool call, same index 0
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_b", "read", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        tc = resp.choices[0].message.tool_calls
        assert len(tc) == 2
        assert tc[0].function.name == "search"
        assert tc[0].function.arguments == '{"q":"hello"}'
        assert tc[1].function.name == "read"
        assert tc[1].function.arguments == '{}'

    def test_content_and_tool_calls_together(self, agent):
        chunks = [
            _make_chunk(content="I'll search"),
            _make_chunk(tool_calls=[_make_tc_delta(0, "call_1", "search", '{}')]),
            _make_chunk(finish_reason="tool_calls"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "I'll search"
        assert len(resp.choices[0].message.tool_calls) == 1

    def test_empty_content_returns_none(self, agent):
        chunks = [_make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content is None
        assert resp.choices[0].message.tool_calls is None

    def test_callback_exception_swallowed(self, agent):
        chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" World"),
            _make_chunk(finish_reason="stop"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)
        agent.stream_delta_callback = MagicMock(side_effect=ValueError("boom"))

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello World"

    def test_model_name_captured(self, agent):
        chunks = [
            _make_chunk(content="Hi", model="gpt-4o"),
            _make_chunk(finish_reason="stop", model="gpt-4o"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.model == "gpt-4o"

    def test_stream_kwarg_injected(self, agent):
        chunks = [_make_chunk(content="x"), _make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        agent._interruptible_streaming_api_call({"messages": [], "model": "test"})

        call_kwargs = agent.client.chat.completions.create.call_args
        assert call_kwargs[1].get("stream") is True or call_kwargs.kwargs.get("stream") is True

    def test_api_exception_falls_back_to_non_streaming(self, agent):
        """When streaming fails before any deltas, fallback to non-streaming is attempted."""
        agent.client.chat.completions.create.side_effect = ConnectionError("fail")
        # Prevent stream retry logic from replacing the mock client
        with patch.object(agent, "_replace_primary_openai_client", return_value=False):
            # The fallback also uses the same client, so it'll fail too
            with pytest.raises(ConnectionError, match="fail"):
                agent._interruptible_streaming_api_call({"messages": []})

    def test_response_has_uuid_id(self, agent):
        chunks = [_make_chunk(content="x"), _make_chunk(finish_reason="stop")]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.id.startswith("stream-")
        assert len(resp.id) > len("stream-")

    def test_empty_choices_chunk_skipped(self, agent):
        empty_chunk = SimpleNamespace(model="gpt-4", choices=[])
        chunks = [
            empty_chunk,
            _make_chunk(content="Hello", model="gpt-4"),
            _make_chunk(finish_reason="stop", model="gpt-4"),
        ]
        agent.client.chat.completions.create.return_value = iter(chunks)

        resp = agent._interruptible_streaming_api_call({"messages": []})

        assert resp.choices[0].message.content == "Hello"
        assert resp.model == "gpt-4"


# ===================================================================
# Interrupt _vprint force=True verification
# ===================================================================


class TestInterruptVprintForceTrue:
    """All interrupt _vprint calls must use force=True so they are always visible."""

    def test_all_interrupt_vprint_have_force_true(self):
        """Scan source for _vprint calls containing 'Interrupt' — each must have force=True."""
        import inspect
        source = inspect.getsource(AIAgent)
        lines = source.split("\n")
        violations = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if "_vprint(" in stripped and "Interrupt" in stripped:
                if "force=True" not in stripped:
                    violations.append(f"line {i}: {stripped}")
        assert not violations, (
            f"Interrupt _vprint calls missing force=True:\n"
            + "\n".join(violations)
        )


# ===================================================================
# Anthropic interrupt handler in _interruptible_api_call
# ===================================================================


class TestAnthropicInterruptHandler:
    """_interruptible_api_call must handle Anthropic mode when interrupted."""

    def test_interruptible_has_anthropic_branch(self):
        """The interrupt handler must check api_mode == 'anthropic_messages'."""
        import inspect
        source = inspect.getsource(AIAgent._interruptible_api_call)
        assert "anthropic_messages" in source, \
            "_interruptible_api_call must handle Anthropic interrupt (api_mode check)"

    def test_interruptible_rebuilds_anthropic_client(self):
        """After interrupting, the Anthropic client should be rebuilt."""
        import inspect
        source = inspect.getsource(AIAgent._interruptible_api_call)
        assert "build_anthropic_client" in source, \
            "_interruptible_api_call must rebuild Anthropic client after interrupt"

    def test_streaming_has_anthropic_branch(self):
        """_streaming_api_call must also handle Anthropic interrupt."""
        import inspect
        source = inspect.getsource(AIAgent._interruptible_streaming_api_call)
        assert "anthropic_messages" in source, \
            "_streaming_api_call must handle Anthropic interrupt"


# ---------------------------------------------------------------------------
# Bugfix: stream_callback forwarding for non-streaming providers
# ---------------------------------------------------------------------------


class TestStreamCallbackNonStreamingProvider:
    """When api_mode != chat_completions, stream_callback must still receive
    the response content so TTS works (batch delivery)."""

    def test_callback_receives_chat_completions_response(self, agent):
        """For chat_completions-shaped responses, callback gets content."""
        agent.api_mode = "anthropic_messages"
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="Hello", tool_calls=None, reasoning_content=None),
                finish_reason="stop", index=0,
            )],
            usage=None, model="test", id="test-id",
        )
        agent._interruptible_api_call = MagicMock(return_value=mock_response)

        received = []
        cb = lambda delta: received.append(delta)
        agent._stream_callback = cb

        _cb = getattr(agent, "_stream_callback", None)
        response = agent._interruptible_api_call({})
        if _cb is not None and response:
            try:
                if agent.api_mode == "anthropic_messages":
                    text_parts = [
                        block.text for block in getattr(response, "content", [])
                        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
                    ]
                    content = " ".join(text_parts) if text_parts else None
                else:
                    content = response.choices[0].message.content
                if content:
                    _cb(content)
            except Exception:
                pass

        # Anthropic format not matched above; fallback via except
        # Test the actual code path by checking chat_completions branch
        received2 = []
        agent.api_mode = "some_other_mode"
        agent._stream_callback = lambda d: received2.append(d)
        _cb2 = agent._stream_callback
        if _cb2 is not None and mock_response:
            try:
                content = mock_response.choices[0].message.content
                if content:
                    _cb2(content)
            except Exception:
                pass
        assert received2 == ["Hello"]

    def test_callback_receives_anthropic_content(self, agent):
        """For Anthropic responses, text blocks are extracted and forwarded."""
        agent.api_mode = "anthropic_messages"
        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello from Claude")],
            stop_reason="end_turn",
        )

        received = []
        cb = lambda d: received.append(d)
        agent._stream_callback = cb
        _cb = agent._stream_callback

        if _cb is not None and mock_response:
            try:
                if agent.api_mode == "anthropic_messages":
                    text_parts = [
                        block.text for block in getattr(mock_response, "content", [])
                        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
                    ]
                    content = " ".join(text_parts) if text_parts else None
                else:
                    content = mock_response.choices[0].message.content
                if content:
                    _cb(content)
            except Exception:
                pass

        assert received == ["Hello from Claude"]


# ---------------------------------------------------------------------------
# Bugfix: API-only user message prefixes must not persist
# ---------------------------------------------------------------------------


class TestPersistUserMessageOverride:
    """Synthetic API-only user prefixes should never leak into transcripts."""

    def test_persist_session_rewrites_current_turn_user_message(self, agent):
        agent._session_db = MagicMock()
        agent.session_id = "session-123"
        agent._last_flushed_db_idx = 0
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = "Hello there"
        messages = [
            {
                "role": "user",
                "content": (
                    "[Voice input — respond concisely and conversationally, "
                    "2-3 sentences max. No code blocks or markdown.] Hello there"
                ),
            },
            {"role": "assistant", "content": "Hi!"},
        ]

        with patch.object(agent, "_save_session_log") as mock_save:
            agent._persist_session(messages, [])

        assert messages[0]["content"] == "Hello there"
        saved_messages = mock_save.call_args.args[0]
        assert saved_messages[0]["content"] == "Hello there"
        first_db_write = agent._session_db.append_message.call_args_list[0].kwargs
        assert first_db_write["content"] == "Hello there"


# ---------------------------------------------------------------------------
# Bugfix: _vprint force=True on error messages during TTS
# ---------------------------------------------------------------------------


class TestVprintForceOnErrors:
    """Error/warning messages must be visible during streaming TTS."""

    def test_forced_message_shown_during_tts(self, agent):
        agent._stream_callback = lambda x: None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("error msg", force=True)
        assert len(printed) == 1

    def test_non_forced_suppressed_during_tts(self, agent):
        agent._stream_callback = lambda x: None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("debug info")
        assert len(printed) == 0

    def test_all_shown_without_tts(self, agent):
        agent._stream_callback = None
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(a)):
            agent._vprint("debug")
            agent._vprint("error", force=True)
        assert len(printed) == 2


class TestNormalizeCodexDictArguments:
    """_normalize_codex_response must produce valid JSON strings for tool
    call arguments, even when the Responses API returns them as dicts."""

    def _make_codex_response(self, item_type, arguments, item_status="completed"):
        """Build a minimal Responses API response with a single tool call."""
        item = SimpleNamespace(
            type=item_type,
            status=item_status,
        )
        if item_type == "function_call":
            item.name = "web_search"
            item.arguments = arguments
            item.call_id = "call_abc123"
            item.id = "fc_abc123"
        elif item_type == "custom_tool_call":
            item.name = "web_search"
            item.input = arguments
            item.call_id = "call_abc123"
            item.id = "fc_abc123"
        return SimpleNamespace(
            output=[item],
            status="completed",
        )

    def test_function_call_dict_arguments_produce_valid_json(self, agent):
        """dict arguments from function_call must be serialised with
        json.dumps, not str(), so downstream json.loads() succeeds."""
        args_dict = {"query": "weather in NYC", "units": "celsius"}
        response = self._make_codex_response("function_call", args_dict)
        msg, _ = agent._normalize_codex_response(response)
        tc = msg.tool_calls[0]
        parsed = json.loads(tc.function.arguments)
        assert parsed == args_dict

    def test_custom_tool_call_dict_arguments_produce_valid_json(self, agent):
        """dict arguments from custom_tool_call must also use json.dumps."""
        args_dict = {"path": "/tmp/test.txt", "content": "hello"}
        response = self._make_codex_response("custom_tool_call", args_dict)
        msg, _ = agent._normalize_codex_response(response)
        tc = msg.tool_calls[0]
        parsed = json.loads(tc.function.arguments)
        assert parsed == args_dict

    def test_string_arguments_unchanged(self, agent):
        """String arguments must pass through without modification."""
        args_str = '{"query": "test"}'
        response = self._make_codex_response("function_call", args_str)
        msg, _ = agent._normalize_codex_response(response)
        tc = msg.tool_calls[0]
        assert tc.function.arguments == args_str


# ---------------------------------------------------------------------------
# OAuth flag and nudge counter fixes (salvaged from PR #1797)
# ---------------------------------------------------------------------------


class TestOAuthFlagAfterCredentialRefresh:
    """_is_anthropic_oauth must update when token type changes during refresh."""

    def test_oauth_flag_updates_api_key_to_oauth(self, agent):
        """Refreshing from API key to OAuth token must set flag to True."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "sk-ant-api-old"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = False

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value="sk-ant-setup-oauth-token"),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._is_anthropic_oauth is True

    def test_oauth_flag_updates_oauth_to_api_key(self, agent):
        """Refreshing from OAuth to API key must set flag to False."""
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._anthropic_api_key = "sk-ant-setup-old"
        agent._anthropic_client = MagicMock()
        agent._is_anthropic_oauth = True

        with (
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value="sk-ant-api03-new-key"),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
        ):
            result = agent._try_refresh_anthropic_client_credentials()

        assert result is True
        assert agent._is_anthropic_oauth is False


class TestFallbackSetsOAuthFlag:
    """_try_activate_fallback must set _is_anthropic_oauth for Anthropic fallbacks."""

    def test_fallback_to_anthropic_oauth_sets_flag(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-setup-oauth-token"

        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent._is_anthropic_oauth is True

    def test_fallback_to_anthropic_api_key_clears_flag(self, agent):
        agent._fallback_activated = False
        agent._fallback_model = {"provider": "anthropic", "model": "claude-sonnet-4-6"}
        agent._fallback_chain = [agent._fallback_model]
        agent._fallback_index = 0

        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com/v1"
        mock_client.api_key = "sk-ant-api03-regular-key"

        with (
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(mock_client, None)),
            patch("agent.anthropic_adapter.build_anthropic_client",
                  return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token",
                  return_value=None),
        ):
            result = agent._try_activate_fallback()

        assert result is True
        assert agent._is_anthropic_oauth is False


class TestMemoryNudgeCounterPersistence:
    """_turns_since_memory must persist across run_conversation calls."""

    def test_counters_initialized_in_init(self):
        """Counters must exist on the agent after __init__."""
        with patch("run_agent.get_tool_definitions", return_value=[]):
            a = AIAgent(
                model="test", api_key="test-key", provider="openrouter",
                skip_context_files=True, skip_memory=True,
            )
        assert hasattr(a, "_turns_since_memory")
        assert hasattr(a, "_iters_since_skill")
        assert a._turns_since_memory == 0
        assert a._iters_since_skill == 0

    def test_counters_not_reset_in_preamble(self):
        """The run_conversation preamble must not zero the nudge counters."""
        import inspect
        src = inspect.getsource(AIAgent.run_conversation)
        # The preamble resets many fields (retry counts, budget, etc.)
        # before the main loop. Find that reset block and verify our
        # counters aren't in it. The reset block ends at iteration_budget.
        preamble_end = src.index("self.iteration_budget = IterationBudget")
        preamble = src[:preamble_end]
        assert "self._turns_since_memory = 0" not in preamble
        assert "self._iters_since_skill = 0" not in preamble


class TestDeadRetryCode:
    """Unreachable retry_count >= max_retries after raise must not exist."""

    def test_no_unreachable_max_retries_after_backoff(self):
        import inspect
        source = inspect.getsource(AIAgent.run_conversation)
        occurrences = source.count("if retry_count >= max_retries:")
        assert occurrences == 2, (
            f"Expected 2 occurrences of 'if retry_count >= max_retries:' "
            f"but found {occurrences}"
        )
