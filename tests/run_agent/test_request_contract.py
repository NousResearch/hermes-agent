"""Outbound request-contract preflight + max-iterations summary tools invariant.

Pins the fix for the codex_responses summary path that used to:

    kwargs = build_api_kwargs(...)   # tools + tool_choice=\"auto\"
    kwargs.pop(\"tools\", None)       # leaves tool_choice → Responses API 400

Also covers the shared preflight gate used by the main loop and summary path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.errors import RuntimeContractViolation
from agent.request_contract import validate_api_kwargs


# ---------------------------------------------------------------------------
# validate_api_kwargs unit tests
# ---------------------------------------------------------------------------


class TestValidateApiKwargs:
    def test_ok_with_tools_and_tool_choice(self):
        validate_api_kwargs(
            {"model": "m", "tools": [{"type": "function"}], "tool_choice": "auto"},
            api_mode="codex_responses",
        )

    def test_ok_toolless(self):
        validate_api_kwargs({"model": "m", "input": []}, api_mode="codex_responses")

    def test_rejects_tool_choice_without_tools(self):
        with pytest.raises(RuntimeContractViolation) as ei:
            validate_api_kwargs(
                {"model": "m", "tool_choice": "auto"},
                api_mode="codex_responses",
                where="unit",
            )
        assert ei.value.field == "tool_choice"
        assert "requires a non-empty tools list" in str(ei.value)

    def test_rejects_parallel_without_tools(self):
        with pytest.raises(RuntimeContractViolation) as ei:
            validate_api_kwargs(
                {"model": "m", "parallel_tool_calls": True},
                api_mode="codex_responses",
            )
        assert ei.value.field == "parallel_tool_calls"

    def test_rejects_tools_none_key(self):
        with pytest.raises(RuntimeContractViolation) as ei:
            validate_api_kwargs({"tools": None, "model": "m"}, api_mode="codex_responses")
        assert ei.value.field == "tools"

    def test_rejects_tool_choice_with_empty_tools(self):
        with pytest.raises(RuntimeContractViolation):
            validate_api_kwargs(
                {"tools": [], "tool_choice": "auto"},
                api_mode="chat_completions",
            )

    def test_ok_chat_with_tools(self):
        validate_api_kwargs(
            {
                "model": "gpt",
                "messages": [],
                "tools": [{"type": "function", "function": {"name": "t"}}],
            },
            api_mode="chat_completions",
        )


# ---------------------------------------------------------------------------
# build_api_kwargs tools= override
# ---------------------------------------------------------------------------


class TestBuildApiKwargsToolsOverride:
    def test_codex_tools_none_omits_tool_fields(self, agent):
        from agent.chat_completion_helpers import build_api_kwargs

        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"
        # Agent still has tools registered (main-loop state).
        assert agent.tools

        kwargs = build_api_kwargs(
            agent,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            tools=None,
        )
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        assert "parallel_tool_calls" not in kwargs
        # Preflight must accept the toolless build.
        validate_api_kwargs(kwargs, api_mode="codex_responses")

    def test_codex_default_includes_tools_when_registered(self, agent):
        from agent.chat_completion_helpers import build_api_kwargs

        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"

        kwargs = build_api_kwargs(
            agent,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        )
        assert "tools" in kwargs and kwargs["tools"]
        assert kwargs.get("tool_choice") == "auto"
        validate_api_kwargs(kwargs, api_mode="codex_responses")

    def test_chat_tools_none_omits_tools(self, agent):
        from agent.chat_completion_helpers import build_api_kwargs

        agent.api_mode = "chat_completions"
        kwargs = build_api_kwargs(
            agent,
            [{"role": "user", "content": "hi"}],
            tools=None,
        )
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
        validate_api_kwargs(kwargs, api_mode="chat_completions")


# ---------------------------------------------------------------------------
# handle_max_iterations summary paths
# ---------------------------------------------------------------------------


def _make_tool_defs(*names: str) -> list:
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
    """Minimal AIAgent with mocked client (mirrors tests/run_agent fixture)."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestMaxIterationsSummaryContract:
    def test_codex_summary_no_tool_choice_when_agent_has_tools(self, agent):
        """Regression: summary must not send tool_choice without tools."""
        agent.api_mode = "codex_responses"
        agent.provider = "openai-codex"
        agent.base_url = "https://chatgpt.com/backend-api/codex"
        agent._base_url_lower = agent.base_url.lower()
        agent._base_url_hostname = "chatgpt.com"
        agent.model = "gpt-5.5"
        agent._cached_system_prompt = "You are helpful."
        assert agent.tools  # tools still registered on the agent
        captured = {}

        def fake_run_codex_stream(kwargs):
            captured.update(kwargs)
            # Preflight would already have run inside run_codex_stream; assert
            # shape here too for the handle_max_iterations construction path.
            validate_api_kwargs(kwargs, api_mode="codex_responses")
            return SimpleNamespace(
                status="completed",
                output=[
                    SimpleNamespace(
                        type="message",
                        status="completed",
                        content=[SimpleNamespace(type="output_text", text="Summary done.")],
                    )
                ],
            )

        with patch.object(agent, "_run_codex_stream", side_effect=fake_run_codex_stream):
            result = agent._handle_max_iterations(
                [{"role": "user", "content": "do stuff"}], 20
            )

        assert "Summary" in result or result == "Summary done."
        assert "tools" not in captured
        assert "tool_choice" not in captured
        assert "parallel_tool_calls" not in captured

    def test_chat_summary_still_works(self, agent):
        agent.api_mode = "chat_completions"
        agent._cached_system_prompt = "You are helpful."
        agent.client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Chat summary.", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        # normalize_response path — mock transport if needed
        with patch.object(
            agent,
            "_ensure_primary_openai_client",
            return_value=agent.client,
        ):
            # Use a simple normalize that returns content
            mock_transport = MagicMock()
            mock_transport.normalize_response.return_value = SimpleNamespace(
                content="Chat summary."
            )
            with patch.object(agent, "_get_transport", return_value=mock_transport):
                result = agent._handle_max_iterations(
                    [{"role": "user", "content": "do stuff"}], 20
                )
        assert "Chat summary" in result or result == "Chat summary."
        kwargs = agent.client.chat.completions.create.call_args.kwargs
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs

    def test_anthropic_summary_tools_none(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent._is_anthropic_oauth = False
        agent._cached_system_prompt = "You are helpful."
        agent.max_tokens = 1024
        agent.reasoning_config = None

        mock_transport = MagicMock()
        mock_transport.build_kwargs.return_value = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1024,
        }
        mock_transport.normalize_response.return_value = SimpleNamespace(
            content="Anthropic summary."
        )

        with (
            patch.object(agent, "_get_transport", return_value=mock_transport),
            patch.object(
                agent,
                "_anthropic_messages_create",
                return_value=SimpleNamespace(content=[]),
            ) as create_mock,
        ):
            result = agent._handle_max_iterations(
                [{"role": "user", "content": "do stuff"}], 20
            )

        assert "Anthropic summary" in result
        # build_kwargs must have been called with tools=None
        assert mock_transport.build_kwargs.call_args.kwargs.get("tools") is None
        create_mock.assert_called()
        sent = create_mock.call_args.args[0]
        assert "tools" not in sent
        assert "tool_choice" not in sent


# ---------------------------------------------------------------------------
# Preflight on main send path
# ---------------------------------------------------------------------------


class TestInterruptibleApiCallPreflight:
    def test_rejects_illegal_kwargs_before_provider(self, agent):
        from agent.chat_completion_helpers import interruptible_api_call
        from agent.errors import RuntimeContractViolation

        agent.api_mode = "codex_responses"
        bad = {"model": "gpt-5.5", "tool_choice": "auto"}  # no tools
        with pytest.raises(RuntimeContractViolation):
            interruptible_api_call(agent, bad)
