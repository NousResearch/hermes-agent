"""Regression test for #101445: terminal 429 responses carry no recovery hint.

Scenario: a classified rate limit exhausts the retry budget.  The terminal
``_final_response`` used to be only::

    API call failed after N retries: HTTP 429: Provider returned error

— the classified ``is_rate_limited`` signal was discarded when rendering the
user-facing message, so ``:free``-tier users could not tell an RPM/RPD cap
from an account problem, nor how to recover (fund, or switch the caller off
the free tier).

The fix mirrors the existing billing/thinking-timeout terminal guidance
patterns: when ``is_rate_limited`` survives to the terminal branch, append
``_rate_limit_terminal_hint(model)`` — tier-shaped text for ``:free`` models
vs paid models.  Retry/backoff behavior is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _rate_limit_terminal_hint
from run_agent import AIAgent


def _make_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _make_agent(model: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key-abcdef12",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            model=model,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._api_max_retries = 2
    return agent


def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - HTTP 429: Provider returned error")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "Provider returned error"}}


class ServerError(Exception):
    status_code = 500

    def __init__(self):
        super().__init__("Error code: 500 - internal server error")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "internal server error"}}


# --- Pure helper unit tests -------------------------------------------------


class TestRateLimitTerminalHintHelper:
    def test_free_tier_hint_names_model_and_recovery(self):
        hint = _rate_limit_terminal_hint("minimax/minimax-m3:free")
        assert "This looks like a `:free`-tier rate limit" in hint
        assert "minimax/minimax-m3:free" in hint
        # The cron escape hatch must reference the bare (non-:free) model.
        assert "hermes cron edit <id> --model minimax/minimax-m3`" in hint
        assert "https://openrouter.ai/credits" in hint

    def test_free_suffix_match_is_case_insensitive(self):
        hint = _rate_limit_terminal_hint("minimax/minimax-m3:Free")
        assert "This looks like a `:free`-tier rate limit" in hint
        assert "hermes cron edit <id> --model minimax/minimax-m3`" in hint

    def test_paid_tier_hint_is_transient_guidance(self):
        hint = _rate_limit_terminal_hint("glm-5.1")
        assert "Rate limit hit on a paid model" in hint
        # Paid guidance must not point at the free-tier funding path.
        assert "openrouter.ai/credits" not in hint
        assert "hermes cron edit" not in hint

    @pytest.mark.parametrize("model", ["", "   ", None])
    def test_blank_model_yields_no_hint(self, model):
        assert _rate_limit_terminal_hint(model) == ""

    def test_hint_always_starts_with_paragraph_break(self):
        # Terminal append contract: the hint is a new paragraph after the
        # summary line, like the billing/thinking-timeout guidance.
        for model in ("minimax/minimax-m3:free", "glm-5.1"):
            assert _rate_limit_terminal_hint(model).startswith("\n\n")


# --- Loop-driven terminal-response tests -------------------------------------


class TestTerminal429ResponseCarriesHint:
    """Drive ``run_conversation`` until the 429 retry budget is exhausted and
    assert the terminal response carries the tier-shaped recovery hint."""

    @pytest.mark.parametrize(
        ("model", "expected_marker", "forbidden_marker"),
        [
            (
                "minimax/minimax-m3:free",
                "This looks like a `:free`-tier rate limit",
                "Rate limit hit on a paid model",
            ),
            (
                "glm-5.1",
                "Rate limit hit on a paid model",
                "This looks like a `:free`-tier rate limit",
            ),
        ],
    )
    def test_429_terminal_response_includes_hint(self, model, expected_marker, forbidden_marker):
        agent = _make_agent(model)

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=RateLimitError()),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_dump_api_request_debug"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.conversation_loop.adaptive_rate_limit_backoff",
                return_value=(0.0, None),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        final = result["final_response"]
        # The original summary line must survive — the hint is appended, never a replacement.
        assert final.startswith("API call failed after 2 retries: ")
        assert expected_marker in final
        assert forbidden_marker not in final
        assert result["completed"] is False
        assert result["failed"] is True
        # The classifier reports either ``rate_limit`` or
        # ``upstream_rate_limit`` for a 429 depending on where the cap was
        # hit; ``is_rate_limited`` covers both, so accept both here.
        assert result["failure_reason"] in ("rate_limit", "upstream_rate_limit")

    def test_non_429_terminal_response_has_no_hint(self):
        """Control: a 500-class failure must not grow a rate-limit hint."""
        agent = _make_agent("glm-5.1")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=ServerError()),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_dump_api_request_debug"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.conversation_loop.adaptive_rate_limit_backoff",
                return_value=(0.0, None),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        final = result["final_response"]
        assert final.startswith("API call failed after 2 retries: ")
        assert "Rate limit hit on a paid model" not in final
        assert "This looks like a `:free`-tier rate limit" not in final
        assert result["failure_reason"] not in ("rate_limit", "upstream_rate_limit")


class TestStreamingWorker429Path:
    """#101445 follow-up (dimkin-eu report): a 429 raised inside the streaming
    worker (``interruptible_streaming_api_call`` → ``chat_completion_helpers``
    sets ``result["error"]``, which the wrapper re-raises) must still reach the
    terminal branch with the rate-limit hint — the worker channel is a
    different *call site*, not a different classifier verdict.

    The reporter hit exactly this shape: ``Streaming failed before delivery:
    ... HTTP 429: Provider returned error`` rendered bare on stock main.  These
    tests force ``_use_streaming`` on and raise the 429 from the streaming
    entry point, then assert the hint is emitted through the same terminal
    branch the non-streaming path uses.
    """

    def _stream_agent(self, model: str) -> AIAgent:
        agent = _make_agent(model)
        # Streaming is decided per-call: no stream consumers + Mock client
        # would flip _use_streaming off. Claim a consumer so the loop routes
        # every attempt through _interruptible_streaming_api_call.
        agent._has_stream_consumers = lambda: True
        agent._disable_streaming = False
        return agent

    @pytest.mark.parametrize(
        ("model", "expected_marker", "forbidden_marker"),
        [
            (
                "minimax/minimax-m3:free",
                "This looks like a `:free`-tier rate limit",
                "Rate limit hit on a paid model",
            ),
            (
                "glm-5.1",
                "Rate limit hit on a paid model",
                "This looks like a `:free`-tier rate limit",
            ),
        ],
    )
    def test_streaming_429_still_emits_hint(self, model, expected_marker, forbidden_marker):
        agent = self._stream_agent(model)

        with (
            # Streaming worker channel: chat_completion_helpers stores the
            # worker error in result["error"]; interruptible_streaming_api_call
            # re-raises it. Raising from the entry point models the same
            # propagation into the main retry loop.
            patch.object(
                agent, "_interruptible_streaming_api_call",
                side_effect=RateLimitError(),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_dump_api_request_debug"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.conversation_loop.adaptive_rate_limit_backoff",
                return_value=(0.0, None),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        final = result["final_response"]
        assert final.startswith("API call failed after 2 retries: ")
        assert expected_marker in final
        assert forbidden_marker not in final
        assert result["failure_reason"] in ("rate_limit", "upstream_rate_limit")

    def test_streaming_non_429_has_no_hint(self):
        """Control through the streaming channel: a 500-class failure stays bare."""
        agent = self._stream_agent("glm-5.1")

        with (
            patch.object(
                agent, "_interruptible_streaming_api_call",
                side_effect=ServerError(),
            ),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_dump_api_request_debug"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.conversation_loop.adaptive_rate_limit_backoff",
                return_value=(0.0, None),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        final = result["final_response"]
        assert final.startswith("API call failed after 2 retries: ")
        assert "Rate limit hit on a paid model" not in final
        assert "This looks like a `:free`-tier rate limit" not in final
