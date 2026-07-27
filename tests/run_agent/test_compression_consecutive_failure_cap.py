"""Regression test: consecutive-failure compression cap is preserved.

The fix for #72451 only resets ``compression_attempts`` when the most
recent compression materially reduced context AND the model responded
successfully.  Identity/no-progress compressions still count toward the
consecutive-failure cap.

Tests verify:
- Identity preflight loop stops at ``max_compression_attempts``
- Effective compressions reset the streak after a successful model response
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_call(i: int):
    return SimpleNamespace(
        id=f"call_{i}",
        type="function",
        function=SimpleNamespace(name="web_search", arguments='{"query": "x"}'),
    )


def _tool_response(i: int):
    msg = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call(i)],
    )
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _stop_response():
    msg = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


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


def _preflight_pressure_compressor() -> MagicMock:
    """Compressor that always needs compression and never defers."""
    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 10_000
    compressor.context_length = 200_000
    compressor.last_prompt_tokens = 150_000
    compressor.should_compress.return_value = True
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None
    return compressor


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=10,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.tool_delay = 0
    a.save_trajectories = False
    a.compression_enabled = True
    a.context_compressor = _preflight_pressure_compressor()
    return a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConsecutiveFailureCompressionCap:

    def test_identity_preflight_stops_at_cap(self, agent):
        """Identity preflight compression stops at max_compression_attempts.

        When the compressor returns messages unchanged, the effectiveness
        tracker never arms, so the counter keeps climbing.  The
        insufficient-progress blocker arms as a secondary backstop.
        """
        assert agent.max_compression_attempts == 3

        compress_calls = []

        def _fake_compress(messages, system_message, **_kwargs):
            compress_calls.append(len(messages))
            return messages, "compressed prompt"

        with (
            patch("agent.conversation_loop._compression_warrants_another_preflight_pass", return_value=True),
            patch.object(agent, "_compress_context", side_effect=_fake_compress),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.handle_function_call", lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True})),
        ):
            agent.client.chat.completions.create.side_effect = [_stop_response()]
            result = agent.run_conversation("do work")

        assert len(compress_calls) <= agent.max_compression_attempts, (
            f"identity preflight loop must stop at or before "
            f"max_compression_attempts ({agent.max_compression_attempts}), "
            f"got {len(compress_calls)}"
        )
        assert result.get("completed") is True

    def test_effective_compression_resets_streak_on_success(self, agent):
        """Effective compression + successful model response → streak reset.

        Post-tool compactions that return new message lists (material progress)
        reset the counter after each successful model response.  Two tool
        iterations → two effective compactions → both reset.
        """
        assert agent.max_compression_attempts == 3

        compress_calls = []

        def _fake_compress_effective(messages, system_message, **_kwargs):
            compress_calls.append(len(messages))
            return messages, "compressed prompt"

        agent.context_compressor._last_compression_made_progress = True
        agent.context_compressor.should_defer_preflight_to_real_usage.return_value = True

        responses = [_tool_response(0), _tool_response(1), _stop_response()]

        with (
            patch.object(agent, "_compress_context", side_effect=_fake_compress_effective),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.handle_function_call", lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True})),
        ):
            agent.client.chat.completions.create.side_effect = responses
            result = agent.run_conversation("do work")

        # Both compactions are effective, so each resets after model response.
        assert len(compress_calls) == 2
        assert result.get("completed") is True
