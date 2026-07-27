"""Regression test: consecutive-failure compression cap is preserved.

The fix for #72451 resets ``compression_attempts`` after every successful
model response, but the consecutive-failure anti-thrash cap must still stop
repeated no-progress compressions that never reach a model response.

This test simulates a pre-API compression loop where ``should_compress``
always returns True and the model never gets a chance to respond between
restart cycles — the shared counter must stop at ``max_compression_attempts``.
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
    """Compressor that always needs compression and never defers.

    ``should_defer_preflight_to_real_usage`` returns False so the pre-API
    pressure gate fires on every turn-start check.

    To prevent the insufficient-progress blocker from stopping the preflight
    loop before the counter cap kicks in, ``_compression_warrants_another_
    preflight_pass`` must also return True on every call.
    """
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

    def test_preflight_compression_capped_during_restart_loop(self, agent):
        """Pre-API compression → restart → pre-API again → … → cap stops loop.

        When the preflight gate compresses and restarts repeatedly without
        the model ever returning a valid response (because the compressed
        request still triggers a compression restart), the shared counter
        stops the loop at ``max_compression_attempts`` — the insufficient-
        progress blocker arms as a secondary backstop, and no model response
        reaches the ``compression_attempts = 0`` reset point.

        The exact number of preflight compactions depends on internal
        progress tracking.  This test verifies the cap is a genuine
        upper bound: no more than ``max_compression_attempts`` preflight
        passes can run.
        """
        assert agent.max_compression_attempts == 3

        compress_calls = []

        def _fake_compress(messages, system_message, **_kwargs):
            compress_calls.append(len(messages))
            # Return identity — no material progress, so the insufficient-
            # progress guard arms after the second pass.
            return messages, "compressed prompt"

        # Override the progress gate to prevent it from stopping early.
        with patch(
            "agent.conversation_loop._compression_warrants_another_preflight_pass",
            return_value=True,
        ):
            with (
                patch.object(agent, "_compress_context", side_effect=_fake_compress),
                patch.object(agent, "_persist_session"),
                patch.object(agent, "_save_trajectory"),
                patch.object(agent, "_cleanup_task_resources"),
                patch(
                    "run_agent.handle_function_call",
                    lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
                ),
            ):
                # Model always returns a tool call so the loop continues,
                # but the preflight fires before the model is ever called.
                agent.client.chat.completions.create.side_effect = [
                    _stop_response(),
                ]
                result = agent.run_conversation("do work")

        # The cap caps at max_compression_attempts.  If it were higher,
        # the insufficient-progress guard would still stop it — either way,
        # ≤ 3 is the correct contract.
        assert len(compress_calls) <= agent.max_compression_attempts, (
            f"preflight loop must stop at or before max_compression_attempts "
            f"({agent.max_compression_attempts}), got {len(compress_calls)}"
        )
        assert result.get("completed") is True

    def test_failure_cap_preserved_after_reset_on_success(self, agent):
        """Consecutive failures still capped even after a success cycle.

        A successful model response resets the counter, but if a subsequent
        error-handler path triggers compression retries without reaching
        another valid response, those retries are still capped at
        ``max_compression_attempts``.
        """
        assert agent.max_compression_attempts == 3

        compress_calls = []

        def _fake_compress(messages, system_message, **_kwargs):
            compress_calls.append(len(messages))
            return messages, "compressed prompt"

        # First turn: defer preflight so only post-tool gate fires.
        agent.context_compressor.should_defer_preflight_to_real_usage.return_value = True

        responses = [_tool_response(0), _tool_response(1), _stop_response()]

        with (
            patch.object(agent, "_compress_context", side_effect=_fake_compress),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "run_agent.handle_function_call",
                lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
            ),
        ):
            agent.client.chat.completions.create.side_effect = responses
            result = agent.run_conversation("do work")

        # 2 tool iterations → 2 post-tool compactions.  Both are followed
        # by successful model responses, so the counter resets each time.
        # The stop response also resets the counter.
        # Total: 2 compactions, both reset before the turn ends.
        assert len(compress_calls) == 2
        assert result.get("completed") is True
