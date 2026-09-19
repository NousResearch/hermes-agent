"""Behavioral regression tests for the post-tool compression attempt cap.

The pre-API pressure gate, the overflow/413 error handlers, and the post-tool
compaction gate all share ``compression_attempts`` as a per-turn backstop,
bounded by the resolved ``compression.max_attempts`` cap (default 3).  Before
the fix the post-tool path neither checked nor incremented the counter, so a
long tool loop could compact after every tool response for the lifetime of
the turn.

These tests drive ``run_conversation()`` through real tool iterations with a
compressor that always demands compression and assert ``_compress_context``
fires at most ``max_compression_attempts`` times per turn — no source
inspection, only observable behavior.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_preflight import compress_after_tool_results
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


def _pressured_compressor() -> MagicMock:
    """A compressor that always reports context pressure after tools run.

    ``should_defer_preflight_to_real_usage`` returns True so the turn-start
    preflight and the pre-API pressure gate stand down — isolating the
    post-tool gate as the only compression site under test.
    """
    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = 10_000
    compressor.context_length = 200_000
    compressor.last_prompt_tokens = 150_000
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.return_value = True
    compressor.should_defer_preflight_to_real_usage.return_value = True
    compressor.get_active_compression_failure_cooldown.return_value = None
    return compressor


@pytest.fixture()
def agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
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
    a.context_compressor = _pressured_compressor()
    return a


def _run_tool_loop(agent, n_tool_iterations: int):
    """Drive one turn: ``n_tool_iterations`` tool calls, then a stop."""
    responses = [_tool_response(i) for i in range(n_tool_iterations)]
    responses.append(_stop_response())
    agent.client.chat.completions.create.side_effect = responses

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    with (
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "model_tools.handle_function_call",
            lambda name, args, task_id=None, **kwargs: json.dumps({"ok": True}),
        ),
    ):
        result = agent.run_conversation("do a lot of tool work")

    return result, compress_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnnounceCompressionStartPreEmit:
    def test_pre_emitted_status_skips_emit_but_arms_lifecycle(self):
        from agent.conversation_compression import _announce_compression_start

        emit = MagicMock()
        agent = SimpleNamespace(
            session_id="s",
            model="test/model",
            context_compressor=SimpleNamespace(emit_automatic_compaction_status=True),
            _emit_status=emit,
        )

        lifecycle = _announce_compression_start(
            agent,
            message_count=2,
            approx_tokens=100,
            focus_topic=None,
            force=False,
            pre_emitted_status="already shown",
        )

        emit.assert_not_called()
        assert lifecycle.status_emitted is True

    def test_without_pre_emitted_status_emits_once(self):
        from agent.conversation_compression import COMPACTION_STATUS, _announce_compression_start

        emit = MagicMock()
        agent = SimpleNamespace(
            session_id="s",
            model="test/model",
            context_compressor=SimpleNamespace(emit_automatic_compaction_status=True),
            _emit_status=emit,
        )

        lifecycle = _announce_compression_start(
            agent,
            message_count=2,
            approx_tokens=100,
            focus_topic=None,
            force=False,
        )

        emit.assert_called_once_with(COMPACTION_STATUS)
        assert lifecycle.status_emitted is True


class TestPostToolCompressionAttemptCap:
    def test_post_tool_compression_announces_before_blocking(self, agent):
        """The remote client gets a compacting frame before compression can block."""
        statuses = []
        agent._emit_status = statuses.append
        del agent.context_compressor.get_automatic_compaction_status_message
        messages = [
            {"role": "user", "content": "save this"},
            {"role": "assistant", "content": None, "tool_calls": []},
            {"role": "tool", "tool_call_id": "call-1", "content": "saved"},
        ]

        def assert_status_announced(_messages):
            assert statuses == [
                "🗜️ Compacting context — summarizing earlier conversation so I can continue..."
            ]

        agent._compress_context = MagicMock(side_effect=lambda received, *_args, **_kwargs: (
            assert_status_announced(received) or (received, "compressed prompt")
        ))
        verdict = compress_after_tool_results(
            agent,
            messages=messages,
            system_message="You are helpful.",
            user_message=messages[0],
            active_system_prompt="You are helpful.",
            conversation_history=messages,
            compression_attempts=0,
            max_compression_attempts=3,
            effective_task_id="test",
            final_response="",
            turn_exit_reason=None,
        )

        assert verdict.compression_attempts == 1
        agent._compress_context.assert_called_once()
        assert agent._compress_context.call_args.kwargs.get("pre_emitted_status") == (
            "🗜️ Compacting context — summarizing earlier conversation so I can continue..."
        )

    def test_post_tool_gate_waits_for_usage_after_native_checkpoint(self, agent):
        agent.context_compressor.awaiting_real_usage_after_compression = True

        result, compress_calls = _run_tool_loop(agent, n_tool_iterations=1)

        assert result["completed"] is True
        assert compress_calls == []

    def test_post_tool_compression_capped_at_default_three(self, agent):
        """7 tool iterations under constant pressure → exactly 3 compactions.

        Before the fix the post-tool gate re-fired after every tool response
        (7 compactions here); the shared per-turn counter caps it at the
        resolved default of 3.
        """
        assert agent.max_compression_attempts == 3  # config default
        result, compress_calls = _run_tool_loop(agent, n_tool_iterations=7)

        assert result["completed"] is True
        assert len(compress_calls) == 3, (
            f"post-tool compression must stop at the per-turn cap (3), "
            f"got {len(compress_calls)} compactions"
        )


    def test_post_tool_compression_shares_counter_with_pre_api_gate(self, agent):
        """Pre-API compactions consume the same per-turn budget.

        Let the pre-API pressure gate fire once (defer disabled for the first
        check), then keep the pressure on through tool iterations: the
        combined total must still respect the cap.
        """
        # First pre-API check does not defer → pre-API gate fires once;
        # afterwards defer again so only the post-tool gate keeps firing.
        defers = iter([False])
        agent.context_compressor.should_defer_preflight_to_real_usage.side_effect = (
            lambda _t: next(defers, True)
        )
        result, compress_calls = _run_tool_loop(agent, n_tool_iterations=7)

        assert result["completed"] is True
        assert len(compress_calls) == 3, (
            "pre-API and post-tool compactions must share one per-turn "
            f"attempt budget, got {len(compress_calls)} total compactions"
        )

    def test_cap_is_per_turn_not_per_session(self, agent):
        """A fresh turn gets a fresh attempt budget."""
        _result, first = _run_tool_loop(agent, n_tool_iterations=5)
        agent.client.chat.completions.create.side_effect = None
        _result, second = _run_tool_loop(agent, n_tool_iterations=5)

        assert len(first) == 3
        assert len(second) == 3
