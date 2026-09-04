"""Behavioral tests for provider-confirmed compression-budget rearming.

``compression_attempts`` is a shared per-turn backstop (pre-API gate,
overflow/413 handlers, post-tool gate). Before the refund fix, *successful*
pre-API compactions consumed it permanently: a marathon tool turn burned all
attempts on compactions that worked, the pre-API gate went dark for the rest
of the turn, and the context grew unchecked until the provider rejected the
request terminally ("max compression attempts (N) reached").

The budget is rearmed only when a completed compaction is followed by a real
provider prompt count below the configured threshold. Rough estimates and
usage-less responses cannot reopen the anti-thrash cap.

These tests drive ``run_conversation()`` through real tool iterations — no
source inspection, only observable compaction counts.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _should_rearm_compression_budget
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Unit: refund decision
# ---------------------------------------------------------------------------


class TestRearmDecision:
    def test_provider_confirmed_recovery_rearms(self):
        assert _should_rearm_compression_budget(
            2,
            completed_compaction_pending=True,
            prompt_tokens=7_999,
            threshold_tokens=10_000,
        )

    @pytest.mark.parametrize(
        ("attempts", "pending", "prompt_tokens", "threshold_tokens"),
        [
            (0, True, 7_999, 10_000),
            (2, False, 7_999, 10_000),
            (2, True, 0, 10_000),
            (2, True, 10_000, 10_000),
            (2, True, 10_001, 10_000),
            (2, True, 7_999, 0),
        ],
    )
    def test_unverified_or_pressured_response_keeps_budget_burned(
        self, attempts, pending, prompt_tokens, threshold_tokens
    ):
        assert not _should_rearm_compression_budget(
            attempts,
            completed_compaction_pending=pending,
            prompt_tokens=prompt_tokens,
            threshold_tokens=threshold_tokens,
        )


# ---------------------------------------------------------------------------
# Behavioral: marathon tool turn keeps compacting past the old cap
# ---------------------------------------------------------------------------


def _tool_call(i: int):
    return SimpleNamespace(
        id=f"call_{i}",
        type="function",
        # Vary the query per call: a real marathon turn issues distinct
        # lookups, and identical (args, result) pairs are now legitimately
        # deduped into reference stubs by the stall-guard subsystem —
        # zero-variance args here would deflate the very context pressure
        # this test exists to exercise.
        function=SimpleNamespace(name="web_search", arguments=f'{{"query": "x{i}"}}'),
    )


def _usage(prompt_tokens: int | None):
    if prompt_tokens is None:
        return None
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        total_tokens=prompt_tokens + 1,
    )


def _tool_response(i: int, prompt_tokens: int | None):
    msg = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call(i)],
    )
    choice = SimpleNamespace(message=msg, finish_reason="tool_calls")
    return SimpleNamespace(
        choices=[choice], model="test/model", usage=_usage(prompt_tokens)
    )


def _stop_response(prompt_tokens: int | None):
    msg = SimpleNamespace(
        content="done",
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice], model="test/model", usage=_usage(prompt_tokens)
    )


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


THRESHOLD = 10_000

# Each tool result is large enough that the assembled request crosses
# THRESHOLD every iteration (estimator is ~chars/4), forcing one pre-API
# compaction per iteration — but stays below the 100K-char per-result
# persistence threshold (tools/budget_config.py) so it reaches the context
# untruncated.
BIG_TOOL_RESULT = "x" * 60_000


def _coherent_compressor() -> MagicMock:
    """A compressor whose should_compress() reflects the passed estimate.

    Unlike the always-True stub in the attempt-cap tests, this models the
    real coupling: pressure at/over threshold → compress; pressure gone →
    healthy. That coupling is what makes the refund safe.
    """
    compressor = MagicMock()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 20
    compressor.threshold_tokens = THRESHOLD
    compressor.context_length = 200_000
    compressor.last_prompt_tokens = 0
    compressor._verify_compaction_cleared_threshold = False
    compressor.awaiting_real_usage_after_compression = False
    compressor.should_compress.side_effect = lambda t=None: (t or 0) >= THRESHOLD
    compressor.should_defer_preflight_to_real_usage.return_value = False
    compressor.get_active_compression_failure_cooldown.return_value = None

    def _update_from_response(usage):
        compressor.last_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        compressor._verify_compaction_cleared_threshold = False
        compressor.awaiting_real_usage_after_compression = False

    compressor.update_from_response.side_effect = _update_from_response
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
            max_iterations=20,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a._disable_streaming = True
    a.tool_delay = 0
    a.save_trajectories = False
    a.compression_enabled = True
    a.context_compressor = _coherent_compressor()
    return a


def _run_marathon_turn(
    agent, n_tool_iterations: int, *, provider_prompt_tokens: int | None
):
    """Drive one turn of ``n_tool_iterations`` oversized tool results."""
    responses = [
        _tool_response(i, provider_prompt_tokens) for i in range(n_tool_iterations)
    ]
    responses.append(_stop_response(provider_prompt_tokens))
    agent.client.chat.completions.create.side_effect = responses

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        # Model a compaction that works: blank out every oversized payload,
        # keeping roles and tool-call pairing intact so sanitization is
        # unaffected. Arm the same provider-verification boundary as the real
        # compression path.
        compress_calls.append(len(messages))
        agent.context_compressor._verify_compaction_cleared_threshold = True
        agent.context_compressor.awaiting_real_usage_after_compression = True
        compacted = [
            dict(m, content="[summarized]")
            if isinstance(m, dict) and len(str(m.get("content") or "")) > 5_000
            else m
            for m in messages
        ]
        return compacted, "compressed prompt"

    with (
        patch.object(agent, "_compress_context", side_effect=_fake_compress),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "model_tools.handle_function_call",
            lambda name, args, task_id=None, **kwargs: json.dumps(
                {"ok": True, "payload": BIG_TOOL_RESULT}
            ),
        ),
    ):
        result = agent.run_conversation("do a lot of tool work")

    return result, compress_calls


class TestCompressionBudgetRefund:
    def test_marathon_turn_compacts_past_the_per_turn_cap(self, agent):
        """8 oversized tool iterations → more compactions than the old cap.

        Pre-refund, the 4th+ pressure spike found the budget exhausted, the
        pre-API gate stayed dark, and the request grew unchecked. With the
        refund, every genuine pressure spike is compacted and the turn
        completes.
        """
        assert agent.max_compression_attempts == 3  # config default
        result, compress_calls = _run_marathon_turn(
            agent,
            n_tool_iterations=8,
            provider_prompt_tokens=THRESHOLD - 1,
        )

        assert result["completed"] is True
        assert len(compress_calls) > 3, (
            "successful compactions must refund the per-turn budget; "
            f"got only {len(compress_calls)} compactions for 8 pressure spikes"
        )

    @pytest.mark.parametrize("provider_prompt_tokens", [None, THRESHOLD])
    def test_unverified_or_pressured_compaction_stays_capped(
        self, agent, provider_prompt_tokens
    ):
        """Missing usage or real usage at threshold cannot recycle the cap."""
        result, compress_calls = _run_marathon_turn(
            agent,
            n_tool_iterations=8,
            provider_prompt_tokens=provider_prompt_tokens,
        )

        assert result["completed"] is True
        assert len(compress_calls) <= agent.max_compression_attempts, (
            "without provider-confirmed headroom the per-turn cap must hold; "
            f"got {len(compress_calls)} compactions"
        )
