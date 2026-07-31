"""Behavioral tests for the per-turn compression budget refund.

``compression_attempts`` is a shared per-turn backstop (pre-API gate,
overflow/413 handlers, post-tool gate). Before the refund fix, *successful*
pre-API compactions consumed it permanently: a marathon tool turn burned all
attempts on compactions that worked, the pre-API gate went dark for the rest
of the turn, and the context grew unchecked until the provider rejected the
request terminally ("max compression attempts (N) reached").

The refund returns the budget when BOTH hold at the top of a loop pass:

* the assembled request sits below ``threshold_tokens *
  _COMPRESSION_BUDGET_REFUND_MARGIN`` (real progress, not a borderline
  shrink), and
* the compressor's own ``should_compress()`` agrees there is no pressure
  (divergent-signal guard — a compressor that still demands compression
  keeps the hard cap's original meaning, see
  test_post_tool_compression_attempt_cap.py).

These tests drive ``run_conversation()`` through real tool iterations — no
source inspection, only observable compaction counts.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _should_refund_compression_budget
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Unit: refund decision
# ---------------------------------------------------------------------------


class TestRefundDecision:
    def test_no_attempts_no_refund(self):
        assert not _should_refund_compression_budget(0, 100, 10_000)

    def test_zero_threshold_no_refund(self):
        assert not _should_refund_compression_budget(2, 100, 0)

    def test_barely_under_threshold_no_refund(self):
        # 9,999 of 10,000 is inside the anti-thrash belt (margin 0.8).
        assert not _should_refund_compression_budget(2, 9_999, 10_000)

    def test_at_margin_no_refund(self):
        assert not _should_refund_compression_budget(2, 8_000, 10_000)

    def test_comfortably_under_margin_refunds(self):
        assert _should_refund_compression_budget(2, 7_999, 10_000)
        assert _should_refund_compression_budget(1, 100, 10_000)


# ---------------------------------------------------------------------------
# Behavioral: marathon tool turn keeps compacting past the old cap
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
    compressor.should_compress.side_effect = lambda t=None: (t or 0) >= THRESHOLD
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


def _run_marathon_turn(agent, n_tool_iterations: int):
    """Drive one turn of ``n_tool_iterations`` oversized tool results."""
    responses = [_tool_response(i) for i in range(n_tool_iterations)]
    responses.append(_stop_response())
    agent.client.chat.completions.create.side_effect = responses

    compress_calls = []

    def _fake_compress(messages, system_message, **_kwargs):
        # Model a compaction that works: blank out every oversized payload,
        # keeping roles and tool-call pairing intact so sanitization is
        # unaffected. Pressure drops far below the refund margin.
        compress_calls.append(len(messages))
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
            "run_agent.handle_function_call",
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
        result, compress_calls = _run_marathon_turn(agent, n_tool_iterations=8)

        assert result["completed"] is True
        assert len(compress_calls) > 3, (
            "successful compactions must refund the per-turn budget; "
            f"got only {len(compress_calls)} compactions for 8 pressure spikes"
        )

    def test_no_refund_when_compressor_still_reports_pressure(self, agent):
        """Divergent signals: compressor demands compression regardless of
        the local estimate → budget stays burnt at the hard cap.

        Mirrors the always-True stub of the attempt-cap regression tests —
        the refund must not reopen that runaway."""
        agent.context_compressor.should_compress.side_effect = None
        agent.context_compressor.should_compress.return_value = True

        result, compress_calls = _run_marathon_turn(agent, n_tool_iterations=8)

        assert result["completed"] is True
        assert len(compress_calls) <= agent.max_compression_attempts, (
            "with should_compress pinned True the per-turn cap must hold; "
            f"got {len(compress_calls)} compactions"
        )
