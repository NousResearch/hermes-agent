"""Behavioral regression tests for the post-tool compression attempt cap.

The pre-API pressure gate, the overflow/413 error handlers, and the post-tool
compaction gate all share ``compression_attempts`` as a per-turn backstop,
bounded by the resolved ``compression.max_attempts`` cap (default 3).

Identity/no-progress compactions that return the same messages unchanged
count toward the consecutive-failure cap — they stop at
``max_compression_attempts``.  Materially-effective compactions that actually
reduce context reset the streak after a successful model response (#72451).
"""

from __future__ import annotations

import copy
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
    compressor.should_compress.return_value = True
    compressor.should_defer_preflight_to_real_usage.return_value = True
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
    a.context_compressor = _pressured_compressor()
    return a


def _run_tool_loop(agent, n_tool_iterations: int, *, effective=False):
    """Drive one turn: ``n_tool_iterations`` tool calls, then a stop.

    When ``effective=True``, the compressor returns a shallow copy of
    ``messages`` so the effectiveness tracker sees material progress
    and resets the streak after each successful model response.
    """
    responses = [_tool_response(i) for i in range(n_tool_iterations)]
    responses.append(_stop_response())
    agent.client.chat.completions.create.side_effect = responses

    compress_calls = []

    def _fake_compress_identity(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    def _fake_compress_effective(messages, system_message, **_kwargs):
        compress_calls.append(len(messages))
        return messages, "compressed prompt"

    _fake_compress = _fake_compress_effective if effective else _fake_compress_identity

    # Mirror the real compressor's progress flag so the effectiveness
    # tracker sees material progress only when the test expects it.
    agent.context_compressor._last_compression_made_progress = effective

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
        result = agent.run_conversation("do a lot of tool work")

    return result, compress_calls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostToolCompressionAttemptCap:

    # ── Identity (no-progress) compressor — cap enforced ──────────────

    def test_identity_compressor_capped_at_default_three(self, agent):
        """7 tool iterations, identity compressor → exactly 3 compactions.

        The compressor returns messages unchanged, so the effectiveness
        tracker never resets the streak.  The per-turn cap of 3 acts as
        the anti-thrash backstop for no-progress cycles.
        """
        assert agent.max_compression_attempts == 3
        result, compress_calls = _run_tool_loop(agent, n_tool_iterations=7)

        assert result["completed"] is True
        assert len(compress_calls) == 3, (
            "identity/no-progress compactions must stop at the per-turn "
            f"cap (3), got {len(compress_calls)}"
        )

    def test_identity_compressor_honors_configured_cap(self, agent):
        """A raised cap allows more no-progress rounds before stopping."""
        agent.max_compression_attempts = 5
        result, compress_calls = _run_tool_loop(agent, n_tool_iterations=8)

        assert result["completed"] is True
        assert len(compress_calls) == 5

    def test_identity_compressor_shares_counter_with_pre_api_gate(self, agent):
        """Pre-API and post-tool sites share the same failure streak.

        The pre-API gate fires once (defer disabled), then the post-tool
        gate fires for the remaining budget.  Identity compressions mean
        no reset — the combined total stays at ``max_compression_attempts``.
        """
        defers = iter([False])
        agent.context_compressor.should_defer_preflight_to_real_usage.side_effect = (
            lambda _t: next(defers, True)
        )
        result, compress_calls = _run_tool_loop(agent, n_tool_iterations=7)

        assert result["completed"] is True
        assert len(compress_calls) == 3, (
            "pre-API and post-tool together must respect the shared "
            f"cap for identity compressions; got {len(compress_calls)}"
        )

    def test_identity_cap_is_per_turn_not_per_session(self, agent):
        """A fresh turn gets a fresh attempt budget."""
        _result, first = _run_tool_loop(agent, n_tool_iterations=5)
        agent.client.chat.completions.create.side_effect = None
        _result, second = _run_tool_loop(agent, n_tool_iterations=5)

        assert len(first) == 3
        assert len(second) == 3

    # ── Effective compressor — streak reset on success ────────────────

    def test_effective_compressor_allows_more_than_three_cycles(self, agent):
        """7 tool iterations, effective compressor → all 7 compactions run.

        Each compaction returns a new messages list (material progress),
        so the streak resets after every successful model response.
        Long tool turns can sustain unlimited maintenance compactions (#72451).
        """
        assert agent.max_compression_attempts == 3
        result, compress_calls = _run_tool_loop(
            agent, n_tool_iterations=7, effective=True,
        )

        assert result["completed"] is True
        assert len(compress_calls) == 7, (
            "effective compactions must reset the streak after every "
            f"successful model response; got {len(compress_calls)}"
        )

    def test_effective_compressor_shares_counter_with_pre_api_gate(self, agent):
        """Effective compactions from both sites reset after success.

        1 pre-API (effective) + 7 post-tool (effective) = 8 compactions
        total, because each cycle resets the streak.
        """
        defers = iter([False])
        agent.context_compressor.should_defer_preflight_to_real_usage.side_effect = (
            lambda _t: next(defers, True)
        )
        result, compress_calls = _run_tool_loop(
            agent, n_tool_iterations=7, effective=True,
        )

        assert result["completed"] is True
        assert len(compress_calls) == 8, (
            "effective pre-API + post-tool cycles must each reset; "
            f"expected 8, got {len(compress_calls)}"
        )

    def test_effective_cap_per_turn_fresh_budget(self, agent):
        """Effective compressions per-turn budget is independent across turns."""
        _result, first = _run_tool_loop(
            agent, n_tool_iterations=5, effective=True,
        )
        agent.client.chat.completions.create.side_effect = None
        _result, second = _run_tool_loop(
            agent, n_tool_iterations=5, effective=True,
        )

        assert len(first) == 5
        assert len(second) == 5
