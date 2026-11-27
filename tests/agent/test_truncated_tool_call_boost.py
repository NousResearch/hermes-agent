"""Truncation retries must send a larger output budget than the failed request (#72770),
without exceeding the model's known output limit (#79715)."""
from types import SimpleNamespace

import pytest

from agent.turn_iteration_prep import apply_retry_restarts
from agent.turn_retry_state import TurnRetryState
from agent.turn_truncation import _retry_truncated_tool_call


def _agent(max_tokens, requested_cap, **extra):
    return SimpleNamespace(
        max_tokens=max_tokens,
        _ephemeral_max_output_tokens=None,
        _buffer_vprint=lambda *a, **k: None,
        _requested_output_cap_from_api_kwargs=lambda kw: requested_cap,
        **extra,
    )


def _tool_call_budgets(agent, attempts=4):
    st = SimpleNamespace(agent=agent, truncated_tool_call_retries=0, is_stub=False)
    st.done = lambda action, result=None: action
    budgets = []
    for _ in range(attempts):
        assert _retry_truncated_tool_call(st, {}) == "continue"
        budgets.append(agent._ephemeral_max_output_tokens)
    return budgets


def _length_continuation_budgets(agent, attempts=4):
    budgets = []
    for n in range(1, attempts + 1):
        _retry = TurnRetryState()
        _retry.restart_with_length_continuation = True
        verdict = apply_retry_restarts(
            agent, _retry=_retry, response=None, interrupted=False, messages=[],
            conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
            final_response=None, retry_count=0, max_retries=3, api_call_count=1,
            restart_count=0, length_continue_retries=n,
            _preflight_compression_blocked=False, _turn_exit_reason="unknown",
        )
        assert verdict.action == "continue"
        budgets.append(agent._ephemeral_max_output_tokens)
    return budgets


SITES = [_tool_call_budgets, _length_continuation_budgets]


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("requested_cap", [32768, 65536])
def test_retry_raises_budget_above_large_requested_cap(site, requested_cap):
    budgets = site(_agent(None, requested_cap))
    assert budgets[0] > requested_cap
    assert max(budgets) <= requested_cap * 2


@pytest.mark.parametrize("site", SITES)
def test_boost_clamped_to_known_model_output_limit(site):
    # claude-sonnet-4-5 outputs at most 64000: never ask for more, and don't double a
    # request that already sits at the ceiling (the provider would 400).
    below = site(_agent(None, 40000, api_mode="anthropic_messages", model="claude-sonnet-4-5"))
    at_limit = site(_agent(None, 64000, api_mode="anthropic_messages", model="claude-sonnet-4-5"))
    assert below == [64000] * 4
    assert at_limit == [64000] * 4


def test_small_explicit_max_tokens_ladder_still_capped_at_floor():
    assert _tool_call_budgets(_agent(4096, None)) == [8192, 16384, 32768, 32768]
