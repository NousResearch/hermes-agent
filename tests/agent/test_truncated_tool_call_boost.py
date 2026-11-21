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


def _tool_call_budgets(agent, attempts=4, response=None):
    st = SimpleNamespace(agent=agent, truncated_tool_call_retries=0, is_stub=False, response=response)
    st.done = lambda action, result=None: action
    budgets = []
    for _ in range(attempts):
        assert _retry_truncated_tool_call(st, {}) == "continue"
        budgets.append(agent._ephemeral_max_output_tokens)
    return budgets


def _length_continuation_budgets(agent, attempts=4, response=None):
    budgets = []
    for n in range(1, attempts + 1):
        _retry = TurnRetryState()
        _retry.restart_with_length_continuation = True
        verdict = apply_retry_restarts(
            agent, _retry=_retry, response=response, interrupted=False, messages=[],
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


def _truncated_response(**usage_fields):
    return SimpleNamespace(usage=SimpleNamespace(**usage_fields))


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("api_mode,usage_fields", [
    ("chat_completions", {"completion_tokens": 16384}),   # chat completions reports completion_tokens
    ("codex_responses", {"output_tokens": 16384}),        # the Responses API reports output_tokens
])
def test_first_retry_outspends_a_response_that_already_hit_the_server_default(site, api_mode, usage_fields):
    """No configured cap and none sent (#153136): the server applied its own default, so the output
    the truncated response already produced IS that ceiling. The first rung must beat it, not reset
    to 8192 — a retry with half the room of the attempt that truncated only truncates again."""
    budgets = site(_agent(None, None, api_mode=api_mode), response=_truncated_response(**usage_fields))
    assert budgets[0] > 16384
    assert sum(budgets) > 16384


@pytest.mark.parametrize("site", SITES)
def test_observed_output_never_lowers_a_configured_or_sent_cap(site):
    """The observed term only fills an empty base: a configured max_tokens (or the cap the failed
    request carried) still sets the ladder, and its first rung stays 2× that cap."""
    response = _truncated_response(completion_tokens=100)
    assert site(_agent(4096, None), response=response)[0] == 8192
    assert site(_agent(None, 8192), response=response)[0] == 16384


def test_observed_usage_absent_keeps_the_legacy_baseline():
    """A response without usage (or with a zero/absent count) keeps today's 4096 baseline."""
    legacy = [8192, 16384, 32768, 32768]
    assert _tool_call_budgets(_agent(None, None), response=SimpleNamespace(usage=None)) == legacy
    zero = _truncated_response(completion_tokens=0, output_tokens=None)
    assert _tool_call_budgets(_agent(None, None), response=zero) == legacy

