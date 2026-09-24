"""Truncated tool-call retries must send a larger output budget than the failed request (#72770)."""
from types import SimpleNamespace

import pytest

from agent.turn_truncation import _retry_truncated_tool_call


def _retry_budgets(max_tokens, requested_cap, attempts=4):
    agent = SimpleNamespace(
        max_tokens=max_tokens,
        _ephemeral_max_output_tokens=None,
        _buffer_vprint=lambda *a, **k: None,
        _requested_output_cap_from_api_kwargs=lambda kw: requested_cap,
    )
    st = SimpleNamespace(agent=agent, truncated_tool_call_retries=0, is_stub=False)
    st.done = lambda action, result=None: action
    budgets = []
    for _ in range(attempts):
        assert _retry_truncated_tool_call(st, {}) == "continue"
        budgets.append(agent._ephemeral_max_output_tokens)
    return budgets


@pytest.mark.parametrize("requested_cap", [32768, 65536])
def test_retry_raises_budget_above_large_requested_cap(requested_cap):
    budgets = _retry_budgets(None, requested_cap)
    assert budgets[0] > requested_cap
    assert max(budgets) <= requested_cap * 2


def test_small_explicit_max_tokens_ladder_still_capped_at_floor():
    assert _retry_budgets(4096, None) == [8192, 16384, 32768, 32768]
