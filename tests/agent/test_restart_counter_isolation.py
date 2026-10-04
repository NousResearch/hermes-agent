"""Counter isolation across actual loop-state phase binding."""
from dataclasses import MISSING, fields

import pytest

from agent.conversation_loop import _LoopState, _run_phase
from agent.turn_iteration_prep import apply_retry_restarts
from agent.turn_retry_state import TurnRetryState
from tests.agent.test_turn_iteration_prep import _agent


@pytest.mark.parametrize("redirect_limit", [10, 100])
@pytest.mark.parametrize("reverse", [False, True])
def test_interleaved_redirect_and_fallback_caps(redirect_limit, reverse):
    state = _LoopState(**{f.name: None for f in fields(_LoopState)
                          if f.default is MISSING and f.default_factory is MISSING})
    state.messages = []
    state.conversation_history = []
    state.max_retries = 3
    state.redirect_restart_limit = redirect_limit
    state._preflight_compression_blocked = True
    state._retry = TurnRetryState()
    agent = _agent()

    def restart(flag):
        setattr(state._retry, flag, True)
        state.api_call_count += 1
        return _run_phase(apply_retry_restarts, agent, state)

    redirect = "restart_with_redirected_messages"
    rebuilt = "restart_with_rebuilt_messages"
    first, first_count, second, second_count = (
        (rebuilt, 3, redirect, redirect_limit) if reverse else (redirect, redirect_limit, rebuilt, 3)
    )
    for _ in range(first_count):
        assert restart(first).action == "continue"
    for _ in range(second_count):
        assert restart(second).action == "continue"
    assert state._preflight_compression_blocked is False
    assert agent.iteration_budget.refunds == redirect_limit + 3
    assert restart(second).action == "break"
    assert state._turn_exit_reason == ("redirect_restart_limit_exceeded" if reverse else "rebuilt_restart_limit_exceeded")
    assert agent.iteration_budget.refunds == redirect_limit + 3
