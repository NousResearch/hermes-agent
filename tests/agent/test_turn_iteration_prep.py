"""``apply_retry_restarts`` bounds the refunding restarts (#106108).

The redirect and rebuilt-for-fallback paths refund the iteration budget and re-issue the
iteration; nothing else in the turn loop counts them, so a flag that keeps re-arming
(a request cancelled on every attempt) refunded forever and held the turn lease.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.turn_iteration_prep import apply_retry_restarts, begin_iteration
from agent.turn_retry_state import TurnRetryState

RESTART_FLAGS = ["restart_with_redirected_messages", "restart_with_rebuilt_messages"]
MAX_RETRIES = 3


def _apply(agent, flag: str | None, restart_count: int, response=None):
    """Run one iteration's restart handling; ``flag=None`` with a response means the model answered."""
    _retry = TurnRetryState()
    if flag:
        setattr(_retry, flag, True)
    return apply_retry_restarts(
        agent, _retry=_retry, response=response, interrupted=False, messages=[],
        conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
        final_response=None, retry_count=0, max_retries=MAX_RETRIES, api_call_count=1,
        restart_count=restart_count, length_continue_retries=0,
        _preflight_compression_blocked=True, _turn_exit_reason="unknown",
    )


def _agent():
    budget = SimpleNamespace(refunds=0)
    budget.refund = lambda: setattr(budget, "refunds", budget.refunds + 1)
    agent = SimpleNamespace(iteration_budget=budget, steered=[])
    agent._drain_pending_redirect = lambda: "last correction"
    agent.steer = agent.steered.append
    return agent


@pytest.mark.parametrize("flag", RESTART_FLAGS)
def test_single_restart_still_reissues_the_iteration(flag):
    """A lone correction / fallback activation keeps its refund-and-continue contract."""
    agent = _agent()
    verdict = _apply(agent, flag, restart_count=0)
    assert verdict.action == "continue"
    assert (agent.iteration_budget.refunds, verdict.api_call_count) == (1, 0)
    if flag == "restart_with_rebuilt_messages":
        assert verdict._preflight_compression_blocked is False  # still the single consumer


@pytest.mark.parametrize("flag", RESTART_FLAGS)
def test_back_to_back_restart_refunds_are_bounded(flag):
    """Re-arming the flag every iteration breaks after ``max_retries`` refunds instead of
    refunding forever (the turn ends, so the session turn lease is released)."""
    agent = _agent()
    restart_count, verdicts = 0, []
    while len(verdicts) < MAX_RETRIES + 5 and (not verdicts or verdicts[-1].action != "break"):
        verdicts.append(_apply(agent, flag, restart_count))
        restart_count = verdicts[-1].restart_count
    assert [v.action for v in verdicts] == ["continue"] * MAX_RETRIES + ["break"]
    assert agent.iteration_budget.refunds == MAX_RETRIES
    assert verdicts[-1]._turn_exit_reason.endswith("restart_limit_exceeded")
    # The correction that tripped the redirect cap is handed back as the next user turn.
    assert agent.steered == (["last correction"] if flag == "restart_with_redirected_messages" else [])


def _interrupted_agent(tool_interrupt_reason):
    agent = SimpleNamespace(
        _interrupt_requested=True, _tool_interrupt_reason=tool_interrupt_reason, quiet_mode=True,
        _drain_pending_redirect=lambda: None, _checkpoint_mgr=SimpleNamespace(new_turn=lambda: None),
    )
    return begin_iteration(
        agent, messages=[], conversation_history=[], original_user_message="hi",
        api_call_count=0, interrupted=False, _turn_exit_reason="unknown",
    )


@pytest.mark.parametrize("tool_interrupt_reason, expected", [
    # Human stops keep the historical reason (every ``interrupt()`` category, and a bare flag write).
    ("explicit stop requested", "interrupted_by_user"),
    ("user sent a new message", "interrupted_by_user"),
    (None, "interrupted_by_user"),
    # A producer that named itself via ``tool_reason`` is booked as the issuer (#112647).
    ("cron inactivity watchdog", "interrupted_by_system(cron_inactivity_watchdog)"),
    ("turn liveness watchdog", "interrupted_by_system(turn_liveness_watchdog)"),
])
def test_interrupt_exit_reason_names_the_system_issuer(tool_interrupt_reason, expected):
    """A watchdog abort must not be recorded as a user stop: the exit reason carries the issuer."""
    verdict = _interrupted_agent(tool_interrupt_reason)
    assert (verdict.action, verdict.interrupted, verdict._turn_exit_reason) == ("break", True, expected)


@pytest.mark.parametrize("flag", RESTART_FLAGS)
def test_response_between_restarts_resets_the_bound(flag):
    """#128000: the user answering clarify cards / sending follow-ups while the model works
    redirects once per request, with a response in between. The bound is for requests that
    are cancelled over and over, so a response starts it over instead of the whole turn
    sharing ``max_retries`` restarts."""
    agent = _agent()
    restart_count, actions = 0, []
    for _ in range(3):  # three rounds of max_retries restarts, each followed by a response
        for _ in range(MAX_RETRIES):
            verdict = _apply(agent, flag, restart_count)
            actions.append(verdict.action)
            restart_count = verdict.restart_count
        verdict = _apply(agent, None, restart_count, response=SimpleNamespace())
        assert (verdict.action, verdict.restart_count) == ("fallthrough", 0)
        restart_count = verdict.restart_count
    assert actions == ["continue"] * (3 * MAX_RETRIES)
    assert agent.steered == []


def test_rebuilt_restarts_allow_full_fallback_chain_when_max_retries_is_one():
    """CF-20: When api_max_retries=1 but _fallback_chain has 3 entries, each fallback hop
    arms restart_with_rebuilt_messages once; the cap must allow len(_fallback_chain) hops
    before breaking with rebuilt_restart_limit_exceeded."""
    agent = _agent()
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"},
        {"provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
        {"provider": "openrouter", "model": "z-ai/glm-5.3"},
    ]
    restart_count, verdicts = 0, []
    for _ in range(4):
        _retry = TurnRetryState(restart_with_rebuilt_messages=True)
        v = apply_retry_restarts(
            agent, _retry=_retry, response=None, interrupted=False, messages=[],
            conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
            final_response=None, retry_count=0, max_retries=1, api_call_count=1,
            restart_count=restart_count, length_continue_retries=0,
            _preflight_compression_blocked=True, _turn_exit_reason="unknown",
        )
        verdicts.append(v)
        restart_count = v.restart_count
    assert [v.action for v in verdicts] == ["continue", "continue", "continue", "break"]
    assert verdicts[-1]._turn_exit_reason == "rebuilt_restart_limit_exceeded"


def test_interleaved_redirect_and_rebuilt_restarts_do_not_starve_each_other():
    """MAJOR-1 & MINOR-1: A user redirect during a failover walk must not starve the 3rd
    fallback hop, and a completed 3-hop failover walk must still allow a user redirect
    before tripping redirect_restart_limit_exceeded; max_retries=0 disables both."""
    agent = _agent()
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "z-ai/glm-5.3-flash"},
        {"provider": "openrouter", "model": "google/gemini-2.5-flash-lite"},
        {"provider": "openrouter", "model": "z-ai/glm-5.3"},
    ]
    restart_count, actions = 0, []
    for flag in [
        "restart_with_redirected_messages",
        "restart_with_rebuilt_messages",
        "restart_with_rebuilt_messages",
        "restart_with_rebuilt_messages",
        "restart_with_redirected_messages",
    ]:
        _retry = TurnRetryState(**{flag: True})
        v = apply_retry_restarts(
            agent, _retry=_retry, response=None, interrupted=False, messages=[],
            conversation_history=[], user_message="hi", api_kwargs={}, current_turn_user_idx=0,
            final_response=None, retry_count=0, max_retries=1, api_call_count=1,
            restart_count=restart_count, length_continue_retries=0,
            _preflight_compression_blocked=True, _turn_exit_reason="unknown",
        )
        actions.append((v.action, v._turn_exit_reason))
        restart_count = v.restart_count
    assert [a[0] for a in actions] == ["continue", "continue", "continue", "continue", "break"]
    assert actions[-1][1] == "redirect_restart_limit_exceeded"

    # max_retries=0 disables rebuilt restarts even when _fallback_chain is non-empty.
    v0 = apply_retry_restarts(
        agent, _retry=TurnRetryState(restart_with_rebuilt_messages=True), response=None,
        interrupted=False, messages=[], conversation_history=[], user_message="hi",
        api_kwargs={}, current_turn_user_idx=0, final_response=None, retry_count=0,
        max_retries=0, api_call_count=1, restart_count=0, length_continue_retries=0,
        _preflight_compression_blocked=True, _turn_exit_reason="unknown",
    )
    assert (v0.action, v0._turn_exit_reason) == ("break", "rebuilt_restart_limit_exceeded")


def test_prepare_iteration_suppresses_noop_reanchor_log_and_logs_real_index_shift(caplog):
    """When _anchors_current_turn is False (e.g. idx=-1 or post-hook content mutation at idx=0)
    but _reanchor returns the identical index, prepare_iteration must not emit 'Re-anchored stale
    current_turn_user_idx X -> X'; when the index genuinely shifts, it must log and update."""
    import logging
    from agent.turn_iteration_prep import prepare_iteration

    req_logger = logging.getLogger("agent.conversation_loop.test_reanchor")
    agent = SimpleNamespace(
        _nous_wire_pending=None,
        step_callback=None,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _adopt_nous_key_before_expiry=lambda: None,
        _drain_pending_steer=lambda: None,
        run_budget_seconds=None,
        iteration_budget=SimpleNamespace(remaining=10, max_iterations=20),
        logger=req_logger,
        _sanitize_args_cursor={},
        _sanitize_tool_call_arguments=lambda *a, **kw: 0,
        session_id="sess-reanchor",
        _reanchor_current_turn_user_idx=lambda msgs, um: (
            0 if any(m.get("role") == "user" for m in msgs) else -1
        ),
    )

    with caplog.at_level(logging.INFO, logger=req_logger.name):
        # Case 1: idx=-1 -> -1 (no user message in messages) -> no-op, must NOT log
        prep_neg = prepare_iteration(
            agent, messages=[{"role": "assistant", "content": "working"}],
            api_call_count=2, user_message="go", current_turn_user_idx=-1,
        )
        assert prep_neg.current_turn_user_idx == -1

        # Case 2: idx=0 -> 0 (mutated user content at index 0, fallback returns 0) -> no-op, must NOT log
        prep_same = prepare_iteration(
            agent, messages=[{"role": "user", "content": "[mutated] go"}],
            api_call_count=2, user_message="go", current_turn_user_idx=0,
        )
        assert prep_same.current_turn_user_idx == 0
        assert [r.getMessage() for r in caplog.records if "Re-anchored stale" in r.getMessage()] == []

        # Case 3: idx=2 -> 0 (post-compaction index shift) -> genuine shift, MUST log and update
        prep_shifted = prepare_iteration(
            agent, messages=[{"role": "user", "content": "[mutated] go"}],
            api_call_count=2, user_message="go", current_turn_user_idx=2,
        )
        assert prep_shifted.current_turn_user_idx == 0
        stale_logs = [r.getMessage() for r in caplog.records if "Re-anchored stale" in r.getMessage()]
        assert stale_logs == ["Re-anchored stale current_turn_user_idx 2 -> 0 (session=sess-reanchor)"]


def test_prepare_iteration_restores_primary_runtime_between_tool_rounds_not_on_restart():
    """CF-60: prepare_iteration must call _restore_primary_runtime on subsequent tool
    iterations (api_call_count > 0 and _rebuilt_restarts_since_response == 0) when
    _fallback_activated is True, but never during an in-flight retry restart."""
    from agent.turn_iteration_prep import prepare_iteration

    restored = []
    agent = SimpleNamespace(
        _nous_wire_pending=None,
        step_callback=None,
        _skill_nudge_interval=0,
        valid_tool_names=set(),
        _adopt_nous_key_before_expiry=lambda: None,
        _adopt_vertex_token_before_expiry=lambda: None,
        _fallback_activated=True,
        _rebuilt_restarts_since_response=1,
        _restore_primary_runtime=lambda: restored.append(True) or True,
        _drain_pending_steer=lambda: None,
        run_budget_seconds=None,
        iteration_budget=SimpleNamespace(remaining=10, max_iterations=20),
        logger=None,
        _sanitize_args_cursor={},
        _sanitize_tool_call_arguments=lambda *a, **kw: 0,
        session_id="sess-cf60",
    )

    # During in-flight fallback restart (_rebuilt_restarts_since_response == 1): must NOT restore primary
    prepare_iteration(agent, messages=[{"role": "user", "content": "hi"}], api_call_count=1)
    assert restored == []

    # Next completed tool round (_rebuilt_restarts_since_response == 0): MUST attempt restore
    agent._rebuilt_restarts_since_response = 0
    prepare_iteration(agent, messages=[{"role": "user", "content": "hi"}], api_call_count=1)
    assert restored == [True]




