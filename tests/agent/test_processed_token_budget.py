"""Processed-token budgets bound diverse, cache-heavy autonomous tool loops."""

from __future__ import annotations

import threading
import sys
from types import SimpleNamespace

import pytest

from agent.processed_token_budget import (
    maybe_inject_processed_token_warning,
    normalize_processed_token_budget,
    note_aux_processed_tokens,
    processed_token_budget_exhausted,
    processed_token_budget_exit_reason,
    reset_turn_processed_tokens,
    turn_processed_tokens,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None), (False, None), (0, None), (-1, None), ("bad", None),
        ("1500000", 1_500_000), (250_000.9, 250_000),
    ],
)
def test_normalize_processed_token_budget(value, expected):
    assert normalize_processed_token_budget(value) == expected


def _agent(**overrides):
    values = {
        "session_total_tokens": 10_000,
        "_turn_processed_tokens_start": 0,
        "_turn_aux_processed_tokens": 0,
        "_processed_token_budget_lock": threading.Lock(),
        "_processed_token_budget_scope": None,
        "_processed_token_budget_warning_injected": False,
        "_interrupt_requested": False,
        "max_processed_tokens_per_turn": 1_500_000,
        "processed_token_warning_ratio": 0.65,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_meter_counts_replayed_main_prompts_and_scoped_aux_usage_without_compression_refund():
    agent = _agent()
    reset_turn_processed_tokens(agent)
    scope = object()
    agent._processed_token_budget_scope = scope

    # Provider totals are cumulative across calls. A compaction changes the transcript,
    # not these counters, so the delta continues from the original turn anchor.
    agent.session_total_tokens += 1_425_000
    note_aux_processed_tokens(agent, 75_000, scope=scope)
    note_aux_processed_tokens(agent, 999_999, scope=object())  # stale prior-turn callback

    assert turn_processed_tokens(agent) == 1_500_000
    assert processed_token_budget_exhausted(agent)
    assert processed_token_budget_exit_reason(agent) == (
        "processed_token_budget_exhausted(1500000/1500000)"
    )


@pytest.mark.parametrize("content", ["result", [{"type": "text", "text": "result"}]])
def test_warning_is_one_shot_and_only_mutates_the_live_tool_tail(content):
    agent = _agent(
        session_total_tokens=985_000,
        _turn_processed_tokens_start=10_000,
    )
    messages = [{"role": "tool", "content": content}]

    assert maybe_inject_processed_token_warning(agent, messages)
    assert "975,000 of 1,500,000" in str(messages[-1]["content"])
    snapshot = str(messages)
    assert not maybe_inject_processed_token_warning(agent, messages)
    assert str(messages) == snapshot


def test_begin_iteration_stops_before_another_provider_call_at_hard_limit(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "agent.conversation_loop",
        SimpleNamespace(
            _apply_active_turn_redirect=lambda *_args, **_kwargs: None,
            _review_input_budget_exhausted=lambda _agent: False,
        ),
    )
    from agent.turn_iteration_prep import begin_iteration

    notices = []
    agent = _agent(
        session_total_tokens=1_510_000,
        _turn_processed_tokens_start=10_000,
        quiet_mode=True,
        _emit_diagnostic_status=notices.append,
        _drain_pending_redirect=lambda: None,
        _checkpoint_mgr=SimpleNamespace(new_turn=lambda: None),
        _review_input_token_budget=None,
        session_input_tokens=0,
    )
    verdict = begin_iteration(
        agent, messages=[], conversation_history=[], original_user_message="work",
        api_call_count=7, interrupted=False, _turn_exit_reason="unknown",
    )

    assert verdict.action == "break"
    assert verdict.api_call_count == 7
    assert verdict._turn_exit_reason == "processed_token_budget_exhausted(1500000/1500000)"
    assert notices and "stopping before the next provider call" in notices[0]


def test_token_budget_finalizer_returns_local_resumable_checkpoint_without_summary_call():
    from agent.turn_finalizer import _resolve_budget_fallback

    agent = _agent(
        session_total_tokens=1_510_000,
        _turn_processed_tokens_start=10_000,
        max_iterations=20,
        iteration_budget=SimpleNamespace(remaining=12),
    )
    agent._handle_max_iterations = lambda *_: pytest.fail("must not make a summary API call")
    response, reason, preserved, interrupted = _resolve_budget_fallback(
        agent, final_response=None, api_call_count=8, interrupted=False, failed=False,
        messages=[], _turn_exit_reason="processed_token_budget_exhausted(1500000/1500000)",
        _pending_verification_response=None,
        _pending_verification_response_previewed=False, logger=SimpleNamespace(warning=lambda *_a, **_k: None),
    )

    assert "resumable checkpoint" in response
    assert reason == "processed_token_budget_exhausted(1500000/1500000)"
    assert not preserved
    assert not interrupted
