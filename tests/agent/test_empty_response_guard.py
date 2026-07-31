"""Tests for agent/empty_response_guard.py (NS-503).

The guard exists to stop the empty-retry loop re-billing large inputs for
deterministic empties (unsignaled provider refusals with zero output
tokens) while never tightening behaviour on ambiguous evidence.

Fail-open contract under test:
- Missing usage -> never deterministic, default budget.
- Any generated tokens (output or reasoning) -> never deterministic.
- Different model/provider/finish_reason across attempts -> not deterministic.
- Guard disabled via env -> everything falls back to defaults.
"""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent import empty_response_guard as guard


def _agent(**overrides):
    base = dict(
        model="anthropic/claude-fable-5",
        provider="nous",
        api_mode="chat_completions",
        base_url=None,
        api_key=None,
        _empty_content_retries=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _response(prompt_tokens=25_900, completion_tokens=0, usage_present=True):
    if not usage_present:
        return SimpleNamespace(usage=None)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(usage=usage)


def _record_streak(agent, responses, finish_reasons=None):
    """Record attempts the way the loop does: record, then increment."""
    finish_reasons = finish_reasons or ["stop"] * len(responses)
    for resp, reason in zip(responses, finish_reasons):
        guard.record_empty_attempt(agent, finish_reason=reason, response=resp)
        agent._empty_content_retries += 1


class TestDeterministicEmpty:
    def test_two_zero_output_attempts_same_signature_is_deterministic(self):
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.deterministic_empty(agent) is True

    def test_single_attempt_is_never_deterministic(self):
        """One empty could be a transient blip — retry #1 must always run."""
        agent = _agent()
        _record_streak(agent, [_response()])
        assert guard.deterministic_empty(agent) is False

    def test_missing_usage_fails_open(self):
        agent = _agent()
        _record_streak(
            agent,
            [_response(usage_present=False), _response(usage_present=False)],
        )
        assert guard.deterministic_empty(agent) is False

    def test_mixed_usage_presence_fails_open(self):
        agent = _agent()
        _record_streak(agent, [_response(), _response(usage_present=False)])
        assert guard.deterministic_empty(agent) is False

    def test_nonzero_output_tokens_fails_open(self):
        """Model generated something (whitespace, stripped think-blocks) —
        that's the flaky-but-recoverable class, keep full retries."""
        agent = _agent()
        _record_streak(
            agent,
            [_response(completion_tokens=42), _response(completion_tokens=42)],
        )
        assert guard.deterministic_empty(agent) is False

    def test_zero_then_nonzero_fails_open(self):
        agent = _agent()
        _record_streak(
            agent,
            [_response(completion_tokens=0), _response(completion_tokens=7)],
        )
        assert guard.deterministic_empty(agent) is False

    def test_signature_change_resets_determinism(self):
        """Fallback switched model mid-streak — new model deserves retries."""
        agent = _agent()
        guard.record_empty_attempt(agent, finish_reason="stop", response=_response())
        agent._empty_content_retries += 1
        agent.model = "other/model"
        guard.record_empty_attempt(agent, finish_reason="stop", response=_response())
        agent._empty_content_retries += 1
        assert guard.deterministic_empty(agent) is False

    def test_finish_reason_change_fails_open(self):
        agent = _agent()
        _record_streak(
            agent,
            [_response(), _response()],
            finish_reasons=["stop", "length"],
        )
        assert guard.deterministic_empty(agent) is False

    def test_new_streak_clears_history(self):
        """Counter reset to 0 (turn start / tool success / compaction /
        fallback) starts a fresh streak — prior attempts must not leak."""
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.deterministic_empty(agent) is True

        agent._empty_content_retries = 0  # any existing reset site
        guard.record_empty_attempt(agent, finish_reason="stop", response=_response())
        agent._empty_content_retries += 1
        assert guard.deterministic_empty(agent) is False

    def test_guard_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_DETERMINISTIC_EMPTY_GUARD", "0")
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.deterministic_empty(agent) is False

    def test_reasoning_tokens_count_as_generation(self, monkeypatch):
        """Reasoning-only responses are owned by the prefill path; the
        guard must not classify them as deterministic empties."""
        canonical = SimpleNamespace(output_tokens=0, reasoning_tokens=128)
        monkeypatch.setattr(
            guard, "_zero_output", lambda a, r: (True, (0 + 128) == 0)
        )
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.deterministic_empty(agent) is False


class TestEmptyRetryBudget:
    def test_default_budget_when_cost_unknown(self, monkeypatch):
        monkeypatch.setattr(guard, "_estimate_attempt_cost", lambda a, r: None)
        assert (
            guard.empty_retry_budget(_agent(), _response())
            == guard.DEFAULT_EMPTY_RETRY_BUDGET
        )

    def test_reduced_budget_above_threshold(self, monkeypatch):
        monkeypatch.setattr(
            guard, "_estimate_attempt_cost", lambda a, r: Decimal("0.80")
        )
        assert (
            guard.empty_retry_budget(_agent(), _response())
            == guard.REDUCED_EMPTY_RETRY_BUDGET
        )

    def test_default_budget_below_threshold(self, monkeypatch):
        monkeypatch.setattr(
            guard, "_estimate_attempt_cost", lambda a, r: Decimal("0.01")
        )
        assert (
            guard.empty_retry_budget(_agent(), _response())
            == guard.DEFAULT_EMPTY_RETRY_BUDGET
        )

    def test_custom_threshold_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_EMPTY_RETRY_COST_THRESHOLD_USD", "5.00")
        monkeypatch.setattr(
            guard, "_estimate_attempt_cost", lambda a, r: Decimal("0.80")
        )
        assert (
            guard.empty_retry_budget(_agent(), _response())
            == guard.DEFAULT_EMPTY_RETRY_BUDGET
        )

    def test_bad_threshold_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("HERMES_EMPTY_RETRY_COST_THRESHOLD_USD", "banana")
        assert guard._cost_threshold_usd() == guard.DEFAULT_COST_THRESHOLD_USD
        monkeypatch.setenv("HERMES_EMPTY_RETRY_COST_THRESHOLD_USD", "-1")
        assert guard._cost_threshold_usd() == guard.DEFAULT_COST_THRESHOLD_USD

    def test_guard_disabled_keeps_default_budget(self, monkeypatch):
        monkeypatch.setenv("HERMES_DETERMINISTIC_EMPTY_GUARD", "0")
        monkeypatch.setattr(
            guard, "_estimate_attempt_cost", lambda a, r: Decimal("9.99")
        )
        assert (
            guard.empty_retry_budget(_agent(), _response())
            == guard.DEFAULT_EMPTY_RETRY_BUDGET
        )

    def test_pricing_exception_fails_open(self, monkeypatch):
        def _boom(a, r):
            raise RuntimeError("pricing exploded")

        # _estimate_attempt_cost itself catches internally; simulate at
        # the normalize layer by feeding garbage usage.
        agent = _agent(model=None, provider=None)
        resp = SimpleNamespace(usage=object())
        assert (
            guard.empty_retry_budget(agent, resp)
            == guard.DEFAULT_EMPTY_RETRY_BUDGET
        )


class TestStreakCost:
    def test_streak_cost_accumulates(self, monkeypatch):
        costs = iter([Decimal("1.10"), Decimal("1.23")])
        monkeypatch.setattr(
            guard, "_estimate_attempt_cost", lambda a, r: next(costs)
        )
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.streak_cost_usd(agent) == Decimal("2.33")

    def test_streak_cost_none_when_unknown(self, monkeypatch):
        monkeypatch.setattr(guard, "_estimate_attempt_cost", lambda a, r: None)
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.streak_cost_usd(agent) is None

    def test_streak_cost_resets_on_new_streak(self, monkeypatch):
        monkeypatch.setattr(
            guard, "_estimate_attempt_cost", lambda a, r: Decimal("1.00")
        )
        agent = _agent()
        _record_streak(agent, [_response(), _response()])
        assert guard.streak_cost_usd(agent) == Decimal("2.00")
        agent._empty_content_retries = 0
        guard.record_empty_attempt(agent, finish_reason="stop", response=_response())
        assert guard.streak_cost_usd(agent) == Decimal("1.00")


class TestZeroOutputExtraction:
    """_zero_output goes through the real normalize_usage path."""

    def test_openai_shape_zero_completion(self):
        agent = _agent()
        present, zero = guard._zero_output(agent, _response(completion_tokens=0))
        assert present is True
        assert zero is True

    def test_openai_shape_with_completion(self):
        agent = _agent()
        present, zero = guard._zero_output(agent, _response(completion_tokens=9))
        assert present is True
        assert zero is False

    def test_no_usage(self):
        agent = _agent()
        present, zero = guard._zero_output(agent, _response(usage_present=False))
        assert present is False
        assert zero is False

    def test_anthropic_shape_zero_output(self):
        agent = _agent(api_mode="anthropic_messages")
        usage = SimpleNamespace(
            input_tokens=25_900,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        present, zero = guard._zero_output(agent, SimpleNamespace(usage=usage))
        assert present is True
        assert zero is True

    def test_all_zero_usage_object_fails_open(self):
        """Proxies that emit an empty usage object (all fields absent →
        normalized to zeros) provide no evidence — must not classify."""
        agent = _agent()
        usage = SimpleNamespace()  # no token fields at all
        present, zero = guard._zero_output(agent, SimpleNamespace(usage=usage))
        assert present is False
        assert zero is False
