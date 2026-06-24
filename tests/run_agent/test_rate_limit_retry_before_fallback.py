"""Tests for agent.rate_limit_retry_before_fallback.

Opt-in knob (default 0 = upstream eager-fallback behavior, #11314 preserved).
When > 0, a transient rate-limit (429) on the primary retries up to N times
before falling back, but ONLY when the 429 looks transient (a short
Retry-After header or a near-future reset_at). Quota-exhaustion 429s still
fall back immediately even when the knob is set.
"""
import time
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent
from agent.conversation_loop import _rate_limit_looks_transient
from agent.error_classifier import ClassifiedError, FailoverReason


def _make_agent(value=None):
    cfg = {"agent": {}}
    if value is not None:
        cfg["agent"]["rate_limit_retry_before_fallback"] = value
    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value=cfg):
        return AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )


def _err_with_retry_after(value):
    return SimpleNamespace(response=SimpleNamespace(headers={"retry-after": value}))


def _classified(reset_at=None):
    ctx = {}
    if reset_at is not None:
        ctx["reset_at"] = reset_at
    return ClassifiedError(reason=FailoverReason.rate_limit, error_context=ctx)


def test_default_is_zero():
    assert _make_agent()._rate_limit_retry_before_fallback == 0


def test_config_override_propagates():
    assert _make_agent(value=5)._rate_limit_retry_before_fallback == 5


def test_negative_clamps_to_zero():
    assert _make_agent(value=-3)._rate_limit_retry_before_fallback == 0


def test_invalid_falls_back_to_zero():
    assert _make_agent(value="nope")._rate_limit_retry_before_fallback == 0


def test_short_retry_after_is_transient():
    assert _rate_limit_looks_transient(_err_with_retry_after("5"), _classified()) is True


def test_long_retry_after_is_not_transient():
    assert _rate_limit_looks_transient(_err_with_retry_after("9000"), _classified()) is False


def test_near_future_reset_at_is_transient():
    assert _rate_limit_looks_transient(
        SimpleNamespace(response=None), _classified(reset_at=time.time() + 10)
    ) is True


def test_far_future_reset_at_is_not_transient():
    assert _rate_limit_looks_transient(
        SimpleNamespace(response=None), _classified(reset_at=time.time() + 7200)
    ) is False


def test_no_signal_is_not_transient():
    assert _rate_limit_looks_transient(
        SimpleNamespace(response=None), _classified()
    ) is False


def test_gate_requires_all_conditions():
    budget = 3
    transient = _rate_limit_looks_transient(_err_with_retry_after("5"), _classified())
    quota = _rate_limit_looks_transient(_err_with_retry_after("9000"), _classified())

    def gate(reason, retry_count, looks_transient):
        return (
            budget > 0
            and reason == FailoverReason.rate_limit
            and retry_count < budget
            and looks_transient
        )

    assert gate(FailoverReason.rate_limit, 0, transient) is True
    assert gate(FailoverReason.rate_limit, 3, transient) is False
    assert gate(FailoverReason.billing, 0, transient) is False
    assert gate(FailoverReason.rate_limit, 0, quota) is False
