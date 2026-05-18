"""Tests for ``agent.eager_rate_limit_fallback``.

This config knob decides what happens when a provider returns a 429 rate-limit
error: fail over to the next provider immediately (eager, the default) or run
the normal retry/backoff loop against the primary first.

Coverage:
  * ``_should_eager_fallback`` — the decision helper, including the invariant
    that billing (HTTP 402) ALWAYS fails over eagerly regardless of the flag.
    402 is permanent; retrying it burns paid requests against an exhausted
    balance, and suppressing its fallback would only drop it into the
    ``is_client_error`` abort path without ever trying a backup provider
    (see #31273 and ``tests/run_agent/test_31273_402_not_retried.py``).
  * Config propagation + default — ``agent.eager_rate_limit_fallback`` reaches
    the agent object and defaults to ``True`` when unset (mirrors
    ``tests/run_agent/test_api_max_retries_config.py``).
  * A source-introspection guard that the production gate in
    ``run_conversation`` actually delegates to the helper (mirrors
    ``tests/agent/test_gemini_fast_fallback.py``).
"""
import inspect
from unittest.mock import patch

from agent import conversation_loop
from agent.conversation_loop import _should_eager_fallback
from agent.error_classifier import FailoverReason
from hermes_cli.config import DEFAULT_CONFIG
from run_agent import AIAgent


# --------------------------------------------------------------------------- #
# Decision helper: _should_eager_fallback(reason, eager_rate_limit_fallback)
# --------------------------------------------------------------------------- #

def test_rate_limit_eager_by_default_falls_back_immediately():
    """eager=True (the default) + 429 → switch to the next provider now."""
    assert _should_eager_fallback(FailoverReason.rate_limit, True) is True


def test_rate_limit_eager_disabled_defers_to_retry_loop():
    """eager=False + 429 → do NOT fall back at this gate; let the normal
    retry/backoff loop run ``api_max_retries`` times against the primary
    before fallback engages."""
    assert _should_eager_fallback(FailoverReason.rate_limit, False) is False


def test_billing_always_falls_back_regardless_of_setting():
    """402 is permanent, so it fails over eagerly whether the flag is True or
    False.  Honoring the flag for billing would only drop a 402 into the
    ``is_client_error`` abort path without ever trying a backup provider
    (#31273)."""
    assert _should_eager_fallback(FailoverReason.billing, True) is True
    assert _should_eager_fallback(FailoverReason.billing, False) is True


def test_unrelated_reasons_are_not_handled_by_this_gate():
    """Only rate_limit and billing reach the eager-fallback gate.  Every other
    reason is handled by its own branch and must return False here regardless
    of the flag (so flipping the knob can never change their routing)."""
    for reason in (
        FailoverReason.overloaded,
        FailoverReason.context_overflow,
        FailoverReason.unknown,
    ):
        assert _should_eager_fallback(reason, True) is False
        assert _should_eager_fallback(reason, False) is False


# --------------------------------------------------------------------------- #
# Config propagation: agent.eager_rate_limit_fallback → the agent object.
# Mirrors tests/run_agent/test_api_max_retries_config.py — the sibling knob
# read one line above this one in agent/agent_init.py.
# --------------------------------------------------------------------------- #

def _make_agent(eager_rate_limit_fallback=None):
    """Build an AIAgent with a mocked ``config.load_config`` that returns a
    config tree containing the given ``agent.eager_rate_limit_fallback`` (or
    omits the key entirely to exercise the default)."""
    cfg = {"agent": {}}
    if eager_rate_limit_fallback is not None:
        cfg["agent"]["eager_rate_limit_fallback"] = eager_rate_limit_fallback

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


def test_default_eager_rate_limit_fallback_is_true():
    """No config override → eager fallback stays on (preserves the prior,
    always-eager behavior)."""
    agent = _make_agent()
    assert agent._eager_rate_limit_fallback is True


def test_eager_rate_limit_fallback_honors_config_false():
    """``agent.eager_rate_limit_fallback: false`` propagates to the agent."""
    agent = _make_agent(eager_rate_limit_fallback=False)
    assert agent._eager_rate_limit_fallback is False


def test_eager_rate_limit_fallback_honors_config_true():
    """Explicit ``true`` propagates too (and matches the documented default)."""
    agent = _make_agent(eager_rate_limit_fallback=True)
    assert agent._eager_rate_limit_fallback is True


def test_default_config_ships_true_default():
    """DEFAULT_CONFIG carries the documented default so the knob is discoverable
    (e.g. via ``hermes config``) even when the user never set it."""
    assert DEFAULT_CONFIG["agent"]["eager_rate_limit_fallback"] is True


# --------------------------------------------------------------------------- #
# Belt-and-suspenders: the production gate must route through the helper, so
# the billing-always-eager / rate-limit-configurable policy actually applies
# inside run_conversation — not just in the unit-tested helper.  Mirrors the
# source guards in test_gemini_fast_fallback.py and test_31273_402_not_retried.py.
# --------------------------------------------------------------------------- #

def test_conversation_loop_gates_eager_fallback_through_helper():
    src = inspect.getsource(conversation_loop.run_conversation)
    assert "should_eager_fallback = _should_eager_fallback(" in src, (
        "run_conversation must compute the eager-fallback decision via "
        "_should_eager_fallback — otherwise the billing/rate-limit policy is "
        "not actually enforced in production."
    )
    assert "if should_eager_fallback and agent._fallback_index" in src, (
        "the eager-fallback gate must branch on the helper's decision, not on "
        "the old combined `is_rate_limited and getattr(...flag...)` shape that "
        "let eager=False wrongly suppress billing (402) fallback."
    )
