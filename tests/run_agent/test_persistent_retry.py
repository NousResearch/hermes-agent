"""Tests for persistent retry on transient provider failures.

Covers the long-horizon / unattended-agent retry behavior added for #35230
and #25689: when ``agent.api_retry_persistent`` is enabled, transient
failures (overloaded, rate/usage/concurrent limits, 5xx, timeouts, empty/
malformed responses) keep retrying past ``api_max_retries`` with backoff,
while authorization and other deterministic errors still fail fast.
"""
import time
from unittest.mock import patch

from run_agent import AIAgent
from agent.error_classifier import FailoverReason


def _make_agent(persistent=None, max_elapsed=None):
    cfg = {"agent": {}}
    if persistent is not None:
        cfg["agent"]["api_retry_persistent"] = persistent
    if max_elapsed is not None:
        cfg["agent"]["api_retry_persistent_max_elapsed_seconds"] = max_elapsed

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


# ── Config surface ──────────────────────────────────────────────────────

def test_persistent_retry_defaults_off():
    agent = _make_agent()
    assert agent._api_retry_persistent is False
    assert agent._api_retry_persistent_max_elapsed_seconds == 0


def test_persistent_retry_honors_config():
    agent = _make_agent(persistent=True)
    assert agent._api_retry_persistent is True


def test_persistent_retry_max_elapsed_parsed():
    agent = _make_agent(persistent=True, max_elapsed=3600)
    assert agent._api_retry_persistent_max_elapsed_seconds == 3600


def test_persistent_retry_max_elapsed_clamps_negative_to_zero():
    agent = _make_agent(persistent=True, max_elapsed=-10)
    assert agent._api_retry_persistent_max_elapsed_seconds == 0


def test_persistent_retry_max_elapsed_invalid_falls_back_to_zero():
    agent = _make_agent(persistent=True, max_elapsed="not-a-number")
    assert agent._api_retry_persistent_max_elapsed_seconds == 0


# ── _should_persist_retry: transient vs deterministic ───────────────────

def test_should_persist_retry_false_when_disabled():
    agent = _make_agent(persistent=False)
    assert agent._should_persist_retry(FailoverReason.overloaded) is False
    assert agent._should_persist_retry(FailoverReason.rate_limit) is False


def test_should_persist_retry_true_for_transient_reasons():
    agent = _make_agent(persistent=True)
    # Provider-side / throttle / transport reasons should persist.
    for reason in (
        FailoverReason.overloaded,
        FailoverReason.rate_limit,    # also covers usage- and concurrent-limit throttles
        FailoverReason.server_error,
        FailoverReason.timeout,
        FailoverReason.unknown,
    ):
        assert agent._should_persist_retry(reason) is True, reason
    # The invalid/empty-response sentinel is passed as a bare string.
    assert agent._should_persist_retry("invalid_response") is True


def test_should_persist_retry_false_for_authorization_and_deterministic():
    agent = _make_agent(persistent=True)
    # These never fix themselves by retrying — must fail fast.
    for reason in (
        FailoverReason.auth,
        FailoverReason.auth_permanent,
        FailoverReason.billing,
        FailoverReason.format_error,
        FailoverReason.content_policy_blocked,
        FailoverReason.model_not_found,
        FailoverReason.provider_policy_blocked,
        FailoverReason.context_overflow,
    ):
        assert agent._should_persist_retry(reason) is False, reason


def test_should_persist_retry_handles_none():
    agent = _make_agent(persistent=True)
    assert agent._should_persist_retry(None) is False


# ── _persistent_retry_time_exhausted: the safety valve ──────────────────

def test_time_exhausted_false_when_no_cap():
    agent = _make_agent(persistent=True, max_elapsed=0)
    # Even an ancient start time never exhausts when cap is 0 (retry forever).
    assert agent._persistent_retry_time_exhausted(time.time() - 10_000) is False


def test_time_exhausted_true_once_cap_exceeded():
    agent = _make_agent(persistent=True, max_elapsed=60)
    assert agent._persistent_retry_time_exhausted(time.time() - 120) is True


def test_time_exhausted_false_within_cap():
    agent = _make_agent(persistent=True, max_elapsed=600)
    assert agent._persistent_retry_time_exhausted(time.time() - 5) is False
