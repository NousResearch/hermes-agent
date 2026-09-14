"""Nous free tier, inference side: the dark-tier 403 keyed on the route, the one-shot model move
after ``model_not_free``, the wrong-host heal, the long-wait rule for structured ``rate_limited``
refusals, and the plain outage sentence once retries are spent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_retry_state import TurnRetryState

WELCOME = "https://welcome-api.nousresearch.com/v1"
PAID = "https://inference-api.nousresearch.com/v1"


class MockAPIError(Exception):
    def __init__(self, message, *, status_code=None, body=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.body = body


def _generic_403():
    body = {"status": 403, "message": "You tried to access something that you don't have permissions for."}
    return MockAPIError(f"Error code: 403 - {body}", status_code=403, body=body)


class TestDarkTier403:
    def test_a_generic_403_from_the_welcome_host_is_the_tier_refusing(self):
        result = classify_api_error(_generic_403(), provider="nous", model="nous/welcome", base_url=WELCOME)
        assert result.reason == FailoverReason.auth_permanent
        assert result.retryable is False and result.should_fallback is True
        assert result.error_context["welcome_route"] == "tier_disabled"

    def test_the_same_403_from_the_paid_host_stays_an_ordinary_403(self):
        result = classify_api_error(_generic_403(), provider="nous", model="nous/welcome", base_url=PAID)
        assert "welcome_route" not in result.error_context

    def test_a_403_from_another_provider_on_any_host_is_untouched(self):
        result = classify_api_error(_generic_403(), provider="openrouter", base_url=WELCOME)
        assert "welcome_route" not in result.error_context


def _agent(**overrides):
    lines = []
    agent = SimpleNamespace(
        provider="nous", model="gpt-5", base_url=WELCOME, log_prefix="",
        _vprint=lambda text, force=False: lines.append(text),
        _try_refresh_nous_client_credentials=lambda **kw: True,
    )
    for k, v in overrides.items():
        setattr(agent, k, v)
    agent.lines = lines
    return agent


class TestOneShotRecoveries:
    def test_model_not_free_moves_the_session_onto_the_alternate_and_retries_once(self):
        from agent.turn_recovery import _recover_welcome_tier
        agent = _agent()
        ctx = {"welcome_refusal": {"reason": "model_not_free", "retry_after": 0,
                                   "alternates": ["nous/welcome"], "upgrade_url": ""}}
        retry = TurnRetryState()
        assert _recover_welcome_tier(agent, SimpleNamespace(error_context=ctx), retry, ctx) is True
        assert agent.model == "nous/welcome"
        assert agent._nous_model_switch == ("gpt-5", "nous/welcome")
        assert "without signing in" in agent.lines[0]
        # Once: a second refusal in the same attempt falls through to the terminal path.
        assert _recover_welcome_tier(agent, SimpleNamespace(error_context=ctx), retry, ctx) is False

    def test_model_not_free_without_an_alternate_does_nothing(self):
        from agent.turn_recovery import _recover_welcome_tier
        agent = _agent()
        ctx = {"welcome_refusal": {"reason": "model_not_free", "retry_after": 0, "alternates": [], "upgrade_url": ""}}
        assert _recover_welcome_tier(agent, SimpleNamespace(error_context=ctx), TurnRetryState(), ctx) is False
        assert agent.model == "gpt-5"

    def test_a_wrong_host_refusal_re_reads_the_route_once(self):
        from agent.turn_recovery import _recover_welcome_tier
        calls = []
        agent = _agent(_try_refresh_nous_client_credentials=lambda **kw: calls.append(kw) or True)
        ctx = {"welcome_route": "anon_on_paid_host"}
        retry = TurnRetryState()
        assert _recover_welcome_tier(agent, SimpleNamespace(error_context=ctx), retry, ctx) is True
        assert calls == [{"force": True}]
        assert _recover_welcome_tier(agent, SimpleNamespace(error_context=ctx), retry, ctx) is False

    def test_a_wrong_host_refusal_whose_heal_fails_falls_through(self):
        from agent.turn_recovery import _recover_welcome_tier
        agent = _agent(_try_refresh_nous_client_credentials=lambda **kw: False)
        ctx = {"welcome_route": "anon_on_paid_host"}
        assert _recover_welcome_tier(agent, SimpleNamespace(error_context=ctx), TurnRetryState(), ctx) is False


class TestLongWaitRule:
    @pytest.mark.parametrize("reason,retry_after,expected", [
        ("rate_limited", 20, True), ("rate_limited", 600, True), ("rate_limited", 19, False),
        ("rate_limited", 0, False), ("at_capacity", 30, False), ("admission_closed", 30, False),
    ])
    def test_only_a_long_rate_limited_refusal_is_an_exhausted_allowance(self, reason, retry_after, expected):
        from agent.nous_rate_guard import is_long_welcome_rate_limit
        ctx = {"welcome_refusal": {"reason": reason, "retry_after": retry_after, "alternates": [], "upgrade_url": ""}}
        assert is_long_welcome_rate_limit(ctx) is expected

    def test_no_refusal_is_not_long(self):
        from agent.nous_rate_guard import is_long_welcome_rate_limit
        assert is_long_welcome_rate_limit({}) is False and is_long_welcome_rate_limit(None) is False


class TestOutageCopy:
    @pytest.mark.parametrize("reason", [FailoverReason.timeout, FailoverReason.overloaded,
                                        FailoverReason.server_error, FailoverReason.unknown])
    def test_a_spent_transport_failure_on_the_welcome_host_reads_as_one_sentence(self, reason):
        from agent.turn_recovery import _welcome_outage_copy
        from hermes_cli.anon_auth import FREE_TIER_OUTAGE_COPY
        assert _welcome_outage_copy(WELCOME, SimpleNamespace(reason=reason)) == FREE_TIER_OUTAGE_COPY

    def test_other_routes_and_other_reasons_keep_the_technical_summary(self):
        from agent.turn_recovery import _welcome_outage_copy
        assert _welcome_outage_copy(PAID, SimpleNamespace(reason=FailoverReason.timeout)) == ""
        assert _welcome_outage_copy(WELCOME, SimpleNamespace(reason=FailoverReason.rate_limit)) == ""
