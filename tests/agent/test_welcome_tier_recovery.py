"""Nous free tier, inference side: the dark-tier 403 keyed on the route, the one-shot model move
after ``model_not_free``, the wrong-host heal, the long-wait rule for structured ``rate_limited``
refusals, and the plain outage sentence once retries are spent.

The recovery hooks are driven from the REAL producer boundary: a gateway body goes through
``classify_api_error`` (and the turn's own ``extract_api_error_context``) exactly as the turn loop
feeds them, so a wiring slip between the two contexts fails here."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import extract_api_error_context
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


def _gateway_error(status: int, body: dict) -> MockAPIError:
    return MockAPIError(f"Error code: {status} - {body}", status_code=status, body=body)


def _generic_403():
    return _gateway_error(403, {"status": 403, "message": "You tried to access something that you don't have permissions for."})


def _refusal(reason: str, *, retry_after: int = 0, alternates=None) -> MockAPIError:
    return _gateway_error(429, {"status": 429, "message": "refused", "reason": reason, "retry_after": retry_after,
                                "alternates": alternates or [], "upgrade_url": "https://portal.example/signup"})


def _classify(err: MockAPIError, *, model: str = "nous/welcome", base_url: str = WELCOME):
    return classify_api_error(err, provider="nous", model=model, base_url=base_url)


class TestDarkTier403:
    def test_a_generic_403_from_the_welcome_host_is_the_tier_refusing(self):
        result = _classify(_generic_403())
        assert result.reason == FailoverReason.auth_permanent
        assert result.retryable is False and result.should_fallback is True
        assert result.error_context["welcome_route"] == "tier_disabled"

    def test_the_same_403_from_the_paid_host_stays_an_ordinary_403(self):
        assert "welcome_route" not in _classify(_generic_403(), base_url=PAID).error_context

    def test_a_403_that_says_something_else_keeps_its_own_classification(self):
        """A safety refusal or a billing wall on the welcome host is not the tier going dark."""
        body = {"status": 403, "message": "This request violates our usage policies."}
        result = _classify(_gateway_error(403, body))
        assert "welcome_route" not in result.error_context
        assert result.reason == FailoverReason.content_policy_blocked

    def test_a_403_from_another_provider_on_any_host_is_untouched(self):
        result = classify_api_error(_generic_403(), provider="openrouter", base_url=WELCOME)
        assert "welcome_route" not in result.error_context


def _agent(**overrides):
    lines = []
    agent = SimpleNamespace(
        provider="nous", model="gpt-5", base_url=WELCOME, log_prefix="", _rate_limit_state=None,
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
        classified = _classify(_refusal("model_not_free", alternates=["nous/welcome"]), model="gpt-5")
        retry = TurnRetryState()
        assert _recover_welcome_tier(agent, classified, retry) is True
        assert agent.model == "nous/welcome"
        assert agent._nous_model_switch == ("gpt-5", "nous/welcome")
        assert "without signing in" in agent.lines[0]
        # Once: a second refusal in the same attempt falls through to the terminal path.
        assert _recover_welcome_tier(agent, classified, retry) is False

    def test_model_not_free_without_an_alternate_does_nothing(self):
        from agent.turn_recovery import _recover_welcome_tier
        agent = _agent()
        classified = _classify(_refusal("model_not_free"), model="gpt-5")
        assert _recover_welcome_tier(agent, classified, TurnRetryState()) is False
        assert agent.model == "gpt-5"

    def test_a_wrong_host_refusal_re_reads_the_route_once(self):
        from agent.turn_recovery import _recover_welcome_tier
        calls = []
        agent = _agent(_try_refresh_nous_client_credentials=lambda **kw: calls.append(kw) or True)
        body = {"status": 400, "message": "Anonymous accounts must use https://welcome-api.nousresearch.com for inference."}
        classified = _classify(_gateway_error(400, body), base_url=PAID)
        assert classified.error_context["welcome_route"] == "anon_on_paid_host"
        retry = TurnRetryState()
        assert _recover_welcome_tier(agent, classified, retry) is True
        assert calls == [{"force": True}]
        assert _recover_welcome_tier(agent, classified, retry) is False

    def test_a_wrong_host_refusal_whose_heal_fails_falls_through(self):
        from agent.turn_recovery import _recover_welcome_tier
        agent = _agent(_try_refresh_nous_client_credentials=lambda **kw: False)
        body = {"status": 400, "message": "Anonymous accounts must use https://welcome-api.nousresearch.com for inference."}
        assert _recover_welcome_tier(agent, _classify(_gateway_error(400, body), base_url=PAID), TurnRetryState()) is False


class TestLongWaitRule:
    @pytest.mark.parametrize("reason,retry_after,expected", [
        ("rate_limited", 20, True), ("rate_limited", 600, True), ("rate_limited", 19, False),
        ("rate_limited", 0, False), ("at_capacity", 30, False), ("admission_closed", 30, False),
    ])
    def test_only_a_long_rate_limited_refusal_is_an_exhausted_allowance(self, reason, retry_after, expected):
        from agent.nous_rate_guard import is_long_welcome_rate_limit
        classified = _classify(_refusal(reason, retry_after=retry_after))
        assert is_long_welcome_rate_limit(classified.error_context) is expected

    def test_no_refusal_is_not_long(self):
        from agent.nous_rate_guard import is_long_welcome_rate_limit
        assert is_long_welcome_rate_limit({}) is False and is_long_welcome_rate_limit(None) is False

    def test_the_turn_records_a_long_refusal_from_the_classifiers_context(self, monkeypatch):
        """The turn hands the guard TWO contexts: its own (``extract_api_error_context``), which
        never carries ``welcome_refusal``, and the classifier's, which does. The breaker must key on
        the latter and record the reset it computed."""
        import agent.nous_rate_guard as guard
        from agent.turn_recovery import _is_genuine_nous_rate_limit
        recorded = []
        monkeypatch.setattr(guard, "record_nous_rate_limit", lambda **kw: recorded.append(kw))
        err = _refusal("rate_limited", retry_after=600)
        turn_ctx = extract_api_error_context(err)
        assert "welcome_refusal" not in turn_ctx
        classified = _classify(err)
        assert _is_genuine_nous_rate_limit(_agent(), err, turn_ctx, classified) is True
        assert recorded and recorded[0]["error_context"]["reset_at"] == classified.error_context["reset_at"]
        # The same body from the PAID host is not an allowance verdict: main's header rule stands.
        recorded.clear()
        assert _is_genuine_nous_rate_limit(_agent(base_url=PAID), err, turn_ctx, classified) is False
        assert recorded == []
        # A short one is not an exhausted allowance: nothing recorded, the turn waits it out.
        recorded.clear()
        short = _refusal("rate_limited", retry_after=5)
        assert _is_genuine_nous_rate_limit(_agent(), short, extract_api_error_context(short), _classify(short)) is False
        assert recorded == []


class TestOutageCopy:
    @pytest.mark.parametrize("reason", [FailoverReason.timeout, FailoverReason.overloaded,
                                        FailoverReason.server_error])
    def test_a_spent_transport_failure_on_the_welcome_host_reads_as_one_sentence(self, reason):
        from agent.turn_recovery import _welcome_outage_copy
        from hermes_cli.anon_auth import FREE_TIER_OUTAGE_COPY
        assert _welcome_outage_copy(WELCOME, SimpleNamespace(reason=reason)) == FREE_TIER_OUTAGE_COPY

    def test_other_routes_and_other_reasons_keep_the_technical_summary(self):
        from agent.turn_recovery import _welcome_outage_copy
        assert _welcome_outage_copy(PAID, SimpleNamespace(reason=FailoverReason.timeout)) == ""
        assert _welcome_outage_copy(WELCOME, SimpleNamespace(reason=FailoverReason.rate_limit)) == ""
        # ``unknown`` is the catch-all for status-less local failures, not the free model's trouble.
        assert _welcome_outage_copy(WELCOME, SimpleNamespace(reason=FailoverReason.unknown)) == ""


class TestTerminalResultsCarryTheFreeTierBlock:
    """The terminal results stamp ``free_tier`` so a client renders the free tier's own card
    (agent/error_surface.py) instead of an OAuth re-login."""

    @staticmethod
    def _terminal_agent():
        agent = _agent(_dump_api_request_debug=lambda *a, **k: None, _flush_status_buffer=lambda: None,
                       _summarize_api_error=lambda e: "HTTP 403: no permissions", _emit_status=lambda *a: None,
                       _persist_session=lambda *a: None, _plines=lambda *a: None, _buffer_status=lambda *a: None,
                       _rate_limit_state=None, _has_pending_fallback=lambda: False)
        return agent

    def test_a_dark_tier_403_is_stamped_disabled_with_the_chat_sentence(self):
        from agent.turn_recovery import nonretryable_client_error_result
        err = _generic_403()
        classified = _classify(err)
        result = nonretryable_client_error_result(
            self._terminal_agent(), err, classified, status_code=403, api_kwargs=None, api_messages=[],
            messages=[], conversation_history=[], api_call_count=1, approx_tokens=10,
            provider="nous", base_url=WELCOME, model="nous/welcome")
        # The chat text names /login; the card text (a button beside it) leaves that tail off.
        assert "switched off" in result["final_response"] and "/login" in result["final_response"]
        assert result["free_tier"]["kind"] == "disabled"
        assert result["final_response"].startswith(result["free_tier"]["message"])
        assert "/login" not in result["free_tier"]["message"]
        assert result["error"] == "HTTP 403: no permissions"      # the technical detail stays in the log line

    def test_an_exhausted_capacity_refusal_is_stamped_at_capacity(self):
        from agent.turn_recovery import max_retries_exhausted_result
        err = _refusal("at_capacity", retry_after=30)
        classified = _classify(err)
        result = max_retries_exhausted_result(
            self._terminal_agent(), err, classified, max_retries=3, is_rate_limited=True, error_msg="429",
            api_kwargs=None, api_messages=[], messages=[], conversation_history=[], api_call_count=3,
            approx_tokens=10, provider="nous", base_url=WELCOME, model="nous/welcome")
        assert result["free_tier"]["kind"] == "at_capacity"
        assert "really busy" in result["free_tier"]["message"] and "/login" not in result["free_tier"]["message"]
        assert "/login" in result["final_response"]

    def test_a_spent_outage_on_the_welcome_host_is_stamped_outage(self):
        from agent.turn_recovery import max_retries_exhausted_result
        err = _gateway_error(503, {"status": 503, "message": "The requested model is currently unavailable."})
        classified = _classify(err)
        result = max_retries_exhausted_result(
            self._terminal_agent(), err, classified, max_retries=3, is_rate_limited=False, error_msg="503",
            api_kwargs=None, api_messages=[], messages=[], conversation_history=[], api_call_count=3,
            approx_tokens=10, provider="nous", base_url=WELCOME, model="nous/welcome")
        assert result["free_tier"]["kind"] == "outage"

    def test_the_same_outage_on_the_paid_host_is_not_stamped(self):
        from agent.turn_recovery import max_retries_exhausted_result
        err = _gateway_error(503, {"status": 503, "message": "The requested model is currently unavailable."})
        result = max_retries_exhausted_result(
            self._terminal_agent(), err, _classify(err, base_url=PAID), max_retries=3, is_rate_limited=False,
            error_msg="503", api_kwargs=None, api_messages=[], messages=[], conversation_history=[],
            api_call_count=3, approx_tokens=10, provider="nous", base_url=PAID, model="hermes-4")
        assert "free_tier" not in result
