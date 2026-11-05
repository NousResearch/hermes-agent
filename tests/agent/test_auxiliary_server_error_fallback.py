"""A persistent upstream 5xx must reach the auxiliary provider fallback, not fail the call.

``_FALLBACK_REASONS`` had no rung for a plain 5xx: a 503 ``overloaded_error`` right after the
same-provider transient retries were exhausted left ``reason is None``, so
``_ladder_provider_fallback`` returned early and the configured ``fallback_chain`` / main-agent
chain / discovery was never tried — the aux call (compression, title, memory flush) just failed
while every other provider sat idle.

The rung is deliberately transient: a 5xx is an upstream fault or a capacity blip, so the candidate
is quarantined for the short TTL instead of the long default that hours-long payment exhaustion owns.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import agent.auxiliary_client as aux
from agent.auxiliary_health import fallback_candidate_quarantine_ttl


class _ApiError(Exception):
    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


def _overloaded():
    return _ApiError("Error code: 503 - {'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}",
                     status_code=503)


def _healthy_client():
    from unittest.mock import MagicMock
    return MagicMock(name="fallback-client")


def _route(provider="xai-oauth"):
    """Explicit (non-auto) provider: a capacity error must still bypass the explicit-provider gate."""
    return aux._LadderRoute(
        None, "compression", "", False, "", provider, None, None, None, None, None,
        {"provider": provider}, None, None)


def _drive(gen, response="fallback-response"):
    """Run a ladder generator to completion: collect the steps it yields, return its value."""
    steps = []
    try:
        step = next(gen)
        while True:
            steps.append(step)
            step = gen.send(response)
    except StopIteration as stop:
        return steps, stop.value


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529, 599])
def test_server_statuses_are_claimed(status):
    assert aux._is_server_error(_ApiError("boom", status_code=status)) is True


@pytest.mark.parametrize("status", [200, 400, 401, 402, 403, 404, 408, 429])
def test_non_5xx_statuses_are_not_claimed(status):
    assert aux._is_server_error(_ApiError("boom", status_code=status)) is False


def test_status_less_error_is_not_claimed():
    assert aux._is_server_error(RuntimeError("auxiliary call failed: boom")) is False


def test_response_status_is_claimed():
    """httpx/openai sometimes carry the status on ``.response`` rather than the exception."""
    assert aux._is_server_error(_ApiError("Error code: 500", response=_Response(500))) is True


def test_overloaded_503_resolves_to_server_error_reason():
    reason = next((label for predicate, label in aux._FALLBACK_REASONS if predicate(_overloaded())), None)
    assert reason == "server error"


def test_specific_reasons_still_win_over_the_5xx_rung():
    labels = {
        "rate limit": _ApiError("Error code: 429 - rate limit reached", status_code=429),
        "payment error": _ApiError("Error code: 402 - payment required", status_code=402),
    }
    for expected, exc in labels.items():
        reason = next((label for predicate, label in aux._FALLBACK_REASONS if predicate(exc)), None)
        assert reason == expected


def test_server_error_uses_the_short_quarantine_hold():
    """The long default TTL would hide a healthy fallback for hours over one overloaded upstream."""
    assert fallback_candidate_quarantine_ttl("server error") == 60.0


def test_persistent_503_falls_back_on_an_explicit_provider():
    """End of the ladder: the 5xx must hand over to the configured chain, not re-raise."""
    healthy = _healthy_client()
    with patch.object(aux, "_try_configured_fallback_chain",
                      return_value=(healthy, "fallback-model", "fallback_chain[0](nous)")) as chain, \
         patch.object(aux, "_try_main_fallback_chain") as main_chain:
        steps, result = _drive(aux._ladder_provider_fallback(_overloaded(), _route()))

    assert result == "fallback-response"
    assert len(steps) == 1 and steps[0].kind == "fallback"
    assert steps[0].args == (healthy, "fallback-model", "fallback_chain[0](nous)")
    assert chain.call_args.kwargs["reason"] == "server error"
    main_chain.assert_not_called()


def test_4xx_still_refuses_an_unconfigured_fallback_on_an_explicit_provider():
    """The rung must not widen the gate: a 400 is a request defect, so no provider hop."""
    with patch.object(aux, "_try_configured_fallback_chain") as chain, \
         patch.object(aux, "_try_main_fallback_chain") as main_chain:
        steps, result = _drive(aux._ladder_provider_fallback(_ApiError("bad request", status_code=400), _route()))
    assert steps == [] and result is None
    chain.assert_not_called()
    main_chain.assert_not_called()
