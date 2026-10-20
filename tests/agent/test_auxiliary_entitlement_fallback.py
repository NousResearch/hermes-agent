"""An entitlement-shaped 403 must reach the provider-fallback rung (#133193).

``_FALLBACK_REASONS`` is a closed wording allowlist: a provider answering 403 with a body
that is neither credential nor billing wording (OpenCode Zen: ``"Model access is
disabled"`` phrased as ``server_error``) matched no predicate, ``reason`` stayed ``None``
and ``_ladder_provider_fallback`` returned before planning any fallback — the aux task
died silently, the provider was never benched, and the usage report showed nothing.

The tests assert the relationship over the wording: an unrecognized 403 yields a fallback
plan plus a health mark, auth/billing wording keeps its own rungs, and errors the table
genuinely does not know (a bare 500) still stop the ladder.
"""

import pytest

import agent.auxiliary_client as ac

ZEN_DISABLED = (
    '{"error":{"type":"server_error","message":"Upstream request failed: '
    'Model access is disabled"}}'
)


class _Err(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status_code = status


class _FakeClient:
    api_key = "sk-test"
    base_url = "https://opencode.example/v1"


def _route(provider="opencode-zen"):
    return ac._LadderRoute(
        client=_FakeClient(),
        task="title_generation",
        tag="",
        async_mode=False,
        base_info=_FakeClient.base_url,
        resolved_provider=provider,
        resolved_model="gemini-3-flash",
        resolved_base_url=None,
        resolved_api_key=None,
        resolved_api_mode=None,
        final_model="gemini-3-flash",
        main_runtime=None,
        route_info={},
        timeout=30.0,
    )


def test_entitlement_403_is_a_capacity_reason_not_auth_or_payment():
    exc = _Err(ZEN_DISABLED, 403)
    assert ac._is_model_access_disabled_error(exc)
    assert not ac._is_auth_error(exc)
    assert not ac._is_payment_error(exc)
    reason = next(
        (label for predicate, label in ac._FALLBACK_REASONS if predicate(exc)), None
    )
    assert reason == "model access disabled"


@pytest.mark.parametrize(
    "body",
    [
        "Model gemini-3-flash is not available on the free tier",
        "Error code: 400 - {'error': {'message': 'quota exceeded, add credits'}}",
        "Unauthenticated: bad-credentials (expired OAuth token)",
    ],
)
def test_auth_and_billing_wording_keep_their_own_rungs(body):
    """First match wins: free-tier/quota bodies stay payment, bad-credentials stays auth —
    the new rung must not swallow neighbouring classifications."""
    exc = _Err(body, 403)
    assert not ac._is_model_access_disabled_error(exc)
    assert ac._is_payment_error(exc) or ac._is_auth_error(exc)


def test_entitlement_403_plans_fallback_and_benches_endpoint(monkeypatch):
    """The core relationship: a 403 no predicate recognized before must still yield a
    fallback plan from an explicit provider (capacity errors bypass the explicit gate)
    and mark the endpoint unhealthy so later aux calls skip it."""
    ac._reset_aux_unhealthy_cache()
    monkeypatch.setattr(ac, "_recoverable_pool_provider", lambda *a, **kw: None)
    monkeypatch.setattr(ac, "_get_auxiliary_task_config", lambda task: {})
    monkeypatch.setattr(
        ac,
        "_try_configured_fallback_chain",
        lambda *a, **kw: (_FakeClient(), "fallback-model", "opencode-go"),
    )

    performed = []

    def perform(step):
        performed.append(step.kind)
        return "fb-response"

    result = ac._drive_ladder(
        ac._ladder_provider_fallback(_Err(ZEN_DISABLED, 403), _route()), perform
    )

    assert performed == ["fallback"], "the ladder must hand the driver a fallback step"
    assert result == "fb-response", "the fallback response must own the outcome"
    assert ac._is_provider_unhealthy("opencode-zen", _FakeClient.base_url)
    ac._reset_aux_unhealthy_cache()


def test_unknown_error_still_stops_the_ladder():
    """Guard against over-broad classification: an error the table genuinely does not
    know (a bare 500) must keep returning None before any provider hop."""
    performed = []

    def perform(step):
        performed.append(step.kind)
        return "should-not-happen"

    result = ac._drive_ladder(
        ac._ladder_provider_fallback(_Err("Internal server error", 500), _route()),
        perform,
    )
    assert result is None
    assert performed == []
