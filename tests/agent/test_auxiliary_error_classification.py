"""Auxiliary error classification shares the main classifier's billing table and labels quarantines truthfully.

Invariants for #107166 (OpenRouter org "Budget limit exceeded" 403 must be a payment error for the aux
ladder, like the main loop), #64144 (an unhealthy mark for absent credentials must not claim a payment
error), and #108349 (a vision fallback never lands on a text-only main model; an upstream-capacity 429
never benches the credential pool).
"""
import logging

import pytest

from agent import auxiliary_client as ac
from agent.error_classifier import FailoverReason, classify_api_error


class _Err(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        self.status_code = status


@pytest.mark.parametrize("body", [
    "Error code: 403 - {'error': {'message': 'Budget limit exceeded (monthly limit). Contact your org admin.', 'code': 403}}",
    "Error code: 403 - {'error': {'message': 'Key limit exceeded (total limit). Contact your org admin.', 'code': 403}}",
])
def test_openrouter_limit_403_is_billing_for_main_loop_and_aux_ladder(body):
    exc = _Err(body, 403)
    assert classify_api_error(exc).reason is FailoverReason.billing
    assert ac._is_payment_error(exc)
    # Every main-classifier billing phrase is an aux payment phrase (one table, no drift).
    assert all(p in ac._PAYMENT_KEYWORDS for p in classify_api_error.__globals__["_BILLING_PATTERNS"])


def test_absent_credentials_quarantine_is_debug_and_names_the_real_reason(caplog):
    ac._reset_aux_unhealthy_cache()
    with caplog.at_level(logging.DEBUG, logger="agent.auxiliary_client"):
        ac._mark_provider_unhealthy("openrouter", ttl=60, reason="OPENROUTER_API_KEY not set", level=logging.DEBUG)
        ac._log_skip_unhealthy("openrouter", "title_generation")
        ac._mark_provider_unhealthy("nous")  # confirmed 402 path keeps the payment wording at WARNING
    marks = [r for r in caplog.records if "marking" in r.getMessage()]
    assert [(r.levelno, "payment" in r.getMessage()) for r in marks] == [(logging.DEBUG, False), (logging.WARNING, True)]
    assert "OPENROUTER_API_KEY not set" in marks[0].getMessage()
    skip = next(r for r in caplog.records if "skipping" in r.getMessage())
    assert "OPENROUTER_API_KEY not set" in skip.getMessage() and "payment" not in skip.getMessage()
    ac._reset_aux_unhealthy_cache()


def test_vision_fallback_skips_text_only_main_model(monkeypatch):
    monkeypatch.setattr(ac, "_read_main_provider", lambda: "zai")
    monkeypatch.setattr(ac, "_read_main_model", lambda: "glm-5.3")
    monkeypatch.setattr(ac, "_main_model_supports_vision", lambda provider, model: False)
    monkeypatch.setattr(ac, "resolve_provider_client", lambda **kw: pytest.fail("must not build a client for a text-only model"))
    assert ac._try_main_agent_model_fallback("nous", "vision", reason="rate limit") == (None, None, "")


@pytest.mark.parametrize("body, benches_pool", [
    ("Error code: 429 - {'status': 429, 'message': \"The requested model is temporarily at capacity upstream. "
     "This is not your API key's rate limit — please retry shortly.\"}", False),
    ("Error code: 429 - {'error': {'message': 'Rate limit exceeded for this API key', 'code': 429}}", True),
])
def test_upstream_capacity_429_is_not_a_credential_to_bench(monkeypatch, body, benches_pool):
    exc = _Err(body, 429)
    assert ac._is_rate_limit_error(exc)
    assert ac._is_overloaded_error(exc) is (not benches_pool)
    assert classify_api_error(exc).reason is (
        FailoverReason.rate_limit if benches_pool else FailoverReason.overloaded)

    # The predicate is only half the contract: _recover_provider_pool must not bench a credential
    # for an upstream-capacity 429, while a real per-key rate limit still rotates the pool.
    class _StubPool:
        def __init__(self):
            self.rotate_calls = []

        def has_credentials(self):
            return True

        def try_refresh_current(self):
            return None

        def mark_exhausted_and_rotate(self, **kwargs):
            self.rotate_calls.append(kwargs)
            return object()  # a next entry exists

    pool = _StubPool()
    monkeypatch.setattr(ac, "load_pool", lambda provider: pool)
    monkeypatch.setattr(ac, "_evict_cached_clients", lambda provider: None)
    recovered = ac._recover_provider_pool("openrouter", exc, failed_api_key="sk-failed")
    assert recovered is benches_pool
    assert len(pool.rotate_calls) == (1 if benches_pool else 0)
    if benches_pool:
        assert pool.rotate_calls[0]["status_code"] == 429
        assert pool.rotate_calls[0]["api_key_hint"] == "sk-failed"


@pytest.mark.parametrize(
    "wrapped,http_status,code,reason,status,retryable,rotate,fallback",
    [
        (wrapped, *case)
        for case in [
            (200, 503, FailoverReason.overloaded, 503, True, False, False),
            (200, 500, FailoverReason.server_error, 500, True, False, False),
            (200, 408, FailoverReason.timeout, 408, True, False, False),
            (200, 401, FailoverReason.auth, 401, False, True, True),
            (
                200,
                "rate_limit_exceeded",
                FailoverReason.rate_limit,
                None,
                True,
                True,
                False,
            ),
            (
                200,
                "insufficient_quota",
                FailoverReason.billing,
                None,
                False,
                True,
                True,
            ),
            (200, "opaque_error", FailoverReason.unknown, None, True, False, False),
        ]
        for wrapped in (False, True)
    ]
    + [
        (
            False,
            401,
            "rate_limit_exceeded",
            FailoverReason.auth,
            401,
            False,
            True,
            True,
        ),
        (
            False,
            503,
            "insufficient_quota",
            FailoverReason.overloaded,
            503,
            True,
            False,
            False,
        ),
        (
            False,
            429,
            "insufficient_quota",
            FailoverReason.billing,
            429,
            False,
            True,
            True,
        ),
        (
            False,
            429,
            "rate_limit_exceeded",
            FailoverReason.rate_limit,
            429,
            True,
            True,
            True,
        ),
    ],
)
def test_auxiliary_classification_preserves_native_recovery(
    wrapped,
    http_status,
    code,
    reason,
    status,
    retryable,
    rotate,
    fallback,
    tmp_path,
    monkeypatch,
):
    from copy import deepcopy

    import httpx

    from agent import auxiliary_call_outcome as outcome
    from agent.error_classifier import ClassifiedError
    from agent.native_usage_openai import UsageValidatedOpenAI

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    error = {"code": code, "message": "Native provider failure"}
    payload = (
        {"success": False, "data": {"error": error}} if wrapped else {"error": error}
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(http_status, json=payload)
            )
        ) as http,
        UsageValidatedOpenAI(
            api_key="fixture",
            base_url="https://fixture.invalid/v1",
            http_client=http,
            max_retries=0,
        ) as client,
    ):
        with pytest.raises(Exception) as caught:
            response = client.chat.completions.create(model="fixture", messages=[])
            ac._validate_llm_response(response)

    native = caught.value
    before = {
        key: deepcopy(value) if key in {"body", "provider_error"} else value
        for key, value in vars(native).items()
    }
    classified = outcome.classify_auxiliary_error(
        native, provider="openai", model="fixture"
    )
    assert (
        classified,
        outcome.error_class(native, provider="openai", model="fixture"),
        vars(native),
    ) == (
        ClassifiedError(
            reason=reason,
            status_code=status,
            provider="openai",
            model="fixture",
            message="Native provider failure",
            retryable=retryable,
            should_rotate_credential=rotate,
            should_fallback=fallback,
        ),
        reason.value,
        before,
    )


@pytest.mark.parametrize("failure", ["unknown", "cancelled", "classifier"])
def test_auxiliary_classification_keeps_unknown_error_isolation(failure, monkeypatch):
    from agent import auxiliary_call_outcome as outcome
    from agent import error_classifier
    from agent.error_classifier import ClassifiedError

    native = (
        KeyboardInterrupt()
        if failure == "cancelled"
        else RuntimeError("Opaque failure")
    )
    native.__cause__ = ValueError("Original cause")
    before = (native.args, vars(native).copy(), native.__cause__)
    if failure == "classifier":

        def reject(*args, **kwargs):
            raise ValueError("Classifier failed")

        monkeypatch.setattr(error_classifier, "classify_api_error", reject)

    assert (
        outcome.classify_auxiliary_error(native),
        outcome.error_class(native),
        (native.args, vars(native), native.__cause__),
    ) == (
        ClassifiedError(
            reason=FailoverReason.unknown,
            provider="",
            model="",
            message="Opaque failure" if failure == "unknown" else "",
        ),
        "unknown",
        before,
    )
