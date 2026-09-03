"""FastAPI-style ``{"detail": "..."}`` error bodies must not read as blank.

Open WebUI, LiteLLM and most self-hosted gateways sit behind FastAPI/Starlette,
which reports errors as ``{"detail": "..."}``. ``_classify_400`` only knew
``error.message`` / ``message`` / ``errorMessage`` / ``errorArgs.reason``. With
none of those keys present ``err_body_msg`` stayed ``""``, the ``is_generic``
heuristic declared the error bare, and the "generic 400 on a large session =
context overflow" rule fired. A healthy session was pushed into the compression
loop and then auto-reset with "max compression attempts reached", while the
real cause (a descriptive upstream router error) never reached the user.

The trap is silent and window-dependent: ``is_large`` is true for *any* session
over 80 messages once ``context_length <= 256_000``, so the same error is a
harmless ``format_error`` on a 262 144-token window and a session-killer on a
200 000-token one. These tests pin the message extraction, which is
window-independent.

Controls matter as much as the fix: a genuinely bare 400 must still route into
compression, and a real overflow reported *through* ``detail`` must still be
recognised as an overflow.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error


class _FakeAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.body = body if body is not None else {}


# Shape of a real LiteLLM router error as forwarded by a FastAPI proxy.
_LITELLM_DETAIL = (
    "litellm.InternalServerError: InternalServerError: OpenAIException - "
    "Failed to generate completions. No fallback model group found for "
    "original model_group=my-model. Fallbacks=[{'my-model_8k': "
    "['my-model_NoRouting']}, {'my-model_16k': ['my-model_NoRouting']}]. "
    "Received Model Group=my-model"
)

# A session that is nowhere near its window but over the absolute
# message/token thresholds that ``is_large`` applies below 256k.
_LIVE_SESSION = dict(
    provider="custom",
    model="my-model",
    approx_tokens=81_520,
    context_length=200_000,
    num_messages=84,
)


def _classify(body, message="Error code: 400", **overrides):
    kwargs = dict(_LIVE_SESSION)
    kwargs.update(overrides)
    return classify_api_error(
        _FakeAPIError(message, status_code=400, body=body),
        **kwargs,
    )


# -- The fix ---------------------------------------------------------------

def test_detail_string_is_not_treated_as_a_bare_error():
    """A descriptive ``detail`` must never enter the compression loop."""
    result = _classify({"detail": _LITELLM_DETAIL})
    assert result.reason is FailoverReason.format_error
    assert result.should_compress is False
    assert result.should_fallback is True


def test_detail_survives_when_the_sdk_does_not_stringify_the_body():
    """``str(error)`` is not a reliable carrier — the body must stand alone.

    The OpenAI SDK happens to render the body into ``str(error)``, which would
    mask a body-only fix. Any SDK that does not (or a wrapped re-raise that
    loses it) must still be classified from the body.
    """
    result = _classify({"detail": _LITELLM_DETAIL}, message="Error code: 400")
    assert result.reason is FailoverReason.format_error


def test_detail_reaches_the_reported_message():
    """The operator must see the router error, not ``Error code: 400``."""
    result = _classify({"detail": _LITELLM_DETAIL})
    assert "no fallback model group found" in result.message.lower()


@pytest.mark.parametrize("detail", [
    # FastAPI request-validation shape.
    [{"loc": ["body", "messages"], "msg": "field required", "type": "value_error.missing"}],
    # Proxies that nest an OpenAI-ish object under detail.
    {"message": "upstream connector refused the request", "code": "connector_down"},
])
def test_non_string_detail_shapes_do_not_crash(detail):
    result = _classify({"detail": detail})
    assert result.reason is not FailoverReason.context_overflow


# -- Controls: the heuristic this fix narrows must still work --------------

def test_genuinely_bare_400_on_a_large_session_still_compresses():
    """Anthropic's bare ``Error`` body is why the heuristic exists."""
    result = _classify({"error": {"message": "Error"}})
    assert result.reason is FailoverReason.context_overflow
    assert result.should_compress is True


def test_empty_body_on_a_large_session_still_compresses():
    result = _classify({})
    assert result.reason is FailoverReason.context_overflow


def test_real_overflow_reported_through_detail_is_still_an_overflow():
    """``detail`` is a carrier, not a verdict — read what is inside it."""
    result = _classify({
        "detail": (
            "This model's maximum context length is 262144 tokens. However, "
            "your messages resulted in 402156 tokens."
        ),
    })
    assert result.reason is FailoverReason.context_overflow
    assert result.should_compress is True


def test_error_message_still_wins_over_detail():
    """Standard OpenAI shape must keep its precedence when both are present."""
    result = _classify({
        "error": {"message": "Error"},
        "detail": _LITELLM_DETAIL,
    })
    assert result.reason is FailoverReason.context_overflow


# -- Subscription usage walls ---------------------------------------------

@pytest.mark.parametrize("message", [
    "You've hit your session limit · resets 2:20pm",
    "You've hit your weekly limit · resets 2:20pm",
])
def test_session_and_weekly_limits_are_transient_rate_limits(message):
    """Subscription backends name the wall by its window, not by the literal
    "usage limit". Without these patterns the error classified as ``unknown``
    (no backoff, no Retry-After) or as a permanent billing failure."""
    result = classify_api_error(
        _FakeAPIError(message, status_code=429, body={"error": {"message": message}}),
        provider="custom",
        model="my-model",
    )
    assert result.reason is FailoverReason.rate_limit
