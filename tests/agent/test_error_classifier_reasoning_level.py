"""A 400 that names the reasoning field and refuses its *value* against a closed level set.

The classifier routes it to ``reasoning_mandatory`` (not the ``format_error`` abort), and the
main-loop rung reacts to what was actually sent: a disable steps up to the floor, an enabled level
the route does not offer (the custom profile's unset default ``medium`` against "off, low, high, or
max") is dropped for the session so the route default applies.
"""
import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from tests.agent.test_reasoning_disable_recovery import _recover, _WireAgent, _wire_reasoning_config

# Alibaba token-plan / DashScope: the set starts at ``minimal``, so ``none`` is refused.
_TOKENPLAN_400 = (
    "Error code: 400 - {'error':{'code':'invalid_parameter_error','param':None,"
    "'message':\"'reasoning_effort' must be one of: 'minimal', 'low', 'medium', 'high', "
    "'xhigh', 'max', 'ultra'\",'type':'invalid_request_error'}}"
)
# A custom OpenAI-compatible endpoint whose set has no ``medium`` (deepseek-v4.1-flash-fast).
_OFF_LOW_HIGH_MAX_400 = "HTTP 400: reasoning_effort must be one of off, low, high, or max."
# Groq gpt-oss: graded levels only.
_GROQ_400 = (
    "Error code: 400 - {'error': {'message': \"'reasoning_effort' must be one of 'low', "
    "'medium', or 'high'\", 'type': 'invalid_request_error'}}"
)


class _APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.parametrize("body", [_TOKENPLAN_400, _OFF_LOW_HIGH_MAX_400, _GROQ_400])
def test_closed_level_set_rejection_is_reasoning_mandatory(body):
    result = classify_api_error(_APIError(body, 400), provider="custom", model="m")
    assert result.reason == FailoverReason.reasoning_mandatory
    assert result.retryable is True and result.should_fallback is False


@pytest.mark.parametrize("body", [
    # The same enum wording about another field is not a reasoning rejection.
    "Error code: 400 - 'tool_choice' must be one of none, auto, required",
    # A paired-field constraint, not a closed level set.
    "Error code: 400 - 'reasoning_effort' must be 'none' when 'enable_thinking' is false",
])
def test_other_enum_wording_keeps_its_classification(body):
    result = classify_api_error(_APIError(body, 400), provider="custom", model="m")
    assert result.reason != FailoverReason.reasoning_mandatory


@pytest.mark.parametrize("body", [_OFF_LOW_HIGH_MAX_400, _GROQ_400])
def test_rejected_enabled_level_retries_with_the_route_default(body):
    agent = _WireAgent({"enabled": True, "effort": "medium"})
    assert _wire_reasoning_config(agent) == {"enabled": True, "effort": "medium"}  # the request that 400ed
    _recover(agent, body)
    assert _wire_reasoning_config(agent) is None, "retry must not resend reasoning_effort=medium"
    assert any("rejects reasoning effort medium" in n for n in agent.notices)
