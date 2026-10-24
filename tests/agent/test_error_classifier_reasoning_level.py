"""A 400 that names the reasoning field and refuses its *value* against a closed level set.

The classifier routes it to ``reasoning_mandatory`` (not the ``format_error`` abort), and the
main-loop rung reacts to what was actually sent: a disable steps up to the floor, an enabled level
the route does not offer (the custom profile's unset default ``medium`` against "off, low, high, or
max") is dropped for the session so the route default applies.

Claude Sonnet 5.5 off the native adapter refuses the disable *value* the same way, naming the one it
takes instead (``between_tools``): also ``reasoning_mandatory``, and a disable refusal (floor), not
a field rejection (strip).
"""
import json

import pytest

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    is_reasoning_field_rejection,
    is_reasoning_required_rejection,
)
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


# Claude Sonnet 5.5 behind an OpenAI-compatible relay / gateway: the route translates the generic
# disable into ``thinking: {"type": "disabled"}`` and the model names its own "off" instead.
_SONNET_55_BETWEEN_TOOLS_400 = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'To turn "
    "thinking off on this model, send \"thinking\": {\"type\": \"between_tools\"} instead of {\"type\": "
    "\"disabled\"}. The model does not think before responding. The short updates it writes between tool "
    "calls come back as thinking blocks.'}, 'request_id': 'req_011CgDChhsS9UybgPXWiAR8Z'}"
)


def test_sonnet_55_between_tools_rejection_is_a_reasoning_mandatory_disable_refusal():
    """Before, this 400 was a format_error: the main loop's one-shot length-continuation ended the turn
    and every thinking-off title call failed."""
    assert is_reasoning_required_rejection(_SONNET_55_BETWEEN_TOOLS_400)
    assert not is_reasoning_field_rejection(_SONNET_55_BETWEEN_TOOLS_400)
    result = classify_api_error(_APIError(_SONNET_55_BETWEEN_TOOLS_400, 400), provider="custom",
                                model="claude-sonnet-5-5")
    assert result.reason == FailoverReason.reasoning_mandatory
    assert result.retryable is True
    assert result.should_fallback is False and result.should_compress is False


@pytest.mark.parametrize("body", [
    # A reasoning-field 400 that doesn't name between_tools keeps its old classification.
    "Error code: 400 - thinking.budget_tokens: Input should be greater than or equal to 1024",
    # "between tool calls" (no underscore) next to "thinking" is not the between_tools type.
    "Error code: 400 - thinking blocks must be preserved between tool calls in this conversation",
])
def test_reasoning_400_without_the_between_tools_type_keeps_its_classification(body):
    result = classify_api_error(_APIError(body, 400), provider="custom", model="claude-sonnet-5-5")
    assert result.reason != FailoverReason.reasoning_mandatory
    assert not is_reasoning_required_rejection(body)


_SONNET_55_INNER = ('To turn thinking off on this model, send "thinking": {"type": "between_tools"} instead of '
                    '{"type": "disabled"}.')
_AGGREGATOR_WRAPPED_400 = "Error code: 400 - " + str({"error": {
    "message": "Provider returned error (reasoning)", "code": 400,
    "metadata": {"provider_name": "Anthropic", "raw": json.dumps({"type": "error", "error": {
        "type": "invalid_request_error", "message": _SONNET_55_INNER}})}}})


@pytest.mark.parametrize("body", [
    # A relay that names a reasoning field well before the model's own message.
    "Error code: 400 - upstream reasoning translation failed (thinking disabled via reasoning_effort none, "
    "mapped for claude-sonnet-5-5 on relay 8317): To turn thinking off on this model, send \"thinking\": "
    "{\"type\": \"between_tools\"} instead of {\"type\": \"disabled\"}.",
    # An aggregator wrapper whose own message mentions reasoning before the raw upstream body.
    _AGGREGATOR_WRAPPED_400,
])
def test_between_tools_match_is_anchored_on_the_marker_not_the_first_reasoning_word(body):
    result = classify_api_error(_APIError(body, 400), provider="custom", model="claude-sonnet-5-5")
    assert result.reason == FailoverReason.reasoning_mandatory


def test_between_tools_far_from_any_reasoning_field_is_not_this_rejection():
    """``between_tools`` must sit next to a reasoning field: a docs pointer 200 chars after an unrelated
    "is required" reasoning 400 keeps that 400's classification (a guard; it passes on main too)."""
    body = ("Error code: 400 - thinking.budget_tokens is required when type is enabled. " + "x" * 200
            + " see docs/between_tools")
    result = classify_api_error(_APIError(body, 400), provider="custom", model="claude-sonnet-5-5")
    assert result.reason != FailoverReason.reasoning_mandatory
