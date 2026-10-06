"""#133850: OpenRouter's reworded guardrail/data-policy 404 must still classify as
``provider_policy_blocked`` (abort retry ladder, fall back immediately).

The old patterns ("no endpoints available matching your ...") are not a substring of the
2026-10 routing wording, so the 404 fell through to ``_V_UNKNOWN`` and was retried 3x with
backoff before fallback kicked in — the docs say a 404 falls back on the first failure.
"""

from agent.error_classifier import FailoverReason, classify_api_error


class _Err(Exception):
    def __init__(self, msg, status_code):
        super().__init__(msg)
        self.status_code = status_code


NEW_ROUTING_404 = (
    "NotFoundError: 0 endpoints out of 3 requested are available matching your guardrail "
    "restrictions and data policy. We removed them for the following reasons ...:\n"
    "Model blocked by guardrail: 3 endpoints excluded; ..."
)


def _classify(msg, status_code=404):
    return classify_api_error(_Err(msg, status_code), provider="openrouter", model="anthropic/claude-opus")


def test_new_routing_headline_classifies_as_policy_blocked():
    result = _classify(NEW_ROUTING_404)
    assert result.reason == FailoverReason.provider_policy_blocked
    assert result.retryable is False
    assert result.should_fallback is True


def test_reason_line_alone_also_matches():
    # The reason-list phrase must work even if the headline wording drifts again.
    result = _classify("404: Model blocked by guardrail: 1 endpoint excluded")
    assert result.reason == FailoverReason.provider_policy_blocked


def test_unrelated_zero_count_404_stays_unknown():
    # "0 endpoints" alone is not a policy signal — a generic routing miss must keep
    # surfacing as unknown instead of silently misreporting a policy block.
    result = _classify("0 endpoints out of 3 requested are available in this region")
    assert result.reason != FailoverReason.provider_policy_blocked
