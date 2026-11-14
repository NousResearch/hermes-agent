"""A provider's 403 for region/territory policy is not an API-key rejection (#148727).

OpenAI answers 403 ``unsupported_country_region_territory`` ("Country, region, or territory not
supported"), and aggregators answer 403 "This model is not available in your country". The request
reaches the provider and the key is accepted, so ``_status_403`` used to fall through to its
bare-403 default (``auth``) and print key-rotation / ``hermes setup`` guidance for a credential
that was never the problem. Each case here is a 403 whose BODY names a region.
"""
import pytest

from agent.error_classifier import classify_api_error
from agent.turn_failure_copy import failure_cause_gloss, nonretryable_copy


class _APIError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


REGION_BODIES = [
    "[unsupported_country_region_territory] Country, region, or territory not supported",
    "This model is not available in your country.",
    "User location is not supported for the API use.",
    "The model is not available in your region.",
]


@pytest.mark.parametrize("body", REGION_BODIES)
def test_403_region_restriction_is_not_auth(body):
    result = classify_api_error(_APIError(body, 403), provider="opencode-go")
    assert result.reason.value == "region_blocked"
    assert result.is_auth is False
    assert result.should_rotate_credential is False
    assert result.retryable is False and result.should_fallback is True


def test_region_copy_names_the_region_not_the_key():
    body = REGION_BODIES[0]
    result = classify_api_error(_APIError(body, 403), provider="opencode-go")
    copy = nonretryable_copy(result, provider="opencode-go", model="gpt-5.6-luna", summary=body)
    assert "region" in copy.lower()
    for wrong in ("API key", "hermes setup", "User-Agent", "firewall"):
        assert wrong not in copy
    assert body in copy


def test_region_gloss_names_the_region_not_a_firewall():
    gloss = failure_cause_gloss("region_blocked")
    assert gloss and "region" in gloss
    assert "firewall" not in gloss and "CDN" not in gloss
