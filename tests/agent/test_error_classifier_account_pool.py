"""An aggregator's exhausted account pool backs off; it is never a context overflow.

The production 503 body is a row in ``test_error_classifier_observed.py``. Its "daily token limit"
tail carries the overflow phrase "token limit", so this pins the other paths that read the same
wording: 500/502/529 share the 5xx rule table, and a status-less relay goes through the message rules.
"""

import pytest

from agent.error_classifier import FailoverReason, classify_api_error

POOL_EXHAUSTED = ("no usable account for realm 'intl': all are disabled, cooling down, expired or parked "
                  "by the daily token limit - add or enable one at the dashboard (/)")


class _APIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = {}


@pytest.mark.parametrize("status_code", [500, 502, 529, None])
def test_exhausted_account_pool_backs_off_instead_of_compressing(status_code):
    result = classify_api_error(_APIError(POOL_EXHAUSTED, status_code), provider="custom", model="test-model")
    assert result.reason == FailoverReason.overloaded
    assert result.should_compress is False
