"""Invalid cooldown metadata must not replace the original provider error."""

import io
import json
from urllib.error import HTTPError

import pytest

from agent.retry_utils import parse_retry_after_seconds


@pytest.mark.parametrize("raw, expected", [
    ("inf", None), ("-Infinity", None), ("NaN", None), ("1e9999", None),
    (float("inf"), None), (float("-inf"), None), (float("nan"), None),
    pytest.param(10 ** 400, None, id="overflowing-integer"),
    ("12.5", 12.5), (12, 12.0), ("-3", 0.0), (0, 0.0),
    ("Wed, 21 Oct 2015 07:28:00 GMT", 0.0),
])
def test_retry_after_is_finite_or_absent(raw, expected):
    for value in (raw, {"Retry-After": raw}, {"retry-after": raw}):
        assert parse_retry_after_seconds(value) == expected


@pytest.mark.parametrize("header", ["inf", "NaN", "1e9999"])
def test_invalid_retry_after_preserves_http_error_contract(monkeypatch, header):
    from hermes_cli import auth_codex, nous_billing

    payload = {"error": "rate_limited", "message": "Try later"}
    headers = {"Retry-After": header}

    def rate_limited(request, **kwargs):
        raise HTTPError(request.full_url, 429, "Too Many Requests", headers,
                        io.BytesIO(json.dumps(payload).encode()))

    monkeypatch.setattr(nous_billing, "_resolve_token_and_base",
                        lambda **kwargs: ("test-token", "https://billing.example.invalid"))
    monkeypatch.setattr(nous_billing.urllib.request, "urlopen", rate_limited)
    with pytest.raises(nous_billing.BillingRateLimited) as caught:
        nous_billing._request("GET", "/billing/state")
    assert caught.value.status == 429
    assert caught.value.payload == payload
    assert caught.value.retry_after is None
    assert auth_codex._parse_retry_after_seconds(headers) is None
