"""Portal-URL resolution for Phase 2b billing errors (nous_billing).

The server emits ``portalUrl`` relative by design (``/billing?topup=open``); the
client must resolve it against the active portal base so deep-links are clickable
on whatever deployment (preview / staging / prod) the user is pointed at.
"""

from __future__ import annotations

import pytest

import hermes_cli.nous_billing as billing
from hermes_cli.nous_billing import (
    BillingError,
    BillingRateLimited,
    _absolutize_portal_url,
    _raise_for_error,
)


class _FakeBillingResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_sizes: list[int] = []

    def __enter__(self) -> "_FakeBillingResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]


class _FakeBillingHTTPError(billing.urllib.error.HTTPError):
    def __init__(
        self,
        body: bytes,
        *,
        code: int = 429,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(
            "https://portal.example.test/api/billing/state",
            code,
            "error",
            headers or {},
            None,
        )
        self.body = body
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]


@pytest.fixture
def _preview(monkeypatch):
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://nas-pr-412.nousresearch.wtf")


@pytest.fixture
def _billing_token(monkeypatch):
    monkeypatch.setattr(
        billing,
        "_resolve_token_and_base",
        lambda *, use_cache=True: ("portal-token", "https://portal.example.test"),
    )


def test_absolutize_resolves_relative(_preview):
    assert (
        _absolutize_portal_url("/billing?topup=open")
        == "https://nas-pr-412.nousresearch.wtf/billing?topup=open"
    )


def test_absolutize_leaves_absolute_unchanged(_preview):
    # Idempotent: an already-absolute URL must NOT be double-prefixed.
    url = "https://other.example/billing?topup=open"
    assert _absolutize_portal_url(url) == url


def test_absolutize_passthrough_empty(_preview):
    assert _absolutize_portal_url(None) is None
    assert _absolutize_portal_url("") == ""


def test_raise_for_error_attaches_absolute_portal_url(_preview):
    # The 403 no_payment_method envelope carries a RELATIVE portalUrl; the raised
    # BillingError must expose it as ABSOLUTE so CLI + TUI render a clickable link.
    with pytest.raises(BillingError) as exc_info:
        _raise_for_error(
            403,
            {"error": "no_payment_method", "portalUrl": "/billing?topup=open"},
        )
    assert (
        exc_info.value.portal_url
        == "https://nas-pr-412.nousresearch.wtf/billing?topup=open"
    )


def test_request_bounds_success_response_read(monkeypatch, _billing_token):
    response = _FakeBillingResponse(b'{"ok": true}')
    captured = []

    def _urlopen(req, timeout):
        captured.append((req, timeout))
        return response

    monkeypatch.setattr(billing.urllib.request, "urlopen", _urlopen)

    result = billing._request("GET", "/api/billing/state", timeout=1)

    assert result == {"ok": True}
    assert response.read_sizes == [billing._BILLING_RESPONSE_BODY_MAX_BYTES + 1]
    assert captured[0][0].full_url == "https://portal.example.test/api/billing/state"
    assert captured[0][1] == 1


def test_request_rejects_oversized_success_response(monkeypatch, _billing_token):
    response = _FakeBillingResponse(
        b"x" * (billing._BILLING_RESPONSE_BODY_MAX_BYTES + 1)
    )
    monkeypatch.setattr(
        billing.urllib.request,
        "urlopen",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(BillingError) as exc_info:
        billing._request("GET", "/api/billing/state")

    assert exc_info.value.error == "response_too_large"
    assert response.read_sizes == [billing._BILLING_RESPONSE_BODY_MAX_BYTES + 1]


def test_request_bounds_oversized_error_response(monkeypatch, _billing_token):
    error = _FakeBillingHTTPError(
        b"x" * (billing._BILLING_ERROR_BODY_MAX_BYTES + 1),
        code=429,
        headers={"Retry-After": "7"},
    )
    monkeypatch.setattr(
        billing.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )

    with pytest.raises(BillingRateLimited) as exc_info:
        billing._request("GET", "/api/billing/state")

    assert exc_info.value.status == 429
    assert exc_info.value.error == "response_too_large"
    assert exc_info.value.retry_after == 7
    assert error.read_sizes == [billing._BILLING_ERROR_BODY_MAX_BYTES + 1]
