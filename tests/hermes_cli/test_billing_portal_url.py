"""Portal-URL resolution for Phase 2b billing errors (nous_billing).

The server emits ``portalUrl`` relative by design (``/billing?topup=open``); the
client must resolve it against the active portal base so deep-links are clickable
on whatever deployment (preview / staging / prod) the user is pointed at.
"""

from __future__ import annotations

import pytest

from hermes_cli.nous_billing import (
    BillingError,
    _absolutize_portal_url,
    _raise_for_error,
)


@pytest.fixture
def _preview(monkeypatch):
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://nas-pr-412.nousresearch.wtf")


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
