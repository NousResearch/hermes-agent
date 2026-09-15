"""A Portal-issued inference URL is validated against the issuing Portal, not a flat allowlist.

The staging Portal returns its own inference gateway (``stg-inference-api``). A production-only
allowlist refused that value and healed it to the PROD host, so a staging-issued JWT was sent to
the production gateway and 401'd. The pairing is keyed on the issuing Portal so the original
protection survives: a PROD-portal session that finds a staging inference URL in its state (the
poisoned-``auth.json`` case the allowlist exists for) is still refused.
"""
from __future__ import annotations

import pytest

from hermes_cli import auth_nous

PROD_PORTAL = "https://portal.nousresearch.com"
STAGING_PORTAL = "https://portal.staging-nousresearch.com"
PROD_INFERENCE = "https://inference-api.nousresearch.com/v1"
STAGING_INFERENCE = "https://stg-inference-api.nousresearch.com/v1"


@pytest.mark.parametrize(
    "portal, url, expected",
    [
        # The bug: the staging Portal's own inference host must survive validation.
        (STAGING_PORTAL, STAGING_INFERENCE, STAGING_INFERENCE),
        # The protection: a prod session must never accept a staging host, however it got there.
        (PROD_PORTAL, STAGING_INFERENCE, None),
        # An unknown/absent Portal keeps the strict production-only allowlist.
        (None, STAGING_INFERENCE, None),
        ("https://portal.evil.example", STAGING_INFERENCE, None),
        # Production remains valid from either Portal; pairing widens, never narrows.
        (PROD_PORTAL, PROD_INFERENCE, PROD_INFERENCE),
        (STAGING_PORTAL, PROD_INFERENCE, PROD_INFERENCE),
    ],
)
def test_inference_host_is_validated_against_the_issuing_portal(portal, url, expected):
    assert auth_nous._validate_nous_inference_url_from_network(url, portal) == expected


def test_pairing_never_admits_a_non_https_or_unknown_host():
    """Pairing widens the host set only — scheme and unknown-host refusals still apply."""
    assert auth_nous._validate_nous_inference_url_from_network(
        "http://stg-inference-api.nousresearch.com/v1", STAGING_PORTAL) is None
    assert auth_nous._validate_nous_inference_url_from_network(
        "https://attacker.example/v1", STAGING_PORTAL) is None


def test_refresh_payload_heals_to_the_portal_that_issued_it():
    """``_healed_nous_inference_url`` carries the Portal through, so a staging refresh keeps
    its staging endpoint instead of being healed onto the production host."""
    refreshed = {"inference_base_url": STAGING_INFERENCE}
    assert auth_nous._healed_nous_inference_url(refreshed, STAGING_PORTAL) == STAGING_INFERENCE
    # Same payload, production Portal: refused and healed to the production default.
    assert auth_nous._healed_nous_inference_url(refreshed, PROD_PORTAL) == (
        auth_nous.DEFAULT_NOUS_INFERENCE_URL)
