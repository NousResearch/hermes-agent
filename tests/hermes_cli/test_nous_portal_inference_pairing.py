"""Nous Portal → inference-host pairing for non-production Portals.

A token minted by the STAGING Portal (``portal.staging-nousresearch.com``) is
meant to be spent at the STAGING inference gateway
(``stg-inference-api.nousresearch.com``), and the staging Portal's refresh
response says exactly that. Before this pairing, that Portal-returned URL was
rejected by ``_ALLOWED_NOUS_INFERENCE_HOSTS`` (prod-only) and healed to the
PRODUCTION default — so a plain ``hermes -p staging`` session (portal env
override set, per the Confluence runbook) shipped its staging-issued JWT to
the prod gateway, which 401s it, unless the operator ALSO set
``NOUS_INFERENCE_BASE_URL`` or ``model.base_url``. ``hermes -p staging status``
then reported ``Inference: https://inference-api.nousresearch.com/v1`` even
with the documented env in place.

The pairing is keyed on the PORTAL host so the original protection (#30611,
#49735) is unchanged for prod sessions: a prod-portal state carrying a staging
inference URL is still a poisoned value and is still healed.
"""

from __future__ import annotations

import hermes_cli.auth as auth
from hermes_cli.auth import (
    DEFAULT_NOUS_INFERENCE_URL,
    DEFAULT_NOUS_PORTAL_URL,
    _ALLOWED_NOUS_INFERENCE_HOSTS,
    _allowed_nous_inference_hosts_for_portal,
    _validate_nous_inference_url_from_network,
)

STAGING_PORTAL = "https://portal.staging-nousresearch.com"
STAGING_INFERENCE = "https://stg-inference-api.nousresearch.com/v1"


class TestAllowedHostsForPortal:
    def test_prod_portal_is_prod_only(self):
        assert (
            _allowed_nous_inference_hosts_for_portal(DEFAULT_NOUS_PORTAL_URL)
            == _ALLOWED_NOUS_INFERENCE_HOSTS
        )

    def test_missing_portal_is_prod_only(self):
        assert _allowed_nous_inference_hosts_for_portal(None) == _ALLOWED_NOUS_INFERENCE_HOSTS
        assert _allowed_nous_inference_hosts_for_portal("") == _ALLOWED_NOUS_INFERENCE_HOSTS

    def test_staging_portal_adds_staging_inference_host_only(self):
        hosts = _allowed_nous_inference_hosts_for_portal(STAGING_PORTAL)
        assert "stg-inference-api.nousresearch.com" in hosts
        assert "inference-api.nousresearch.com" in hosts
        assert hosts - _ALLOWED_NOUS_INFERENCE_HOSTS == {"stg-inference-api.nousresearch.com"}

    def test_unknown_portal_host_gets_no_extra(self):
        assert (
            _allowed_nous_inference_hosts_for_portal("https://portal.evil.example")
            == _ALLOWED_NOUS_INFERENCE_HOSTS
        )


class TestValidatorPairing:
    def test_staging_url_accepted_for_staging_portal(self):
        assert (
            _validate_nous_inference_url_from_network(STAGING_INFERENCE, STAGING_PORTAL)
            == STAGING_INFERENCE
        )

    def test_staging_url_still_rejected_for_prod_portal(self):
        """The 2026-07 poisoned-auth.json case: prod session, staging URL."""
        assert (
            _validate_nous_inference_url_from_network(STAGING_INFERENCE, DEFAULT_NOUS_PORTAL_URL)
            is None
        )

    def test_staging_url_still_rejected_when_portal_unknown(self):
        """Callers that pass no portal keep the strict prod-only behaviour."""
        assert _validate_nous_inference_url_from_network(STAGING_INFERENCE) is None

    def test_prod_url_accepted_for_staging_portal(self):
        assert (
            _validate_nous_inference_url_from_network(DEFAULT_NOUS_INFERENCE_URL, STAGING_PORTAL)
            == DEFAULT_NOUS_INFERENCE_URL
        )

    def test_pairing_does_not_open_other_hosts(self):
        assert (
            _validate_nous_inference_url_from_network(
                "https://stg-inference-api.evil.example/v1", STAGING_PORTAL
            )
            is None
        )


class TestRefreshKeepsStagingPairing:
    """End-to-end through ``refresh_nous_oauth_from_state``: a staging-portal
    state refreshed against the staging Portal keeps the staging inference
    URL it is handed, instead of healing it to prod."""

    def _patch(self, monkeypatch, returned_inference_url):
        monkeypatch.setattr(auth, "_nous_invoke_jwt_status", lambda *a, **k: "needs_refresh")
        monkeypatch.setattr(
            auth,
            "_refresh_access_token",
            lambda **k: {
                "access_token": "newtok",
                "refresh_token": "newrtok",
                "expires_in": 3600,
                "inference_base_url": returned_inference_url,
            },
        )
        monkeypatch.setattr(auth, "_assert_nous_inference_jwt_usable", lambda *a, **k: None)
        monkeypatch.setattr(auth, "_select_nous_invoke_jwt", lambda *a, **k: None)

    def _state(self, portal):
        return {
            "access_token": "tok",
            "refresh_token": "rtok",
            "client_id": "hermes-cli",
            "portal_base_url": portal,
            "inference_base_url": DEFAULT_NOUS_INFERENCE_URL,
        }

    def test_staging_portal_keeps_staging_inference_url(self, monkeypatch):
        self._patch(monkeypatch, STAGING_INFERENCE)
        result = auth.refresh_nous_oauth_from_state(
            self._state(STAGING_PORTAL), force_refresh=True
        )
        assert result["inference_base_url"] == STAGING_INFERENCE

    def test_prod_portal_still_heals_staging_inference_url(self, monkeypatch):
        self._patch(monkeypatch, STAGING_INFERENCE)
        result = auth.refresh_nous_oauth_from_state(
            self._state(DEFAULT_NOUS_PORTAL_URL), force_refresh=True
        )
        assert result["inference_base_url"] == DEFAULT_NOUS_INFERENCE_URL
