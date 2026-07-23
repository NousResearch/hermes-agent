"""Regression tests for refresh_nous_oauth_pure's portal_base_url allowlist.

resolve_nous_access_token() and resolve_nous_runtime_credentials() already
validate a stored/network-sourced portal_base_url against
_NOUS_PORTAL_ALLOWED_HOSTS before using it, healing to DEFAULT_NOUS_PORTAL_URL
on rejection. refresh_nous_oauth_pure() -- the third distinct call site that
POSTs the user's refresh_token to portal_base_url -- did not: it took
whatever value the caller passed (falling back to the default only if
empty/falsy) with no host check at all.

The caller-supplied value can originate from the cross-profile shared store
(_try_import_shared_nous_state -> refresh_nous_oauth_from_state ->
refresh_nous_oauth_pure), which this function has no way to know is
untampered -- so it must re-validate here, not trust it as-is. A poisoned
portal_base_url in that file (a stale pre-allowlist write, a corrupted file,
or local tampering) would otherwise have the refresh_token replayed to it on
every subsequent rehydrate, forever.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

from hermes_cli.auth import (
    DEFAULT_NOUS_PORTAL_URL,
    _NOUS_PORTAL_ALLOWED_HOSTS,
    refresh_nous_oauth_pure,
)


def _mock_refresh(monkeypatch, captured: dict):
    def fake_refresh_access_token(*, client, portal_base_url, client_id, refresh_token):
        captured["portal_base_url"] = portal_base_url
        return {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
        }

    import hermes_cli.auth as auth_mod

    monkeypatch.setattr(auth_mod, "_refresh_access_token", fake_refresh_access_token)
    monkeypatch.setattr(auth_mod, "_nous_invoke_jwt_status", lambda *a, **k: "expired")
    monkeypatch.setattr(auth_mod, "_assert_nous_inference_jwt_usable", lambda state: None)
    monkeypatch.setattr(auth_mod, "_select_nous_invoke_jwt", lambda state: None)


class TestPortalBaseUrlAllowlist:
    def test_attacker_host_healed_to_default(self, monkeypatch, caplog):
        """A poisoned portal_base_url must never receive the refresh_token —
        the exact bug: previously it was POSTed there unconditionally."""
        captured: dict = {}
        _mock_refresh(monkeypatch, captured)

        with caplog.at_level(logging.WARNING, logger="hermes_cli.auth"):
            result = refresh_nous_oauth_pure(
                access_token="old-token",
                refresh_token="refresh-abc",
                client_id="hermes-cli",
                portal_base_url="https://attacker.evil.com",
                inference_base_url="https://inference-api.nousresearch.com",
            )

        assert captured["portal_base_url"] == DEFAULT_NOUS_PORTAL_URL
        assert result["portal_base_url"] == DEFAULT_NOUS_PORTAL_URL
        assert any("attacker.evil.com" in rec.message for rec in caplog.records)

    def test_legitimate_host_passes_through_unchanged(self, monkeypatch):
        captured: dict = {}
        _mock_refresh(monkeypatch, captured)

        result = refresh_nous_oauth_pure(
            access_token="old-token",
            refresh_token="refresh-abc",
            client_id="hermes-cli",
            portal_base_url="https://portal.nousresearch.com",
            inference_base_url="https://inference-api.nousresearch.com",
        )

        assert captured["portal_base_url"] == "https://portal.nousresearch.com"
        assert result["portal_base_url"] == "https://portal.nousresearch.com"

    def test_env_override_bypasses_allowlist_and_wins_outright(self, monkeypatch):
        """HERMES_PORTAL_BASE_URL / NOUS_PORTAL_BASE_URL is the documented
        dev/staging escape hatch -- must win even over an attacker-controlled
        caller-supplied value, matching resolve_nous_access_token."""
        captured: dict = {}
        _mock_refresh(monkeypatch, captured)
        monkeypatch.setenv(
            "HERMES_PORTAL_BASE_URL", "https://portal.staging-nousresearch.com"
        )

        result = refresh_nous_oauth_pure(
            access_token="old-token",
            refresh_token="refresh-abc",
            client_id="hermes-cli",
            portal_base_url="https://attacker.evil.com",
            inference_base_url="https://inference-api.nousresearch.com",
        )

        assert captured["portal_base_url"] == "https://portal.staging-nousresearch.com"
        assert result["portal_base_url"] == "https://portal.staging-nousresearch.com"

    def test_no_refresh_needed_skips_validation_path_harmlessly(self, monkeypatch):
        """When the access token isn't expiring, no refresh POST happens at
        all -- portal_base_url still ends up validated in the returned
        state (it's computed unconditionally), so a poisoned value can't
        silently survive untouched even on this path."""
        import hermes_cli.auth as auth_mod

        monkeypatch.setattr(auth_mod, "_nous_invoke_jwt_status", lambda *a, **k: None)
        monkeypatch.setattr(auth_mod, "_assert_nous_inference_jwt_usable", lambda state: None)
        monkeypatch.setattr(auth_mod, "_select_nous_invoke_jwt", lambda state: None)

        result = refresh_nous_oauth_pure(
            access_token="still-valid-token",
            refresh_token="refresh-abc",
            client_id="hermes-cli",
            portal_base_url="https://attacker.evil.com",
            inference_base_url="https://inference-api.nousresearch.com",
        )

        assert result["portal_base_url"] == DEFAULT_NOUS_PORTAL_URL

    def test_default_portal_url_is_in_allowlist(self):
        """Sanity check mirroring the inference-url suite: the default must
        itself validate, or every install would get healed away from it."""
        from urllib.parse import urlparse

        assert urlparse(DEFAULT_NOUS_PORTAL_URL).hostname in _NOUS_PORTAL_ALLOWED_HOSTS
