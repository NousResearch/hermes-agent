"""Regression tests for the Nous Portal env-override bypassing the host
allowlist, mirroring the existing NOUS_INFERENCE_BASE_URL /
_ALLOWED_NOUS_INFERENCE_HOSTS treatment.

Real incident (2026-07): a hosted agent provisioned by nous-account-service
on the `staging` Vercel environment is stamped with
``HERMES_PORTAL_BASE_URL=https://portal.staging-nousresearch.com`` in its
container env (the documented dev/staging override), while its bootstrap
``auth.json`` ALSO persists ``portal_base_url`` to the same staging host.

Before this fix, ``resolve_nous_access_token`` / ``resolve_nous_runtime_
credentials`` read ``state.get("portal_base_url")`` FIRST via a plain ``or``
chain, so whenever the stored state had ANY value the env vars were never
even consulted — and whichever value won (state or env) was then run through
``_NOUS_PORTAL_ALLOWED_HOSTS``, which only recognised the production host.
The staging host was silently rewritten back to prod on every refresh, so a
staging-issued refresh token got replayed against the PROD token endpoint.
Prod correctly rejected that with ``invalid_grant``, which triggered
``_quarantine_nous_oauth_state`` and wiped the entire credential pool.

The correct fix (mirroring ``_nous_inference_env_override()``): the env
override is a TRUSTED value the operator/deployment set themselves — it must
win outright (even over a stored value) and bypass the allowlist entirely.
The allowlist exists only to reject an untrusted NETWORK-provided value
(a poisoned portal_base_url written to auth.json by a compromised Portal
response), never a value the operator explicitly configured.
"""

from __future__ import annotations

import json
import logging

from hermes_cli.auth import (
    DEFAULT_NOUS_PORTAL_URL,
    _NOUS_PORTAL_ALLOWED_HOSTS,
    _nous_portal_env_override,
)


class TestPortalEnvOverrideHelper:
    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
        monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
        assert _nous_portal_env_override() is None

    def test_hermes_portal_base_url_wins(self, monkeypatch):
        monkeypatch.setenv(
            "HERMES_PORTAL_BASE_URL", "https://portal.staging-nousresearch.com/"
        )
        monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
        assert (
            _nous_portal_env_override() == "https://portal.staging-nousresearch.com"
        )

    def test_nous_portal_base_url_used_as_fallback(self, monkeypatch):
        monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
        monkeypatch.setenv(
            "NOUS_PORTAL_BASE_URL", "https://portal.staging-nousresearch.com"
        )
        assert (
            _nous_portal_env_override() == "https://portal.staging-nousresearch.com"
        )

    def test_env_override_not_gated_by_allowlist(self, monkeypatch):
        """The whole point: an env-set staging host is NOT in
        _NOUS_PORTAL_ALLOWED_HOSTS, and the helper must return it anyway —
        gating happens only for network-provenance values."""
        monkeypatch.setenv(
            "HERMES_PORTAL_BASE_URL", "https://portal.staging-nousresearch.com"
        )
        assert "portal.staging-nousresearch.com" not in _NOUS_PORTAL_ALLOWED_HOSTS
        assert (
            _nous_portal_env_override() == "https://portal.staging-nousresearch.com"
        )


class TestResolveAccessTokenEnvOverrideWins:
    """End-to-end: resolve_nous_access_token must use the env override for
    the refresh call, bypassing the allowlist, even when state also has a
    portal_base_url set (the exact incident shape)."""

    def _write_auth_file(self, tmp_path, *, stored_portal_url):
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "active_provider": "nous",
                    "providers": {
                        "nous": {
                            "portal_base_url": stored_portal_url,
                            "access_token": "expired-access",
                            "refresh_token": "staging-refresh",
                            "client_id": "hermes-cli-vps",
                            "expires_at": "2000-01-01T00:00:00+00:00",
                        }
                    },
                }
            )
        )
        return auth_file

    def _run_and_capture(self, monkeypatch, auth):
        seen_portal_urls = []

        def _fake_refresh(*, client, portal_base_url, client_id, refresh_token):
            seen_portal_urls.append(portal_base_url)
            return {
                "access_token": "new-access",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

        monkeypatch.setattr(auth, "_refresh_access_token", _fake_refresh)

        caplog_records = []
        logger = logging.getLogger("hermes_cli.auth")
        handler = logging.Handler()
        handler.emit = lambda record: caplog_records.append(record.getMessage())
        logger.addHandler(handler)
        try:
            auth.resolve_nous_access_token()
        finally:
            logger.removeHandler(handler)
        return seen_portal_urls, caplog_records

    def test_env_override_wins_even_with_staging_state_stored(
        self, monkeypatch, tmp_path
    ):
        """The real incident: state ALSO has the staging host stored (from
        a prior HERMES_AUTH_JSON_BOOTSTRAP seed), and the env var is set to
        the same staging host. Both must resolve to staging, and the
        allowlist-rejection warning must never fire."""
        import hermes_cli.auth as auth

        staging_portal = "https://portal.staging-nousresearch.com"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PORTAL_BASE_URL", staging_portal)
        self._write_auth_file(tmp_path, stored_portal_url=staging_portal)

        seen_portal_urls, records = self._run_and_capture(monkeypatch, auth)

        assert seen_portal_urls == [staging_portal]
        assert not any(
            "ignoring invalid portal_base_url" in msg for msg in records
        ), "env override must bypass the allowlist gate entirely"

    def test_env_override_wins_over_prod_state(self, monkeypatch, tmp_path):
        """Even when the STORED state is the prod host (e.g. a stale/healed
        value from before the env var was set), the env override must still
        win for the actual refresh call."""
        import hermes_cli.auth as auth

        staging_portal = "https://portal.staging-nousresearch.com"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_PORTAL_BASE_URL", staging_portal)
        self._write_auth_file(tmp_path, stored_portal_url=DEFAULT_NOUS_PORTAL_URL)

        seen_portal_urls, _records = self._run_and_capture(monkeypatch, auth)

        assert seen_portal_urls == [staging_portal]

    def test_no_env_override_stored_staging_host_heals_to_prod(
        self, monkeypatch, tmp_path
    ):
        """Without the env override set, a stored staging host is untrusted
        network provenance and correctly heals to prod (this is the
        allowlist's actual job — preserved, not regressed, by this fix)."""
        import hermes_cli.auth as auth

        staging_portal = "https://portal.staging-nousresearch.com"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
        monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
        self._write_auth_file(tmp_path, stored_portal_url=staging_portal)

        seen_portal_urls, records = self._run_and_capture(monkeypatch, auth)

        assert seen_portal_urls == [DEFAULT_NOUS_PORTAL_URL]
        assert any("ignoring invalid portal_base_url" in msg for msg in records)

    def test_no_env_no_staging_state_prod_url_used_unmodified(
        self, monkeypatch, tmp_path
    ):
        """Baseline: no override, no staging state — prod is used and the
        allowlist never even logs a warning (nothing was rejected)."""
        import hermes_cli.auth as auth

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
        monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
        self._write_auth_file(tmp_path, stored_portal_url=DEFAULT_NOUS_PORTAL_URL)

        seen_portal_urls, records = self._run_and_capture(monkeypatch, auth)

        assert seen_portal_urls == [DEFAULT_NOUS_PORTAL_URL]
        assert not any("ignoring invalid portal_base_url" in msg for msg in records)
