"""Tests for _resolve_requests_verify() env var precedence.

Verifies that custom provider `/models` fetches honour the three supported
CA bundle env vars (HERMES_CA_BUNDLE, REQUESTS_CA_BUNDLE, SSL_CERT_FILE)
in the documented priority order, and that non-existent paths are
skipped gracefully rather than breaking the request.

Also verifies that per-provider ``ssl_ca_cert``/``ssl_verify`` settings are
threaded through ``fetch_endpoint_model_metadata`` and ``get_pricing_entry``
so a custom endpoint's provider-scoped CA applies to its metadata/pricing
probes — not just to its inference clients.

No filesystem or network I/O required — we use tmp_path to create real
CA bundle stand-in files and monkeypatch env vars.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from agent.model_metadata import _resolve_requests_verify


_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE")


@pytest.fixture
def clean_env(monkeypatch):
    """Clear all three SSL env vars so each test starts from a known state."""
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


@pytest.fixture
def bundle_file(tmp_path: Path) -> str:
    """Create a placeholder CA bundle file and return its absolute path."""
    path = tmp_path / "ca.pem"
    path.write_text("-----BEGIN CERTIFICATE-----\nstub\n-----END CERTIFICATE-----\n")
    return str(path)


class TestResolveRequestsVerify:
    def test_no_env_returns_true(self, clean_env):
        assert _resolve_requests_verify() is True

    def test_hermes_ca_bundle_returns_path(self, clean_env, bundle_file):
        clean_env.setenv("HERMES_CA_BUNDLE", bundle_file)
        assert _resolve_requests_verify() == bundle_file

    def test_requests_ca_bundle_returns_path(self, clean_env, bundle_file):
        clean_env.setenv("REQUESTS_CA_BUNDLE", bundle_file)
        assert _resolve_requests_verify() == bundle_file

    def test_ssl_cert_file_returns_path(self, clean_env, bundle_file):
        clean_env.setenv("SSL_CERT_FILE", bundle_file)
        assert _resolve_requests_verify() == bundle_file

    def test_priority_hermes_over_requests(self, clean_env, tmp_path, bundle_file):
        other = tmp_path / "other.pem"
        other.write_text("stub")
        clean_env.setenv("HERMES_CA_BUNDLE", bundle_file)
        clean_env.setenv("REQUESTS_CA_BUNDLE", str(other))
        assert _resolve_requests_verify() == bundle_file

    def test_priority_requests_over_ssl_cert_file(self, clean_env, tmp_path, bundle_file):
        other = tmp_path / "other.pem"
        other.write_text("stub")
        clean_env.setenv("REQUESTS_CA_BUNDLE", bundle_file)
        clean_env.setenv("SSL_CERT_FILE", str(other))
        assert _resolve_requests_verify() == bundle_file

    def test_nonexistent_path_falls_through(self, clean_env, tmp_path, bundle_file):
        missing = tmp_path / "does_not_exist.pem"
        clean_env.setenv("HERMES_CA_BUNDLE", str(missing))
        clean_env.setenv("REQUESTS_CA_BUNDLE", bundle_file)
        assert _resolve_requests_verify() == bundle_file

    def test_all_nonexistent_returns_true(self, clean_env, tmp_path):
        missing1 = tmp_path / "a.pem"
        missing2 = tmp_path / "b.pem"
        missing3 = tmp_path / "c.pem"
        clean_env.setenv("HERMES_CA_BUNDLE", str(missing1))
        clean_env.setenv("REQUESTS_CA_BUNDLE", str(missing2))
        clean_env.setenv("SSL_CERT_FILE", str(missing3))
        assert _resolve_requests_verify() is True

    def test_empty_string_env_var_ignored(self, clean_env, bundle_file):
        clean_env.setenv("HERMES_CA_BUNDLE", "")
        clean_env.setenv("REQUESTS_CA_BUNDLE", bundle_file)
        assert _resolve_requests_verify() == bundle_file


# ──────────────────────────────────────────────────────────────────────────
# Provider-scoped TLS threading through fetch_endpoint_model_metadata and
# get_pricing_entry. Issue #66544: per-provider ssl_ca_cert must reach the
# /models and pricing probes, not just the inference clients.
# ──────────────────────────────────────────────────────────────────────────


def _make_models_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    resp.ok = True
    return resp


class TestFetchEndpointModelMetadataProviderTLS:
    """fetch_endpoint_model_metadata must pass provider ssl_ca_cert as verify=."""

    def test_provider_ca_bundle_passed_to_requests_verify(self, clean_env, tmp_path):
        """Explicit provider ssl_ca_cert is used as verify= for the /models probe."""
        from agent.model_metadata import fetch_endpoint_model_metadata

        provider_ca = tmp_path / "provider-ca.pem"
        provider_ca.write_text("provider-ca")

        resp = _make_models_response({"data": [{"id": "test-model", "name": "Test"}]})

        with patch("agent.model_metadata.requests.get", return_value=resp) as mock_get, \
             patch("agent.model_metadata.is_local_endpoint", return_value=False):
            fetch_endpoint_model_metadata(
                "https://custom.example.com/v1",
                api_key="key",
                ssl_ca_cert=str(provider_ca),
                force_refresh=True,
            )

        assert mock_get.called
        assert mock_get.call_args.kwargs["verify"] == str(provider_ca)

    def test_no_provider_ssl_falls_back_to_default(self, clean_env):
        """Without provider ssl_ca_cert, verify= is True (current default)."""
        from agent.model_metadata import fetch_endpoint_model_metadata

        resp = _make_models_response({"data": [{"id": "test-model"}]})

        with patch("agent.model_metadata.requests.get", return_value=resp) as mock_get, \
             patch("agent.model_metadata.is_local_endpoint", return_value=False):
            fetch_endpoint_model_metadata(
                "https://custom.example.com/v1",
                api_key="key",
                force_refresh=True,
            )

        assert mock_get.call_args.kwargs["verify"] is True

    def test_provider_ssl_verify_false_disables_verification(self, clean_env):
        """Provider ssl_verify=False propagates to verify=False on the probe."""
        from agent.model_metadata import fetch_endpoint_model_metadata

        resp = _make_models_response({"data": [{"id": "test-model"}]})

        with patch("agent.model_metadata.requests.get", return_value=resp) as mock_get, \
             patch("agent.model_metadata.is_local_endpoint", return_value=False):
            fetch_endpoint_model_metadata(
                "https://custom.example.com/v1",
                api_key="key",
                ssl_verify=False,
                force_refresh=True,
            )

        assert mock_get.call_args.kwargs["verify"] is False


class TestGetPricingEntryProviderTLS:
    """get_pricing_entry must forward provider ssl_ca_cert to fetch_endpoint_model_metadata."""

    def test_provider_ca_forwarded_to_endpoint_probe(self, clean_env, tmp_path):
        from agent.usage_pricing import get_pricing_entry

        provider_ca = tmp_path / "provider-ca.pem"
        provider_ca.write_text("provider-ca")

        captured = {}

        def fake_fetch(base_url, api_key="", **kwargs):
            captured["ssl_ca_cert"] = kwargs.get("ssl_ca_cert")
            captured["ssl_verify"] = kwargs.get("ssl_verify")
            return {}

        with patch("agent.usage_pricing.fetch_endpoint_model_metadata", side_effect=fake_fetch):
            get_pricing_entry(
                "some-model",
                provider="custom",
                base_url="https://custom.example.com/v1",
                api_key="key",
                ssl_ca_cert=str(provider_ca),
                ssl_verify=True,
            )

        assert captured["ssl_ca_cert"] == str(provider_ca)
        assert captured["ssl_verify"] is True

    def test_no_provider_ssl_passes_none(self, clean_env):
        """get_pricing_entry without TLS args passes None through (backward compat)."""
        from agent.usage_pricing import get_pricing_entry

        captured = {}

        def fake_fetch(base_url, api_key="", **kwargs):
            captured["ssl_ca_cert"] = kwargs.get("ssl_ca_cert")
            captured["ssl_verify"] = kwargs.get("ssl_verify")
            return {}

        with patch("agent.usage_pricing.fetch_endpoint_model_metadata", side_effect=fake_fetch):
            get_pricing_entry(
                "some-model",
                provider="custom",
                base_url="https://custom.example.com/v1",
                api_key="key",
            )

        assert captured["ssl_ca_cert"] is None
        assert captured["ssl_verify"] is None
