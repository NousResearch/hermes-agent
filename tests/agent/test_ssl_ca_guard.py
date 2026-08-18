"""Tests for the preventive SSL CA bundle guard."""

from pathlib import Path

import certifi
import pytest

from agent.errors import SSLConfigurationError
from agent.ssl_guard import verify_ca_bundle, verify_ca_bundle_with_fallback


def test_healthy_bundle_passes(monkeypatch):
    """A real, non-empty certifi bundle must verify without raising."""
    for key in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(key, raising=False)
    bundle = Path(certifi.where())
    assert bundle.exists()
    assert bundle.stat().st_size > 1024
    verify_ca_bundle()


def test_empty_certifi_bundle_raises_ssl_error(monkeypatch, tmp_path):
    """Empty file is treated as a corrupted bundle."""
    fake = tmp_path / "empty.pem"
    fake.write_bytes(b"")
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    assert "too small" in str(exc.value).lower()


@pytest.mark.parametrize("env_var", ["HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_missing_explicit_ca_bundle_env_raises_before_httpx(monkeypatch, tmp_path, env_var):
    """Bad CA-bundle env vars should be reported before OpenAI/httpx init."""
    fake = tmp_path / "missing.pem"
    monkeypatch.setenv(env_var, str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    message = str(exc.value)
    assert env_var in message
    assert str(fake) in message
    assert "force-reinstall" in message


def test_truststore_get_ca_certs_not_implemented_is_accepted(monkeypatch, tmp_path):
    """A truststore-backed SSLContext (Windows OS trust store) raises
    NotImplementedError from get_ca_certs(). The guard must accept the
    already-loaded bundle rather than fail.

    Regression for the empty-message ``Failed to initialize OpenAI client:``
    seen on every fresh agent init on Windows (str(NotImplementedError()) == "").
    """
    from agent import ssl_guard

    bundle = tmp_path / "bundle.pem"
    bundle.write_text(
        "-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n",
        encoding="utf-8",
    )

    class _TruststoreLikeContext:
        def get_ca_certs(self, binary_form=False):  # noqa: ARG002 - mirror ssl API
            raise NotImplementedError()

    # create_default_context(cafile=...) loads the bundle fine; only the
    # post-load introspection is unsupported under truststore.
    monkeypatch.setattr(
        ssl_guard.ssl, "create_default_context", lambda *a, **k: _TruststoreLikeContext()
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))

    # Must not raise on the explicit env bundle nor the certifi check.
    verify_ca_bundle()
    verify_ca_bundle_with_fallback()
