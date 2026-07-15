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


def test_missing_certifi_bundle_raises_ssl_error(monkeypatch, tmp_path):
    """Point certifi.where() at a non-existent path; expect a clear error."""
    fake = tmp_path / "nope.pem"
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    message = str(exc.value).lower()
    assert "certifi" in message
    assert "missing" in message
    assert "force-reinstall" in message


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


def test_invalid_explicit_ca_bundle_env_raises(monkeypatch, tmp_path):
    """An existing but invalid explicit bundle should get a user-facing error."""
    fake = tmp_path / "broken.pem"
    fake.write_text("not a cert bundle", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    assert "cannot be loaded" in str(exc.value)


def test_verify_ca_bundle_with_fallback_keeps_same_contract(monkeypatch, tmp_path):
    """The compatibility wrapper still rejects broken explicit CA paths."""
    fake = tmp_path / "missing.pem"
    monkeypatch.setenv("SSL_CERT_FILE", str(fake))
    with pytest.raises(SSLConfigurationError):
        verify_ca_bundle_with_fallback()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_skip_env_var_bypasses_guard(monkeypatch, tmp_path, value):
    """HERMES_SKIP_SSL_GUARD is an intentional escape hatch for managed trust stores."""
    fake = tmp_path / "missing.pem"
    monkeypatch.setenv("HERMES_SKIP_SSL_GUARD", value)
    monkeypatch.setenv("SSL_CERT_FILE", str(fake))
    verify_ca_bundle()
    verify_ca_bundle_with_fallback()
