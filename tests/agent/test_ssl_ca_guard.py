"""Tests for the preventive SSL CA bundle guard."""

import os
import ssl
from pathlib import Path
from unittest.mock import patch

import certifi
import pytest

from agent.errors import SSLConfigurationError
from agent.ssl_guard import (
    verify_ca_bundle,
    verify_ca_bundle_with_fallback,
)


def test_healthy_bundle_passes(tmp_path, monkeypatch):
    """A real, non-empty certifi bundle must verify without raising."""
    # Sanity: certifi.where() must point to a real file in the test venv.
    bundle = Path(certifi.where())
    assert bundle.exists()
    assert bundle.stat().st_size > 1024
    verify_ca_bundle()  # should not raise


def test_missing_bundle_raises_ssl_error(monkeypatch, tmp_path):
    """Point certifi.where() at a non-existent path; expect a clear error."""
    fake = tmp_path / "nope.pem"
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    assert "not found" in str(exc.value).lower()


def test_empty_bundle_raises_ssl_error(monkeypatch, tmp_path):
    """Empty file is treated as a corrupted bundle."""
    fake = tmp_path / "empty.pem"
    fake.write_bytes(b"")
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    with pytest.raises(SSLConfigurationError) as exc:
        verify_ca_bundle()
    assert "corrupted" in str(exc.value).lower() or "empty" in str(exc.value).lower()


def test_skip_env_var_disables_guard(monkeypatch, tmp_path):
    """HERMES_SKIP_SSL_GUARD=1 must make the guard a no-op."""
    monkeypatch.setenv("HERMES_SKIP_SSL_GUARD", "1")
    fake = tmp_path / "nope.pem"  # would raise if guard ran
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    verify_ca_bundle()  # should not raise


def test_macos_fallback_allows_startup(monkeypatch, tmp_path):
    """On Darwin, an unloadable certifi bundle must fall back to system trust."""
    fake = tmp_path / "broken.pem"
    fake.write_bytes(b"not a real bundle")
    monkeypatch.setattr(certifi, "where", lambda: str(fake))
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    fake_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with patch("ssl.create_default_context", return_value=fake_ctx):
        # Should NOT raise — macOS fallback lets startup proceed.
        verify_ca_bundle_with_fallback()
