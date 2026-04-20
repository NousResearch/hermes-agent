"""Tests for _resolve_verify returning ssl.SSLContext instead of str for CA bundles.

This test verifies the fix for bug #12706: httpx deprecates verify=<str> and
expects ssl.SSLContext when a custom CA bundle is configured.

The test should:
- FAIL before the fix (returns str path)
- PASS after the fix (returns ssl.SSLContext)
"""

import os
import ssl

import pytest

from hermes_cli.auth import _resolve_verify


# Use the system's default CA bundle for testing
DEFAULT_CA_FILE = ssl.get_default_verify_paths().cafile


class TestResolveVerifySslContext:
    """Test that _resolve_verify returns ssl.SSLContext for CA bundles."""

    def test_resolve_verify_returns_ssl_context_for_ca_bundle(self, monkeypatch):
        """When a CA bundle path is provided, _resolve_verify returns an ssl.SSLContext."""
        # Clear any env vars that might interfere
        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        # Use the system's actual CA bundle which is a valid PEM file
        result = _resolve_verify(ca_bundle=DEFAULT_CA_FILE)

        # The result should be an ssl.SSLContext, NOT a string
        assert isinstance(result, ssl.SSLContext), (
            f"Expected ssl.SSLContext but got {type(result).__name__}: {result!r}. "
            "httpx deprecates verify=<str> and requires ssl.SSLContext."
        )

    def test_resolve_verify_returns_true_when_no_ca_bundle(self, monkeypatch):
        """When no CA bundle is configured, _resolve_verify returns True (not a path)."""
        # Clear any env vars that might interfere
        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        result = _resolve_verify()
        assert result is True, f"Expected True but got {result!r}"

    def test_resolve_verify_returns_true_for_missing_ca_bundle_path(self, monkeypatch):
        """When a CA bundle path is configured but doesn't exist, returns True."""
        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        result = _resolve_verify(ca_bundle="/nonexistent/path/to/ca-bundle.crt")
        assert result is True, f"Expected True for missing CA bundle but got {result!r}"

    def test_resolve_verify_returns_false_when_insecure_is_true(self, monkeypatch):
        """When insecure=True, _resolve_verify returns False (skip SSL verification)."""
        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)

        result = _resolve_verify(insecure=True)
        assert result is False, f"Expected False for insecure=True but got {result!r}"

    def test_resolve_verify_returns_ssl_context_from_hermes_ca_bundle_env(self, monkeypatch):
        """SSLContext is returned when HERMES_CA_BUNDLE env var is set."""
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.setenv("HERMES_CA_BUNDLE", DEFAULT_CA_FILE)

        result = _resolve_verify()
        assert isinstance(result, ssl.SSLContext), (
            f"Expected ssl.SSLContext from HERMES_CA_BUNDLE env var, got {type(result).__name__}"
        )

    def test_resolve_verify_returns_ssl_context_from_ssl_cert_file_env(self, monkeypatch):
        """SSLContext is returned when SSL_CERT_FILE env var is set."""
        monkeypatch.delenv("HERMES_CA_BUNDLE", raising=False)
        monkeypatch.setenv("SSL_CERT_FILE", DEFAULT_CA_FILE)

        result = _resolve_verify()
        assert isinstance(result, ssl.SSLContext), (
            f"Expected ssl.SSLContext from SSL_CERT_FILE env var, got {type(result).__name__}"
        )
