"""Tests for hermes_cli.auth._default_verify platform-aware fallback.

On macOS with Homebrew Python, the system OpenSSL cannot locate the
system trust store, so we explicitly load certifi's bundle. On other
platforms we defer to httpx's own default (which itself uses certifi).

Most tests use monkeypatching — no real SSL handshakes. A handful use
an openssl-generated self-signed cert via the `real_bundle_file`
fixture because `ssl.create_default_context(cafile=...)` parses the
bundle and refuses stubs.
"""

import os
import shutil
import ssl
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hermes_cli.auth import _default_verify, _resolve_verify


@pytest.fixture
def real_bundle_file(tmp_path: Path) -> str:
    """Return a path to a real openssl-generated self-signed cert.

    Skips the test when the `openssl` binary isn't on PATH, so CI images
    without it degrade gracefully instead of erroring out.
    """
    if shutil.which("openssl") is None:
        pytest.skip("openssl binary not available")
    cert = tmp_path / "ca.pem"
    key = tmp_path / "key.pem"
    result = subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key), "-out", str(cert),
            "-sha256", "-days", "1", "-nodes",
            "-subj", "/CN=test",
        ],
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        pytest.skip(f"openssl failed: {result.stderr.decode('utf-8', 'ignore')[:200]}")
    return str(cert)


class TestDefaultVerify:
    def test_returns_ssl_context_on_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        result = _default_verify()
        assert isinstance(result, ssl.SSLContext)

    def test_returns_true_on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert _default_verify() is True

    def test_returns_true_on_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert _default_verify() is True

    def test_darwin_falls_back_to_true_when_certifi_missing(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "certifi":
                raise ImportError("simulated missing certifi")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        assert _default_verify() is True


class TestResolveVerifyIntegration:
    """_resolve_verify should defer to _default_verify in the no-CA path."""

    def test_no_ca_uses_default_verify_on_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        for var in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            monkeypatch.delenv(var, raising=False)
        result = _resolve_verify()
        assert isinstance(result, ssl.SSLContext)

    def test_no_ca_uses_default_verify_on_linux(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        for var in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            monkeypatch.delenv(var, raising=False)
        assert _resolve_verify() is True

    def test_requests_ca_bundle_respected(self, monkeypatch, real_bundle_file):
        for var in ("HERMES_CA_BUNDLE", "SSL_CERT_FILE"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", real_bundle_file)
        result = _resolve_verify()
        assert isinstance(result, ssl.SSLContext)

    def test_missing_ca_path_falls_back_to_default_verify(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(tmp_path / "missing.pem"))
        for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            monkeypatch.delenv(var, raising=False)
        assert _resolve_verify() is True

    def test_insecure_wins_over_everything(self, monkeypatch, tmp_path):
        bundle = tmp_path / "ca.pem"
        bundle.write_text("stub")
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(bundle))
        assert _resolve_verify(insecure=True) is False
