"""Regression tests for gateway SSL certificate environment repair."""

import os
import ssl
import sys
from types import SimpleNamespace

import pytest

from hermes_cli.ssl_certs import ensure_ssl_certs


def test_ensure_ssl_certs_ignores_stale_ssl_cert_file(monkeypatch, tmp_path):
    """A missing SSL_CERT_FILE should be treated as unset, not trusted."""
    cert_file = tmp_path / "cacert.pem"
    cert_file.write_text("dummy cert bundle", encoding="utf-8")
    stale_file = tmp_path / "missing.pem"

    monkeypatch.setenv("SSL_CERT_FILE", str(stale_file))
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(cafile=None, openssl_cafile=None),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: str(cert_file)),
    )

    ensure_ssl_certs()

    assert stale_file.exists() is False
    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_ensure_ssl_certs_preserves_configured_bundle(monkeypatch, tmp_path):
    configured_bundle = tmp_path / "corporate-ca.pem"
    configured_bundle.write_text("configured bundle", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(configured_bundle))

    ensure_ssl_certs()

    assert os.environ["SSL_CERT_FILE"] == str(configured_bundle)


@pytest.mark.macos_only
def test_ensure_ssl_certs_prefers_certifi_to_compiled_macos_bundle(
    monkeypatch,
    tmp_path,
):
    import certifi

    system_bundle = tmp_path / "system-cert.pem"
    system_bundle.write_text("incomplete system bundle", encoding="utf-8")
    certifi_bundle = tmp_path / "certifi.pem"
    certifi_bundle.write_text("certifi bundle", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(
            cafile=str(system_bundle),
            openssl_cafile=str(system_bundle),
        ),
    )
    monkeypatch.setattr(certifi, "where", lambda: str(certifi_bundle))

    ensure_ssl_certs()

    assert os.environ["SSL_CERT_FILE"] == str(certifi_bundle)
