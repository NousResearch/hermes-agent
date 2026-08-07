"""Regression tests for gateway SSL certificate environment repair."""

import os
import ssl
import sys
from types import SimpleNamespace

from gateway.run import _ensure_ssl_certs


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

    _ensure_ssl_certs()

    assert stale_file.exists() is False
    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_ensure_ssl_certs_ignores_directory_ssl_cert_file(monkeypatch, tmp_path):
    """A directory is not a valid SSL certificate bundle."""
    cert_file = tmp_path / "cacert.pem"
    cert_file.write_text("dummy cert bundle", encoding="utf-8")
    cert_directory = tmp_path / "cert-directory"
    cert_directory.mkdir()

    monkeypatch.setenv("SSL_CERT_FILE", str(cert_directory))
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

    _ensure_ssl_certs()

    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_ensure_ssl_certs_keeps_existing_ssl_cert_file(monkeypatch, tmp_path):
    """A valid user-provided SSL_CERT_FILE must not be overwritten."""
    cert_file = tmp_path / "existing.pem"
    cert_file.write_text("dummy cert bundle", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(cert_file))

    _ensure_ssl_certs()

    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_ensure_ssl_certs_prefers_homebrew_bundle_for_problematic_darwin_default(monkeypatch):
    """A problematic macOS venv default uses the Homebrew trust bundle."""
    homebrew_bundle = "/opt/homebrew/etc/openssl@3/cert.pem"
    python_default = "/private/etc/ssl/cert.pem"
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(cafile=python_default, openssl_cafile=None),
    )
    monkeypatch.setattr(
        "gateway.run.os.path.exists",
        lambda path: path in {homebrew_bundle, python_default},
    )

    _ensure_ssl_certs()

    assert os.environ["SSL_CERT_FILE"] == homebrew_bundle


def test_ensure_ssl_certs_keeps_non_problematic_darwin_default(monkeypatch):
    """A working non-problematic Python default remains authoritative."""
    homebrew_bundle = "/opt/homebrew/etc/openssl@3/cert.pem"
    python_default = "/custom/python/cert.pem"
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: SimpleNamespace(cafile=python_default, openssl_cafile=None),
    )
    monkeypatch.setattr(
        "gateway.run.os.path.exists",
        lambda path: path in {homebrew_bundle, python_default},
    )

    _ensure_ssl_certs()

    assert os.environ["SSL_CERT_FILE"] == python_default
