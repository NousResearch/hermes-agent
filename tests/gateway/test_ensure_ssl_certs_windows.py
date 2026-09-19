"""Tests for gateway/run.py _ensure_ssl_certs with agent.win_ca_bundle (#43294)."""

import os
import ssl
from types import SimpleNamespace

import pytest

# Import the real module (established practice in this repo).
from gateway import run as gateway_run


class TestEnsureSslCertsWindowsBundle:
    def test_merged_path_sets_env(self, monkeypatch, tmp_path):
        bundle_path = tmp_path / "merged.pem"
        bundle_path.write_text("stub pem", encoding="utf-8")
        monkeypatch.setattr(
            "agent.win_ca_bundle.windows_merged_ca_bundle",
            lambda: str(bundle_path),
        )
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.setattr(
            ssl,
            "get_default_verify_paths",
            lambda: SimpleNamespace(cafile=None, openssl_cafile=None, capath=None, openssl_capath=None),
        )
        gateway_run._ensure_ssl_certs()
        assert os.environ["SSL_CERT_FILE"] == str(bundle_path)

    def test_merged_none_falls_through_to_certifi(self, monkeypatch, tmp_path):
        cert_file = tmp_path / "certifi_bundle.pem"
        cert_file.write_text("certifi stub", encoding="utf-8")
        monkeypatch.setattr("agent.win_ca_bundle.windows_merged_ca_bundle", lambda: None)
        monkeypatch.delenv("SSL_CERT_FILE", raising=False)
        monkeypatch.setattr(
            ssl,
            "get_default_verify_paths",
            lambda: SimpleNamespace(cafile=None, openssl_cafile=None, capath=None, openssl_capath=None),
        )
        monkeypatch.setitem(
            __import__("sys").modules,
            "certifi",
            SimpleNamespace(where=lambda: str(cert_file)),
        )
        gateway_run._ensure_ssl_certs()
        assert os.environ["SSL_CERT_FILE"] == str(cert_file)

    def test_existing_env_respected_merged_hook_never_called(self, monkeypatch, tmp_path):
        existing = tmp_path / "existing.pem"
        existing.write_text("existing", encoding="utf-8")
        monkeypatch.setenv("SSL_CERT_FILE", str(existing))
        # If the merged hook is consulted, this raises — proving early return.
        monkeypatch.setattr(
            "agent.win_ca_bundle.windows_merged_ca_bundle",
            lambda: (_ for _ in ()).throw(AssertionError("merged hook should not be called")),
        )
        gateway_run._ensure_ssl_certs()
        assert os.environ["SSL_CERT_FILE"] == str(existing)
