"""Metadata discovery uses chat's resolved TLS policy without ambient CA files."""
from __future__ import annotations

import ssl

import httpx
import pytest

from agent.model_metadata_http import resolve_verify


@pytest.mark.parametrize("variable", ["HERMES_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_ambient_ca_paths_do_not_break_metadata_clients(monkeypatch, tmp_path, variable):
    monkeypatch.setenv(variable, str(tmp_path / "missing-ca"))
    verify = resolve_verify()
    with httpx.Client(verify=verify) as client:
        context = client._transport._pool._ssl_context
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname
