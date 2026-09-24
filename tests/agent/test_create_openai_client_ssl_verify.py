"""Regression: keepalive httpx client must honor custom CA bundles for HTTPS providers."""


import httpx
import pytest

from agent.ssl_verify import resolve_httpx_verify
from run_agent import AIAgent

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY")


@pytest.fixture
def clean_tls_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)






def test_build_keepalive_http_client_ssl_verify_false(clean_tls_env):
    verify = resolve_httpx_verify(ssl_verify=False)
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert client._transport._pool._ssl_context.check_hostname is False
