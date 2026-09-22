"""Regression: keepalive httpx client must honor custom CA bundles for HTTPS providers."""

import ssl

import certifi
import httpx
import pytest

from agent.ssl_verify import resolve_httpx_verify
from run_agent import AIAgent

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY")


@pytest.fixture
def clean_tls_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_build_keepalive_http_client_uses_hermes_ca_bundle(clean_tls_env, monkeypatch):
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    verify = resolve_httpx_verify()
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert isinstance(client._transport._pool._ssl_context, ssl.SSLContext)


def test_build_keepalive_http_client_honors_per_provider_ssl_ca_cert(clean_tls_env):
    verify = resolve_httpx_verify(ca_bundle=certifi.where())
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert isinstance(client._transport._pool._ssl_context, ssl.SSLContext)


def test_build_keepalive_http_client_ssl_verify_false(clean_tls_env):
    verify = resolve_httpx_verify(ssl_verify=False)
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert client._transport._pool._ssl_context.check_hostname is False
