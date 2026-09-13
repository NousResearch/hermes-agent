"""Regression: the keepalive httpx client must carry the resolved TLS trust.

Trust is the OS certificate store; a per-provider ``ssl_ca_cert`` replaces it
for that one client. Both shapes have to survive the trip into httpx's
transport, which is what these assert.
"""

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


def test_build_keepalive_http_client_uses_the_platform_trust_store(clean_tls_env):
    verify = resolve_httpx_verify()
    assert verify is True

    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    # httpx builds its own context from the patched ssl.SSLContext, so the
    # pool's context IS the platform-verified one.
    ctx = client._transport._pool._ssl_context
    assert type(ctx).__module__.startswith("truststore")


def test_build_keepalive_http_client_honors_per_provider_ssl_ca_cert(clean_tls_env):
    from truststore._ssl_constants import _original_SSLContext

    verify = resolve_httpx_verify(ca_bundle=certifi.where())
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    # The pinned bundle must reach the transport UNPATCHED — a truststore
    # context here would mean the pin silently became "trust the machine".
    ctx = client._transport._pool._ssl_context
    assert isinstance(ctx, _original_SSLContext)
    assert not type(ctx).__module__.startswith("truststore")


def test_build_keepalive_http_client_ssl_verify_false(clean_tls_env):
    verify = resolve_httpx_verify(ssl_verify=False)
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert client._transport._pool._ssl_context.check_hostname is False
    assert client._transport._pool._ssl_context.verify_mode == ssl.CERT_NONE
