"""Regression: keepalive httpx client must honor custom CA bundles for HTTPS providers."""

import ssl

import certifi
import httpx
import pytest

from agent.ssl_verify import resolve_httpx_verify
from run_agent import AIAgent

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY")

# install_truststore() rebinds ssl.SSLContext to the truststore subclass process-wide;
# explicit-bundle contexts are deliberately built from the ORIGINAL base class (an
# explicit bundle replaces OS trust). Pin the base class before any resolver call so
# isinstance checks below assert against real TLS contexts either way.
try:
    from truststore._ssl_constants import _original_SSLContext as _AnyTlsContext
except ImportError:  # pragma: no cover - truststore layout changed
    _AnyTlsContext = ssl.SSLContext


@pytest.fixture
def clean_tls_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_build_keepalive_http_client_ssl_cert_file_uses_shared_context(clean_tls_env, monkeypatch):
    # The PM resolver passes a platform context when SSL_CERT_FILE is set
    # (httpx reads CA env vars before the injected verifier gets control).
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    verify = resolve_httpx_verify()
    assert isinstance(verify, (ssl.SSLContext, _AnyTlsContext))
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert isinstance(client._transport._pool._ssl_context, (ssl.SSLContext, _AnyTlsContext))


def test_build_keepalive_http_client_honors_per_provider_ssl_ca_cert(clean_tls_env):
    # An explicit bundle replaces OS trust: the context is an ORIGINAL-class
    # SSLContext loaded with the bundle, never a truststore-injected subclass.
    verify = resolve_httpx_verify(ca_bundle=certifi.where())
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert isinstance(client._transport._pool._ssl_context, _AnyTlsContext)


def test_build_keepalive_http_client_ssl_verify_false(clean_tls_env):
    verify = resolve_httpx_verify(ssl_verify=False)
    client = AIAgent._build_keepalive_http_client(
        "https://ollama.example.com/v1", verify=verify,
    )
    assert isinstance(client, httpx.Client)
    assert client._transport._pool._ssl_context.check_hostname is False
