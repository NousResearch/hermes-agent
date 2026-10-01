"""Regression: auxiliary-client keepalive httpx client must honor custom CA bundles.

The main OpenAI client resolves per-provider ``ssl_ca_cert`` / ``ssl_verify`` and
``HERMES_CA_BUNDLE`` via ``agent.ssl_verify.resolve_httpx_verify``. Auxiliary calls
(compression, vision, web_extract, title generation, session_search) build their own
keepalive client through ``agent.process_bootstrap.build_keepalive_http_client`` and must
apply the same TLS settings — otherwise an HTTPS custom_providers endpoint signed by a
private CA works for chat but fails ``APIConnectionError`` on every auxiliary task.
"""

import ssl

import certifi
import httpx
import pytest

from agent.process_bootstrap import build_keepalive_http_client

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY")


@pytest.fixture
def clean_tls_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_build_keepalive_http_client_forwards_verify_context(clean_tls_env):
    ctx = ssl.create_default_context(cafile=certifi.where())
    client = build_keepalive_http_client("https://ollama.example.com/v1", verify=ctx)
    assert isinstance(client, httpx.Client)
    assert client._transport._pool._ssl_context is ctx








def test_resolve_aux_verify_ssl_verify_false(clean_tls_env, monkeypatch):
    import hermes_cli.config as cfg
    import hermes_cli.config_providers as config_providers
    from agent import auxiliary_client

    monkeypatch.setattr(
        cfg,
        "get_custom_provider_tls_settings",
        lambda *a, **k: {"ssl_verify": False},
    )
    monkeypatch.setattr(
        config_providers,
        "get_custom_provider_tls_settings",
        lambda *a, **k: {"ssl_verify": False},
    )
    assert auxiliary_client._resolve_aux_verify("https://ollama.example.com/v1") is False


