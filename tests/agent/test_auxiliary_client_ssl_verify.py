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


def test_build_keepalive_http_client_verify_false_disables_hostname_check(clean_tls_env):
    client = build_keepalive_http_client("https://ollama.example.com/v1", verify=False)
    assert isinstance(client, httpx.Client)
    assert client._transport._pool._ssl_context.check_hostname is False


def test_build_keepalive_http_client_default_verify_true(clean_tls_env):
    client = build_keepalive_http_client("https://ollama.example.com/v1")
    assert isinstance(client, httpx.Client)


def test_resolve_aux_verify_uses_per_provider_ssl_ca_cert(clean_tls_env, monkeypatch):
    """_resolve_aux_verify should mirror the main-client resolution for a matched base_url."""
    import hermes_cli.config as cfg
    from agent import auxiliary_client

    # get_custom_provider_tls_settings is imported inside the function from
    # hermes_cli.config, so patch it at the source module.
    monkeypatch.setattr(
        cfg,
        "get_custom_provider_tls_settings",
        lambda *a, **k: {"ssl_ca_cert": certifi.where()},
    )
    verify = auxiliary_client._resolve_aux_verify("https://ollama.example.com/v1")
    assert isinstance(verify, ssl.SSLContext)


def test_resolve_aux_verify_ssl_verify_false(clean_tls_env, monkeypatch):
    import hermes_cli.config as cfg
    from agent import auxiliary_client

    monkeypatch.setattr(
        cfg,
        "get_custom_provider_tls_settings",
        lambda *a, **k: {"ssl_verify": False},
    )
    assert auxiliary_client._resolve_aux_verify("https://ollama.example.com/v1") is False


def test_resolve_aux_verify_no_match_defaults_true(clean_tls_env, monkeypatch):
    import hermes_cli.config as cfg
    from agent import auxiliary_client

    monkeypatch.setattr(cfg, "get_custom_provider_tls_settings", lambda *a, **k: {})
    assert auxiliary_client._resolve_aux_verify("https://openrouter.ai/api/v1") is True
