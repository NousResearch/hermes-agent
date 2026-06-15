"""Regression guard: ssl_verify must reach the PRIMARY chat client (#28260).

PR #28271 first threaded the custom-provider ``ssl_verify`` field into the
auxiliary client only (``agent/auxiliary_client.py``).  But the bug in #28260 is
``hermes chat`` — the main ``AIAgent`` client.  That client takes a different
path: ``create_openai_client`` builds its ``http_client`` via
``AIAgent._build_keepalive_http_client``, whose ``httpx.Client`` was constructed
with httpx's default certificate verification.  So ``ssl_verify: false`` (or a
CA-bundle path) never reached the client ``hermes chat`` actually uses, and a
self-signed endpoint still raised ``APIConnectionError`` / ``CERTIFICATE_VERIFY_FAILED``.

These tests pin that:
  * ``_build_keepalive_http_client(verify=...)`` lands the value on the transport
    that performs the TLS handshake (httpx ignores ``verify`` on ``Client`` when
    a custom ``transport`` is supplied, so it MUST be on the transport).
  * ``create_openai_client`` resolves the custom-provider override for the active
    base_url and threads it into that keepalive client — the main-path coverage
    the reviewer asked for, for both ``ssl_verify: false`` and a CA-bundle path.
  * the default (no override) still uses full verification.
"""
import ssl

import certifi
import httpx
import pytest

from run_agent import AIAgent


def _make_agent():
    return AIAgent(
        api_key="test-key",
        base_url="https://gpu.internal:8443/v1",
        provider="custom",
        model="local-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def _ssl_context_of(http_client: httpx.Client) -> ssl.SSLContext:
    """The TLS context the keepalive client's base transport will hand to the OS."""
    return http_client._transport._pool._ssl_context


def _extract_http_client(call_kwargs: dict) -> httpx.Client:
    return call_kwargs.get("http_client")


# --------------------------------------------------------------------------- #
# Builder-level: verify lands on the transport, not the Client.
# --------------------------------------------------------------------------- #

def test_keepalive_builder_disables_verification_when_false():
    client = AIAgent._build_keepalive_http_client("https://gpu.internal:8443/v1", verify=False)
    try:
        ctx = _ssl_context_of(client)
        assert ctx.verify_mode == ssl.CERT_NONE
        assert ctx.check_hostname is False
    finally:
        client.close()


def test_keepalive_builder_keeps_full_verification_by_default():
    # verify omitted -> default httpx behaviour (full verification) preserved.
    client = AIAgent._build_keepalive_http_client("https://api.openai.com/v1")
    try:
        ctx = _ssl_context_of(client)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is True
    finally:
        client.close()


def test_keepalive_builder_accepts_ca_bundle_path():
    # A CA-bundle path string must be accepted and still verify (against that CA).
    client = AIAgent._build_keepalive_http_client(
        "https://corp.internal:8443/v1", verify=certifi.where()
    )
    try:
        ctx = _ssl_context_of(client)
        assert ctx.verify_mode == ssl.CERT_REQUIRED
    finally:
        client.close()


# --------------------------------------------------------------------------- #
# Main path: create_openai_client resolves the override and threads it through.
# --------------------------------------------------------------------------- #

@pytest.fixture
def _patch_openai(monkeypatch):
    """Capture the kwargs passed to ``OpenAI(**client_kwargs)``."""
    calls = {}

    def _fake_openai(**kwargs):
        calls["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("run_agent.OpenAI", _fake_openai)
    return calls


def test_main_path_threads_ssl_verify_false(monkeypatch, _patch_openai):
    """ssl_verify: false on the active custom provider disables verification on
    the primary chat client's keepalive httpx.Client."""
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_ssl_verify",
        lambda base_url, *a, **k: False,
    )
    agent = _make_agent()
    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://gpu.internal:8443/v1"},
        reason="test", shared=False,
    )

    http_client = _extract_http_client(_patch_openai["kwargs"])
    assert isinstance(http_client, httpx.Client), (
        "primary client was not given a keepalive httpx.Client"
    )
    try:
        assert _ssl_context_of(http_client).verify_mode == ssl.CERT_NONE, (
            "ssl_verify:false did not reach the main chat client's transport"
        )
    finally:
        http_client.close()


def test_main_path_threads_ca_bundle_path(monkeypatch, _patch_openai):
    """A CA-bundle path resolves through to the primary client without error."""
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_ssl_verify",
        lambda base_url, *a, **k: certifi.where(),
    )
    agent = _make_agent()
    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://corp.internal:8443/v1"},
        reason="test", shared=False,
    )

    http_client = _extract_http_client(_patch_openai["kwargs"])
    assert isinstance(http_client, httpx.Client)
    try:
        # Custom CA bundle => still verifying (just against the provided CA).
        assert _ssl_context_of(http_client).verify_mode == ssl.CERT_REQUIRED
    finally:
        http_client.close()


def test_main_path_keeps_default_verification_without_override(monkeypatch, _patch_openai):
    """No ssl_verify configured -> primary client keeps full verification."""
    monkeypatch.setattr(
        "hermes_cli.config.get_custom_provider_ssl_verify",
        lambda base_url, *a, **k: None,
    )
    agent = _make_agent()
    agent._create_openai_client(
        {"api_key": "test-key", "base_url": "https://api.openai.com/v1"},
        reason="test", shared=False,
    )

    http_client = _extract_http_client(_patch_openai["kwargs"])
    assert isinstance(http_client, httpx.Client)
    try:
        assert _ssl_context_of(http_client).verify_mode == ssl.CERT_REQUIRED
    finally:
        http_client.close()


# --------------------------------------------------------------------------- #
# Resolver: reads the normalized field from both config schemas.
# --------------------------------------------------------------------------- #

def test_resolver_reads_false_and_ca_path_across_schemas():
    from hermes_cli.config import get_custom_provider_ssl_verify

    config = {
        # legacy list schema
        "custom_providers": [
            {
                "name": "gpu",
                "base_url": "https://gpu.internal:8443/v1",
                "ssl_verify": False,
            },
        ],
        # newer keyed schema
        "providers": {
            "corp": {
                "base_url": "https://corp.internal:8443/v1",
                "ssl_verify": "/etc/ssl/certs/corp-ca.pem",
            },
        },
    }

    assert get_custom_provider_ssl_verify(
        "https://gpu.internal:8443/v1", config=config) is False
    # trailing-slash insensitive
    assert get_custom_provider_ssl_verify(
        "https://corp.internal:8443/v1/", config=config) == "/etc/ssl/certs/corp-ca.pem"
    # no match / no override -> None (default verification preserved)
    assert get_custom_provider_ssl_verify(
        "https://other.example.com/v1", config=config) is None
    assert get_custom_provider_ssl_verify("", config=config) is None
