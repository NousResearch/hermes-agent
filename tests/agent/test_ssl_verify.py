"""Tests for agent.ssl_verify TLS resolvers."""

import ssl

import certifi
import pytest

from agent.ssl_verify import resolve_httpx_verify, resolve_requests_verify

_CA_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")


@pytest.fixture
def clean_ca_env(monkeypatch):
    for var in _CA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_ssl_verify_false_disables_verification(clean_ca_env):
    assert resolve_httpx_verify(ssl_verify=False) is False


def test_hermes_ca_bundle_returns_ssl_context(clean_ca_env, monkeypatch):
    monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
    result = resolve_httpx_verify()
    assert isinstance(result, ssl.SSLContext)


def test_explicit_ca_bundle_param(clean_ca_env):
    result = resolve_httpx_verify(ca_bundle=certifi.where())
    assert isinstance(result, ssl.SSLContext)


def test_missing_ca_bundle_falls_back_to_true(clean_ca_env, monkeypatch):
    monkeypatch.setenv("HERMES_CA_BUNDLE", "/nonexistent/root-ca.pem")
    assert resolve_httpx_verify() is True


def test_default_without_env_is_true(clean_ca_env):
    assert resolve_httpx_verify() is True


# ──────────────────────────────────────────────────────────────────────────
# resolve_requests_verify — provider-scoped TLS for `requests`-based probes
# (fetch_endpoint_model_metadata /models, /v1/props, lm-studio native).
# Mirrors resolve_httpx_verify semantics but returns bool|str (a filesystem
# path or True/False) because `requests.get(verify=...)` does not accept an
# ssl.SSLContext.
# ──────────────────────────────────────────────────────────────────────────


class TestResolveRequestsVerify:
    """Provider-scoped TLS resolution for requests-based metadata probes."""

    def test_no_args_no_env_returns_true(self, clean_ca_env):
        assert resolve_requests_verify() is True

    def test_explicit_ca_bundle_returns_path(self, clean_ca_env):
        result = resolve_requests_verify(ca_bundle=certifi.where())
        assert result == certifi.where()

    def test_provider_ca_overrides_env(self, clean_ca_env, monkeypatch, tmp_path):
        env_bundle = tmp_path / "env-ca.pem"
        env_bundle.write_text("env-ca")
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(env_bundle))
        # Provider ca_bundle wins over the env var.
        assert resolve_requests_verify(ca_bundle=certifi.where()) == certifi.where()

    def test_ssl_verify_false_disables(self, clean_ca_env):
        assert resolve_requests_verify(ssl_verify=False) is False

    def test_ssl_verify_false_string_disables(self, clean_ca_env):
        assert resolve_requests_verify(ssl_verify="false") is False

    def test_no_provider_falls_back_to_env(self, clean_ca_env, monkeypatch):
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", certifi.where())
        assert resolve_requests_verify() == certifi.where()

    def test_missing_provider_ca_falls_back_to_env(self, clean_ca_env, monkeypatch):
        # A non-existent provider ca_bundle path falls through to env/default,
        # matching resolve_httpx_verify's fallback behaviour.
        monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
        assert resolve_requests_verify(ca_bundle="/nonexistent/provider-ca.pem") == certifi.where()

    def test_missing_provider_ca_no_env_returns_true(self, clean_ca_env):
        assert resolve_requests_verify(ca_bundle="/nonexistent/provider-ca.pem") is True

    def test_env_precedence_hermes_over_requests(self, clean_ca_env, monkeypatch, tmp_path):
        other = tmp_path / "other.pem"
        other.write_text("other")
        monkeypatch.setenv("HERMES_CA_BUNDLE", certifi.where())
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(other))
        assert resolve_requests_verify() == certifi.where()
