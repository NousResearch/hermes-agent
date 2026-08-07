"""The Anthropic client must honour Hermes' TLS verify resolution.

``agent.ssl_verify.resolve_httpx_verify`` exists because httpx pins certifi's
bundle and will not pick up ``HERMES_CA_BUNDLE`` / ``SSL_CERT_FILE`` /
``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE`` or the per-provider
``ssl_ca_cert`` / ``ssl_verify`` config on its own — the settings have to be
passed explicitly. ``create_openai_client`` and the auxiliary clients already
route through it; ``build_anthropic_client`` never did, so it always used
certifi and there was no way to configure it.

Asserts the observable contract: the CA certificates the constructed client
would actually trust. No network.
"""

from __future__ import annotations

import pathlib
import ssl

import certifi
import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.ssl_verify import resolve_httpx_verify

_CA_ENV_VARS = (
    "HERMES_CA_BUNDLE",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


@pytest.fixture(autouse=True)
def _clean_ca_env(monkeypatch):
    for name in _CA_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def tiny_ca(tmp_path):
    """A one-certificate CA bundle — the count makes the assertion unambiguous."""
    first = pathlib.Path(certifi.where()).read_text(encoding="utf-8").split(
        "-----END CERTIFICATE-----"
    )[0]
    path = tmp_path / "corp-ca.pem"
    path.write_text(first + "-----END CERTIFICATE-----\n", encoding="utf-8")
    return path


def _ssl_context(client):
    """The SSL context the built client's transport would actually use."""
    return client._client._transport._pool._ssl_context


def _trusted_count(client) -> int:
    return len(_ssl_context(client).get_ca_certs())


class TestCaBundleIsHonoured:
    @pytest.mark.parametrize("env_var", _CA_ENV_VARS)
    def test_every_documented_ca_env_var_reaches_the_client(
        self, tiny_ca, env_var, monkeypatch
    ):
        monkeypatch.setenv(env_var, str(tiny_ca))
        client = build_anthropic_client("sk-ant-test", base_url="https://api.anthropic.com")
        assert _trusted_count(client) == 1

    def test_per_provider_ssl_ca_cert_reaches_the_client(self, tiny_ca):
        client = build_anthropic_client(
            "sk-ant-test", base_url="https://proxy.example", ssl_ca_cert=str(tiny_ca)
        )
        assert _trusted_count(client) == 1

    def test_matches_what_the_shared_resolver_returns(self, tiny_ca, monkeypatch):
        """Same answer as the OpenAI path gets — that is the whole point."""
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(tiny_ca))
        expected = resolve_httpx_verify()
        assert isinstance(expected, ssl.SSLContext)
        client = build_anthropic_client("sk-ant-test", base_url="https://api.anthropic.com")
        assert _trusted_count(client) == len(expected.get_ca_certs())

    def test_oauth_token_path_is_covered_too(self, tiny_ca, monkeypatch):
        """OAuth/Bearer auth takes a different kwargs branch; TLS must still apply."""
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(tiny_ca))
        client = build_anthropic_client(
            "sk-ant-oat01-testtoken", base_url="https://api.anthropic.com"
        )
        assert _trusted_count(client) == 1


class TestSslVerifyFalse:
    def test_verification_can_be_disabled(self):
        client = build_anthropic_client(
            "sk-ant-test", base_url="https://local.example", ssl_verify=False
        )
        assert _ssl_context(client).verify_mode == ssl.CERT_NONE

    def test_string_false_is_accepted(self):
        client = build_anthropic_client(
            "sk-ant-test", base_url="https://local.example", ssl_verify="false"
        )
        assert _ssl_context(client).verify_mode == ssl.CERT_NONE


class TestDefaultPathUnchanged:
    def test_no_tls_config_keeps_the_sdk_default_bundle(self):
        client = build_anthropic_client("sk-ant-test", base_url="https://api.anthropic.com")
        assert _trusted_count(client) > 1        # certifi's full bundle
        assert _ssl_context(client).verify_mode == ssl.CERT_REQUIRED

    def test_missing_ca_file_falls_back_to_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(tmp_path / "nope.pem"))
        client = build_anthropic_client("sk-ant-test", base_url="https://api.anthropic.com")
        assert _trusted_count(client) > 1
        assert _ssl_context(client).verify_mode == ssl.CERT_REQUIRED

    def test_client_still_builds_and_keeps_its_settings(self, tiny_ca, monkeypatch):
        monkeypatch.setenv("HERMES_CA_BUNDLE", str(tiny_ca))
        client = build_anthropic_client(
            "sk-ant-test", base_url="https://api.anthropic.com", timeout=42.0
        )
        assert client.max_retries == 0
        assert client.timeout.read == 42.0
        assert client.timeout.connect == 10.0
