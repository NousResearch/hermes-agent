"""Tests for per-provider TLS settings in custom_providers config."""

from hermes_cli.config import (
    apply_custom_provider_tls_to_client_kwargs,
    get_custom_provider_tls_settings,
)


def test_get_custom_provider_tls_settings_matches_base_url():
    providers = [
        {
            "name": "Ollama",
            "base_url": "https://ollama.example.com/v1",
            "ssl_ca_cert": "/etc/ssl/mkcert-root.pem",
        }
    ]
    tls = get_custom_provider_tls_settings(
        "https://ollama.example.com/v1/",
        custom_providers=providers,
    )
    assert tls == {"ssl_ca_cert": "/etc/ssl/mkcert-root.pem"}


def test_apply_custom_provider_tls_to_client_kwargs():
    client_kwargs = {"api_key": "x", "base_url": "https://ollama.example.com/v1"}
    providers = [
        {
            "name": "Ollama",
            "base_url": "https://ollama.example.com/v1",
            "ssl_ca_cert": "/etc/ssl/mkcert-root.pem",
            "ssl_verify": True,
        }
    ]
    apply_custom_provider_tls_to_client_kwargs(
        client_kwargs,
        "https://ollama.example.com/v1",
        custom_providers=providers,
    )
    assert client_kwargs["ssl_ca_cert"] == "/etc/ssl/mkcert-root.pem"
    assert client_kwargs["ssl_verify"] is True


def test_get_custom_provider_tls_settings_matches_case_insensitively():
    """A config base_url with mixed case must still match a lowercased runtime base_url."""
    providers = [
        {
            "name": "Ollama",
            "base_url": "https://Ollama.Example.com/v1",
            "ssl_ca_cert": "/etc/ssl/mkcert-root.pem",
        }
    ]
    tls = get_custom_provider_tls_settings(
        "https://ollama.example.com/v1",
        custom_providers=providers,
    )
    assert tls == {"ssl_ca_cert": "/etc/ssl/mkcert-root.pem"}


def test_get_custom_provider_tls_settings_no_substring_bypass():
    """A base_url that is only a prefix of an entry must NOT match."""
    providers = [
        {
            "name": "Ollama",
            "base_url": "https://ollama.example.com/v1",
            "ssl_verify": False,
        }
    ]
    # A different host that shares a prefix must not pick up ssl_verify:false.
    assert get_custom_provider_tls_settings(
        "https://ollama.example.com.attacker.test/v1",
        custom_providers=providers,
    ) == {}


def test_get_custom_provider_tls_settings_preserves_extra_path_segment():
    providers = [
        {
            "base_url": "https://ollama.example.com/v1//",
            "ssl_verify": False,
        }
    ]

    assert get_custom_provider_tls_settings(
        "https://ollama.example.com/v1",
        custom_providers=providers,
    ) == {}
