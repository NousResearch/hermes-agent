"""Unit tests for find_custom_provider_identity (base_url → custom:<name>).

Reverse lookup used by tui_gateway session persistence to recover a named
``providers:`` / ``custom_providers:`` entry from the only durable fact the
session row keeps once the provider has been resolved to the literal string
"custom": the endpoint URL. See
tests/tui_gateway/test_custom_provider_session_persistence.py for the
end-to-end persist/resume round-trip.
"""

import hermes_cli.runtime_provider as rp


def test_matches_legacy_custom_providers_list(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {"name": "MiMo v2.5 Pro", "base_url": "https://api.mimo.example/v1"}
            ]
        },
    )
    assert (
        rp.find_custom_provider_identity("https://api.mimo.example/v1")
        == "custom:mimo-v2.5-pro"
    )


def test_matches_providers_dict_by_key(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {"providers": {"local": {"api": "http://127.0.0.1:8000/v1"}}},
    )
    assert (
        rp.find_custom_provider_identity("http://127.0.0.1:8000/v1")
        == "custom:local"
    )


def test_match_ignores_trailing_slash_and_case(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {"name": "local", "base_url": "http://Localhost:8000/v1/"}
            ]
        },
    )
    assert (
        rp.find_custom_provider_identity("http://localhost:8000/v1")
        == "custom:local"
    )


def test_no_match_returns_none(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {"name": "other", "base_url": "https://elsewhere.example/v1"}
            ]
        },
    )
    assert rp.find_custom_provider_identity("https://api.mimo.example/v1") is None


def test_empty_base_url_returns_none(monkeypatch):
    monkeypatch.setattr(
        rp, "load_config", lambda: {"custom_providers": [{"name": "x"}]}
    )
    assert rp.find_custom_provider_identity("") is None
    assert rp.find_custom_provider_identity(None) is None


def test_identity_resolves_back_through_named_lookup(monkeypatch):
    """The returned slug must be accepted by _get_named_custom_provider —
    that is the whole point of persisting it."""
    config = {
        "custom_providers": [
            {
                "name": "mimo-v2.5-pro",
                "base_url": "https://api.mimo.example/v1",
                "api_key": "sk-entry",
            }
        ]
    }
    monkeypatch.setattr(rp, "load_config", lambda: config)

    slug = rp.find_custom_provider_identity("https://api.mimo.example/v1")
    assert slug == "custom:mimo-v2.5-pro"

    entry = rp._get_named_custom_provider(slug)
    assert entry is not None
    assert entry["base_url"] == "https://api.mimo.example/v1"
    assert entry["api_key"] == "sk-entry"
