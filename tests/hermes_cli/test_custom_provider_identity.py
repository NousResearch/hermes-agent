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


def test_candidate_lookup_preserves_case_sensitive_paths(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {"name": "tenant-upper", "base_url": "https://gateway.example/TenantA"},
                {"name": "tenant-lower", "base_url": "https://gateway.example/tenanta"},
            ]
        },
    )

    assert rp.find_custom_provider_identities("https://GATEWAY.example/tenanta/") == [
        "custom:tenant-lower"
    ]


def test_candidate_lookup_returns_all_shared_endpoint_identities(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "model": {"provider": "custom:tenant-b"},
            "custom_providers": [
                {"name": "tenant-a", "base_url": "https://gateway.example/v1"},
                {"name": "tenant-b", "base_url": "https://gateway.example/v1/"},
            ]
        },
    )

    assert rp.find_custom_provider_identities("https://gateway.example/v1") == [
        "custom:tenant-a",
        "custom:tenant-b",
    ]
    assert rp.find_custom_provider_identity("https://gateway.example/v1") is None
    assert rp.canonical_custom_identity(base_url="https://gateway.example/v1") is None


def test_normalized_name_collision_is_ambiguous(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {"name": "Tenant A", "base_url": "https://gateway.example/v1"},
                {"name": "tenant-a", "base_url": "https://gateway.example/v1"},
            ]
        },
    )

    assert rp.find_custom_provider_identities("https://gateway.example/v1") == [
        "custom:tenant-a",
        "custom:tenant-a",
    ]
    assert rp.find_custom_provider_identity("https://gateway.example/v1") is None


def test_candidate_lookup_skips_disabled_provider_blocks(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "disabled": {
                    "enabled": False,
                    "api": "https://gateway.example/v1",
                },
                "enabled": {
                    "api": "https://gateway.example/v1",
                },
            },
            "custom_providers": [
                {
                    "name": "legacy-disabled",
                    "enabled": False,
                    "base_url": "https://legacy-disabled.example/v1",
                    "api_key": "must-not-resolve",
                }
            ],
        },
    )

    assert rp.find_custom_provider_identities("https://gateway.example/v1") == [
        "custom:enabled"
    ]
    assert rp.find_custom_provider_identities("https://legacy-disabled.example/v1") == []
