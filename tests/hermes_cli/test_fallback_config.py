from __future__ import annotations

from hermes_cli.fallback_config import (
    get_configured_fallback_chain,
    get_fallback_chain,
    get_fallback_policy,
)


def test_missing_policy_preserves_legacy_any_behavior():
    cfg = {
        "fallback_providers": [
            {"provider": "openrouter", "model": "remote"},
            {
                "provider": "custom",
                "model": "local",
                "base_url": "http://127.0.0.1:8000/v1",
            },
        ]
    }

    assert get_fallback_policy(cfg) == "any"
    assert get_fallback_chain(cfg) == cfg["fallback_providers"]


def test_off_keeps_configured_order_but_has_no_eligible_routes():
    cfg = {
        "fallback_policy": "off",
        "fallback_providers": [
            {"provider": "openrouter", "model": "first"},
            {"provider": "anthropic", "model": "second"},
        ],
    }

    assert get_configured_fallback_chain(cfg) == cfg["fallback_providers"]
    assert get_fallback_chain(cfg) == []


def test_local_only_uses_endpoint_metadata_not_model_names(monkeypatch):
    monkeypatch.setenv("LM_BASE_URL", "http://10.55.0.3:1234/v1")
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {"provider": "opencode-zen", "model": "local-looking-name"},
            {"provider": "lmstudio", "model": "cloud-looking-name"},
            {"provider": "mystery", "model": "definitely-local"},
        ],
    }

    assert get_fallback_chain(cfg) == [
        {"provider": "lmstudio", "model": "cloud-looking-name"}
    ]


def test_local_only_does_not_reclassify_builtin_cloud_provider_via_env(
    monkeypatch,
):
    monkeypatch.setenv("OPENCODE_ZEN_BASE_URL", "http://127.0.0.1:9999/v1")
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {"provider": "opencode-zen", "model": "remote-model"},
        ],
    }

    assert get_fallback_chain(cfg) == []


def test_local_only_rejects_builtin_anthropic_even_with_explicit_local_url():
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {
                "provider": "anthropic",
                "model": "claude",
                "base_url": "http://localhost:9000/v1",
            }
        ],
    }

    assert get_fallback_chain(cfg) == []


def test_local_only_allows_explicit_user_provider_redefinition():
    cfg = {
        "fallback_policy": "local-only",
        "providers": {
            "anthropic": {
                "base_url": "http://10.55.0.3:9000/v1",
            }
        },
        "fallback_providers": [
            {"provider": "anthropic", "model": "local-compatible"}
        ],
    }

    assert get_fallback_chain(cfg) == [
        {"provider": "anthropic", "model": "local-compatible"}
    ]


def test_invalid_explicit_policy_fails_closed():
    cfg = {
        "fallback_policy": "ANY",
        "fallback_providers": [{"provider": "openrouter", "model": "remote"}],
    }

    assert get_fallback_policy(cfg) == "off"
    assert get_fallback_chain(cfg) == []
