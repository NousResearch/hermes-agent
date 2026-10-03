"""Credential pools must never cross provider or custom-endpoint boundaries."""

from types import SimpleNamespace
from unittest.mock import patch

from agent.credential_pool import (
    custom_provider_pool_key_candidates,
    credential_pool_matches_provider,
    resolve_runtime_pool_key,
)
from hermes_cli import runtime_provider as rp


def test_provider_match_requires_exact_non_custom_identity():
    assert credential_pool_matches_provider("deepseek", "deepseek")
    assert not credential_pool_matches_provider("openai-codex", "deepseek")
    assert not credential_pool_matches_provider("", "deepseek")


def test_custom_pool_match_is_scoped_by_endpoint():
    with patch(
        "agent.credential_pool.get_custom_provider_pool_key",
        return_value="custom:lab",
    ):
        assert credential_pool_matches_provider(
            "custom:lab", "custom", base_url="https://lab.example/v1"
        )
        assert not credential_pool_matches_provider(
            "custom:other", "custom", base_url="https://lab.example/v1"
        )


def test_named_custom_pool_match_requires_configured_identity_and_endpoint():
    configured = [
        (
            "gemini-display",
            {
                "name": "Gemini Display",
                "provider_key": "gemini-no-filter",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/",
            },
        )
    ]
    with patch("agent.credential_pool._iter_custom_providers", return_value=configured):
        assert credential_pool_matches_provider(
            "custom:gemini-display",
            "gemini-no-filter",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        assert credential_pool_matches_provider(
            "custom:gemini-display",
            "custom:gemini-no-filter",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )
        assert not credential_pool_matches_provider(
            "custom:gemini-display",
            "gemini-no-filter",
            base_url="https://fallback.example/v1",
        )
        assert not credential_pool_matches_provider(
            "custom:gemini-display",
            "custom:gemini-no-filter",
            base_url="https://fallback.example/v1",
        )
        assert not credential_pool_matches_provider(
            "custom:gemini-display",
            "other-provider",
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )


def test_runtime_pool_key_resolves_all_custom_runtime_identities():
    endpoint = "https://generativelanguage.googleapis.com/v1beta"
    configured = [
        (
            "sibling-display",
            {
                "name": "Sibling Display",
                "provider_key": "sibling-provider",
                "base_url": endpoint,
            },
        ),
        (
            "gemini-display",
            {
                "name": "Gemini Display",
                "provider_key": "gemini-no-filter",
                "base_url": endpoint,
            },
        )
    ]
    with patch("agent.credential_pool._iter_custom_providers", return_value=configured):
        assert resolve_runtime_pool_key("custom", endpoint) == "sibling-provider"
        assert (
            resolve_runtime_pool_key("gemini-no-filter", endpoint)
            == "gemini-no-filter"
        )
        assert (
            resolve_runtime_pool_key("custom:gemini-no-filter", endpoint)
            == "gemini-no-filter"
        )
        assert (
            resolve_runtime_pool_key(
                "gemini-no-filter",
                "https://fallback.example/v1",
            )
            == "gemini-no-filter"
        )


def test_runtime_pool_key_resolves_modern_provider_in_mixed_config():
    endpoint = "https://generativelanguage.googleapis.com/v1beta"
    config = {
        "custom_providers": [
            {
                "name": "Legacy Provider",
                "base_url": "https://legacy.example/v1",
            }
        ],
        "providers": {
            "gemini-no-filter": {
                "name": "Gemini Display",
                "api": endpoint,
            }
        },
    }

    with patch("agent.credential_pool._load_config_safe", return_value=config):
        assert (
            resolve_runtime_pool_key("gemini-no-filter", endpoint)
            == "gemini-no-filter"
        )
        assert (
            resolve_runtime_pool_key("custom:gemini-no-filter", endpoint)
            == "gemini-no-filter"
        )
        assert (
            resolve_runtime_pool_key(
                "custom:gemini-no-filter",
                "https://fallback.example/v1",
            )
            == "custom:gemini-no-filter"
        )


def test_keyed_provider_pool_matches_runtime_aliases():
    configured = [
        (
            "b.ai",
            {
                "name": "B.AI",
                "provider_key": "b-ai",
                "base_url": "https://api.b.ai/v1",
            },
        )
    ]
    with patch("agent.credential_pool._iter_custom_providers", return_value=configured):
        assert credential_pool_matches_provider(
            "b-ai", "b-ai", base_url="https://api.b.ai/v1"
        )
        assert credential_pool_matches_provider(
            "b-ai", "custom", base_url="https://api.b.ai/v1"
        )
        assert credential_pool_matches_provider(
            "b-ai", "custom:b.ai", base_url="https://api.b.ai/v1"
        )
        assert not credential_pool_matches_provider(
            "b-ai", "custom", base_url="https://other.example/v1"
        )
        assert not credential_pool_matches_provider(
            "b-ai", "deepseek", base_url="https://api.b.ai/v1"
        )


def test_runtime_pool_key_prefers_durable_provider_slug():
    endpoint = "https://api.b.ai/v1"
    configured = [
        (
            "b.ai",
            {
                "name": "B.AI",
                "provider_key": "b-ai",
                "base_url": endpoint,
            },
        )
    ]
    with patch("agent.credential_pool._iter_custom_providers", return_value=configured):
        assert resolve_runtime_pool_key("b-ai", endpoint) == "b-ai"
        assert resolve_runtime_pool_key("custom", endpoint) == "b-ai"
        assert resolve_runtime_pool_key("custom:b.ai", endpoint) == "b-ai"


def test_runtime_pool_key_preserves_non_custom_identity():
    with patch("agent.credential_pool._iter_custom_providers", return_value=[]):
        assert (
            resolve_runtime_pool_key("openai-codex", "https://chatgpt.com/backend-api")
            == "openai-codex"
        )


def test_runtime_ignores_pool_loaded_for_different_provider(monkeypatch):
    entry = SimpleNamespace(
        provider="openai-codex",
        access_token="wrong-token",
        runtime_api_key="wrong-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(
        provider="openai-codex",
        has_credentials=lambda: True,
        select=lambda **_kwargs: entry,
    )
    monkeypatch.setattr(rp, "load_pool", lambda _provider: pool)
    monkeypatch.setattr(rp, "resolve_provider", lambda *_a, **_kw: "deepseek")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "deepseek", "default": "deepseek-chat"},
    )
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "deepseek",
            "api_key": "deepseek-key",
            "base_url": "https://api.deepseek.com/v1",
            "source": "env",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="deepseek")

    assert resolved["provider"] == "deepseek"
    assert resolved["api_key"] == "deepseek-key"
    assert resolved["base_url"] == "https://api.deepseek.com/v1"


# ── Same-URL sibling providers must not kill each other's pool ──────────────
# runtime_provider.py stamps provider="custom" for ALL named custom providers,
# so agent.provider is the bare string "custom" while pools are keyed
# "custom:<name>". custom_provider_pool_key_candidates() used to return on the
# FIRST entry whose base_url matched, so with two providers sharing an endpoint
# (e.g. cline-free and cline-free-glm on api.cline.bot) the second sibling's
# pool failed credential_pool_matches_provider and agent_init dropped
# _credential_pool entirely — 429 recovery then had no keys to rotate.


def _sibling_entries():
    return [
        ("cline-free", {"name": "cline-free", "base_url": "https://api.cline.bot/api/v1"}),
        ("cline-free-glm", {"name": "cline-free-glm", "base_url": "https://api.cline.bot/api/v1"}),
    ]


def test_candidates_collect_all_same_url_siblings():
    """The URL match loop must collect EVERY sibling entry, not return the first."""
    with patch("agent.credential_pool._iter_custom_providers", return_value=_sibling_entries()):
        keys = custom_provider_pool_key_candidates("https://api.cline.bot/api/v1")
    assert "custom:cline-free" in keys
    assert "custom:cline-free-glm" in keys


def test_bare_custom_agent_keeps_named_sibling_pool():
    """agent.provider='custom' (runtime stamp) + pool=custom:cline-free-glm
    + same base_url as sibling cline-free -> pool must SURVIVE."""
    with patch("agent.credential_pool._iter_custom_providers", return_value=_sibling_entries()):
        assert credential_pool_matches_provider(
            "custom:cline-free-glm", "custom", base_url="https://api.cline.bot/api/v1"
        )


def test_bare_custom_agent_keeps_first_provider_pool_too():
    """Sibling order must not matter: pool=custom:cline-free also survives."""
    with patch("agent.credential_pool._iter_custom_providers", return_value=_sibling_entries()):
        assert credential_pool_matches_provider(
            "custom:cline-free", "custom", base_url="https://api.cline.bot/api/v1"
        )


def test_bare_custom_agent_still_rejects_foreign_url():
    """Fail-closed preserved: a pool for URL A must not match an agent on URL B."""
    with patch("agent.credential_pool._iter_custom_providers", return_value=_sibling_entries()):
        assert not credential_pool_matches_provider(
            "custom:cline-free-glm", "custom", base_url="https://api.other-host.dev/v1"
        )


def test_bare_custom_agent_still_rejects_unconfigured_pool():
    """A pool keyed to a name absent from config still fails closed."""
    with patch("agent.credential_pool._iter_custom_providers", return_value=_sibling_entries()):
        assert not credential_pool_matches_provider(
            "custom:not-configured", "custom", base_url="https://api.cline.bot/api/v1"
        )