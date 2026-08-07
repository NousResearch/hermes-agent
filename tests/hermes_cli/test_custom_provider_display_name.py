"""Tests for resolve_custom_provider_display_name (status/UI labels).

Named ``providers:`` / ``custom_providers:`` entries resolve to billing class
``"custom"``. Status surfaces must recover the config.yaml entry name so
``/status`` can show ``Model: gpt (openlux)`` instead of ``(custom)``.
"""

import hermes_cli.runtime_provider as rp


def test_strips_custom_menu_key():
    assert (
        rp.resolve_custom_provider_display_name("custom:openlux") == "openlux"
    )
    assert (
        rp.resolve_custom_provider_display_name("custom:My Local") == "my-local"
    )


def test_passthrough_builtin_providers():
    assert rp.resolve_custom_provider_display_name("deepseek") == "deepseek"
    assert rp.resolve_custom_provider_display_name("anthropic") == "anthropic"


def test_recovers_providers_dict_by_base_url(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "openlux": {"base_url": "https://api.openlux.ai/v1"},
                "onerouter": {"base_url": "https://www.onerouter.one/v1"},
            }
        },
    )
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://api.openlux.ai/v1",
        )
        == "openlux"
    )
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://www.onerouter.one/v1/",
        )
        == "onerouter"
    )


def test_recovers_legacy_custom_providers_list(monkeypatch):
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
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://api.mimo.example/v1",
        )
        == "mimo-v2.5-pro"
    )


def test_shared_gateway_prefers_model_catalog_match(monkeypatch):
    """Multiple providers on one URL: prefer the entry that serves the model."""
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "openlux-claude": {
                    "base_url": "https://api.openlux.ai/v1",
                    "models": ["claude-sonnet-5", "claude-opus-5"],
                },
                "openlux-codex": {
                    "base_url": "https://api.openlux.ai/v1",
                    "models": ["gpt-5.6-sol", "gpt-5.6-terra"],
                },
            }
        },
    )
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://api.openlux.ai/v1",
            model="gpt-5.6-terra",
        )
        == "openlux-codex"
    )
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://api.openlux.ai/v1",
            model="claude-sonnet-5",
        )
        == "openlux-claude"
    )


def test_direct_named_provider_key(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "openlux": {"base_url": "https://api.openlux.ai/v1"},
            }
        },
    )
    assert rp.resolve_custom_provider_display_name("openlux") == "openlux"


def test_unmatched_custom_stays_custom(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: {"providers": {}})
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://adhoc.example/v1",
        )
        == "custom"
    )


def test_unmatched_url_ignores_configured_provider(monkeypatch):
    """Explicit ad-hoc session URL must not inherit global model.provider.

    Regression for PR review: canonical_custom_identity falls back to
    config_provider, which would mis-label an unmatched custom endpoint as
    the unrelated named provider from config.yaml.
    """
    cfg = {
        "model": {
            "provider": "openlux",
            "default": "gpt-5.6-sol",
            "base_url": "https://api.openlux.ai/v1",
        },
        "providers": {
            "openlux": {
                "base_url": "https://api.openlux.ai/v1",
                "models": ["gpt-5.6-sol"],
            }
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            base_url="https://adhoc.example/v1",
            config_provider="openlux",
            model="gpt-5.6-sol",
        )
        == "custom"
    )


def test_no_url_may_use_configured_provider(monkeypatch):
    """Without a session URL, config.model.provider remains a valid recovery."""
    cfg = {
        "model": {"provider": "openlux", "default": "gpt-5.6-sol"},
        "providers": {
            "openlux": {
                "base_url": "https://api.openlux.ai/v1",
                "models": ["gpt-5.6-sol"],
            }
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda: cfg)
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            config_provider="openlux",
        )
        == "openlux"
    )


def test_recovers_by_model_when_base_url_missing(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "local-llm": {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "default_model": "qwen3-coder",
                }
            }
        },
    )
    assert (
        rp.resolve_custom_provider_display_name(
            "custom",
            model="qwen3-coder",
        )
        == "local-llm"
    )
