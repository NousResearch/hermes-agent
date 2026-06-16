from __future__ import annotations

from hermes_cli import fallback_config
from hermes_cli import models as models_mod
from hermes_cli.fallback_config import get_fallback_chain


def _promotions_config(**overrides):
    cfg = {
        "fallback_promotions": {
            "enabled": True,
            "providers": ["nous"],
            "position": "prepend",
        }
    }
    cfg["fallback_promotions"].update(overrides)
    return cfg


def test_missing_promotions_config_preserves_static_only_behavior(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [
            {
                "provider": "nous",
                "model": "stepfun/step-3.7-flash:free",
                "supports_tools": True,
            }
        ],
    )

    assert get_fallback_chain({}) == []


def test_free_nous_promotion_entries_read_current_portal_recommendations(monkeypatch):
    calls = []

    def _fake_fetch(portal_base_url, timeout=5.0, **kwargs):
        calls.append((portal_base_url, timeout, kwargs))
        return {
            "freeRecommendedModels": [
                {"modelName": "stepfun/step-3.7-flash:free"},
                {"modelName": " stepfun/step-3.7-flash:free "},
                {"modelName": ""},
                {},
            ]
        }

    monkeypatch.setattr(models_mod, "_resolve_nous_portal_url", lambda: "https://portal.example")
    monkeypatch.setattr(models_mod, "fetch_nous_recommended_models", _fake_fetch)

    assert fallback_config._free_nous_promotion_entries() == [
        {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "supports_tools": True,
            "source": "dynamic-free-promotion",
        }
    ]
    assert calls == [("https://portal.example", 1.5, {})]


def test_nous_free_promotions_prepend_to_static_fallbacks(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [
            {
                "provider": "nous",
                "model": "stepfun/step-3.7-flash:free",
                "supports_tools": True,
                "source": "dynamic-free-promotion",
            }
        ],
    )

    cfg = _promotions_config()
    cfg["fallback_providers"] = [
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
    ]

    chain = get_fallback_chain(cfg)

    assert chain == [
        {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "supports_tools": True,
            "source": "dynamic-free-promotion",
        },
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
    ]


def test_explicit_fallback_entry_wins_over_dynamic_duplicate(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [
            {
                "provider": "nous",
                "model": "stepfun/step-3.7-flash:free",
                "supports_tools": True,
                "source": "dynamic-free-promotion",
            }
        ],
    )

    cfg = _promotions_config()
    cfg["fallback_providers"] = [
        {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "supports_tools": False,
            "note": "operator override",
        }
    ]

    assert get_fallback_chain(cfg) == cfg["fallback_providers"]


def test_promotions_can_append_after_static_chain(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [{"provider": "nous", "model": "stepfun/step-3.7-flash:free"}],
    )

    cfg = _promotions_config(position="append")
    cfg["fallback_providers"] = [{"provider": "taro", "model": "qwen3.6-27b-256k"}]

    assert get_fallback_chain(cfg) == [
        {"provider": "taro", "model": "qwen3.6-27b-256k"},
        {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
    ]


def test_promotions_skip_when_provider_is_not_authenticated(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: False)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [{"provider": "nous", "model": "stepfun/step-3.7-flash:free"}],
    )

    cfg = _promotions_config()
    cfg["fallback_providers"] = [{"provider": "opencode-zen", "model": "deepseek-v4-flash-free"}]

    assert get_fallback_chain(cfg) == [
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"}
    ]


def test_env_override_can_disable_promotions(monkeypatch):
    monkeypatch.setenv("HERMES_FALLBACK_PROMOTIONS", "0")
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [{"provider": "nous", "model": "stepfun/step-3.7-flash:free"}],
    )

    assert get_fallback_chain(_promotions_config()) == []
