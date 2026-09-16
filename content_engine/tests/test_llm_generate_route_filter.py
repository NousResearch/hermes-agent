"""Tests for llm_generate._llm_configs — governed route-chain filtering.

The chain must:
- keep registry slots whose route_model_id matches the primary model
  (slug or namespaced form),
- keep registry-sanctioned cross-model slots (text fallback, local final
  resort) marked with route_slot + route_class,
- drop unmarked entries that serve a different model,
- drop duplicate deployments of the primary (runtime would skip them).
"""
import llm_generate as G

FAKE_BASE = "https://api.xkiro.com/v1"


def _fake_config():
    return {
        "model": {
            "default": "deepseek/deepseek-v4-flash",
            "provider": "custom:xkiro-free",
            "base_url": FAKE_BASE,
        },
        "fallback_providers": [
            {  # duplicate of primary (registry slot 1) -> dropped by dedup
                "provider": "custom:xkiro-free",
                "model": "deepseek/deepseek-v4-flash",
                "base_url": FAKE_BASE,
                "route_slot": "slot-dsflash-1",
                "route_class": "perm",
                "route_model_id": "deepseek-v4-flash",
            },
            {  # same model, different provider -> kept
                "provider": "ollama-cloud",
                "model": "deepseek-v4-flash",
                "base_url": "https://ollama.com/v1",
                "route_slot": "slot-dsflash-4",
                "route_class": "perm",
                "route_model_id": "deepseek-v4-flash",
            },
            {  # cross-model sanctioned text fallback -> kept
                "provider": "openai-codex",
                "model": "gpt-5.6-luna",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "route_slot": "codex-fallback",
                "route_class": "perm",
            },
            {  # local final resort -> kept
                "provider": "custom:turbofit-local",
                "model": "active:main",
                "base_url": "http://127.0.0.1:8091/v1",
                "route_slot": "local-final",
                "route_class": "perm",
            },
            {  # legacy entry, same exact model, no registry marks -> kept
                "provider": "custom:legacy",
                "model": "deepseek/deepseek-v4-flash",
                "base_url": "https://legacy.example/v1",
            },
            {  # different model, no registry marks -> dropped
                "provider": "custom:other",
                "model": "totally-different-model",
                "base_url": "https://other.example/v1",
            },
        ],
    }


def test_chain_filter_and_dedup(monkeypatch):
    import hermes_cli.fallback_config as fc

    monkeypatch.setattr(fc, "get_fallback_chain",
                        lambda config: list(config["fallback_providers"]))

    def fake_resolve(**kwargs):
        return {
            "base_url": kwargs.get("explicit_base_url"),
            "api_key": "test-key",
            "api_mode": "chat_completions",
            "credential_pool": None,
        }

    monkeypatch.setattr(G, "_resolve_runtime", fake_resolve)

    cfgs = G._llm_configs(config=_fake_config())
    providers = [c["provider"] for c in cfgs]
    assert providers == [
        "custom:xkiro-free",      # primary
        "ollama-cloud",
        "openai-codex",
        "custom:turbofit-local",
        "custom:legacy",
    ]
    models = [c["model"] for c in cfgs]
    assert models[0] == "deepseek/deepseek-v4-flash"
    assert "totally-different-model" not in models


def test_primary_slug_match(monkeypatch):
    """route_model_id equal to the slug (post-slash form) passes."""
    import hermes_cli.fallback_config as fc
    monkeypatch.setattr(fc, "get_fallback_chain",
                        lambda config: list(config["fallback_providers"]))
    monkeypatch.setattr(
        G, "_resolve_runtime",
        lambda **kw: {"base_url": kw.get("explicit_base_url"),
                      "api_key": "k", "api_mode": "chat_completions",
                      "credential_pool": None},
    )
    cfg = _fake_config()
    cfg["fallback_providers"] = [cfg["fallback_providers"][1]]  # ollama slot
    out = G._llm_configs(config=cfg)
    assert [c["provider"] for c in out] == ["custom:xkiro-free", "ollama-cloud"]
