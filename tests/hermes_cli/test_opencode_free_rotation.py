from hermes_cli import models as model_catalog
from hermes_cli.fallback_chain import normalize_fallback_entries


def test_provider_model_ids_prefers_opencode_live_catalog(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "_fetch_opencode_live_models",
        lambda provider, **kwargs: ["qwen3.6-plus-free", "paid-live"],
    )
    monkeypatch.setattr(
        "agent.models_dev.list_agentic_models",
        lambda provider: ["from-models-dev"],
    )

    out = model_catalog.provider_model_ids("opencode-zen")

    assert out[:2] == ["qwen3.6-plus-free", "paid-live"]
    assert "from-models-dev" in out
    assert "big-pickle" in out


def test_opencode_free_model_ids_filters_live_free_models(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "provider_model_ids",
        lambda provider, **kwargs: [
            "paid-model",
            "big-pickle",
            "qwen3.6-plus-free",
            "minimax-m2.5-free",
        ],
    )

    assert model_catalog.opencode_free_model_ids() == [
        "big-pickle",
        "qwen3.6-plus-free",
        "minimax-m2.5-free",
    ]


def test_normalize_fallback_entries_expands_opencode_free_sentinel(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda: ["big-pickle", "qwen3.6-plus-free"],
    )

    out = normalize_fallback_entries([
        {"provider": "opencode-zen", "model": "auto-free"},
    ])

    assert out == [
        {"provider": "opencode-zen", "model": "big-pickle"},
        {"provider": "opencode-zen", "model": "qwen3.6-plus-free"},
    ]


def test_normalize_fallback_entries_supports_virtual_provider(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda: ["deepseek-v4-flash-free"],
    )

    out = normalize_fallback_entries({"provider": "opencode-free"})

    assert out == [
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"},
    ]


def test_normalize_fallback_entries_dedupes_expanded_and_explicit(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda: ["big-pickle", "qwen3.6-plus-free"],
    )

    out = normalize_fallback_entries([
        {"provider": "opencode-zen", "model": "auto-free"},
        {"provider": "opencode-zen", "model": "big-pickle"},
        {"provider": "openai", "model": "gpt-5.4"},
    ])

    assert out == [
        {"provider": "opencode-zen", "model": "big-pickle"},
        {"provider": "opencode-zen", "model": "qwen3.6-plus-free"},
        {"provider": "openai", "model": "gpt-5.4"},
    ]


def test_resolve_config_model_id_expands_auto_free_primary(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda **kwargs: ["big-pickle", "qwen3.6-plus-free"],
    )

    assert model_catalog.resolve_config_model_id("opencode-zen", "auto-free") == "big-pickle"
    assert model_catalog.resolve_config_model_id("opencode-zen", "") == "big-pickle"


def test_resolve_config_model_id_preserves_explicit_model(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda **kwargs: ["big-pickle"],
    )

    assert model_catalog.resolve_config_model_id("opencode-zen", "glm-5-free") == "glm-5-free"


def test_get_default_model_for_provider_prefers_opencode_free(monkeypatch):
    monkeypatch.setattr(
        model_catalog,
        "opencode_free_model_ids",
        lambda **kwargs: ["deepseek-v4-flash-free"],
    )

    assert model_catalog.get_default_model_for_provider("opencode-zen") == "deepseek-v4-flash-free"
