"""Tests for gateway projection of lower-domain turn model facts."""

from gateway import model_runtime_facts as facts


def test_provider_default_uses_canonical_selection_with_preferred_cache(monkeypatch):
    monkeypatch.setattr(
        facts,
        "static_provider_model_ids",
        lambda _provider: ("expensive/model", "safe/model"),
    )
    monkeypatch.setattr(
        facts,
        "static_provider_default_preference",
        lambda _provider: "safe/model",
    )
    monkeypatch.setattr(
        "gateway.model_catalog_runtime.cached_default_model",
        lambda _provider: "safe/model",
    )

    assert facts.provider_default_model("openrouter") == "safe/model"


def test_provider_default_preserves_curated_order_without_preference(monkeypatch):
    monkeypatch.setattr(
        facts,
        "static_provider_model_ids",
        lambda _provider: ("first/model", "second/model"),
    )
    monkeypatch.setattr(
        facts,
        "static_provider_default_preference",
        lambda _provider: "",
    )

    assert facts.provider_default_model("openai-codex") == "first/model"


def test_runtime_model_normalization_delegates_to_canonical_identity(monkeypatch):
    seen = {}
    monkeypatch.setattr(facts, "is_aggregator", lambda _provider: False)
    monkeypatch.setattr(
        facts,
        "static_provider_model_ids",
        lambda _provider: ("vendor/model",),
    )

    def normalize(provider, model, *, known_ids):
        seen.update(provider=provider, model=model, known_ids=known_ids)
        return "vendor/model"

    monkeypatch.setattr(facts, "normalize_model_id", normalize)
    assert facts.normalize_runtime_model("openai", "model") == "vendor/model"
    assert seen == {
        "provider": "openai",
        "model": "model",
        "known_ids": ("vendor/model",),
    }


def test_aggregator_model_identity_is_not_rewritten(monkeypatch):
    monkeypatch.setattr(facts, "is_aggregator", lambda _provider: True)
    monkeypatch.setattr(
        facts,
        "normalize_model_id",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("aggregator model must pass through")
        ),
    )

    assert facts.normalize_runtime_model("openrouter", "vendor/model") == "vendor/model"
