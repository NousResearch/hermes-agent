
import application_model_pricing
import application_model_selection_defaults as defaults
from models import catalog_static as static


def test_preferred_silent_default_uses_cached_label(monkeypatch):
    monkeypatch.setattr(
        "models.catalog_runtime.cached_default_model",
        lambda _path, provider: "provider/default",
    )
    assert defaults.preferred_silent_default_model("provider") == "provider/default"


def test_preferred_silent_default_falls_back_to_curated_constant(monkeypatch):
    monkeypatch.setattr(
        "models.catalog_runtime.cached_default_model",
        lambda _path, provider: None,
    )
    assert (
        defaults.preferred_silent_default_model("openrouter")
        == static.PREFERRED_SILENT_DEFAULT_MODEL
    )


def test_provider_default_uses_first_static_model_for_ordinary_provider(monkeypatch):
    monkeypatch.setattr(
        static,
        "_PROVIDER_MODELS",
        {"provider-x": ["model-a", "model-b"]},
    )
    monkeypatch.setattr(static, "_SILENT_DEFAULT_PROVIDERS", frozenset())
    selection = defaults.select_provider_default("provider-x")
    assert defaults.selected_model_id(selection) == "model-a"


def test_silent_provider_trusts_preferred_without_static_catalog(monkeypatch):
    monkeypatch.setattr(static, "_PROVIDER_MODELS", {"provider-x": []})
    monkeypatch.setattr(
        static,
        "_SILENT_DEFAULT_PROVIDERS",
        frozenset({"provider-x"}),
    )
    monkeypatch.setattr(
        defaults,
        "preferred_silent_default_model",
        lambda provider="": "model-default",
    )
    selection = defaults.select_provider_default("provider-x")
    assert defaults.selected_model_id(selection) == "model-default"


def test_silent_default_uses_candidate_order_when_label_missing(monkeypatch):
    monkeypatch.setattr(
        defaults,
        "preferred_silent_default_model",
        lambda provider="": "not/listed",
    )
    selection = defaults.select_silent_default(
        "provider-x",
        ["model-b", "model-a"],
    )
    assert defaults.selected_model_id(selection) == "model-b"


def _patch_nous_default_facts(monkeypatch, *, allowed):
    from hermes_cli import models as catalog
    import application_model_pricing as pricing

    monkeypatch.setattr(
        catalog,
        "get_curated_nous_model_ids",
        lambda: ["vendor/blocked", "vendor/allowed"],
    )
    monkeypatch.setattr(catalog, "check_nous_free_tier", lambda **_kwargs: False)
    monkeypatch.setattr(
        "application_nous_recommendations.fetch_recommended_models",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(application_model_pricing, "get_pricing_for_provider", lambda _provider: {})
    monkeypatch.setattr(application_model_pricing, "nous_policy_allowed_ids", lambda: allowed)
    monkeypatch.setattr(
        "hermes_cli.auth.get_provider_auth_state",
        lambda _provider: {},
    )
    monkeypatch.setattr(
        defaults,
        "preferred_silent_default_model",
        lambda provider="": "not/listed",
    )


def test_nous_default_filters_org_policy_before_selection(monkeypatch):
    _patch_nous_default_facts(monkeypatch, allowed={"vendor/allowed"})
    selection, free_tier = defaults.select_nous_recommended_default()
    assert free_tier is False
    assert defaults.selected_model_id(selection) == "vendor/allowed"


def test_nous_default_preserves_unrestricted_catalog_order(monkeypatch):
    _patch_nous_default_facts(monkeypatch, allowed=None)
    selection, free_tier = defaults.select_nous_recommended_default()
    assert free_tier is False
    assert defaults.selected_model_id(selection) == "vendor/blocked"
