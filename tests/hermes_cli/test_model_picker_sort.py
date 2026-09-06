"""Picker ordering: curated head keeps its hand-picked order, live-only tail is alphabetized,
and providers whose curated list is a membership filter (HuggingFace) sort in full."""
from types import SimpleNamespace

from hermes_cli.models import _merge_picker_models, _profile_live_catalog
from hermes_cli.models_catalog_static import (
    _FULLY_ALPHABETIZE_PICKER_PROVIDERS,
    _LIVE_FIRST_PICKER_PROVIDERS,
)


def test_curated_head_kept_live_tail_alphabetized():
    curated = ["z-newest", "m-older"]  # deliberate newest-first order
    live = ["m-older", "delta", "Alpha", "charlie"]
    assert _merge_picker_models("zai", curated, live) == ["z-newest", "m-older", "Alpha", "charlie", "delta"]


def test_no_curated_list_is_alphabetized_outright():
    assert _merge_picker_models("ollama-cloud", [], ["qwen", "Gemma", "llama"]) == ["Gemma", "llama", "qwen"]


def test_no_live_returns_curated_verbatim():
    assert _merge_picker_models("zai", ["b", "a"], []) == ["b", "a"]


def test_live_first_provider_keeps_raw_live_order():
    provider = next(iter(_LIVE_FIRST_PICKER_PROVIDERS))
    live = ["zeta", "alpha", "mid"]
    curated = ["stale-curated", "alpha"]
    # live leads verbatim, curated-only entries append in their own order, nothing sorted
    assert _merge_picker_models(provider, curated, live) == ["zeta", "alpha", "mid", "stale-curated"]


def test_fully_alphabetized_provider_sorts_curated_head_too():
    assert "huggingface" in _FULLY_ALPHABETIZE_PICKER_PROVIDERS
    curated = ["zephyr", "mistral"]
    live = ["qwen", "aya", "mistral"]
    assert _merge_picker_models("huggingface", curated, live) == ["aya", "mistral", "qwen", "zephyr"]


def test_dedup_is_case_insensitive_and_keeps_curated_spelling():
    assert _merge_picker_models("zai", ["GLM-5"], ["glm-5", "b-model"]) == ["GLM-5", "b-model"]


def test_profile_live_catalog_routes_through_merge(monkeypatch):
    profile = SimpleNamespace(
        auth_type="api_key",
        base_url="https://example.invalid/v1",
        fallback_models=["curated-two", "curated-one"],
        fetch_models=lambda api_key, base_url: ["zulu", "curated-one", "alpha"],
    )
    monkeypatch.setattr("providers.get_provider_profile", lambda name: profile)
    monkeypatch.setattr("hermes_cli.models._api_key_credentials", lambda name: ("key", None))
    monkeypatch.setattr("hermes_cli.models._PROVIDER_MODELS", {})
    assert _profile_live_catalog("some-plugin-provider") == ["curated-two", "curated-one", "alpha", "zulu"]
