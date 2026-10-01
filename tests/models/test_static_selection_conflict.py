"""Static session.create model/provider coherence belongs to the model domain."""
from __future__ import annotations

from hermes_cli.models_validate import static_model_provider_conflict as legacy
from models.catalog_static import static_provider_model_ids
from models.selection_conflict import static_model_provider_conflict


def test_legacy_validation_uses_exact_lower_owner():
    assert legacy is static_model_provider_conflict


def test_strict_oauth_rejects_definite_foreign_family():
    foreign = next(name for name in static_provider_model_ids("anthropic") if name.startswith("claude"))
    result = static_model_provider_conflict(foreign, "openai-codex")
    assert result is not None
    assert result["provider"] == "openai-codex"
    assert result["model"] == foreign
    assert result["suggestions"]


def test_own_family_and_custom_routes_stay_permissive():
    assert static_model_provider_conflict("gpt-unlisted-private", "openai-codex") is None
    assert static_model_provider_conflict("totally-new-private", "anthropic") is None
    assert static_model_provider_conflict("anything", "custom:new-endpoint") is None
    assert static_model_provider_conflict("anything", "openrouter") is None


def test_known_foreign_native_model_gets_useful_suggestion():
    foreign = next(name for name in static_provider_model_ids("openai-codex") if name.startswith("gpt"))
    result = static_model_provider_conflict(foreign, "anthropic")
    assert result is not None
    assert result["provider"] == "anthropic"
    assert isinstance(result["message"], str)
