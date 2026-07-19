"""Test for issue #59560: custom_providers.models list should not be ignored when live /models discovery returns a different list."""

import pytest
from unittest.mock import patch, MagicMock

def test_configured_models_merged_with_live_models():
    """
    When custom_providers have configured models AND live discovery returns models,
    the result should MERGE them (not replace configured with live).
    """
    from hermes_cli.model_switch import list_authenticated_providers
    
    configured_models = [
        {"id": "configured-model-1", "name": "Configured Model 1"},
        {"id": "configured-model-2", "name": "Configured Model 2"}
    ]
    # fetch_api_models returns list[str] — model slugs
    live_models = ["live-model-1", "live-model-2"]
    
    with patch('hermes_cli.models.fetch_api_models', return_value=live_models):
        providers = list_authenticated_providers(
            custom_providers=[
                {
                    "name": "Test Provider",
                    "api": "http://localhost:9999/v1",
                    "api_key": "sk-test",
                    "models": configured_models,
                    "discover_models": True
                }
            ]
        )
    
    test_provider = None
    for p in providers:
        if p.get("name") == "Test Provider":
            test_provider = p
            break
    
    assert test_provider is not None, "Test provider not found in results"
    
    result_models = test_provider.get("models", [])
    # After merge, models_list contains strings (declared_model_ids) + live strings
    result_slugs = {m if isinstance(m, str) else m.get("slug", m.get("id", "")) for m in result_models}
    
    assert "configured-model-1" in result_slugs, "Configured model 1 missing"
    assert "configured-model-2" in result_slugs, "Configured model 2 missing"
    assert "live-model-1" in result_slugs, "Live model 1 missing"
    assert "live-model-2" in result_slugs, "Live model 2 missing"
    assert len(result_models) == 4, f"Expected 4 merged models, got {len(result_models)}"


def test_configured_models_preserved_when_live_discovery_fails():
    """When live discovery fails, configured models should still be present."""
    from hermes_cli.model_switch import list_authenticated_providers
    
    configured_models = [
        {"id": "configured-only", "name": "Configured Only"}
    ]
    
    with patch('hermes_cli.models.fetch_api_models', side_effect=Exception("Network error")):
        providers = list_authenticated_providers(
            custom_providers=[
                {
                    "name": "Test Provider 2",
                    "api": "http://localhost:9998/v1",
                    "api_key": "sk-test",
                    "models": configured_models,
                    "discover_models": True
                }
            ]
        )
    
    test_provider = None
    for p in providers:
        if p.get("name") == "Test Provider 2":
            test_provider = p
            break
    
    assert test_provider is not None
    result_models = test_provider.get("models", [])
    assert len(result_models) == 1, "Configured models should be preserved when discovery fails"
    # _declared_model_ids extracts "id" from dict, so result is string
    assert result_models[0] == "configured-only"


def test_no_duplicates_when_live_returns_same_model():
    """
    If a model exists in both configured and live, it should not be duplicated.
    """
    from hermes_cli.model_switch import list_authenticated_providers
    
    configured_models = [
        {"id": "shared-model", "name": "Shared Model"},
        {"id": "only-configured", "name": "Only Configured"}
    ]
    live_models = ["shared-model", "only-live"]
    
    with patch('hermes_cli.models.fetch_api_models', return_value=live_models):
        providers = list_authenticated_providers(
            custom_providers=[
                {
                    "name": "Test Provider 3",
                    "api": "http://localhost:9997/v1",
                    "api_key": "sk-test",
                    "models": configured_models,
                    "discover_models": True
                }
            ]
        )
    
    test_provider = None
    for p in providers:
        if p.get("name") == "Test Provider 3":
            test_provider = p
            break
    
    assert test_provider is not None
    result_models = test_provider.get("models", [])
    result_slugs = [m if isinstance(m, str) else m.get("slug", m.get("id", "")) for m in result_models]
    
    # shared-model should appear only once
    assert result_slugs.count("shared-model") == 1, "Duplicate model found"
    assert len(result_models) == 3, f"Expected 3 unique models, got {len(result_models)}: {result_slugs}"
