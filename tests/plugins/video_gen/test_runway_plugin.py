"""Smoke tests for the RunwayML video generation provider."""

import os
import pytest

# Import the provider directly from the plugin
from plugins.video_gen.runway import (
    RunwayMLProvider,
    RUNWAYML_MODELS,
    DEFAULT_MODEL,
    RATIO_MAP,
    register,
)


@pytest.fixture
def provider():
    return RunwayMLProvider()


class TestRunwayMLRegistration:
    """Test provider registration and basic properties."""

    def test_name(self, provider):
        assert provider.name == "runwayml"

    def test_display_name(self, provider):
        assert provider.display_name == "RunwayML"

    def test_default_model(self, provider):
        assert provider.default_model() == "gen4.5"

    def test_list_models(self, provider):
        models = provider.list_models()
        assert len(models) >= 1
        model_ids = [m["id"] for m in models]
        assert "gen4.5" in model_ids


class TestRunwayMLCapabilities:
    """Test provider capabilities."""

    def test_capabilities(self, provider):
        caps = provider.capabilities()
        assert "text" in caps["modalities"]
        assert "image" in caps["modalities"]
        assert caps["min_duration"] == 2
        assert caps["max_duration"] == 10
        assert caps["supports_audio"] is False
        assert caps["supports_negative_prompt"] is False

    def test_ratio_map(self):
        """Test aspect ratio to pixel ratio mapping."""
        assert RATIO_MAP["16:9"] == "1280:720"
        assert RATIO_MAP["9:16"] == "720:1280"
        assert RATIO_MAP["1:1"] == "960:960"
        assert RATIO_MAP["4:3"] == "1104:832"


class TestRunwayMLAvailability:
    """Test API key availability detection."""

    def test_not_available_without_key(self, provider, monkeypatch):
        monkeypatch.delenv("RUNWAYML_API_KEY", raising=False)
        assert provider.is_available() is False

    def test_available_with_key(self, provider, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_KEY", "test-key-123")
        assert provider.is_available() is True


class TestRunwayMLSetupSchema:
    """Test setup schema generation."""

    def test_setup_schema(self, provider):
        schema = provider.get_setup_schema()
        assert schema["name"] == "RunwayML"
        assert schema["badge"] == "paid"
        env_vars = schema["env_vars"]
        assert len(env_vars) == 1
        assert env_vars[0]["key"] == "RUNWAYML_API_KEY"


class TestRunwayMLModelResolution:
    """Test model resolution logic."""

    def test_resolve_model_explicit(self, provider):
        assert provider._resolve_model("gen4.5") == "gen4.5"

    def test_resolve_model_env(self, provider, monkeypatch):
        monkeypatch.setenv("RUNWAYML_VIDEO_MODEL", "gen4.5")
        assert provider._resolve_model(None) == "gen4.5"

    def test_resolve_model_default(self, provider, monkeypatch):
        monkeypatch.delenv("RUNWAYML_VIDEO_MODEL", raising=False)
        assert provider._resolve_model(None) == "gen4.5"
