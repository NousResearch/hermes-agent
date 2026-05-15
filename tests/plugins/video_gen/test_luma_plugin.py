"""Smoke tests for the Luma Dream Machine video generation provider."""

import os
import pytest

# Import the provider directly from the plugin
from plugins.video_gen.luma import (
    LumaProvider,
    LUMAAI_MODELS,
    DEFAULT_MODEL,
    VALID_ASPECT_RATIOS,
    VALID_DURATIONS,
    VALID_RESOLUTIONS,
    register,
)


@pytest.fixture
def provider():
    return LumaProvider()


class TestLumaRegistration:
    """Test provider registration and basic properties."""

    def test_name(self, provider):
        assert provider.name == "luma"

    def test_display_name(self, provider):
        assert provider.display_name == "Luma Dream Machine"

    def test_default_model(self, provider):
        assert provider.default_model() == "ray-2"

    def test_list_models(self, provider):
        models = provider.list_models()
        assert len(models) >= 2
        model_ids = [m["id"] for m in models]
        assert "ray-2" in model_ids
        assert "ray-flash-2" in model_ids


class TestLumaCapabilities:
    """Test provider capabilities."""

    def test_capabilities(self, provider):
        caps = provider.capabilities()
        assert "text" in caps["modalities"]
        assert "image" in caps["modalities"]
        assert caps["min_duration"] == 5
        assert caps["max_duration"] == 9
        assert caps["supports_audio"] is False
        assert caps["supports_negative_prompt"] is False

    def test_valid_aspect_ratios(self):
        assert "16:9" in VALID_ASPECT_RATIOS
        assert "9:16" in VALID_ASPECT_RATIOS
        assert "1:1" in VALID_ASPECT_RATIOS

    def test_valid_durations(self):
        assert "5s" in VALID_DURATIONS
        assert "9s" in VALID_DURATIONS

    def test_valid_resolutions(self):
        assert "720p" in VALID_RESOLUTIONS
        assert "1080p" in VALID_RESOLUTIONS


class TestLumaAvailability:
    """Test API key availability detection."""

    def test_not_available_without_key(self, provider, monkeypatch):
        monkeypatch.delenv("LUMAAI_API_KEY", raising=False)
        assert provider.is_available() is False

    def test_available_with_key(self, provider, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "test-key-123")
        assert provider.is_available() is True


class TestLumaSetupSchema:
    """Test setup schema generation."""

    def test_setup_schema(self, provider):
        schema = provider.get_setup_schema()
        assert schema["name"] == "Luma Dream Machine"
        assert schema["badge"] == "paid"
        env_vars = schema["env_vars"]
        assert len(env_vars) == 1
        assert env_vars[0]["key"] == "LUMAAI_API_KEY"


class TestLumaModelResolution:
    """Test model resolution logic."""

    def test_resolve_model_explicit(self, provider):
        assert provider._resolve_model("ray-2") == "ray-2"
        assert provider._resolve_model("ray-flash-2") == "ray-flash-2"

    def test_resolve_model_env(self, provider, monkeypatch):
        monkeypatch.setenv("LUMAAI_VIDEO_MODEL", "ray-flash-2")
        assert provider._resolve_model(None) == "ray-flash-2"

    def test_resolve_model_default(self, provider, monkeypatch):
        monkeypatch.delenv("LUMAAI_VIDEO_MODEL", raising=False)
        assert provider._resolve_model(None) == "ray-2"
