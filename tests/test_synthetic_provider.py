"""Tests for Synthetic (synthetic.new) provider integration."""
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS
from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS, _API_KEY_PROVIDER_VISION_MODELS


class TestSyntheticProviderRegistry:
    """Test Synthetic is properly registered as a provider."""
    
    def test_synthetic_in_provider_registry(self):
        """Synthetic should be in PROVIDER_REGISTRY with correct config."""
        assert "synthetic" in PROVIDER_REGISTRY
        config = PROVIDER_REGISTRY["synthetic"]
        assert config.id == "synthetic"
        assert config.name == "Synthetic"
        assert config.auth_type == "api_key"
        assert config.inference_base_url == "https://api.synthetic.new/openai/v1"
        assert "SYNTHETIC_API_KEY" in config.api_key_env_vars

    def test_synthetic_in_provider_models(self):
        """Synthetic should have a model catalog."""
        assert "synthetic" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["synthetic"]
        # Should have at least 15 models from official catalog
        assert len(models) >= 15
        # Check key models are present
        assert "hf:moonshotai/Kimi-K2.5" in models
        assert "hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4" in models
        # Ensure Synthetic metadata entries use hf: prefix to avoid collisions
        assert "hf:zai-org/GLM-5" in DEFAULT_CONTEXT_LENGTHS
        assert DEFAULT_CONTEXT_LENGTHS["hf:zai-org/GLM-5"] == 192000
        # Check original GLM-5 still has its correct context
        assert "zai-org/GLM-5" in DEFAULT_CONTEXT_LENGTHS
        assert DEFAULT_CONTEXT_LENGTHS["zai-org/GLM-5"] == 202752

    def test_synthetic_in_provider_labels(self):
        """Synthetic should have a display label."""
        assert "synthetic" in _PROVIDER_LABELS
        assert _PROVIDER_LABELS["synthetic"] == "Synthetic"


class TestSyntheticAuxiliaryModels:
    """Test auxiliary model routing for Synthetic."""
    
    def test_default_aux_model(self):
        """Default aux model should be Nemotron (fast/cheap)."""
        assert "synthetic" in _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS["synthetic"] == "hf:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"

    def test_vision_aux_model(self):
        """Vision aux model should be Kimi (multimodal)."""
        assert "synthetic" in _API_KEY_PROVIDER_VISION_MODELS
        assert _API_KEY_PROVIDER_VISION_MODELS["synthetic"] == "hf:moonshotai/Kimi-K2.5"


class TestSyntheticModelIds:
    """Test that Synthetic model IDs follow correct format."""
    
    def test_all_models_have_hf_prefix(self):
        """All Synthetic models should use hf: prefix."""
        models = _PROVIDER_MODELS.get("synthetic", [])
        for model in models:
            assert model.startswith("hf:"), f"Model {model} missing hf: prefix"
