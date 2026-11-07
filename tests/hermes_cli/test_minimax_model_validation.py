"""Tests for MiniMax model validation via static catalog (issues #12611, #12460, #12399, #12547).

MiniMax and MiniMax-CN providers don't expose /v1/models, so validate_requested_model()
must validate against the static catalog instead of probing the live API.
"""

from unittest.mock import patch

import pytest

from hermes_cli.models_validate import validate_requested_model


class TestMiniMaxModelValidation:
    """Test that validate_requested_model handles MiniMax providers correctly."""

    @pytest.fixture(autouse=True)
    def _isolate_minimax(self):
        """Ensure MiniMax catalog is used even if a live /v1/models endpoint exists."""
        # Simulate fetch_api_models returning None (i.e., /v1/models is unreachable),
        # proving that the catalog path is taken.
        probe_payload = {
            "models": None,
            "probed_url": "https://api.minimax.io/v1/models",
            "resolved_base_url": "https://api.minimax.io/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
            yield

    # -------------------------------------------------------------------------
    # Test 1: A known MiniMax model is accepted with recognized=True
    # -------------------------------------------------------------------------
    def test_valid_minimax_model_accepted(self):
        result = validate_requested_model("MiniMax-M2.7", "minimax")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    # -------------------------------------------------------------------------
    # Test 1b: Case-insensitive lookup matches catalog entries
    # -------------------------------------------------------------------------

    def test_valid_minimax_model_uppercase(self):
        result = validate_requested_model("MINIMAX-M2.7", "minimax")
        assert result["accepted"] is True
        assert result["recognized"] is True

    # -------------------------------------------------------------------------
    # Test 2: A near-match model on minimax-cn triggers a suggestion (not auto-correct)
    # -------------------------------------------------------------------------
    def test_near_match_minimax_cn_suggests_similar(self):
        # "MiniMax-M2.7-highspeed" is somewhat similar to "MiniMax-M2.7" (ratio ~0.71)
        # but below the 0.9 auto-correct cutoff. It should be accepted with a
        # recognized=False and a similar-models suggestion (ratio > 0.5).
        result = validate_requested_model("MiniMax-M2.7-highspeed", "minimax-cn")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is False
        # Should NOT auto-correct (ratio 0.71 < 0.9)
        assert "corrected_model" not in result
        # But should suggest similar models (ratio 0.71 > 0.5)
        assert "MiniMax-M2.7" in result["message"]

    # -------------------------------------------------------------------------
    # Test 3: A completely unknown model is accepted (not rejected) with a warning
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Test 4: Verify catalog path is used (probe_api_models returns None)
    # -------------------------------------------------------------------------



    # -------------------------------------------------------------------------
    # MiniMax-M3.1-Flash: curated entry + case-insensitive match, and the live
    # models_url so catalog discovery probes {base}/v1/models (not {base}/models).
    # -------------------------------------------------------------------------
    def test_minimax_m31_flash_recognized_lowercase(self):
        result = validate_requested_model("minimax-m3.1-flash", "minimax")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_minimax_m31_flash_recognized_canonical_case(self):
        result = validate_requested_model("MiniMax-M3.1-Flash", "minimax")
        assert result["accepted"] is True
        assert result["recognized"] is True

    def test_minimax_curated_catalog_lists_m31_flash(self):
        from hermes_cli.models_catalog_static import _MINIMAX_MODELS
        assert "MiniMax-M3.1-Flash" in _MINIMAX_MODELS
        # MiniMax-M3 stays first: it is the non-interactive fallback default (models[0]).
        assert _MINIMAX_MODELS[0] == "MiniMax-M3"

    def test_minimax_profiles_expose_live_models_url(self):
        from providers import get_provider_profile
        profile = get_provider_profile("minimax")
        assert profile.models_url == "https://api.minimax.io/anthropic/v1/models"
        cn = get_provider_profile("minimax-cn")
        assert cn.models_url == "https://api.minimaxi.com/anthropic/v1/models"
