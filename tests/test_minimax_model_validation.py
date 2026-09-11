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
    def test_minimax_uses_catalog_not_api_probe(self):
        """Ensure that when fetch_api_models returns None, the catalog is still checked."""
        # The _isolate_minimax fixture already patches fetch_api_models to return None.
        # If we reach the catalog path, MiniMax-M2.5 should be found and recognized.
        result = validate_requested_model("MiniMax-M2.5", "minimax")
        assert result["accepted"] is True
        assert result["recognized"] is True
        assert result["message"] is None


