"""Regression test for MoA virtual provider validation (issue #60512).

MoA is a local virtual provider, not a remote model. When a user switches
to "moa" via /model or the model picker, the validation logic should skip
API probing and accept it immediately.

The bug occurred when a custom provider was configured and the user tried
to use MoA with it. The old code checked `provider == "moa"`, but in this
case the provider is "custom" and the model_name is "moa", so the check
failed and the code probed the remote endpoint for a "moa" model.
"""

from unittest.mock import patch

import pytest

from hermes_cli.models import validate_requested_model


class TestMoAVirtualProviderValidation:
    """Test that validate_requested_model handles MoA virtual provider correctly."""

    @pytest.fixture(autouse=True)
    def _isolate_from_remote_probes(self):
        """Ensure we don't hit the network during tests."""
        # Simulate fetch_api_models returning an empty list,
        # proving that we accept MoA without probing the endpoint.
        probe_payload = {
            "models": [],
            "probed_url": "https://custom.example.com/v1/models",
            "resolved_base_url": "https://custom.example.com/v1",
            "suggested_base_url": None,
            "used_fallback": False,
        }
        with patch("hermes_cli.models.fetch_api_models", return_value=None), \
             patch("hermes_cli.models.probe_api_models", return_value=probe_payload):
            yield

    def test_moa_model_name_accepted_without_probe(self):
        """When model_name is 'moa', accept immediately without API probe."""
        result = validate_requested_model("moa", "custom")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_moa_model_name_case_insensitive(self):
        """MoA model name is case-insensitive."""
        result = validate_requested_model("MOA", "custom")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_moa_model_name_mixed_case(self):
        """MoA model name with mixed case should also work."""
        result = validate_requested_model("MoA", "custom")
        assert result["accepted"] is True
        assert result["persist"] is True
        assert result["recognized"] is True
        assert result["message"] is None

    def test_moa_model_name_with_spaces_rejected(self):
        """Model names with spaces are still rejected, even for MoA."""
        result = validate_requested_model("moa ", "custom")
        assert result["accepted"] is False
        assert result["persist"] is False
        assert result["message"] == "Model names cannot contain spaces."

    def test_moa_with_different_providers(self):
        """MoA model name should be accepted regardless of provider."""
        # Test with various providers
        for provider in ["custom", "custom:my-endpoint", "openrouter"]:
            result = validate_requested_model("moa", provider)
            assert result["accepted"] is True, f"Provider {provider} rejected MoA"
            assert result["persist"] is True
            assert result["recognized"] is True
            assert result["message"] is None

    def test_non_moa_models_still_validated(self):
        """Non-MoA models should still go through normal validation."""
        # This test verifies that our fix doesn't break normal model validation.
        # If the model is not "moa", it should be rejected if not in the catalog.
        result = validate_requested_model("not-a-real-model", "custom")
        # Since we mocked the API to return an empty list, this should be
        # accepted with a warning (the remote endpoint behavior).
        assert result["accepted"] is True
        assert result["recognized"] is False
        assert "was not found in this custom endpoint's model listing" in result["message"]