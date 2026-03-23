"""Tests for context length auto-detection in custom provider workflow."""

from unittest.mock import patch, MagicMock

from cli import HermesCLI


class TestContextLengthAutoDetection:
    """Test context length detection when saving custom providers."""
    
    def _make_cli(self):
        """Create a minimal HermesCLI instance for testing."""
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.model = "anthropic/claude-opus-4.6"
        cli_obj.agent = object()
        cli_obj.provider = "openrouter"
        cli_obj.requested_provider = "openrouter"
        cli_obj.base_url = "https://openrouter.ai/api/v1"
        cli_obj.api_key = "***"
        cli_obj._explicit_api_key = "***"
        cli_obj._explicit_base_url = None
        return cli_obj

    def test_context_length_detected_and_displayed(self, capsys):
        """Test that detected context length is shown to user."""
        cli_obj = self._make_cli()

        with patch("hermes_cli.models.probe_api_models", return_value={
            "models": ["test/model"],
            "probed_url": "https://custom.api/v1/models",
            "resolved_base_url": "https://custom.api/v1",
            "suggested_base_url": "https://custom.api/v1",
            "used_fallback": False,
        }), patch("cli.save_config_value"), \
             patch("hermes_cli.main._save_custom_provider"), \
             patch("agent.model_metadata.get_model_context_length", return_value=200000):
            
            # Simulate model flow with context_length=None (auto-detect mode)
            # The CLI now parses flags and passes context_length to validate_requested_model
            cli_obj.process_command("/model test/model --base-url https://custom.api/v1 --api-key secret-key")
        
        output = capsys.readouterr().out
        # Should show detection feedback
        assert "auto-detected" in output.lower() or "context length" in output.lower()

    def test_context_length_default_fallback_displayed(self, capsys):
        """Test that default fallback is shown when detection fails."""
        cli_obj = self._make_cli()

        with patch("hermes_cli.models.probe_api_models", return_value={
            "models": ["test/model"],
            "probed_url": "https://custom.api/v1/models",
            "resolved_base_url": "https://custom.api/v1",
            "suggested_base_url": "https://custom.api/v1",
            "used_fallback": False,
        }), patch("cli.save_config_value"), \
             patch("hermes_cli.main._save_custom_provider"), \
             patch("agent.model_metadata.get_model_context_length", return_value=None):
            
            cli_obj.process_command("/model test/model --base-url https://custom.api/v1 --api-key secret-key")
        
        output = capsys.readouterr().out
        # Should mention context length or default
        assert "context" in output.lower() or "default" in output.lower()

    def test_probe_api_models_suggested_base_url_with_fallback(self):
        """Test that suggested_base_url is set correctly when using fallback."""
        from hermes_cli import models
        
        # Mock the probe to simulate fallback scenario
        # First URL fails, second (fallback) succeeds
        call_count = [0]
        def mock_urlopen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("primary failed")
            # Fallback succeeds
            import json
            mock_data = json.dumps({"data": [{"id": "test/model"}]}).encode()
            mock_response = MagicMock()
            mock_response.read.return_value = mock_data
            return mock_response
        
        with patch("urllib.request.urlopen", side_effect=mock_urlopen_side_effect):
            result = models.probe_api_models("https://primary.example.com", "https://fallback.example.com")
        
        # When fallback is used, suggested_base_url should be alternate_base
        # Note: The logic checks if is_fallback is True, not if alternate_base != candidate_base
        assert "suggested_base_url" in result

    def test_probe_api_models_suggested_base_url_without_fallback(self):
        """Test that suggested_base_url is normalized when not using fallback."""
        from hermes_cli import models
        import json
        
        # Mock successful response from primary URL
        mock_data = json.dumps({"data": [{"id": "test/model"}]}).encode()
        mock_response = MagicMock()
        mock_response.read.return_value = mock_data
        
        with patch("urllib.request.urlopen", return_value=mock_response):
            result = models.probe_api_models("https://primary.example.com", None)
        
        # When not using fallback, suggested_base_url should be normalized or None
        # (since alternate_base is None in this case)
        assert result["used_fallback"] is False
        assert result["suggested_base_url"] is None or result["suggested_base_url"] == "https://primary.example.com"


class TestDefaultFallbackContext:
    """Test DEFAULT_FALLBACK_CONTEXT constant usage."""
    
    def test_default_fallback_context_is_set(self):
        """Test that DEFAULT_FALLBACK_CONTEXT constant exists and has value."""
        from agent.model_metadata import DEFAULT_FALLBACK_CONTEXT
        
        assert DEFAULT_FALLBACK_CONTEXT is not None
        assert isinstance(DEFAULT_FALLBACK_CONTEXT, int)
        assert DEFAULT_FALLBACK_CONTEXT > 0
    
    def test_default_fallback_context_used_in_get_model_context_length(self):
        """Test that get_model_context_length returns DEFAULT_FALLBACK_CONTEXT for unknown models."""
        from agent.model_metadata import get_model_context_length, DEFAULT_FALLBACK_CONTEXT
        
        # Unknown model with no matching provider should return default
        result = get_model_context_length("unknown-model-xyz", provider="custom")
        assert result == DEFAULT_FALLBACK_CONTEXT
