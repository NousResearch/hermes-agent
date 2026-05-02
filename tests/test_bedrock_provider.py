"""Tests for Amazon Bedrock provider integration."""

import os
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, get_auth_status
from hermes_cli.runtime_provider import resolve_runtime_provider


class TestBedrockProviderRegistry:
    """Test that Bedrock is registered correctly."""

    def test_bedrock_in_registry(self):
        """Bedrock should be in the provider registry."""
        assert "bedrock" in PROVIDER_REGISTRY

    def test_bedrock_config(self):
        """Bedrock provider config should be correct."""
        config = PROVIDER_REGISTRY["bedrock"]
        assert config.id == "bedrock"
        assert config.name == "AWS Bedrock"
        assert config.auth_type == "aws_sdk"
        assert config.inference_base_url == "https://bedrock-runtime.us-east-1.amazonaws.com"


class TestBedrockAliases:
    """Test provider aliases that map to bedrock."""

    def test_aws_alias(self):
        """'aws' should be an alias for 'bedrock'."""
        from hermes_cli.auth import resolve_provider

        # Mock AWS credentials so auto-resolution picks bedrock
        with patch("agent.bedrock_adapter.has_aws_credentials", return_value=True):
            with patch("hermes_cli.auth.os.getenv") as mock_getenv:
                # No OpenRouter credentials
                mock_getenv.return_value = ""
                result = resolve_provider("aws")
                assert result == "bedrock"

    def test_aws_bedrock_alias(self):
        """'aws-bedrock' should be an alias for 'bedrock'."""
        from hermes_cli.auth import resolve_provider

        with patch("agent.bedrock_adapter.has_aws_credentials", return_value=True):
            with patch("hermes_cli.auth.os.getenv", return_value=""):
                result = resolve_provider("aws-bedrock")
                assert result == "bedrock"


class TestBedrockCredentialDetection:
    """Test credential detection for Bedrock."""

    @patch("agent.bedrock_adapter.has_aws_credentials")
    def test_bedrock_auto_detection(self, mock_has_creds):
        """Bedrock should be auto-detected when AWS credentials exist."""
        from hermes_cli.auth import resolve_provider

        mock_has_creds.return_value = True
        with patch("hermes_cli.auth.os.getenv", return_value=""):
            result = resolve_provider("auto")
            assert result == "bedrock"

    @patch("agent.bedrock_adapter.has_aws_credentials")
    def test_bedrock_explicit_selection(self, mock_has_creds):
        """Bedrock should fail gracefully when no AWS credentials exist."""
        from hermes_cli.auth import resolve_provider, AuthError

        mock_has_creds.return_value = False
        with patch("hermes_cli.auth.os.getenv", return_value=""):
            # Explicit selection should fail if no AWS credentials
            with pytest.raises(AuthError):
                resolve_provider("bedrock")


class TestBedrockAuthStatus:
    """Test auth status for Bedrock provider."""

    @patch("agent.bedrock_adapter.has_aws_credentials")
    def test_bedrock_logged_in_status(self, mock_has_creds):
        """Bedrock auth status should reflect AWS credential availability."""
        mock_has_creds.return_value = True
        status = get_auth_status("bedrock")
        assert status["logged_in"] is True
        assert status["provider"] == "bedrock"

    @patch("agent.bedrock_adapter.has_aws_credentials")
    def test_bedrock_logged_out_status(self, mock_has_creds):
        """Bedrock auth status should be logged_out when no AWS credentials."""
        mock_has_creds.return_value = False
        status = get_auth_status("bedrock")
        assert status["logged_in"] is False
        assert status["provider"] == "bedrock"


class TestBedrockRuntimeResolution:
    """Test runtime resolution for Bedrock."""

    @patch("agent.bedrock_adapter.resolve_bedrock_region")
    @patch("agent.bedrock_adapter.has_aws_credentials")
    @patch("hermes_cli.config.load_config")
    def test_bedrock_runtime_resolution(
        self, mock_load_config, mock_has_creds, mock_resolve_region
    ):
        """Runtime provider resolution should handle Bedrock."""
        mock_has_creds.return_value = True
        mock_resolve_region.return_value = "us-east-1"
        mock_load_config.return_value = {
            "model": {"default": "us.anthropic.claude-sonnet-4-6", "provider": "bedrock"},
            "bedrock": {},
        }

        # Mock credential resolution to avoid actual AWS calls
        with patch(
            "agent.bedrock_adapter.get_bedrock_client_with_region"
        ) as mock_client:
            with patch("agent.bedrock_adapter.is_anthropic_bedrock_model", return_value=True):
                runtime = resolve_runtime_provider(
                    requested="bedrock", target_model="us.anthropic.claude-sonnet-4-6"
                )
                assert runtime["provider"] == "bedrock"
                assert runtime["base_url"] == "https://bedrock-runtime.us-east-1.amazonaws.com"

    @patch("agent.bedrock_adapter.resolve_bedrock_region")
    @patch("agent.bedrock_adapter.has_aws_credentials")
    @patch("hermes_cli.config.load_config")
    def test_bedrock_region_env_override(
        self, mock_load_config, mock_has_creds, mock_resolve_region
    ):
        """HERMES_BEDROCK_REGION env var should override region."""
        mock_has_creds.return_value = True
        mock_resolve_region.return_value = "us-west-2"
        mock_load_config.return_value = {"model": {"provider": "bedrock"}, "bedrock": {}}

        with patch.dict(os.environ, {"HERMES_BEDROCK_REGION": "eu-west-1"}):
            with patch("agent.bedrock_adapter.is_anthropic_bedrock_model", return_value=False):
                with patch("agent.bedrock_adapter.get_bedrock_client_with_region"):
                    runtime = resolve_runtime_provider(requested="bedrock")
                    # The region from env var should be used in the base URL
                    assert "eu-west-1" in runtime.get("base_url", "")


class TestBedrockConfigYaml:
    """Test Bedrock configuration parsing from config.yaml."""

    @patch("agent.bedrock_adapter.has_aws_credentials", return_value=True)
    @patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1")
    @patch("hermes_cli.config.load_config")
    def test_bedrock_config_parsing(
        self, mock_load_config, mock_resolve_region, mock_has_creds
    ):
        """Bedrock config from cli-config.yaml should be parsed correctly."""
        mock_load_config.return_value = {
            "model": {
                "default": "us.anthropic.claude-opus-4-1-20250514",
                "provider": "bedrock",
            },
            "bedrock": {"region": "eu-west-1", "model": "us.anthropic.claude-opus-4-1-20250514"},
        }

        with patch("agent.bedrock_adapter.is_anthropic_bedrock_model", return_value=True):
            with patch("agent.bedrock_adapter.get_bedrock_client_with_region"):
                runtime = resolve_runtime_provider(requested="bedrock")
                assert runtime["provider"] == "bedrock"
                # Region from config should take priority
                assert "eu-west-1" in runtime.get("base_url", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
