"""Integration tests for the AWS Bedrock provider wiring.

Verifies that the Bedrock provider is correctly registered in the
provider registry, model catalog, and runtime resolution pipeline.
These tests do NOT require AWS credentials or boto3 — all AWS calls
are mocked.

Note: Tests that import ``hermes_cli.auth`` or ``hermes_cli.runtime_provider``
require Python 3.10+ due to ``str | None`` type syntax in the import chain.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestProviderRegistry:
    """Verify Bedrock is registered in PROVIDER_REGISTRY."""

    def test_bedrock_in_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "bedrock" in PROVIDER_REGISTRY

    def test_bedrock_auth_type_is_aws_sdk(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["bedrock"]
        assert pconfig.auth_type == "aws_sdk"

    def test_bedrock_has_no_api_key_env_vars(self):
        """Bedrock uses the AWS SDK credential chain, not API keys."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["bedrock"]
        assert pconfig.api_key_env_vars == ()

    def test_bedrock_base_url_env_var(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["bedrock"]
        assert pconfig.base_url_env_var == "BEDROCK_BASE_URL"


class TestProviderAliases:
    """Verify Bedrock aliases resolve correctly."""

    def test_aws_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("aws") == "bedrock"

    def test_aws_bedrock_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("aws-bedrock") == "bedrock"

    def test_amazon_bedrock_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("amazon-bedrock") == "bedrock"

    def test_amazon_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("amazon") == "bedrock"


class TestProviderLabels:
    """Verify Bedrock appears in provider labels."""

    def test_bedrock_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS.get("bedrock") == "AWS Bedrock"


class TestModelCatalog:
    """Verify Bedrock has a static model fallback list."""

    def test_bedrock_has_curated_models(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("bedrock", [])
        assert len(models) > 0

    def test_bedrock_models_include_claude(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("bedrock", [])
        claude_models = [m for m in models if "anthropic.claude" in m]
        assert len(claude_models) > 0

    def test_bedrock_models_include_nova(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("bedrock", [])
        nova_models = [m for m in models if "amazon.nova" in m]
        assert len(nova_models) > 0


class TestResolveProvider:
    """Verify resolve_provider() handles bedrock correctly."""

    def test_explicit_bedrock_resolves(self, monkeypatch):
        """When user explicitly requests 'bedrock', it should resolve."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        # bedrock is in the registry, so resolve_provider should return it
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("bedrock")
        assert result == "bedrock"

    def test_aws_alias_resolves_to_bedrock(self):
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("aws")
        assert result == "bedrock"

    def test_amazon_bedrock_alias_resolves(self):
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("amazon-bedrock")
        assert result == "bedrock"

    def test_auto_detect_with_aws_credentials(self, monkeypatch):
        """When AWS credentials are present and no other provider is configured,
        auto-detect should find bedrock."""
        from hermes_cli.auth import resolve_provider

        # Clear all other provider env vars
        for var in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_TOKEN", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY"]:
            monkeypatch.delenv(var, raising=False)

        # Set AWS credentials
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        # Mock the auth store to have no active provider
        with patch("hermes_cli.auth._load_auth_store", return_value={}):
            result = resolve_provider("auto")
        assert result == "bedrock"


class TestRuntimeProvider:
    """Verify resolve_runtime_provider() handles bedrock correctly."""

    def test_bedrock_runtime_resolution(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")

        # Mock resolve_provider to return bedrock
        with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "bedrock"}):
            result = resolve_runtime_provider(requested="bedrock")

        assert result["provider"] == "bedrock"
        assert result["api_mode"] == "bedrock_converse"
        assert result["region"] == "eu-west-1"
        assert "bedrock-runtime.eu-west-1.amazonaws.com" in result["base_url"]
        assert result["api_key"] == "aws-sdk"

    def test_bedrock_runtime_default_region(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        monkeypatch.setenv("AWS_PROFILE", "default")
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

        with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "bedrock"}):
            result = resolve_runtime_provider(requested="bedrock")

        assert result["region"] == "us-east-1"

    def test_bedrock_runtime_no_credentials_raises_on_auto_detect(self, monkeypatch):
        """When bedrock is auto-detected (not explicitly requested) and no
        credentials are found, runtime resolution should raise AuthError."""
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.auth import AuthError

        # Clear all AWS env vars
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE",
                     "AWS_BEARER_TOKEN_BEDROCK", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
                     "AWS_WEB_IDENTITY_TOKEN_FILE"]:
            monkeypatch.delenv(var, raising=False)

        # Mock both the provider resolution and boto3's credential chain
        mock_session = MagicMock()
        mock_session.get_credentials.return_value = None
        with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "bedrock"}), \
             patch("hermes_cli.runtime_provider.resolve_requested_provider", return_value="auto"), \
             patch.dict("sys.modules", {"botocore": MagicMock(), "botocore.session": MagicMock()}):
            import botocore.session as _bs
            _bs.get_session = MagicMock(return_value=mock_session)
            with pytest.raises(AuthError, match="No AWS credentials"):
                resolve_runtime_provider(requested="auto")

    def test_bedrock_runtime_explicit_skips_credential_check(self, monkeypatch):
        """When user explicitly requests bedrock, trust boto3's credential chain
        even if env-var detection finds nothing (covers IMDS, SSO, etc.)."""
        from hermes_cli.runtime_provider import resolve_runtime_provider

        # No AWS env vars set — but explicit bedrock request should not raise
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_PROFILE",
                     "AWS_BEARER_TOKEN_BEDROCK"]:
            monkeypatch.delenv(var, raising=False)

        with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
             patch("hermes_cli.runtime_provider._get_model_config", return_value={"provider": "bedrock"}):
            result = resolve_runtime_provider(requested="bedrock")
        assert result["provider"] == "bedrock"
        assert result["api_mode"] == "bedrock_converse"


# ---------------------------------------------------------------------------
# providers.py integration
# ---------------------------------------------------------------------------

class TestProvidersModule:
    """Verify bedrock is wired into hermes_cli/providers.py."""

    def test_bedrock_alias_in_providers(self):
        from hermes_cli.providers import ALIASES
        assert ALIASES.get("bedrock") is None  # "bedrock" IS the canonical name, not an alias
        assert ALIASES.get("aws") == "bedrock"
        assert ALIASES.get("aws-bedrock") == "bedrock"

    def test_bedrock_transport_mapping(self):
        from hermes_cli.providers import TRANSPORT_TO_API_MODE
        assert TRANSPORT_TO_API_MODE.get("bedrock_converse") == "bedrock_converse"

    def test_determine_api_mode_from_bedrock_url(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode(
            "unknown", "https://bedrock-runtime.us-east-1.amazonaws.com"
        ) == "bedrock_converse"

    def test_label_override(self):
        from hermes_cli.providers import _LABEL_OVERRIDES
        assert _LABEL_OVERRIDES.get("bedrock") == "AWS Bedrock"


# ---------------------------------------------------------------------------
# Error classifier integration
# ---------------------------------------------------------------------------

class TestErrorClassifierBedrock:
    """Verify Bedrock error patterns are in the global error classifier."""

    def test_throttling_in_rate_limit_patterns(self):
        from agent.error_classifier import _RATE_LIMIT_PATTERNS
        assert "throttlingexception" in _RATE_LIMIT_PATTERNS

    def test_context_overflow_patterns(self):
        from agent.error_classifier import _CONTEXT_OVERFLOW_PATTERNS
        assert "input is too long" in _CONTEXT_OVERFLOW_PATTERNS


# ---------------------------------------------------------------------------
# pyproject.toml bedrock extra
# ---------------------------------------------------------------------------

class TestPackaging:
    """Verify bedrock optional dependency is declared."""

    def test_bedrock_extra_exists(self):
        import configparser
        from pathlib import Path
        # Read pyproject.toml to verify [bedrock] extra
        toml_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = toml_path.read_text()
        assert 'bedrock = ["boto3' in content

    def test_bedrock_in_all_extra(self):
        from pathlib import Path
        content = (Path(__file__).parent.parent.parent / "pyproject.toml").read_text()
        assert '"hermes-agent[bedrock]"' in content
