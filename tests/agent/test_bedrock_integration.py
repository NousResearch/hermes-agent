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


# ---------------------------------------------------------------------------
# Model ID dot preservation — regression for #11976
# ---------------------------------------------------------------------------
# AWS Bedrock inference-profile model IDs embed structural dots:
#
#   global.anthropic.claude-opus-4-7
#   us.anthropic.claude-sonnet-4-5-20250929-v1:0
#   apac.anthropic.claude-haiku-4-5
#
# ``agent.anthropic_adapter.normalize_model_name`` converts dots to hyphens
# unless the caller opts in via ``preserve_dots=True``.  Before this fix,
# ``AIAgent._anthropic_preserve_dots`` returned False for the ``bedrock``
# provider, so Claude-on-Bedrock requests went out with
# ``global-anthropic-claude-opus-4-7`` (all dots mangled to hyphens) and
# Bedrock rejected them with:
#
#   HTTP 400: The provided model identifier is invalid.
#
# The fix adds ``bedrock`` to the preserve-dots provider allowlist and
# ``bedrock-runtime.`` to the base-URL heuristic, mirroring the shape of
# the opencode-go fix for #5211 (commit f77be22c), which extended this
# same allowlist.


class TestBedrockPreserveDotsFlag:
    """``AIAgent._anthropic_preserve_dots`` must return True on Bedrock so
    inference-profile IDs survive the normalize step intact."""

    def test_bedrock_provider_preserves_dots(self):
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="bedrock", base_url="")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_bedrock_runtime_us_east_1_url_preserves_dots(self):
        """Defense-in-depth: even without an explicit ``provider="bedrock"``,
        a ``bedrock-runtime.us-east-1.amazonaws.com`` base URL must not
        mangle dots."""
        from types import SimpleNamespace
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://bedrock-runtime.us-east-1.amazonaws.com",
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_bedrock_runtime_ap_northeast_2_url_preserves_dots(self):
        """Reporter-reported region (ap-northeast-2) exercises the same
        base-URL heuristic."""
        from types import SimpleNamespace
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://bedrock-runtime.ap-northeast-2.amazonaws.com",
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_non_bedrock_aws_url_does_not_preserve_dots(self):
        """Unrelated AWS endpoints (e.g. ``s3.us-east-1.amazonaws.com``)
        must not accidentally activate the dot-preservation heuristic —
        the heuristic is scoped to the ``bedrock-runtime.`` substring
        specifically."""
        from types import SimpleNamespace
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://s3.us-east-1.amazonaws.com",
        )
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_anthropic_native_still_does_not_preserve_dots(self):
        """Canary: adding Bedrock to the allowlist must not weaken the
        existing Anthropic native behaviour — ``claude-sonnet-4.6`` still
        becomes ``claude-sonnet-4-6`` for the Anthropic API."""
        from types import SimpleNamespace
        agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com")
        from run_agent import AIAgent
        assert AIAgent._anthropic_preserve_dots(agent) is False


class TestBedrockModelNameNormalization:
    """End-to-end: ``normalize_model_name`` + the preserve-dots flag
    reproduce the exact production request shape for each Bedrock model
    family, confirming the fix resolves the reporter's HTTP 400."""

    def test_global_anthropic_inference_profile_preserved(self):
        """The reporter's exact model ID."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "global.anthropic.claude-opus-4-7", preserve_dots=True
        ) == "global.anthropic.claude-opus-4-7"

    def test_us_anthropic_dated_inference_profile_preserved(self):
        """Regional + dated Sonnet inference profile."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            preserve_dots=True,
        ) == "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_apac_anthropic_haiku_inference_profile_preserved(self):
        """APAC inference profile — same structural-dot shape."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "apac.anthropic.claude-haiku-4-5", preserve_dots=True
        ) == "apac.anthropic.claude-haiku-4-5"

    def test_bedrock_prefix_preserved_without_preserve_dots(self):
        """Bedrock inference profile IDs are auto-detected by prefix and
        always returned unmangled -- ``preserve_dots`` is irrelevant for
        these IDs because the dots are namespace separators, not version
        separators.  Regression for #12295."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "global.anthropic.claude-opus-4-7", preserve_dots=False
        ) == "global.anthropic.claude-opus-4-7"

    def test_bare_foundation_model_id_preserved(self):
        """Non-inference-profile Bedrock IDs
        (e.g. ``anthropic.claude-3-5-sonnet-20241022-v2:0``) use dots as
        vendor separators and must also survive intact under
        ``preserve_dots=True``."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            preserve_dots=True,
        ) == "anthropic.claude-3-5-sonnet-20241022-v2:0"


class TestBedrockBuildAnthropicKwargsEndToEnd:
    """Integration: calling ``build_anthropic_kwargs`` with a Bedrock-
    shaped model ID and ``preserve_dots=True`` produces the unmangled
    model string in the outgoing kwargs — the exact body sent to the
    ``bedrock-runtime.`` endpoint.  This is the integration-level
    regression for the reporter's HTTP 400."""

    def test_bedrock_inference_profile_survives_build_kwargs(self):
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="global.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            preserve_dots=True,
        )
        assert kwargs["model"] == "global.anthropic.claude-opus-4-7", (
            "Bedrock inference-profile ID was mangled in build_anthropic_kwargs: "
            f"{kwargs['model']!r}"
        )

    def test_bedrock_model_preserved_without_preserve_dots(self):
        """Bedrock inference profile IDs survive ``build_anthropic_kwargs``
        even without ``preserve_dots=True`` -- the prefix auto-detection
        in ``normalize_model_name`` is the load-bearing piece.
        Regression for #12295."""
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="global.anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            preserve_dots=False,
        )
        assert kwargs["model"] == "global.anthropic.claude-opus-4-7"


class TestBedrockModelIdDetection:
    """Tests for ``_is_bedrock_model_id`` and the auto-detection that
    makes ``normalize_model_name`` preserve dots for Bedrock IDs
    regardless of ``preserve_dots``.  Regression for #12295."""

    def test_bare_bedrock_id_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("anthropic.claude-opus-4-7") is True

    def test_regional_us_prefix_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("us.anthropic.claude-sonnet-4-5-v1:0") is True

    def test_regional_global_prefix_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("global.anthropic.claude-opus-4-7") is True

    def test_regional_eu_prefix_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("eu.anthropic.claude-sonnet-4-6") is True

    def test_openrouter_format_not_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("claude-opus-4.6") is False

    def test_bare_claude_not_detected(self):
        from agent.anthropic_adapter import _is_bedrock_model_id
        assert _is_bedrock_model_id("claude-opus-4-7") is False

    def test_bare_bedrock_id_preserved_without_flag(self):
        """The primary bug from #12295: ``anthropic.claude-opus-4-7``
        sent to bedrock-mantle via auxiliary clients that don't pass
        ``preserve_dots=True``."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "anthropic.claude-opus-4-7", preserve_dots=False
        ) == "anthropic.claude-opus-4-7"

    def test_openrouter_dots_still_converted(self):
        """Non-Bedrock dotted model names must still be converted."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name("claude-opus-4.6") == "claude-opus-4-6"

    def test_bare_bedrock_id_survives_build_kwargs(self):
        """End-to-end: bare Bedrock ID through ``build_anthropic_kwargs``
        without ``preserve_dots=True`` -- the auxiliary client path."""
        from agent.anthropic_adapter import build_anthropic_kwargs
        kwargs = build_anthropic_kwargs(
            model="anthropic.claude-opus-4-7",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            max_tokens=1024,
            reasoning_config=None,
            preserve_dots=False,
        )
        assert kwargs["model"] == "anthropic.claude-opus-4-7"


# ---------------------------------------------------------------------------
# auxiliary_client Bedrock resolution — fix for #13919
# ---------------------------------------------------------------------------
# Before the fix, resolve_provider_client("bedrock", ...) fell through to the
# "unhandled auth_type" warning and returned (None, None), breaking all
# auxiliary tasks (compression, memory, summarization) for Bedrock users.


class TestAuxiliaryClientBedrockResolution:
    """Verify resolve_provider_client handles Bedrock's aws_sdk auth type."""

    def test_bedrock_returns_client_with_credentials(self, monkeypatch):
        """With valid AWS credentials, Bedrock should return a usable client."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "us-west-2")

        mock_anthropic_bedrock = MagicMock()
        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   return_value=mock_anthropic_bedrock):
            from agent.auxiliary_client import resolve_provider_client, AnthropicAuxiliaryClient
            client, model = resolve_provider_client("bedrock", None)

        assert client is not None, (
            "resolve_provider_client('bedrock') returned None — "
            "aws_sdk auth type is not handled"
        )
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert model is not None
        assert client.api_key == "aws-sdk"
        assert "us-west-2" in client.base_url

    def test_bedrock_returns_none_without_credentials(self, monkeypatch):
        """Without AWS credentials, Bedrock should return (None, None) gracefully."""
        with patch("agent.bedrock_adapter.has_aws_credentials", return_value=False):
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("bedrock", None)

        assert client is None
        assert model is None

    def test_bedrock_uses_configured_region(self, monkeypatch):
        """Bedrock client base_url should reflect AWS_REGION."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "eu-central-1")

        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   return_value=MagicMock()):
            from agent.auxiliary_client import resolve_provider_client
            client, _ = resolve_provider_client("bedrock", None)

        assert client is not None
        assert "eu-central-1" in client.base_url

    def test_bedrock_respects_explicit_model(self, monkeypatch):
        """When caller passes an explicit model, it should be used."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   return_value=MagicMock()):
            from agent.auxiliary_client import resolve_provider_client
            _, model = resolve_provider_client(
                "bedrock", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            )

        assert "claude-sonnet" in model

    def test_bedrock_async_mode(self, monkeypatch):
        """Async mode should return an AsyncAnthropicAuxiliaryClient."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   return_value=MagicMock()):
            from agent.auxiliary_client import resolve_provider_client, AsyncAnthropicAuxiliaryClient
            client, model = resolve_provider_client("bedrock", None, async_mode=True)

        assert client is not None
        assert isinstance(client, AsyncAnthropicAuxiliaryClient)

    def test_bedrock_default_model_is_haiku(self, monkeypatch):
        """Default auxiliary model for Bedrock should be Haiku (fast, cheap)."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   return_value=MagicMock()):
            from agent.auxiliary_client import resolve_provider_client
            _, model = resolve_provider_client("bedrock", None)

        assert "haiku" in model.lower()
