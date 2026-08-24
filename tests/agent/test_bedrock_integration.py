"""Integration tests for the AWS Bedrock provider wiring.

Verifies that the Bedrock provider is correctly registered in the
provider registry, model catalog, and runtime resolution pipeline.
These tests do NOT require AWS credentials or boto3 — all AWS calls
are mocked.

Note: Tests that import ``hermes_cli.auth`` or ``hermes_cli.runtime_provider``
require Python 3.10+ due to ``str | None`` type syntax in the import chain.
"""

from unittest.mock import MagicMock, patch

import pytest


_BOTO_PREFIXES = ("botocore", "boto3")


@pytest.fixture(autouse=True)
def _boto_sys_modules_hygiene():
    """Snapshot/restore boto* sys.modules around every test.

    Tests here plant fake botocore/boto3 modules; a fake that leaks (or a
    real submodule first-imported inside a stub window) poisons later
    imports of the real ``botocore.exceptions`` with
    ``No module named 'botocore.vendored'`` (PR #92617 CI flake). This
    fixture makes stub windows airtight regardless of test ordering.
    """
    import sys as _sys

    saved = {
        name: mod
        for name, mod in _sys.modules.items()
        if name.split(".", 1)[0] in _BOTO_PREFIXES
    }
    yield
    for name in [n for n in _sys.modules if n.split(".", 1)[0] in _BOTO_PREFIXES]:
        _sys.modules.pop(name, None)
    _sys.modules.update(saved)



class TestProviderRegistry:
    """Verify Bedrock is registered in PROVIDER_REGISTRY."""

    def test_bedrock_in_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "bedrock" in PROVIDER_REGISTRY


    def test_bedrock_has_no_api_key_env_vars(self):
        """Bedrock uses the AWS SDK credential chain, not API keys."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["bedrock"]
        assert pconfig.api_key_env_vars == ()



class TestProviderAliases:
    """Verify Bedrock aliases resolve correctly."""

    def test_aws_alias(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("aws") == "bedrock"





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
        # bedrock is in the registry, so resolve_provider should return it
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("bedrock")
        assert result == "bedrock"

    def test_aws_alias_resolves_to_bedrock(self):
        from hermes_cli.auth import resolve_provider
        result = resolve_provider("aws")
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

    def test_bedrock_openai_models_route_to_mantle_responses(self, monkeypatch):
        """Bedrock's OpenAI models (GPT-5.5 / GPT-5.6 family) are not Converse
        models — they only answer on the Mantle /openai/v1 Responses surface.
        Every allowlisted ID must route there, with the aws-sdk IAM sentinel."""
        from agent.bedrock_adapter import BEDROCK_OPENAI_RESPONSES_MODEL_IDS
        from hermes_cli.runtime_provider import resolve_runtime_provider

        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "us-east-2")

        assert "openai.gpt-5.5" in BEDROCK_OPENAI_RESPONSES_MODEL_IDS
        for suffix in ("sol", "terra", "luna"):
            assert f"openai.gpt-5.6-{suffix}" in BEDROCK_OPENAI_RESPONSES_MODEL_IDS

        for model_id in BEDROCK_OPENAI_RESPONSES_MODEL_IDS:
            with patch("hermes_cli.runtime_provider.resolve_provider", return_value="bedrock"), \
                 patch("hermes_cli.runtime_provider._get_model_config", return_value={
                     "provider": "bedrock",
                     "default": model_id,
                 }):
                result = resolve_runtime_provider(requested="bedrock")

            assert result["api_mode"] == "codex_responses", model_id
            assert result["model"] == model_id
            assert result["base_url"] == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
            assert result["api_key"] == "aws-sdk"
            assert result["bedrock_openai"] is True, model_id

    def test_bedrock_openai_context_length_is_272k(self):
        """AWS model cards list a 272K context window for the Mantle OpenAI
        models; make sure we do not fall back to the 128K default."""
        from agent.bedrock_adapter import (
            BEDROCK_OPENAI_RESPONSES_MODEL_IDS,
            get_bedrock_context_length,
        )
        for model_id in BEDROCK_OPENAI_RESPONSES_MODEL_IDS:
            assert get_bedrock_context_length(model_id) == 272_000


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


    def test_determine_api_mode_from_bedrock_url(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode(
            "unknown", "https://bedrock-runtime.us-east-1.amazonaws.com"
        ) == "bedrock_converse"



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
    """Verify Bedrock remains a declared lazy optional dependency."""

    @staticmethod
    def _optional_dependencies():
        import tomllib
        from pathlib import Path

        content = (Path(__file__).parent.parent.parent / "pyproject.toml").read_text()
        return tomllib.loads(content)["project"]["optional-dependencies"]

    def test_bedrock_extra_exists(self):
        extras = self._optional_dependencies()
        assert "bedrock" in extras
        assert any(dep.startswith("boto3==") for dep in extras["bedrock"])

    def test_bedrock_is_not_eager_installed_by_all_extra(self):
        extras = self._optional_dependencies()
        assert "hermes-agent[bedrock]" not in extras["all"]


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



    def test_bedrock_prefix_preserved_without_preserve_dots(self):
        """Bedrock inference profile IDs are auto-detected by prefix and
        always returned unmangled -- ``preserve_dots`` is irrelevant for
        these IDs because the dots are namespace separators, not version
        separators.  Regression for #12295."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "global.anthropic.claude-opus-4-7", preserve_dots=False
        ) == "global.anthropic.claude-opus-4-7"



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






    def test_bare_bedrock_id_preserved_without_flag(self):
        """The primary bug from #12295: ``anthropic.claude-opus-4-7``
        sent to bedrock-mantle via auxiliary clients that don't pass
        ``preserve_dots=True``."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name(
            "anthropic.claude-opus-4-7", preserve_dots=False
        ) == "anthropic.claude-opus-4-7"




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




    def test_bedrock_default_model_is_haiku(self, monkeypatch):
        """Default auxiliary model for Bedrock should be Haiku (fast, cheap)."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        with patch("agent.anthropic_adapter.build_anthropic_bedrock_client",
                   return_value=MagicMock()):
            from agent.auxiliary_client import resolve_provider_client
            _, model = resolve_provider_client("bedrock", None)

        assert "haiku" in model.lower()





    def test_bedrock_converse_shim_stream_returns_complete_response(self, monkeypatch):
        """stream=True is not supported by the shim — a complete response comes
        back and call_llm's streaming consumer downgrades gracefully."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIO...MPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        from agent.auxiliary_client import BedrockAuxiliaryClient

        client = BedrockAuxiliaryClient("us-east-1", "openai.gpt-oss-20b-1:0")
        sentinel = object()
        with patch("agent.bedrock_adapter.call_converse", return_value=sentinel) as mock_converse:
            resp = client.chat.completions.create(
                model="openai.gpt-oss-20b-1:0",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        # Non-streaming call_converse is still used; the caller's
        # got-final-object downgrade path handles the rest.
        assert resp is sentinel
        assert mock_converse.call_count == 1

    def test_bedrock_shim_uncapped_when_caller_omits_max_tokens(self, monkeypatch):
        """No caller max_tokens → the shim passes None through and the wire
        request carries no inferenceConfig.maxTokens, so Bedrock uses the
        model's maximum allowed output (#10809 on the Bedrock wire).

        Guards against the shim's old hardcoded ``else 4096`` fallback, which
        kept aux vision descriptions capped after the vision call sites
        dropped their own caps."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIO...MPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")

        from agent.auxiliary_client import BedrockAuxiliaryClient

        client = BedrockAuxiliaryClient("us-east-1", "openai.gpt-oss-20b-1:0")
        boto3_client = MagicMock()
        with patch("agent.bedrock_adapter._get_bedrock_runtime_client",
                   return_value=boto3_client), \
             patch("agent.bedrock_adapter.normalize_converse_response"):
            # Aux vision-style call: no max_tokens key at all.
            client.chat.completions.create(
                model="openai.gpt-oss-20b-1:0",
                messages=[{"role": "user", "content": "describe"}],
                temperature=0.1,
            )
            wire_kwargs = boto3_client.converse.call_args.kwargs
            assert "maxTokens" not in wire_kwargs.get("inferenceConfig", {})

            # An explicit caller cap still lands on the wire unchanged.
            client.chat.completions.create(
                model="openai.gpt-oss-20b-1:0",
                messages=[{"role": "user", "content": "describe"}],
                max_tokens=1234,
            )
            wire_kwargs = boto3_client.converse.call_args.kwargs
            assert wire_kwargs["inferenceConfig"]["maxTokens"] == 1234

    def test_bedrock_mantle_config_region_beats_env_region(self, monkeypatch):
        """bedrock.region in config.yaml must win over AWS_REGION for auxiliary
        Mantle calls — the same priority the main runtime resolver uses (#65076
        review: aux resolution previously derived its region env-first and
        could leave the primary runtime's configured region)."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "eu-central-1")
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bearer")

        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.api_key = kwargs.get("api_key")
                self.base_url = kwargs.get("base_url")

            def close(self):
                pass

        with patch("hermes_cli.config.load_config_readonly",
                   return_value={"bedrock": {"region": "us-west-2"}}), \
             patch("agent.auxiliary_client.OpenAI", _FakeOpenAI):
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("bedrock", "openai.gpt-5.6-sol")

        assert client is not None
        assert model == "openai.gpt-5.6-sol"
        assert "us-west-2" in captured.get("base_url", ""), (
            "Mantle auxiliary base_url ignored config.yaml bedrock.region"
        )

    def test_bedrock_openai_aux_uses_responses_client(self, monkeypatch):
        """Auxiliary tasks on Bedrock GPT models use the Mantle Responses
        path (SigV4 http client + aws-sdk sentinel), not the Anthropic shim."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAIOSFODNN7EXAMPLE")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        monkeypatch.setenv("AWS_REGION", "us-east-2")

        with patch("agent.auxiliary_client.OpenAI", return_value=MagicMock()) as mock_openai, \
             patch("agent.bedrock_adapter.build_bedrock_openai_http_client", return_value=MagicMock()):
            from agent.auxiliary_client import resolve_provider_client, CodexAuxiliaryClient
            client, model = resolve_provider_client("bedrock", "openai.gpt-5.5")

        assert model == "openai.gpt-5.5"
        assert isinstance(client, CodexAuxiliaryClient)
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["api_key"] == "aws-sdk"
        assert kwargs["base_url"] == "https://bedrock-mantle.us-east-2.api.aws/openai/v1"
        assert "http_client" in kwargs
