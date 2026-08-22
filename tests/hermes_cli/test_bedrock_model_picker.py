"""Tests for AWS Bedrock integration in the model picker and provider catalog.

Covers the three paths changed by fix/bedrock-provider-model-ids-live-discovery:

  1. provider_model_ids("bedrock") — uses live discover_bedrock_models() instead
     of the static _PROVIDER_MODELS table, with curated fallback.

  2. list_authenticated_providers() Section 2 (HERMES_OVERLAYS) — bedrock
     appears when AWS credentials are present; model list comes from live
     discovery keyed by the resolved region, NOT the static us.* table.

  3. Region resolution — resolve_bedrock_region() reads from botocore profile
     when no AWS_REGION / AWS_DEFAULT_REGION env vars are set, so EU/AP users
     in eu-central-1 get eu.* profile IDs, not us.* ones.

All Bedrock API calls are mocked — no real AWS credentials needed.
"""

from contextlib import contextmanager
from types import ModuleType
from unittest.mock import MagicMock, patch

from agent.bedrock_adapter import BEDROCK_OPENAI_RESPONSES_MODEL_IDS

_MANTLE_MODELS = list(BEDROCK_OPENAI_RESPONSES_MODEL_IDS)
_MANTLE_SET = {m.lower() for m in _MANTLE_MODELS}


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _mock_botocore_session(*, return_value=None):
    """Patch botocore.session even when botocore is not installed."""
    botocore_mod = ModuleType("botocore")
    session_mod = ModuleType("botocore.session")
    session_mod.get_session = MagicMock(return_value=return_value)
    botocore_mod.session = session_mod
    with patch.dict("sys.modules", {"botocore": botocore_mod, "botocore.session": session_mod}):
        yield session_mod.get_session


_EU_MODELS = [
    {"id": "eu.anthropic.claude-sonnet-4-6-20250514-v1:0", "name": "Claude Sonnet 4.6 (EU)", "provider": "inference-profile"},
    {"id": "eu.anthropic.claude-haiku-4-5-20251015-v1:0",  "name": "Claude Haiku 4.5 (EU)",  "provider": "inference-profile"},
    {"id": "eu.amazon.nova-pro-v1:0",                       "name": "Nova Pro (EU)",           "provider": "inference-profile"},
]

_US_MODELS = [
    {"id": "us.anthropic.claude-sonnet-4-6-20250514-v1:0", "name": "Claude Sonnet 4.6 (US)", "provider": "inference-profile"},
    {"id": "us.amazon.nova-pro-v1:0",                       "name": "Nova Pro (US)",           "provider": "inference-profile"},
]


def _mock_discover(region: str):
    """Return EU models for eu-* regions, US models otherwise."""
    return _EU_MODELS if region.startswith("eu-") else _US_MODELS


# ---------------------------------------------------------------------------
# 1. provider_model_ids("bedrock")
# ---------------------------------------------------------------------------

class TestProviderModelIdsBedrock:
    """provider_model_ids("bedrock") should use live Bedrock discovery."""

    def test_returns_live_discovered_model_ids(self, monkeypatch):
        """Live discovery result is returned as a flat list of model ID strings."""
        from hermes_cli.models import provider_model_ids

        monkeypatch.setenv("AWS_REGION", "eu-central-1")

        with patch("agent.bedrock_adapter.discover_bedrock_models", side_effect=_mock_discover), \
             patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="eu-central-1"):
            result = provider_model_ids("bedrock")

        assert "eu.anthropic.claude-sonnet-4-6-20250514-v1:0" in result
        assert "eu.anthropic.claude-haiku-4-5-20251015-v1:0" in result
        for _m in _MANTLE_MODELS:
            assert _m in result
        assert len(result) == len(_EU_MODELS) + len(_MANTLE_MODELS)

    def test_region_determines_model_ids(self, monkeypatch):
        """Different regions produce different model ID prefixes (eu.* vs us.*)."""
        from hermes_cli.models import provider_model_ids

        with patch("agent.bedrock_adapter.discover_bedrock_models", side_effect=_mock_discover):
            with patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="eu-central-1"):
                eu_result = provider_model_ids("bedrock")
            with patch("agent.bedrock_adapter.resolve_bedrock_region", return_value="us-east-1"):
                us_result = provider_model_ids("bedrock")

        assert all(m.startswith("eu.") or m.lower() in _MANTLE_SET for m in eu_result)
        assert all(m.startswith("us.") or m.lower() in _MANTLE_SET for m in us_result)
        assert eu_result != us_result





# ---------------------------------------------------------------------------
# 2. list_authenticated_providers() — bedrock via HERMES_OVERLAYS (Section 2)
# ---------------------------------------------------------------------------

class TestListAuthenticatedProvidersBedrock:
    """Bedrock should appear in the /model picker when AWS creds are present."""





    def test_bedrock_not_shown_without_credentials(self, monkeypatch):
        """Bedrock must not appear when no AWS credentials are present."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
        monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", raising=False)

        with patch("agent.bedrock_adapter.has_aws_credentials", return_value=False):
            providers = list_authenticated_providers(current_provider="openai")

        bedrock = next((p for p in providers if p["slug"] == "bedrock"), None)
        assert bedrock is None, "bedrock should NOT appear when AWS credentials are absent"

    def test_non_bedrock_picker_does_not_probe_full_aws_chain(self, monkeypatch):
        """Non-Bedrock provider discovery must not touch boto3's full credential chain."""
        from hermes_cli.model_switch import list_authenticated_providers

        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)
        monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", raising=False)
        monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", raising=False)

        calls = {"has_aws_credentials": 0}

        def _has_aws_credentials():
            calls["has_aws_credentials"] += 1
            return False

        with patch("agent.bedrock_adapter.has_aws_credentials", side_effect=_has_aws_credentials):
            providers = list_authenticated_providers(current_provider="openrouter", max_models=0)

        assert calls["has_aws_credentials"] == 0
        assert all(p["slug"] != "bedrock" for p in providers)


# ---------------------------------------------------------------------------
# 3. Region routing: EU/AP users see regional model IDs
# ---------------------------------------------------------------------------

class TestBedrockRegionRouting:
    """End-to-end: region from botocore profile is used for discovery, so EU/AP
    users get eu.*/ap.* model IDs rather than the hardcoded us-east-1 list."""

    def test_eu_region_from_botocore_profile_yields_eu_models(self):
        """When botocore resolves eu-central-1, picker shows eu.* model IDs."""
        from hermes_cli.model_switch import list_authenticated_providers

        mock_session = MagicMock()
        mock_session.get_config_variable.return_value = "eu-central-1"

        with patch("agent.bedrock_adapter.has_aws_credentials", return_value=True), \
             patch("agent.bedrock_adapter.discover_bedrock_models", side_effect=_mock_discover), \
             _mock_botocore_session(return_value=mock_session):
            providers = list_authenticated_providers(current_provider="bedrock")

        bedrock = next((p for p in providers if p["slug"] == "bedrock"), None)
        assert bedrock is not None
        for _m in _MANTLE_MODELS:
            assert _m in bedrock["models"]
        for model_id in bedrock["models"]:
            assert model_id.startswith("eu.") or model_id.lower() in _MANTLE_SET, \
                f"Expected eu.* or Bedrock OpenAI model ID from eu-central-1 profile, got {model_id!r}"


    def test_env_var_takes_priority_over_botocore_profile(self, monkeypatch):
        """AWS_REGION env var wins over botocore profile region."""
        from agent.bedrock_adapter import resolve_bedrock_region

        monkeypatch.setenv("AWS_REGION", "us-west-2")

        mock_session = MagicMock()
        mock_session.get_config_variable.return_value = "eu-central-1"

        with _mock_botocore_session(return_value=mock_session):
            region = resolve_bedrock_region()

        assert region == "us-west-2", "env var should override botocore profile"


# ---------------------------------------------------------------------------
# 4. providers.py overlay registration
# ---------------------------------------------------------------------------

class TestBedrockOverlayRegistration:
    """bedrock entry in HERMES_OVERLAYS is correctly configured."""

    def test_bedrock_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "bedrock" in HERMES_OVERLAYS


    def test_bedrock_label(self):
        from hermes_cli.providers import get_label
        label = get_label("bedrock")
        assert label  # non-empty
        assert "bedrock" in label.lower() or "aws" in label.lower()

    def test_bedrock_aliases_resolve(self):
        from hermes_cli.providers import normalize_provider
        for alias in ("aws", "aws-bedrock", "amazon-bedrock", "amazon"):
            assert normalize_provider(alias) == "bedrock", \
                f"alias {alias!r} should normalize to 'bedrock'"
