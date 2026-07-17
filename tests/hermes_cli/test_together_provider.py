"""First-class Together AI provider wiring tests."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from hermes_cli.auth import resolve_api_key_provider_credentials
from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS, normalize_provider


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_BASE_URL", raising=False)


@pytest.mark.parametrize(
    "alias",
    ["together", "together-ai", "togetherai", "TOGETHER-AI", " TogetherAI "],
)
def test_models_normalize_provider(alias):
    assert normalize_provider(alias) == "together"


@pytest.mark.parametrize("alias", ["together", "together-ai", "togetherai"])
def test_providers_normalize_provider(alias):
    from hermes_cli.providers import normalize_provider as normalize_in_providers

    assert normalize_in_providers(alias) == "together"


def test_provider_is_in_canonical_catalog():
    assert "together" in [entry.slug for entry in CANONICAL_PROVIDERS]
    assert CANONICAL_PROVIDERS[1].slug == "together"
    assert _PROVIDER_LABELS["together"] == "Together AI"


def test_secret_registry_and_overlay():
    from hermes_cli.config import OPTIONAL_ENV_VARS
    from hermes_cli.providers import HERMES_OVERLAYS

    secret = OPTIONAL_ENV_VARS["TOGETHER_API_KEY"]
    assert secret["category"] == "provider"
    assert secret["password"] is True
    assert "TOGETHER_BASE_URL" not in OPTIONAL_ENV_VARS

    overlay = HERMES_OVERLAYS["together"]
    assert overlay.transport == "openai_chat"
    assert overlay.base_url_override == "https://api.together.ai/v1"
    assert not overlay.base_url_env_var
    assert not overlay.is_aggregator


def test_credentials_use_canonical_endpoint(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "together-test-key")

    creds = resolve_api_key_provider_credentials("together")

    assert creds["provider"] == "together"
    assert creds["api_key"] == "together-test-key"
    assert creds["base_url"] == "https://api.together.ai/v1"


def test_auto_resolution_detects_together_key(monkeypatch):
    from hermes_cli.auth import resolve_provider

    monkeypatch.setenv("TOGETHER_API_KEY", "together-test-key")
    with patch("hermes_cli.config.load_config", return_value={}):
        assert resolve_provider("auto") == "together"


def test_runtime_fails_closed_without_key(monkeypatch):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError, match="TOGETHER_API_KEY"):
        resolve_runtime_provider(requested="together")


def test_auxiliary_client_uses_profile_defaults(monkeypatch):
    from agent.auxiliary_client import resolve_provider_client

    monkeypatch.setenv("TOGETHER_API_KEY", "together-test-key")
    with patch("agent.auxiliary_client.OpenAI") as mock_openai:
        mock_openai.return_value = object()
        client, model = resolve_provider_client("together-ai")

    assert client is not None
    assert model == "Qwen/Qwen3.5-9B"
    kwargs = mock_openai.call_args.kwargs
    assert kwargs["base_url"] == "https://api.together.ai/v1"
    assert "HTTP-Referer" not in kwargs.get("default_headers", {})
    assert "X-Title" not in kwargs.get("default_headers", {})


def test_setup_uses_profile_fallback_when_catalogs_are_unavailable(monkeypatch):
    from hermes_cli.config import load_config
    from hermes_cli.model_setup_flows import _model_flow_api_key_provider
    from providers import get_provider_profile

    profile = get_provider_profile("together")
    seen = {}

    def capture_models(models, **kwargs):
        seen["models"] = list(models)
        return None

    monkeypatch.setenv("TOGETHER_API_KEY", "together-test-key")
    monkeypatch.setattr(profile, "fetch_models", lambda **kwargs: None)
    with (
        patch(
            "hermes_cli.main._prompt_api_key",
            return_value=("together-test-key", False),
        ),
        patch("agent.models_dev.list_agentic_models", return_value=[]),
        patch(
            "hermes_cli.auth._prompt_model_selection",
            side_effect=capture_models,
        ),
        patch("builtins.input", return_value=""),
    ):
        _model_flow_api_key_provider(load_config(), "together")

    assert seen["models"] == list(profile.fallback_models)


def test_shared_model_probe_accepts_together_bare_list():
    from hermes_cli.models import probe_api_models

    response = io.BytesIO(
        json.dumps([{"id": "MiniMaxAI/MiniMax-M3"}]).encode()
    )
    with patch(
        "hermes_cli.models._urlopen_model_catalog_request",
        return_value=response,
    ):
        result = probe_api_models(
            "test-key",
            "https://api.together.ai/v1",
        )

    assert result["models"] == ["MiniMaxAI/MiniMax-M3"]


@pytest.mark.parametrize(
    "url",
    ["https://api.together.ai/v1", "https://api.together.xyz/v1"],
)
def test_model_metadata_infers_canonical_and_legacy_hosts(url):
    from agent.model_metadata import _infer_provider_from_url

    assert _infer_provider_from_url(url) == "together"
