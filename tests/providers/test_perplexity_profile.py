"""Bundled Perplexity Agent API provider: credential scoping, routing, runtime resolution, headers, catalog."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.auxiliary_client import get_text_auxiliary_client
from agent.credential_pool import load_pool
from agent.models_dev import lookup_models_dev_context
from hermes_cli import auth, models
from hermes_cli.auth_commands import auth_add_command
from hermes_cli.inventory import ConfigContext, build_models_payload
from hermes_cli.main import _has_any_provider_configured
from hermes_cli.model_switch import resolve_startup_model_route, switch_model
from hermes_cli.providers import determine_api_mode, normalize_provider
from hermes_cli.runtime_provider import resolve_runtime_provider
from hermes_cli.web_server_config import _infer_provider_on_model_change
from hermes_constants import get_hermes_home

KEY = "pplx-test-key-1234567890"
# models.dev lists the Agent API as ``perplexity-agent``; its ``perplexity`` entry has no vendor-prefixed ids.
MODELS_DEV = {
    "perplexity": {"id": "perplexity", "name": "Perplexity", "env": ["PERPLEXITY_API_KEY"], "models": {}},
    "perplexity-agent": {"id": "perplexity-agent", "models": {"anthropic/claude-opus-5-5": {
        "id": "anthropic/claude-opus-5-5", "limit": {"context": 1000000, "output": 128000}}}},
}


@pytest.fixture
def perplexity_key(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", KEY)


@pytest.fixture
def perplexity_status_only(monkeypatch):
    """Only Perplexity's own auth status: a gh login on the host must not decide the first-run check."""
    status = auth.get_auth_status
    monkeypatch.setattr(auth, "get_auth_status", lambda pid: status(pid) if pid in ("perplexity", "pplx") else {})


def test_search_key_alone_never_selects_perplexity(perplexity_key):
    """``PERPLEXITY_API_KEY`` is also the web-search key: auto-detection must not make it the model provider."""
    with pytest.raises(auth.AuthError):
        auth.resolve_provider("auto")


def test_search_key_alone_is_not_an_auxiliary_fallback(perplexity_key):
    assert get_text_auxiliary_client() == (None, None)


def test_search_key_alone_is_not_an_explicit_provider(perplexity_key):
    """Neither the env key nor the pool row seeded from it lists Perplexity in ``explicit_only`` pickers."""
    load_pool("perplexity")
    row = {"slug": "perplexity", "name": "Perplexity", "models": ["anthropic/claude-opus-5-5"], "total_models": 1,
           "is_current": False, "is_user_defined": False, "source": "hermes"}
    ctx = ConfigContext(current_provider="openrouter", current_model="openai/gpt-5.5", current_base_url="",
                        user_providers={}, custom_providers=[])
    with patch("hermes_cli.model_switch.list_authenticated_providers", return_value=[row]):
        payload = build_models_payload(ctx, explicit_only=True)

    assert [entry["source"] for entry in auth.read_credential_pool("perplexity")] == ["env:PERPLEXITY_API_KEY"]
    assert auth.is_provider_explicitly_configured("perplexity") is False
    assert "perplexity" not in [entry["slug"] for entry in payload["providers"]]


def test_search_key_alone_still_reaches_first_run_setup(perplexity_key, perplexity_status_only):
    assert _has_any_provider_configured() is False


def test_hermes_auth_add_makes_perplexity_a_configured_provider(perplexity_status_only):
    """A credential from ``hermes auth add perplexity`` is a choice of the provider, unlike the env key."""
    auth_add_command(SimpleNamespace(provider="perplexity", api_key=KEY, label="work", auth_type="", priority=None))

    assert [entry["source"] for entry in auth.read_credential_pool("perplexity")] == ["manual"]
    assert auth.is_provider_explicitly_configured("perplexity") is True
    assert _has_any_provider_configured() is True


def test_selecting_perplexity_makes_it_explicit_with_a_small_auxiliary_model(perplexity_key):
    (get_hermes_home() / "config.yaml").write_text(
        "model:\n  provider: perplexity\n  default: anthropic/claude-opus-5-5\n"
        "auxiliary:\n  title:\n    provider: perplexity\n", encoding="utf-8")
    client, model = get_text_auxiliary_client("title")

    assert auth.is_provider_explicitly_configured("perplexity") is True
    assert _has_any_provider_configured() is True
    assert (type(client).__name__, model) == ("CodexAuxiliaryClient", "anthropic/claude-haiku-5-5")


def test_vendor_slug_change_stays_on_perplexity(perplexity_key, monkeypatch):
    """Perplexity serves ``vendor/model`` ids itself: a model change must not re-route to OpenRouter."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key-1234567890")
    assert _infer_provider_on_model_change("openai/gpt-5.5", "perplexity") == ("", "openai/gpt-5.5")


def test_bare_model_name_resolves_in_the_perplexity_catalog(perplexity_key):
    """``/model glm-5.3`` in a Perplexity session picks the vendor-prefixed id its catalog serves."""
    with patch("hermes_cli.model_switch.list_provider_models", return_value=["openai/gpt-5.5", "perplexity/glm-5.3"]):
        result = switch_model("glm-5.3", current_provider="perplexity", current_model="openai/gpt-5.5")

    assert (result.success, result.target_provider, result.new_model) == (True, "perplexity", "perplexity/glm-5.3")


def test_pplx_alias_routes_model_selection(perplexity_key):
    """``-m pplx:<model>`` at startup and ``/model <model> --provider pplx`` in a session."""
    route = resolve_startup_model_route("pplx:anthropic/claude-opus-5-5", current_provider="openrouter")
    result = switch_model("anthropic/claude-opus-5-5", current_provider="openrouter", current_model="openai/gpt-5.5",
                          explicit_provider="pplx")

    assert models.normalize_provider("pplx") == "perplexity"
    assert (route.provider, route.model) == ("perplexity", "anthropic/claude-opus-5-5")
    assert (result.success, result.target_provider, result.new_model, result.api_mode, result.base_url) == (
        True, "perplexity", "anthropic/claude-opus-5-5", "codex_responses", "https://api.perplexity.ai/v1")


@pytest.mark.parametrize("requested", ["perplexity", "pplx"])
def test_explicit_provider_resolves_to_the_agent_api(perplexity_key, requested):
    with patch("agent.models_dev.fetch_models_dev", return_value=MODELS_DEV):
        runtime = resolve_runtime_provider(requested=requested)
        switch_mode = determine_api_mode(requested, runtime["base_url"])

    assert (runtime["provider"], runtime["api_mode"], runtime["base_url"], runtime["api_key"]) == (
        "perplexity", "codex_responses", "https://api.perplexity.ai/v1", KEY)
    assert normalize_provider(requested) == "perplexity"
    assert switch_mode == "codex_responses"


def test_client_carries_the_harness_identity_headers(perplexity_key):
    from run_agent import AIAgent

    runtime = resolve_runtime_provider(requested="perplexity")
    with patch("model_tools.get_tool_definitions", return_value=[]), \
         patch("model_tools.check_toolset_requirements", return_value={}), \
         patch("agent.process_bootstrap.OpenAI") as openai_client:
        AIAgent(provider=runtime["provider"], model="anthropic/claude-opus-5-5", base_url=runtime["base_url"],
                api_key=runtime["api_key"], api_mode=runtime["api_mode"], quiet_mode=True,
                skip_context_files=True, skip_memory=True)

    headers = openai_client.call_args.kwargs["default_headers"]
    assert headers["X-Pplx-Integration"] == "hermes-agent"
    assert headers["User-Agent"].startswith("Hermes-Agent/")


def test_context_window_comes_from_the_agent_api_catalog():
    with patch("agent.models_dev.fetch_models_dev", return_value=MODELS_DEV):
        assert lookup_models_dev_context("perplexity", "anthropic/claude-opus-5-5") == 1000000
