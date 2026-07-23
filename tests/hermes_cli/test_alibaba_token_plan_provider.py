"""Tests for the Alibaba Cloud Token Plan provider plugin."""

import importlib.util
import os
import pathlib

import pytest

from providers import get_provider_profile
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import _PROVIDER_MODELS, provider_group_for_slug
from hermes_cli.providers import HERMES_OVERLAYS, normalize_provider

PLUGIN_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "plugins"
    / "model-providers"
    / "alibaba-token-plan"
)

# Capture only an explicitly exported integration credential before the global
# autouse fixture scrubs provider keys from each test for hermeticity.
_LIVE_API_KEY = os.environ.get("ALIBABA_TOKEN_PLAN_API_KEY")


def _load_plugin():
    init_file = PLUGIN_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "plugins.model_providers.alibaba_token_plan_test",
        init_file,
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plugin_registers_and_resolves_by_alias():
    _load_plugin()
    prof = get_provider_profile("alibaba-token-plan")
    assert prof is not None
    assert prof.name == "alibaba-token-plan"
    # alias resolution
    assert get_provider_profile("token-plan").name == "alibaba-token-plan"


def test_plugin_endpoint_is_token_plan_exclusive():
    _load_plugin()
    prof = get_provider_profile("alibaba-token-plan")
    assert prof is not None
    assert prof.base_url == (
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    )
    assert prof.env_vars[0] == "ALIBABA_TOKEN_PLAN_API_KEY"
    assert "DASHSCOPE_API_KEY" not in prof.env_vars


def test_provider_is_registered_across_core_surfaces():
    provider = PROVIDER_REGISTRY["alibaba-token-plan"]
    assert provider.api_key_env_vars == ("ALIBABA_TOKEN_PLAN_API_KEY",)
    assert provider.base_url_env_var == "ALIBABA_TOKEN_PLAN_BASE_URL"
    assert HERMES_OVERLAYS["alibaba-token-plan"].transport == "openai_chat"
    assert normalize_provider("token-plan") == "alibaba-token-plan"
    assert provider_group_for_slug("alibaba-token-plan") == "qwen"


def test_static_catalog_contains_only_agent_chat_models():
    models = _PROVIDER_MODELS["alibaba-token-plan"]
    assert "qwen3.8-max-preview" in models
    assert "deepseek-v4-pro" in models
    assert not any(model.startswith("wan") for model in models)


@pytest.mark.integration
def test_fetch_models_against_live_endpoint():
    """Live probe — requires a real ALIBABA_TOKEN_PLAN_API_KEY.

    The integration test is opt-in and only reads explicit process
    environment variables; unit tests never inspect a user's Hermes profile.
    """
    key = _LIVE_API_KEY
    if not key:
        pytest.skip("ALIBABA_TOKEN_PLAN_API_KEY not available")

    _load_plugin()
    prof = get_provider_profile("alibaba-token-plan")
    assert prof is not None
    models = prof.fetch_models(api_key=key, timeout=20)
    assert models
    expected = {"qwen3.8-max-preview", "qwen3.7-max", "qwen3.6-flash", "deepseek-v4-pro"}
    assert expected.issubset(set(models))
    assert not any(model.startswith("wan") for model in models)
