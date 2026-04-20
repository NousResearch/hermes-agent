"""Tests for FastRouter provider integration across the codebase.

Covers: constants, auth registry, aliases, auto-detection, model normalization,
model catalog, auxiliary client, AIAgent URL detection, API kwargs, and pricing.
"""

import json
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

# Ensure dotenv doesn't interfere
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


# ── Constants ─────────────────────────────────────────────────────────────────

class TestFastRouterConstants:
    def test_base_url_defined(self):
        from hermes_constants import FASTROUTER_BASE_URL
        assert FASTROUTER_BASE_URL == "https://api.fastrouter.ai/api/v1"

    def test_models_url_derived_from_base(self):
        from hermes_constants import FASTROUTER_BASE_URL, FASTROUTER_MODELS_URL
        assert FASTROUTER_MODELS_URL == f"{FASTROUTER_BASE_URL}/models"

    def test_models_url_value(self):
        from hermes_constants import FASTROUTER_MODELS_URL
        assert FASTROUTER_MODELS_URL == "https://api.fastrouter.ai/api/v1/models"


# ── Provider Registry & Auth ──────────────────────────────────────────────────

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "FASTROUTER_API_KEY", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY",
    "KIMI_API_KEY", "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY",
    "AI_GATEWAY_API_KEY", "KILOCODE_API_KEY", "HF_TOKEN",
    "NOUS_API_KEY", "GITHUB_TOKEN", "GH_TOKEN",
    "DASHSCOPE_API_KEY", "OPENCODE_ZEN_API_KEY", "OPENCODE_GO_API_KEY",
)


@pytest.fixture()
def _clean_env(monkeypatch):
    """Clear all provider env vars to isolate auto-detection tests."""
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})


class TestFastRouterRegistry:
    def test_registered_in_provider_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        assert "fastrouter" in PROVIDER_REGISTRY

    def test_registry_entry_fields(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["fastrouter"]
        assert cfg.id == "fastrouter"
        assert cfg.name == "FastRouter"
        assert cfg.auth_type == "api_key"
        assert cfg.inference_base_url == "https://api.fastrouter.ai/api/v1"
        assert "FASTROUTER_API_KEY" in cfg.api_key_env_vars
        assert cfg.base_url_env_var == "FASTROUTER_BASE_URL"


class TestFastRouterResolveProvider:
    def test_explicit_fastrouter(self):
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("fastrouter") == "fastrouter"

    def test_alias_fast_router_hyphen(self):
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("fast-router") == "fastrouter"

    def test_alias_fast_router_underscore(self):
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("fast_router") == "fastrouter"

    def test_alias_case_insensitive(self):
        from hermes_cli.auth import resolve_provider
        assert resolve_provider("FastRouter") == "fastrouter"
        assert resolve_provider("FAST-ROUTER") == "fastrouter"

    def test_auto_detects_fastrouter_key(self, _clean_env, monkeypatch):
        from hermes_cli.auth import resolve_provider
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-test-key")
        assert resolve_provider("auto") == "fastrouter"

    def test_openrouter_takes_priority_over_fastrouter(self, _clean_env, monkeypatch):
        from hermes_cli.auth import resolve_provider
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-key")
        assert resolve_provider("auto") == "openrouter"


class TestFastRouterCredentials:
    def test_unconfigured_status(self, _clean_env):
        from hermes_cli.auth import get_api_key_provider_status
        status = get_api_key_provider_status("fastrouter")
        assert status["configured"] is False

    def test_configured_status(self, _clean_env, monkeypatch):
        from hermes_cli.auth import get_api_key_provider_status
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-test-key-123")
        status = get_api_key_provider_status("fastrouter")
        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["key_source"] == "FASTROUTER_API_KEY"
        assert "fastrouter" in status["base_url"].lower()

    def test_custom_base_url_override(self, _clean_env, monkeypatch):
        from hermes_cli.auth import get_api_key_provider_status
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-key")
        monkeypatch.setenv("FASTROUTER_BASE_URL", "https://custom.fastrouter.example/v1")
        status = get_api_key_provider_status("fastrouter")
        assert status["base_url"] == "https://custom.fastrouter.example/v1"

    def test_resolve_credentials(self, _clean_env, monkeypatch):
        from hermes_cli.auth import resolve_api_key_provider_credentials
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-key-456")
        creds = resolve_api_key_provider_credentials("fastrouter")
        assert creds["api_key"] == "fr-key-456"
        assert "fastrouter" in creds["base_url"].lower()


# ── Model Normalization ───────────────────────────────────────────────────────

class TestFastRouterModelNormalization:
    def test_fastrouter_is_aggregator(self):
        from hermes_cli.model_normalize import _AGGREGATOR_PROVIDERS
        assert "fastrouter" in _AGGREGATOR_PROVIDERS

    def test_fastrouter_prepends_vendor(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider("claude-sonnet-4.6", "fastrouter")
        assert result == "anthropic/claude-sonnet-4.6"

    def test_fastrouter_prepends_openai_vendor(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider("gpt-5.4", "fastrouter")
        assert result == "openai/gpt-5.4"

    def test_fastrouter_vendor_already_present(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider("anthropic/claude-sonnet-4.6", "fastrouter")
        assert result == "anthropic/claude-sonnet-4.6"

    def test_fastrouter_preserves_dots(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider("minimax/minimax-m2.7", "fastrouter")
        assert result == "minimax/minimax-m2.7"


# ── Model Catalog ─────────────────────────────────────────────────────────────

class TestFastRouterModels:
    def test_static_list_structure(self):
        from hermes_cli.models import FASTROUTER_MODELS
        for entry in FASTROUTER_MODELS:
            assert isinstance(entry, tuple) and len(entry) == 2
            mid, desc = entry
            assert isinstance(mid, str) and len(mid) > 0
            assert isinstance(desc, str)

    def test_static_list_has_minimum_models(self):
        from hermes_cli.models import FASTROUTER_MODELS
        assert len(FASTROUTER_MODELS) >= 5

    def test_first_model_is_recommended(self):
        from hermes_cli.models import FASTROUTER_MODELS
        _, desc = FASTROUTER_MODELS[0]
        assert desc == "recommended"

    def test_all_models_have_vendor_prefix(self):
        from hermes_cli.models import FASTROUTER_MODELS
        for mid, _ in FASTROUTER_MODELS:
            assert "/" in mid, f"FastRouter model '{mid}' missing vendor/ prefix"

    def test_in_provider_models(self):
        from hermes_cli.models import _PROVIDER_MODELS, FASTROUTER_MODELS
        assert "fastrouter" in _PROVIDER_MODELS
        expected_ids = [mid for mid, _ in FASTROUTER_MODELS]
        assert _PROVIDER_MODELS["fastrouter"] == expected_ids

    def test_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        provider_slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "fastrouter" in provider_slugs


class TestFetchFastRouterModels:
    def test_live_fetch_recomputes_free_tags(self, monkeypatch):
        import hermes_cli.models as _models_mod

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps({
                    "data": [
                        {"id": "anthropic/claude-opus-4.6", "pricing": {"prompt": "0.000015", "completion": "0.000075"}},
                        {"id": "anthropic/claude-sonnet-4.6", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
                        {"id": "google/gemini-3-flash-preview", "pricing": {"prompt": "0", "completion": "0"}},
                    ]
                }).encode()

        monkeypatch.setattr(_models_mod, "_fastrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", return_value=_Resp()):
            from hermes_cli.models import fetch_fastrouter_models
            models = fetch_fastrouter_models(force_refresh=True)

        assert models[0] == ("anthropic/claude-opus-4.6", "recommended")
        assert ("anthropic/claude-sonnet-4.6", "") in models
        assert ("google/gemini-3-flash-preview", "free") in models

    def test_falls_back_to_static_on_error(self, monkeypatch):
        import hermes_cli.models as _models_mod
        from hermes_cli.models import FASTROUTER_MODELS

        monkeypatch.setattr(_models_mod, "_fastrouter_catalog_cache", None)
        with patch("hermes_cli.models.urllib.request.urlopen", side_effect=OSError("network error")):
            from hermes_cli.models import fetch_fastrouter_models
            models = fetch_fastrouter_models(force_refresh=True)

        assert models == FASTROUTER_MODELS

    def test_returns_cached_on_second_call(self, monkeypatch):
        import hermes_cli.models as _models_mod

        cached = [("anthropic/claude-opus-4.6", "recommended"), ("openai/gpt-5.4", "")]
        monkeypatch.setattr(_models_mod, "_fastrouter_catalog_cache", cached)

        from hermes_cli.models import fetch_fastrouter_models
        result = fetch_fastrouter_models()
        assert result == cached


# ── Auxiliary Client ──────────────────────────────────────────────────────────

class TestTryFastRouter:
    def test_returns_client_when_key_set(self, monkeypatch, _clean_env):
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-aux-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            from agent.auxiliary_client import _try_fastrouter
            client, model = _try_fastrouter()

        assert client is not None
        assert model == "google/gemini-3-flash-preview"
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "fr-aux-key"
        assert "fastrouter" in call_kwargs["base_url"].lower()

    def test_returns_none_when_no_key(self, monkeypatch, _clean_env):
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            from agent.auxiliary_client import _try_fastrouter
            client, model = _try_fastrouter()

        assert client is None
        assert model is None

    def test_pool_entry_takes_priority(self, _clean_env):
        class _Entry:
            access_token = "pooled-fr-token"
            agent_key = None
            inference_base_url = "https://pool.fastrouter.example/v1"

        class _Pool:
            def has_credentials(self):
                return True
            def select(self):
                return _Entry()

        with patch("agent.auxiliary_client.load_pool", return_value=_Pool()), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            from agent.auxiliary_client import _try_fastrouter
            client, model = _try_fastrouter()

        assert client is not None
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["api_key"] == "pooled-fr-token"
        assert call_kwargs["base_url"] == "https://pool.fastrouter.example/v1"


class TestExplicitFastRouterRouting:
    def test_resolve_provider_client_fastrouter(self, monkeypatch, _clean_env):
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-explicit")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("fastrouter")

        assert client is not None
        assert model is not None

    def test_resolve_provider_client_fastrouter_no_key_warns(self, _clean_env):
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            from agent.auxiliary_client import resolve_provider_client
            client, model = resolve_provider_client("fastrouter")

        assert client is None
        assert model is None


class TestFastRouterInProviderChain:
    def test_chain_includes_fastrouter(self):
        from agent.auxiliary_client import _get_provider_chain
        chain = _get_provider_chain()
        labels = [label for label, _ in chain]
        assert "fastrouter" in labels

    def test_fastrouter_after_openrouter_in_chain(self):
        from agent.auxiliary_client import _get_provider_chain
        chain = _get_provider_chain()
        labels = [label for label, _ in chain]
        or_idx = labels.index("openrouter")
        fr_idx = labels.index("fastrouter")
        assert fr_idx == or_idx + 1

    def test_fastrouter_in_aggregator_providers(self):
        from agent.auxiliary_client import _AGGREGATOR_PROVIDERS
        assert "fastrouter" in _AGGREGATOR_PROVIDERS

    def test_fastrouter_in_auto_provider_labels(self):
        from agent.auxiliary_client import _AUTO_PROVIDER_LABELS
        assert "_try_fastrouter" in _AUTO_PROVIDER_LABELS
        assert _AUTO_PROVIDER_LABELS["_try_fastrouter"] == "fastrouter"


class TestFastRouterVisionRouting:
    def test_strict_vision_backend_dispatches_to_try_fastrouter(self, monkeypatch, _clean_env):
        """provider='fastrouter' in _resolve_strict_vision_backend calls _try_fastrouter."""
        monkeypatch.setenv("FASTROUTER_API_KEY", "fr-vision-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            from agent.auxiliary_client import _resolve_strict_vision_backend
            client, model = _resolve_strict_vision_backend("fastrouter")

        assert client is not None


# ── AIAgent URL Detection ─────────────────────────────────────────────────────

def _make_agent(monkeypatch, provider, base_url, model=None):
    from run_agent import AIAgent

    def _tool_defs(*names):
        return [{"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": {"type": "object", "properties": {}}}} for n in names]

    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: _tool_defs("web_search", "terminal"))
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.api_key = kw.get("api_key", "test")
            self.base_url = kw.get("base_url", "http://test")
        def close(self):
            pass

    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    kwargs = dict(
        api_key="test-key",
        base_url=base_url,
        provider=provider,
        api_mode="chat_completions",
        max_iterations=4,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    if model:
        kwargs["model"] = model
    return AIAgent(**kwargs)


class TestAIAgentFastRouterURLDetection:
    def test_is_fastrouter_url_true(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        assert agent._is_fastrouter_url() is True

    def test_is_fastrouter_url_false_for_openrouter(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter", "https://openrouter.ai/api/v1")
        assert agent._is_fastrouter_url() is False

    def test_is_fastrouter_url_false_for_custom(self, monkeypatch):
        agent = _make_agent(monkeypatch, "custom", "http://localhost:1234/v1")
        assert agent._is_fastrouter_url() is False

    def test_is_aggregator_router_includes_fastrouter(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        assert agent._is_aggregator_router_url() is True

    def test_is_aggregator_router_includes_openrouter(self, monkeypatch):
        agent = _make_agent(monkeypatch, "openrouter", "https://openrouter.ai/api/v1")
        assert agent._is_aggregator_router_url() is True

    def test_is_aggregator_router_false_for_custom(self, monkeypatch):
        agent = _make_agent(monkeypatch, "custom", "http://localhost:1234/v1")
        assert agent._is_aggregator_router_url() is False


# ── AIAgent Build API Kwargs ──────────────────────────────────────────────────

class TestBuildApiKwargsFastRouter:
    def test_uses_chat_completions_format(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "messages" in kwargs
        assert "model" in kwargs
        assert kwargs["messages"][-1]["content"] == "hi"

    def test_no_responses_api_fields(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "input" not in kwargs
        assert "instructions" not in kwargs
        assert "store" not in kwargs

    def test_includes_tools(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "tools" in kwargs
        tool_names = [t["function"]["name"] for t in kwargs["tools"]]
        assert "web_search" in tool_names

    def test_includes_reasoning_in_extra_body_for_claude(self, monkeypatch):
        agent = _make_agent(
            monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4-20250514",
        )
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert "reasoning" in extra
        assert extra["reasoning"]["enabled"] is True

    def test_provider_preferences_in_extra_body(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        agent.providers_allowed = ["anthropic"]
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        extra = kwargs.get("extra_body", {})
        assert extra.get("provider", {}).get("only") == ["anthropic"]

    def test_claude_gets_max_tokens_on_fastrouter(self, monkeypatch):
        agent = _make_agent(
            monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1",
            model="anthropic/claude-opus-4.6",
        )
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "max_tokens" in kwargs

    def test_prompt_caching_enabled_for_claude_on_fastrouter(self, monkeypatch):
        agent = _make_agent(
            monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1",
            model="anthropic/claude-opus-4.6",
        )
        assert agent._use_prompt_caching is True


# ── Client Headers ────────────────────────────────────────────────────────────

class TestFastRouterClientHeaders:
    def test_fastrouter_default_headers(self, monkeypatch):
        agent = _make_agent(monkeypatch, "fastrouter", "https://api.fastrouter.ai/api/v1")
        headers = agent._client_kwargs.get("default_headers", {})
        assert headers.get("X-Title") == "Hermes Agent"
        assert "hermes-agent" in headers.get("HTTP-Referer", "").lower()


# ── Model Switch Curated List ────────────────────────────────────────────────

class TestFastRouterModelSwitch:
    def test_fastrouter_in_alias_fallback_providers(self):
        from hermes_cli.model_switch import _resolve_alias_fallback
        import inspect
        src = inspect.getsource(_resolve_alias_fallback)
        assert "fastrouter" in src

    def test_fastrouter_curated_list_populated(self):
        from hermes_cli.models import FASTROUTER_MODELS, _PROVIDER_MODELS
        assert "fastrouter" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["fastrouter"]) == len(FASTROUTER_MODELS)


# ── models.dev mapping ───────────────────────────────────────────────────────

class TestFastRouterModelsDev:
    def test_provider_to_models_dev_mapping(self):
        from agent.models_dev import PROVIDER_TO_MODELS_DEV
        assert "fastrouter" in PROVIDER_TO_MODELS_DEV
        assert PROVIDER_TO_MODELS_DEV["fastrouter"] == "fastrouter"


# ── Config / .env ─────────────────────────────────────────────────────────────

class TestFastRouterConfig:
    def test_fastrouter_in_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "FASTROUTER_API_KEY" in OPTIONAL_ENV_VARS

    def test_env_var_metadata(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        meta = OPTIONAL_ENV_VARS["FASTROUTER_API_KEY"]
        assert meta["category"] == "provider"
        assert meta["password"] is True
        assert "fastrouter" in meta["url"].lower()
