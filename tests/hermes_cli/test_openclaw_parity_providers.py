"""Tests for OpenClaw-parity OpenAI-compatible model provider plugins.

These providers are thin ``ProviderProfile`` plugins under
``plugins/model-providers/``. They should auto-wire into:

- ``providers`` registry
- ``PROVIDER_REGISTRY`` (auth)
- ``CANONICAL_PROVIDERS`` / model picker
- alias normalization (auth + models)
"""

from __future__ import annotations

import pytest

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
from hermes_cli.models import CANONICAL_PROVIDERS, normalize_provider
from providers import get_provider_profile, list_providers


# (canonical_id, sample_alias, api_key_env, expected_base_url_substr)
NEW_PROVIDERS = [
    ("groq", "groq-cloud", "GROQ_API_KEY", "api.groq.com"),
    ("mistral", "mistral-ai", "MISTRAL_API_KEY", "api.mistral.ai"),
    ("cohere", "cohere-ai", "COHERE_API_KEY", "api.cohere.ai"),
    ("together", "together-ai", "TOGETHER_API_KEY", "api.together.xyz"),
    ("cerebras", "cerebras-ai", "CEREBRAS_API_KEY", "api.cerebras.ai"),
    ("venice", "venice-ai", "VENICE_API_KEY", "api.venice.ai"),
    ("featherless", "featherless-ai", "FEATHERLESS_API_KEY", "api.featherless.ai"),
    ("baseten", "baseten-ai", "BASETEN_API_KEY", "inference.baseten.co"),
    ("chutes", "chutes-ai", "CHUTES_API_KEY", "llm.chutes.ai"),
    ("longcat", "longcat-ai", "LONGCAT_API_KEY", "api.longcat.chat"),
    ("vercel-ai-gateway", "vercel", "AI_GATEWAY_API_KEY", "ai-gateway.vercel.sh"),
    ("vllm", "vllm-local", "VLLM_API_KEY", "127.0.0.1:8000"),
    ("sglang", "sglang-local", "SGLANG_API_KEY", "127.0.0.1:30000"),
    ("litellm", "litellm-proxy", "LITELLM_API_KEY", "127.0.0.1:4000"),
    ("ollama-local", "ollama-local-daemon", "OLLAMA_LOCAL_API_KEY", "127.0.0.1:11434"),
    ("qianfan", "baidu-qianfan", "QIANFAN_API_KEY", "qianfan.baidubce.com"),
    ("byteplus", "byteplus-ark", "BYTEPLUS_API_KEY", "bytepluses.com"),
    ("volcengine", "doubao", "VOLCANO_ENGINE_API_KEY", "volces.com"),
]


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    keys = {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    }
    for _pid, _alias, env, _url in NEW_PROVIDERS:
        keys.add(env)
        keys.add(env.replace("_API_KEY", "_BASE_URL"))
        if env.endswith("_API_KEY"):
            # also clear common base url forms
            keys.add(env[: -len("_API_KEY")] + "_BASE_URL")
    # Explicit extras
    keys.update(
        {
            "GROQ_BASE_URL",
            "MISTRAL_BASE_URL",
            "COHERE_BASE_URL",
            "TOGETHER_BASE_URL",
            "CEREBRAS_BASE_URL",
            "VENICE_BASE_URL",
            "FEATHERLESS_BASE_URL",
            "BASETEN_BASE_URL",
            "CHUTES_BASE_URL",
            "LONGCAT_BASE_URL",
            "AI_GATEWAY_BASE_URL",
            "VLLM_BASE_URL",
            "SGLANG_BASE_URL",
            "LITELLM_BASE_URL",
            "OLLAMA_LOCAL_BASE_URL",
            "QIANFAN_BASE_URL",
            "BYTEPLUS_BASE_URL",
            "VOLCANO_ENGINE_BASE_URL",
        }
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)


class TestOpenClawParityProfiles:
    @pytest.mark.parametrize("pid,alias,env,url_part", NEW_PROVIDERS)
    def test_profile_registered(self, pid, alias, env, url_part):
        profile = get_provider_profile(pid)
        assert profile is not None, f"missing profile for {pid}"
        assert profile.name == pid
        assert profile.auth_type == "api_key"
        assert url_part in (profile.base_url or "")
        assert env in profile.env_vars

    @pytest.mark.parametrize("pid,alias,env,url_part", NEW_PROVIDERS)
    def test_auth_registry(self, pid, alias, env, url_part):
        assert pid in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY[pid]
        assert cfg.auth_type == "api_key"
        assert env in cfg.api_key_env_vars
        assert url_part in (cfg.inference_base_url or "")

    @pytest.mark.parametrize("pid,alias,env,url_part", NEW_PROVIDERS)
    def test_canonical_picker(self, pid, alias, env, url_part):
        slugs = {p.slug for p in CANONICAL_PROVIDERS}
        assert pid in slugs

    @pytest.mark.parametrize("pid,alias,env,url_part", NEW_PROVIDERS)
    def test_alias_resolve_auth(self, pid, alias, env, url_part, monkeypatch):
        monkeypatch.setenv(env, f"test-key-{pid}")
        assert resolve_provider(pid) == pid
        assert resolve_provider(alias) == pid

    @pytest.mark.parametrize("pid,alias,env,url_part", NEW_PROVIDERS)
    def test_alias_normalize_models(self, pid, alias, env, url_part):
        assert normalize_provider(pid) == pid
        # bare "ollama" remains the legacy custom alias — not ollama-local
        if alias != "ollama":
            assert normalize_provider(alias) == pid


class TestOpenClawParityEnvInjection:
    def test_optional_env_vars_include_new_keys(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        # MISTRAL_API_KEY pre-existed as a tool (Voxtral STT/TTS) env var before
        # the chat provider plugin; injection skips already-listed keys.
        pre_existing_tool_keys = {"MISTRAL_API_KEY"}

        for _pid, _alias, env, _url in NEW_PROVIDERS:
            assert env in OPTIONAL_ENV_VARS, f"{env} missing from OPTIONAL_ENV_VARS"
            assert OPTIONAL_ENV_VARS[env]["password"] is True
            if env in pre_existing_tool_keys:
                assert OPTIONAL_ENV_VARS[env]["category"] in {"provider", "tool"}
            else:
                assert OPTIONAL_ENV_VARS[env]["category"] == "provider"


class TestOllamaNamingContract:
    def test_bare_ollama_still_means_custom(self):
        # Historical Hermes contract: bare "ollama" → custom endpoint path.
        assert normalize_provider("ollama") == "custom"

    def test_ollama_cloud_unchanged(self):
        assert normalize_provider("ollama-cloud") == "ollama-cloud"
        assert normalize_provider("ollama_cloud") == "ollama-cloud"

    def test_ollama_local_first_class(self):
        assert get_provider_profile("ollama-local") is not None
        assert "ollama-local" in PROVIDER_REGISTRY


class TestListProvidersIncludesNew:
    def test_all_new_providers_listed(self):
        names = {p.name for p in list_providers()}
        missing = [pid for pid, *_ in NEW_PROVIDERS if pid not in names]
        assert missing == []
