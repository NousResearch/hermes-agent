"""Tests for Cloudflare Workers AI provider support."""

from __future__ import annotations

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)


_ACCOUNT = "0123456789abcdef0123456789abcdef"
_EXPECTED_URL = f"https://api.cloudflare.com/client/v4/accounts/{_ACCOUNT}/ai/v1"

_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
    "XAI_API_KEY", "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY", "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN", "GITHUB_TOKEN", "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in _OTHER_PROVIDER_KEYS + (
        "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestWorkersAIRegistry:
    def test_registered(self):
        assert "workers-ai" in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["workers-ai"]
        assert pconfig.name == "Cloudflare Workers AI"
        assert pconfig.auth_type == "api_key"
        assert pconfig.api_key_env_vars == ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_API_KEY")
        assert pconfig.base_url_env_var == "CLOUDFLARE_BASE_URL"
        assert pconfig.inference_base_url == ""

    def test_profile_exists(self):
        from providers import get_provider_profile
        profile = get_provider_profile("workers-ai")
        assert profile is not None
        assert profile.display_name == "Cloudflare Workers AI"
        assert "@cf/moonshotai/kimi-k2.6" in profile.fallback_models


class TestWorkersAIAliases:
    @pytest.mark.parametrize("alias", [
        "workers-ai", "cloudflare", "cloudflare-workers-ai",
        "cf-workers-ai", "workersai", "cloudflare-ai",
    ])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        assert resolve_provider(alias) == "workers-ai"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("cloudflare") == "workers-ai"
        assert normalize_provider("cloudflare-workers-ai") == "workers-ai"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as np
        assert np("cloudflare") == "workers-ai"
        assert np("workersai") == "workers-ai"


class TestWorkersAICredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        status = get_api_key_provider_status("workers-ai")
        assert status["configured"]
        assert status["base_url"] == _EXPECTED_URL

    def test_openrouter_key_does_not_configure(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("workers-ai")
        assert not status["configured"]

    def test_resolve_credentials_from_account_id(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-direct-key")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        creds = resolve_api_key_provider_credentials("workers-ai")
        assert creds["api_key"] == "cf-direct-key"
        assert creds["base_url"] == _EXPECTED_URL

    def test_api_key_alias(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-legacy-key")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        creds = resolve_api_key_provider_credentials("workers-ai")
        assert creds["api_key"] == "cf-legacy-key"

    def test_base_url_override_wins(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-direct-key")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://example.test/ai/v1")
        creds = resolve_api_key_provider_credentials("workers-ai")
        assert creds["base_url"] == "https://example.test/ai/v1"

    def test_token_preferred_over_api_key(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "token-wins")
        monkeypatch.setenv("CLOUDFLARE_API_KEY", "key-loses")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        creds = resolve_api_key_provider_credentials("workers-ai")
        assert creds["api_key"] == "token-wins"


class TestWorkersAIModelCatalog:
    def test_static_model_list(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "workers-ai" in _PROVIDER_MODELS
        assert "@cf/moonshotai/kimi-k2.6" in _PROVIDER_MODELS["workers-ai"]
        assert len(_PROVIDER_MODELS["workers-ai"]) >= 8

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "workers-ai" in slugs

    def test_live_first_picker(self):
        from hermes_cli.models import _LIVE_FIRST_PICKER_PROVIDERS
        assert "workers-ai" in _LIVE_FIRST_PICKER_PROVIDERS


class TestWorkersAIProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["workers-ai"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "CLOUDFLARE_BASE_URL"
        assert "CLOUDFLARE_API_TOKEN" in overlay.extra_env_vars
        assert not overlay.is_aggregator

    def test_optional_env_vars(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS
        assert "CLOUDFLARE_API_TOKEN" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["CLOUDFLARE_API_TOKEN"]["password"] is True
        assert "CLOUDFLARE_ACCOUNT_ID" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["CLOUDFLARE_ACCOUNT_ID"]["password"] is False

    def test_doctor_hints(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS
        assert "CLOUDFLARE_API_TOKEN" in _PROVIDER_ENV_HINTS


class TestWorkersAIUrlMapping:
    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "workers-ai" in _PROVIDER_PREFIXES
        assert "cloudflare" in _PROVIDER_PREFIXES

    def test_hostname(self, monkeypatch):
        from providers import get_provider_profile
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", _ACCOUNT)
        profile = get_provider_profile("workers-ai")
        assert profile.get_hostname() == "api.cloudflare.com"


class TestWorkersAICatalogPayload:
    def _mod(self):
        from providers import get_provider_profile
        import sys
        profile = get_provider_profile("workers-ai")
        return sys.modules[profile.resolve_base_url.__self__.__class__.__module__]

    def test_openai_and_cloudflare_envelopes(self):
        workers = self._mod()
        openai_shape = {"data": [{"id": "@cf/zai-org/glm-5.3"}, {"id": "@cf/moonshotai/kimi-k2.6"}]}
        cf_shape = {"success": True, "result": [{"id": "@cf/zai-org/glm-5.3"}, {"name": "@cf/moonshotai/kimi-k2.6"}]}
        assert workers._ids_from_catalog_payload(openai_shape) == [
            "@cf/zai-org/glm-5.3", "@cf/moonshotai/kimi-k2.6",
        ]
        assert workers._ids_from_catalog_payload(cf_shape) == [
            "@cf/zai-org/glm-5.3", "@cf/moonshotai/kimi-k2.6",
        ]

    def test_prefers_name_over_uuid_id(self):
        workers = self._mod()
        payload = {"success": True, "result": [
            {
                "id": "f9f2250b-1048-4a52-9910-d0bf976616a1",
                "name": "@cf/zai-org/glm-5.3",
                "task": {"name": "Text Generation"},
            },
            {
                "id": "e8e8abe4-a372-4c13-815f-4688ba655c8e",
                "name": "@cf/baai/bge-large-en-v1.5",
                "task": {"name": "Text Embeddings"},
            },
        ]}
        assert workers._ids_from_catalog_payload(payload, text_generation_only=True) == [
            "@cf/zai-org/glm-5.3",
        ]
        assert workers._ids_from_catalog_payload(payload) == [
            "@cf/zai-org/glm-5.3", "@cf/baai/bge-large-en-v1.5",
        ]

    def test_filters_non_text_models(self):
        workers = self._mod()
        payload = {"success": True, "result": [
            {"id": "@cf/zai-org/glm-5.3", "task": {"name": "Text Generation"}},
            {"id": "@cf/baai/bge-large-en-v1.5", "task": {"name": "Text Embeddings"}},
            {"id": "@cf/black-forest-labs/flux-1-schnell", "task": {"name": "Text-to-Image"}},
            {"id": "@cf/openai/whisper", "task": {"name": "Automatic Speech Recognition"}},
        ]}
        assert workers._ids_from_catalog_payload(payload, text_generation_only=True) == [
            "@cf/zai-org/glm-5.3",
        ]

    def test_search_url(self):
        workers = self._mod()
        url = workers.models_search_url(_ACCOUNT, page=2)
        assert f"/accounts/{_ACCOUNT}/ai/models/search" in url
        assert "page=2" in url

    def test_account_id_from_base_url(self):
        workers = self._mod()
        assert workers.account_id_from_base_url(_EXPECTED_URL) == _ACCOUNT

    def test_fetch_models_uses_search(self, monkeypatch):
        workers = self._mod()
        urls = []

        class _Resp:
            def read(self):
                return b'{"success":true,"result":[{"id":"@cf/zai-org/glm-5.3","task":{"name":"Text Generation"}},{"id":"@cf/moonshotai/kimi-k2.6","task":{"name":"Text Generation"}}],"result_info":{"total_pages":1}}'

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_open(req, timeout=8.0):
            urls.append(req.full_url)
            return _Resp()

        monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", fake_open)
        ids = workers.fetch_workers_ai_models(
            api_key="tok", account_id=_ACCOUNT, openai_base_url=_EXPECTED_URL)
        assert ids == ["@cf/zai-org/glm-5.3", "@cf/moonshotai/kimi-k2.6"]
        assert any("/ai/models/search" in u for u in urls)

    def test_setup_flow_uses_live_catalog(self, monkeypatch):
        from hermes_cli.model_setup_flows import _workers_ai_models
        monkeypatch.setattr(
            "providers.get_provider_profile",
            lambda name: type("P", (), {
                "fetch_models": staticmethod(lambda **kw: [
                    "@cf/zai-org/glm-5.3", "@cf/deepseek-ai/deepseek-v4-pro-0813",
                ]),
            })() if name == "workers-ai" else None,
        )
        out = _workers_ai_models(None, ["@cf/moonshotai/kimi-k2.6"], "tok", _EXPECTED_URL)
        assert out[0] == "@cf/zai-org/glm-5.3"
        assert "@cf/moonshotai/kimi-k2.6" in out
        assert len(out) == 3


class TestWorkersAISetupBase:
    def test_url_for_account(self):
        from hermes_cli.model_setup_flows import _workers_ai_url_for_account
        assert _workers_ai_url_for_account(_ACCOUNT) == _EXPECTED_URL
        assert _workers_ai_url_for_account("") == ""
        assert _workers_ai_url_for_account(f'"{_ACCOUNT}"') == _EXPECTED_URL

    def test_first_time_prompts_and_saves(self, monkeypatch):
        from hermes_cli.model_setup_flows import _workers_ai_effective_base
        saved: dict[str, str] = {}
        monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: "")
        monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
        monkeypatch.setattr("hermes_cli.model_setup_flows._ask", lambda *a, **k: _ACCOUNT)
        assert _workers_ai_effective_base() == _EXPECTED_URL
        assert saved["CLOUDFLARE_ACCOUNT_ID"] == _ACCOUNT
        assert saved["CLOUDFLARE_BASE_URL"] == ""

    def test_keep_existing_account_id(self, monkeypatch):
        from hermes_cli.model_setup_flows import _workers_ai_effective_base
        saved: dict[str, str] = {}
        monkeypatch.setattr(
            "hermes_cli.config.get_env_value",
            lambda name: _ACCOUNT if name == "CLOUDFLARE_ACCOUNT_ID" else "",
        )
        monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
        monkeypatch.setattr("hermes_cli.model_setup_flows._ask", lambda *a, **k: "k")
        assert _workers_ai_effective_base() == _EXPECTED_URL
        assert saved == {}

    def test_replace_account_id(self, monkeypatch):
        from hermes_cli.model_setup_flows import _workers_ai_effective_base
        new_id = "ffffffffffffffffffffffffffffffff"
        answers = iter(["r", new_id])
        saved: dict[str, str] = {}
        monkeypatch.setattr(
            "hermes_cli.config.get_env_value",
            lambda name: _ACCOUNT if name == "CLOUDFLARE_ACCOUNT_ID" else "",
        )
        monkeypatch.setattr("hermes_cli.config.save_env_value", lambda k, v: saved.__setitem__(k, v))
        monkeypatch.setattr("hermes_cli.model_setup_flows._ask", lambda *a, **k: next(answers))
        assert _workers_ai_effective_base() == (
            f"https://api.cloudflare.com/client/v4/accounts/{new_id}/ai/v1"
        )
        assert saved["CLOUDFLARE_ACCOUNT_ID"] == new_id
        assert saved["CLOUDFLARE_BASE_URL"] == ""

    def test_first_time_cancel(self, monkeypatch):
        from hermes_cli.model_setup_flows import _workers_ai_effective_base
        monkeypatch.setattr("hermes_cli.config.get_env_value", lambda name: "")
        monkeypatch.setattr("hermes_cli.model_setup_flows._ask", lambda *a, **k: "")
        assert _workers_ai_effective_base() == ""
