"""Tests for SiliconFlow China provider support."""

import sys
import types

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)


_OTHER_PROVIDER_KEYS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "DASHSCOPE_API_KEY", "XAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY", "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "ARCEEAI_API_KEY", "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN", "GITHUB_TOKEN", "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
)


class TestSiliconFlowCnProviderRegistry:
    def test_registered(self):
        assert "siliconflow-cn" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["siliconflow-cn"].name == "SiliconFlow (China)"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["siliconflow-cn"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["siliconflow-cn"].inference_base_url == "https://api.siliconflow.cn/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["siliconflow-cn"].api_key_env_vars == ("SILICONFLOW_CN_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["siliconflow-cn"].base_url_env_var == "SILICONFLOW_CN_BASE_URL"


class TestSiliconFlowCnAliases:
    def test_alias_resolves(self, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("SILICONFLOW_CN_API_KEY", "sf-cn-test-12345678")
        assert resolve_provider("siliconflow-cn") == "siliconflow-cn"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider

        assert normalize_provider("siliconflow-cn") == "siliconflow-cn"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider

        assert normalize_provider("siliconflow-cn") == "siliconflow-cn"


class TestSiliconFlowCnAutoDetection:
    def test_auto_detect(self, monkeypatch):
        for var in _OTHER_PROVIDER_KEYS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("SILICONFLOW_CN_API_KEY", "sf-cn-test-12345678")
        assert resolve_provider("auto") == "siliconflow-cn"


class TestSiliconFlowCnCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("SILICONFLOW_CN_API_KEY", "sf-cn-test-12345678")
        status = get_api_key_provider_status("siliconflow-cn")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("SILICONFLOW_CN_API_KEY", raising=False)
        status = get_api_key_provider_status("siliconflow-cn")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_provider_configured(self, monkeypatch):
        monkeypatch.delenv("SILICONFLOW_CN_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("siliconflow-cn")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("SILICONFLOW_CN_API_KEY", "sf-cn-test-12345678")
        monkeypatch.delenv("SILICONFLOW_CN_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("siliconflow-cn")
        assert creds["api_key"] == "sf-cn-test-12345678"
        assert creds["base_url"] == "https://api.siliconflow.cn/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("SILICONFLOW_CN_API_KEY", "sf-cn-test-12345678")
        monkeypatch.setenv("SILICONFLOW_CN_BASE_URL", "https://custom.siliconflow.cn.example/v1")
        creds = resolve_api_key_provider_credentials("siliconflow-cn")
        assert creds["base_url"] == "https://custom.siliconflow.cn.example/v1"


class TestSiliconFlowCnModelCatalog:
    def test_static_model_list_fallback(self):
        from hermes_cli.models import _PROVIDER_MODELS

        assert "siliconflow-cn" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["siliconflow-cn"]
        assert models == [
            "Qwen/Qwen3.5-397B-A17B",
            "Pro/deepseek-ai/DeepSeek-V3.2",
            "Pro/zai-org/GLM-5.1",
            "Pro/zai-org/GLM-4.7",
            "Pro/moonshotai/Kimi-K2.5",
            "Pro/MiniMaxAI/MiniMax-M2.5",
        ]

    def test_setup_fallback_models_exist(self):
        from hermes_cli.setup import _DEFAULT_PROVIDER_MODELS

        assert "siliconflow-cn" in _DEFAULT_PROVIDER_MODELS
        assert _DEFAULT_PROVIDER_MODELS["siliconflow-cn"] == [
            "Qwen/Qwen3.5-397B-A17B",
            "Pro/deepseek-ai/DeepSeek-V3.2",
            "Pro/zai-org/GLM-5.1",
            "Pro/zai-org/GLM-4.7",
            "Pro/moonshotai/Kimi-K2.5",
            "Pro/MiniMaxAI/MiniMax-M2.5",
        ]
        assert "Pro/zai-org/GLM-5.1" in _DEFAULT_PROVIDER_MODELS["siliconflow-cn"]

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "siliconflow-cn" in slugs


class TestSiliconFlowCnNormalization:
    def test_is_aggregator_provider(self):
        from hermes_cli.model_normalize import _AGGREGATOR_PROVIDERS

        assert "siliconflow-cn" in _AGGREGATOR_PROVIDERS

    def test_passthrough_when_vendor_already_present(self):
        from hermes_cli.model_normalize import normalize_model_for_provider

        model = "Pro/zai-org/GLM-5.1"
        assert normalize_model_for_provider(model, "siliconflow-cn") == model

    def test_provider_module_marks_as_aggregator(self):
        from hermes_cli.providers import is_aggregator

        assert is_aggregator("siliconflow-cn") is True


class TestSiliconFlowCnURLMapping:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.siliconflow.cn") == "siliconflow-cn"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "siliconflow-cn" in _PROVIDER_PREFIXES

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.siliconflow.cn/v1") == "siliconflow-cn"

    def test_trajectory_compressor_detects_siliconflow_cn(self):
        import trajectory_compressor as tc

        comp = tc.TrajectoryCompressor.__new__(tc.TrajectoryCompressor)
        comp.config = types.SimpleNamespace(base_url="https://api.siliconflow.cn/v1")
        assert comp._detect_provider() == "siliconflow-cn"


class TestSiliconFlowCnProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "siliconflow-cn" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["siliconflow-cn"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "SILICONFLOW_CN_BASE_URL"
        assert overlay.is_aggregator

    def test_label(self):
        from hermes_cli.providers import get_label

        assert get_label("siliconflow-cn") == "SiliconFlow (China)"

    def test_get_provider(self):
        from hermes_cli.providers import get_provider

        pdef = get_provider("siliconflow-cn")
        assert pdef is not None
        assert pdef.id == "siliconflow-cn"
        assert pdef.transport == "openai_chat"
        assert pdef.is_aggregator is True


class TestSiliconFlowCnDoctor:
    def test_provider_env_hints(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "SILICONFLOW_CN_API_KEY" in _PROVIDER_ENV_HINTS


class TestSiliconFlowCnAgentInit:
    def test_api_mode_is_chat_completions(self):
        from hermes_cli.providers import HERMES_OVERLAYS, TRANSPORT_TO_API_MODE

        overlay = HERMES_OVERLAYS["siliconflow-cn"]
        api_mode = TRANSPORT_TO_API_MODE[overlay.transport]
        assert api_mode == "chat_completions"
