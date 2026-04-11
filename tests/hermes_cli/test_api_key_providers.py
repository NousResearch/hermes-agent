"""Tests for API-key provider support (z.ai/GLM, Kimi, MiniMax, AI Gateway)."""

import os

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    ProviderConfig,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
    get_auth_status,
    AuthError,
    KIMI_CODE_BASE_URL,
    STEPFUN_STEP_PLAN_INTL_BASE_URL,
    STEPFUN_STEP_PLAN_CN_BASE_URL,
    _resolve_kimi_base_url,
)
from hermes_cli.copilot_auth import _try_gh_cli_token


# =============================================================================
# Provider Registry tests
# =============================================================================

class TestProviderRegistry:
    """Test that new providers are correctly registered."""

    @pytest.mark.parametrize("provider_id,name,auth_type", [
        ("copilot-acp", "GitHub Copilot ACP", "external_process"),
        ("copilot", "GitHub Copilot", "api_key"),
        ("huggingface", "Hugging Face", "api_key"),
        ("zai", "Z.AI / GLM", "api_key"),
        ("xai", "xAI", "api_key"),
        ("nvidia", "NVIDIA NIM", "api_key"),
        ("kimi-coding", "Kimi / Moonshot", "api_key"),
        ("stepfun", "StepFun Step Plan", "api_key"),
        ("minimax", "MiniMax", "api_key"),
        ("minimax-cn", "MiniMax (China)", "api_key"),
        ("ai-gateway", "Vercel AI Gateway", "api_key"),
        ("kilocode", "Kilo Code", "api_key"),
        ("gmi", "GMI Cloud", "api_key"),
    ])
    def test_provider_registered(self, provider_id, name, auth_type):
        assert provider_id in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY[provider_id]
        assert pconfig.name == name
        assert pconfig.auth_type == auth_type
        assert pconfig.inference_base_url  # must have a default base URL

    def test_zai_env_vars(self):
        pconfig = PROVIDER_REGISTRY["zai"]
        assert pconfig.api_key_env_vars == ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY")
        assert pconfig.base_url_env_var == "GLM_BASE_URL"

    def test_xai_env_vars(self):
        pconfig = PROVIDER_REGISTRY["xai"]
        assert pconfig.api_key_env_vars == ("XAI_API_KEY",)
        assert pconfig.base_url_env_var == "XAI_BASE_URL"
        assert pconfig.inference_base_url == "https://api.x.ai/v1"

    def test_nvidia_env_vars(self):
        pconfig = PROVIDER_REGISTRY["nvidia"]
        assert pconfig.api_key_env_vars == ("NVIDIA_API_KEY",)
        assert pconfig.base_url_env_var == "NVIDIA_BASE_URL"
        assert pconfig.inference_base_url == "https://integrate.api.nvidia.com/v1"

    def test_copilot_env_vars(self):
        pconfig = PROVIDER_REGISTRY["copilot"]
        assert pconfig.api_key_env_vars == ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
        assert pconfig.base_url_env_var == "COPILOT_API_BASE_URL"

    def test_kimi_env_vars(self):
        pconfig = PROVIDER_REGISTRY["kimi-coding"]
        # KIMI_API_KEY is the primary env var; KIMI_CODING_API_KEY is a
        # secondary fallback for Kimi Code sk-kimi- keys so users don't
        # have to overload the same variable.
        assert "KIMI_API_KEY" in pconfig.api_key_env_vars
        assert "KIMI_CODING_API_KEY" in pconfig.api_key_env_vars
        assert pconfig.base_url_env_var == "KIMI_BASE_URL"

    def test_minimax_env_vars(self):
        pconfig = PROVIDER_REGISTRY["minimax"]
        assert pconfig.api_key_env_vars == ("MINIMAX_API_KEY",)
        assert pconfig.base_url_env_var == "MINIMAX_BASE_URL"

    def test_stepfun_env_vars(self):
        pconfig = PROVIDER_REGISTRY["stepfun"]
        assert pconfig.api_key_env_vars == ("STEPFUN_API_KEY",)
        assert pconfig.base_url_env_var == "STEPFUN_BASE_URL"

    def test_minimax_cn_env_vars(self):
        pconfig = PROVIDER_REGISTRY["minimax-cn"]
        assert pconfig.api_key_env_vars == ("MINIMAX_CN_API_KEY",)
        assert pconfig.base_url_env_var == "MINIMAX_CN_BASE_URL"

    def test_ai_gateway_env_vars(self):
        pconfig = PROVIDER_REGISTRY["ai-gateway"]
        assert pconfig.api_key_env_vars == ("AI_GATEWAY_API_KEY",)
        assert pconfig.base_url_env_var == "AI_GATEWAY_BASE_URL"

    def test_kilocode_env_vars(self):
        pconfig = PROVIDER_REGISTRY["kilocode"]
        assert pconfig.api_key_env_vars == ("KILOCODE_API_KEY",)
        assert pconfig.base_url_env_var == "KILOCODE_BASE_URL"

    def test_gmi_env_vars(self):
        pconfig = PROVIDER_REGISTRY["gmi"]
        assert pconfig.api_key_env_vars == ("GMI_API_KEY",)
        assert pconfig.base_url_env_var == "GMI_BASE_URL"

    def test_huggingface_env_vars(self):
        pconfig = PROVIDER_REGISTRY["huggingface"]
        assert pconfig.api_key_env_vars == ("HF_TOKEN",)
        assert pconfig.base_url_env_var == "HF_BASE_URL"

    def test_base_urls(self):
        assert PROVIDER_REGISTRY["copilot"].inference_base_url == "https://api.githubcopilot.com"
        assert PROVIDER_REGISTRY["copilot-acp"].inference_base_url == "acp://copilot"
        assert PROVIDER_REGISTRY["zai"].inference_base_url == "https://api.z.ai/api/paas/v4"
        assert PROVIDER_REGISTRY["kimi-coding"].inference_base_url == "https://api.moonshot.ai/v1"
        assert PROVIDER_REGISTRY["stepfun"].inference_base_url == STEPFUN_STEP_PLAN_INTL_BASE_URL
        assert PROVIDER_REGISTRY["minimax"].inference_base_url == "https://api.minimax.io/anthropic"
        assert PROVIDER_REGISTRY["minimax-cn"].inference_base_url == "https://api.minimaxi.com/anthropic"
        assert PROVIDER_REGISTRY["ai-gateway"].inference_base_url == "https://ai-gateway.vercel.sh/v1"
        assert PROVIDER_REGISTRY["kilocode"].inference_base_url == "https://api.kilo.ai/api/gateway"
        assert PROVIDER_REGISTRY["gmi"].inference_base_url == "https://api.gmi-serving.com/v1"
        assert PROVIDER_REGISTRY["huggingface"].inference_base_url == "https://router.huggingface.co/v1"

    def test_oauth_providers_unchanged(self):
        """Ensure we didn't break the existing OAuth providers."""
        assert "nous" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["nous"].auth_type == "oauth_device_code"
        assert "openai-codex" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["openai-codex"].auth_type == "oauth_external"


# =============================================================================
# Provider Resolution tests
# =============================================================================

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    "LM_API_KEY", "LM_BASE_URL",
    "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY",
    "KIMI_API_KEY", "KIMI_BASE_URL", "STEPFUN_API_KEY", "STEPFUN_BASE_URL",
    "MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_CN_API_KEY", "MINIMAX_CN_BASE_URL",
    "AI_GATEWAY_API_KEY", "AI_GATEWAY_BASE_URL",
    "KILOCODE_API_KEY", "KILOCODE_BASE_URL",
    "GMI_API_KEY", "GMI_BASE_URL",
    "DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL",
    "XAI_API_KEY", "XAI_BASE_URL", "HF_TOKEN", "HF_BASE_URL", "XIAOMI_API_KEY", "XIAOMI_BASE_URL",
    "OPENCODE_ZEN_API_KEY", "OPENCODE_ZEN_BASE_URL", "OPENCODE_GO_API_KEY", "OPENCODE_GO_BASE_URL",
    "NOUS_API_KEY", "GITHUB_TOKEN", "GH_TOKEN",
    "HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH",
    "HERMES_COPILOT_ACP_ARGS", "COPILOT_ACP_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})


class TestResolveProvider:
    """Test resolve_provider() with new providers."""

    def test_explicit_zai(self):
        assert resolve_provider("zai") == "zai"

    def test_explicit_kimi_coding(self):
        assert resolve_provider("kimi-coding") == "kimi-coding"

    def test_explicit_stepfun(self):
        assert resolve_provider("stepfun") == "stepfun"

    def test_explicit_minimax(self):
        assert resolve_provider("minimax") == "minimax"

    def test_explicit_minimax_cn(self):
        assert resolve_provider("minimax-cn") == "minimax-cn"

    def test_explicit_ai_gateway(self):
        assert resolve_provider("ai-gateway") == "ai-gateway"

    def test_explicit_gmi(self):
        assert resolve_provider("gmi") == "gmi"

    def test_alias_glm(self):
        assert resolve_provider("glm") == "zai"

    def test_alias_z_ai(self):
        assert resolve_provider("z-ai") == "zai"

    def test_alias_zhipu(self):
        assert resolve_provider("zhipu") == "zai"

    def test_alias_kimi(self):
        assert resolve_provider("kimi") == "kimi-coding"

    def test_alias_moonshot(self):
        assert resolve_provider("moonshot") == "kimi-coding"

    def test_alias_step(self):
        assert resolve_provider("step") == "stepfun"

    def test_alias_minimax_underscore(self):
        assert resolve_provider("minimax_cn") == "minimax-cn"

    def test_alias_aigateway(self):
        assert resolve_provider("aigateway") == "ai-gateway"

    def test_alias_vercel(self):
        assert resolve_provider("vercel") == "ai-gateway"

    def test_alias_gmi_cloud(self):
        assert resolve_provider("gmi-cloud") == "gmi"

    def test_explicit_kilocode(self):
        assert resolve_provider("kilocode") == "kilocode"

    def test_alias_kilo(self):
        assert resolve_provider("kilo") == "kilocode"

    def test_alias_kilo_code(self):
        assert resolve_provider("kilo-code") == "kilocode"

    def test_alias_kilo_gateway(self):
        assert resolve_provider("kilo-gateway") == "kilocode"

    def test_alias_case_insensitive(self):
        assert resolve_provider("GLM") == "zai"
        assert resolve_provider("Z-AI") == "zai"
        assert resolve_provider("Kimi") == "kimi-coding"

    def test_alias_github_copilot(self):
        assert resolve_provider("github-copilot") == "copilot"

    def test_alias_github_models(self):
        assert resolve_provider("github-models") == "copilot"

    def test_alias_github_copilot_acp(self):
        assert resolve_provider("github-copilot-acp") == "copilot-acp"
        assert resolve_provider("copilot-acp-agent") == "copilot-acp"

    def test_explicit_huggingface(self):
        assert resolve_provider("huggingface") == "huggingface"

    def test_alias_hf(self):
        assert resolve_provider("hf") == "huggingface"

    def test_alias_hugging_face(self):
        assert resolve_provider("hugging-face") == "huggingface"

    def test_alias_huggingface_hub(self):
        assert resolve_provider("huggingface-hub") == "huggingface"

    def test_unknown_provider_raises(self):
        with pytest.raises(AuthError):
            resolve_provider("nonexistent-provider-xyz")

    def test_auto_detects_glm_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")
        assert resolve_provider("auto") == "zai"

    def test_auto_detects_zai_key(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
        assert resolve_provider("auto") == "zai"

    def test_auto_detects_z_ai_key(self, monkeypatch):
        monkeypatch.setenv("Z_AI_API_KEY", "test-z-ai-key")
        assert resolve_provider("auto") == "zai"

    def test_auto_detects_kimi_key(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
        assert resolve_provider("auto") == "kimi-coding"

    def test_auto_detects_stepfun_key(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "test-stepfun-key")
        assert resolve_provider("auto") == "stepfun"

    def test_auto_detects_minimax_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-mm-key")
        assert resolve_provider("auto") == "minimax"

    def test_auto_detects_minimax_cn_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-mm-cn-key")
        assert resolve_provider("auto") == "minimax-cn"

    def test_auto_detects_ai_gateway_key(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-gw-key")
        assert resolve_provider("auto") == "ai-gateway"

    def test_auto_detects_gmi_key(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "test-gmi-key")
        assert resolve_provider("auto") == "gmi"

    def test_auto_detects_kilocode_key(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "test-kilo-key")
        assert resolve_provider("auto") == "kilocode"

    def test_auto_detects_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_test_token")
        assert resolve_provider("auto") == "huggingface"

    def test_openrouter_takes_priority_over_glm(self, monkeypatch):
        """OpenRouter API key should win over GLM in auto-detection."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        assert resolve_provider("auto") == "openrouter"

    def test_auto_does_not_select_copilot_from_github_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "gh-test-token")
        with pytest.raises(AuthError, match="No inference provider configured"):
            resolve_provider("auto")


# =============================================================================
# API Key Provider Status tests
# =============================================================================

class TestApiKeyProviderStatus:

    def test_unconfigured_provider(self):
        status = get_api_key_provider_status("zai")
        assert status["configured"] is False
        assert status["logged_in"] is False

    def test_configured_provider(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-key-123")
        status = get_api_key_provider_status("zai")
        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["key_source"] == "GLM_API_KEY"
        assert "z.ai" in status["base_url"].lower() or "api.z.ai" in status["base_url"]

    def test_fallback_env_var(self, monkeypatch):
        """ZAI_API_KEY should work when GLM_API_KEY is not set."""
        monkeypatch.setenv("ZAI_API_KEY", "zai-fallback-key")
        status = get_api_key_provider_status("zai")
        assert status["configured"] is True
        assert status["key_source"] == "ZAI_API_KEY"

    def test_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://custom.kimi.example/v1")
        status = get_api_key_provider_status("kimi-coding")
        assert status["base_url"] == "https://custom.kimi.example/v1"

    def test_stepfun_status_uses_configured_base_url(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", STEPFUN_STEP_PLAN_CN_BASE_URL)
        status = get_api_key_provider_status("stepfun")
        assert status["configured"] is True
        assert status["base_url"] == STEPFUN_STEP_PLAN_CN_BASE_URL

    def test_copilot_status_uses_gh_cli_token(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "gho_gh_cli_token")
        status = get_api_key_provider_status("copilot")
        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["key_source"] == "gh auth token"
        assert status["base_url"] == "https://api.githubcopilot.com"

    def test_get_auth_status_dispatches_to_api_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        status = get_auth_status("minimax")
        assert status["configured"] is True
        assert status["provider"] == "minimax"

    def test_copilot_acp_status_detects_local_cli(self, monkeypatch):
        monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio --debug")
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/local/bin/{command}")

        status = get_external_process_provider_status("copilot-acp")

        assert status["configured"] is True
        assert status["logged_in"] is True
        assert status["command"] == "copilot"
        assert status["resolved_command"] == "/usr/local/bin/copilot"
        assert status["args"] == ["--acp", "--stdio", "--debug"]
        assert status["base_url"] == "acp://copilot"

    def test_get_auth_status_dispatches_to_external_process(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/opt/bin/{command}")

        status = get_auth_status("copilot-acp")

        assert status["configured"] is True
        assert status["provider"] == "copilot-acp"

    def test_non_api_key_provider(self):
        status = get_api_key_provider_status("nous")
        assert status["configured"] is False


# =============================================================================
# Credential Resolution tests
# =============================================================================

class TestResolveApiKeyProviderCredentials:

    def test_resolve_zai_with_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-secret-key")
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["provider"] == "zai"
        assert creds["api_key"] == "glm-secret-key"
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"
        assert creds["source"] == "GLM_API_KEY"

    def test_resolve_copilot_with_github_token(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "gh-env-secret")
        creds = resolve_api_key_provider_credentials("copilot")
        assert creds["provider"] == "copilot"
        assert creds["api_key"] == "gh-env-secret"
        assert creds["base_url"] == "https://api.githubcopilot.com"
        assert creds["source"] == "GITHUB_TOKEN"

    def test_resolve_copilot_with_gh_cli_fallback(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "gho_cli_secret")
        creds = resolve_api_key_provider_credentials("copilot")
        assert creds["provider"] == "copilot"
        assert creds["api_key"] == "gho_cli_secret"
        assert creds["base_url"] == "https://api.githubcopilot.com"
        assert creds["source"] == "gh auth token"

    def test_resolve_lmstudio_uses_token_and_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("LM_API_KEY", "lm-token")
        monkeypatch.setenv("LM_BASE_URL", "http://lmstudio.remote:4321/v1")

        creds = resolve_api_key_provider_credentials("lmstudio")

        assert creds["provider"] == "lmstudio"
        assert creds["api_key"] == "lm-token"
        assert creds["base_url"] == "http://lmstudio.remote:4321/v1"

    def test_resolve_lmstudio_no_api_key_substitutes_placeholder(self, monkeypatch):
        # No-auth LM Studio: when LM_API_KEY isn't set, runtime credentials
        # carry a placeholder so gateway/TUI/cron paths see the local server
        # as configured. get_api_key_provider_status still reports unconfigured.
        monkeypatch.delenv("LM_API_KEY", raising=False)
        monkeypatch.delenv("LM_BASE_URL", raising=False)

        creds = resolve_api_key_provider_credentials("lmstudio")

        assert creds["provider"] == "lmstudio"
        assert creds["api_key"] == "dummy-lm-api-key"
        assert creds["base_url"] == "http://127.0.0.1:1234/v1"

    def test_try_gh_cli_token_uses_homebrew_path_when_not_on_path(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.copilot_auth.shutil.which", lambda command: None)
        monkeypatch.setattr(
            "hermes_cli.copilot_auth.os.path.isfile",
            lambda path: path == "/opt/homebrew/bin/gh",
        )
        monkeypatch.setattr(
            "hermes_cli.copilot_auth.os.access",
            lambda path, mode: path == "/opt/homebrew/bin/gh" and mode == os.X_OK,
        )

        calls = []

        class _Result:
            returncode = 0
            stdout = "gh-cli-secret\n"

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Result()

        monkeypatch.setattr("hermes_cli.copilot_auth.subprocess.run", _fake_run)

        assert _try_gh_cli_token() == "gh-cli-secret"
        assert calls == [["/opt/homebrew/bin/gh", "auth", "token"]]

    def test_resolve_copilot_acp_with_local_cli(self, monkeypatch):
        monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio")
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/local/bin/{command}")

        creds = resolve_external_process_provider_credentials("copilot-acp")

        assert creds["provider"] == "copilot-acp"
        assert creds["api_key"] == "copilot-acp"
        assert creds["base_url"] == "acp://copilot"
        assert creds["command"] == "/usr/local/bin/copilot"
        assert creds["args"] == ["--acp", "--stdio"]
        assert creds["source"] == "process"

    def test_resolve_kimi_with_key(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["provider"] == "kimi-coding"
        assert creds["api_key"] == "kimi-secret-key"
        assert creds["base_url"] == "https://api.moonshot.ai/v1"

    def test_resolve_stepfun_with_key(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-secret-key")
        creds = resolve_api_key_provider_credentials("stepfun")
        assert creds["provider"] == "stepfun"
        assert creds["api_key"] == "stepfun-secret-key"
        assert creds["base_url"] == STEPFUN_STEP_PLAN_INTL_BASE_URL

    def test_resolve_stepfun_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-secret-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", STEPFUN_STEP_PLAN_CN_BASE_URL)
        creds = resolve_api_key_provider_credentials("stepfun")
        assert creds["base_url"] == STEPFUN_STEP_PLAN_CN_BASE_URL

    def test_resolve_minimax_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-secret-key")
        creds = resolve_api_key_provider_credentials("minimax")
        assert creds["provider"] == "minimax"
        assert creds["api_key"] == "mm-secret-key"
        assert creds["base_url"] == "https://api.minimax.io/anthropic"

    def test_resolve_minimax_cn_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "mmcn-secret-key")
        creds = resolve_api_key_provider_credentials("minimax-cn")
        assert creds["provider"] == "minimax-cn"
        assert creds["api_key"] == "mmcn-secret-key"
        assert creds["base_url"] == "https://api.minimaxi.com/anthropic"

    def test_resolve_ai_gateway_with_key(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-secret-key")
        creds = resolve_api_key_provider_credentials("ai-gateway")
        assert creds["provider"] == "ai-gateway"
        assert creds["api_key"] == "gw-secret-key"
        assert creds["base_url"] == "https://ai-gateway.vercel.sh/v1"

    def test_resolve_kilocode_with_key(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-secret-key")
        creds = resolve_api_key_provider_credentials("kilocode")
        assert creds["provider"] == "kilocode"
        assert creds["api_key"] == "kilo-secret-key"
        assert creds["base_url"] == "https://api.kilo.ai/api/gateway"

    def test_resolve_gmi_with_key(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-secret-key")
        creds = resolve_api_key_provider_credentials("gmi")
        assert creds["provider"] == "gmi"
        assert creds["api_key"] == "gmi-secret-key"
        assert creds["base_url"] == "https://api.gmi-serving.com/v1"

    def test_resolve_gmi_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-key")
        monkeypatch.setenv("GMI_BASE_URL", "https://custom.gmi.example/v1")
        creds = resolve_api_key_provider_credentials("gmi")
        assert creds["base_url"] == "https://custom.gmi.example/v1"

    def test_resolve_kilocode_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-key")
        monkeypatch.setenv("KILOCODE_BASE_URL", "https://custom.kilo.example/v1")
        creds = resolve_api_key_provider_credentials("kilocode")
        assert creds["base_url"] == "https://custom.kilo.example/v1"

    def test_resolve_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        monkeypatch.setenv("GLM_BASE_URL", "https://custom.glm.example/v4")
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://custom.glm.example/v4"

    def test_resolve_without_key_returns_empty(self):
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == ""
        assert creds["source"] == "default"

    def test_resolve_invalid_provider_raises(self):
        with pytest.raises(AuthError):
            resolve_api_key_provider_credentials("nous")

    def test_glm_key_priority(self, monkeypatch):
        """GLM_API_KEY takes priority over ZAI_API_KEY."""
        monkeypatch.setenv("GLM_API_KEY", "primary")
        monkeypatch.setenv("ZAI_API_KEY", "secondary")
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == "primary"
        assert creds["source"] == "GLM_API_KEY"

    def test_zai_key_fallback(self, monkeypatch):
        """ZAI_API_KEY used when GLM_API_KEY not set."""
        monkeypatch.setenv("ZAI_API_KEY", "secondary")
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == "secondary"
        assert creds["source"] == "ZAI_API_KEY"


# =============================================================================
# Runtime Provider Resolution tests
# =============================================================================

class TestRuntimeProviderResolution:

    def test_runtime_zai(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="zai")
        assert result["provider"] == "zai"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "glm-key"
        assert "z.ai" in result["base_url"] or "api.z.ai" in result["base_url"]

    def test_runtime_kimi(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "kimi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="kimi-coding")
        assert result["provider"] == "kimi-coding"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "kimi-key"

    def test_runtime_stepfun(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-key")
        monkeypatch.setenv("STEPFUN_BASE_URL", STEPFUN_STEP_PLAN_CN_BASE_URL)
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="stepfun")
        assert result["provider"] == "stepfun"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "stepfun-key"
        assert result["base_url"] == STEPFUN_STEP_PLAN_CN_BASE_URL

    def test_runtime_minimax(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="minimax")
        assert result["provider"] == "minimax"
        assert result["api_key"] == "mm-key"

    def test_runtime_ai_gateway(self, monkeypatch):
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="ai-gateway")
        assert result["provider"] == "ai-gateway"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gw-key"
        assert "ai-gateway.vercel.sh" in result["base_url"]

    def test_runtime_kilocode(self, monkeypatch):
        monkeypatch.setenv("KILOCODE_API_KEY", "kilo-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="kilocode")
        assert result["provider"] == "kilocode"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "kilo-key"
        assert "kilo.ai" in result["base_url"]

    def test_runtime_gmi(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="gmi")
        assert result["provider"] == "gmi"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gmi-key"
        assert result["base_url"] == "https://api.gmi-serving.com/v1"

    def test_runtime_auto_detects_api_key_provider(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "auto-kimi-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="auto")
        assert result["provider"] == "kimi-coding"
        assert result["api_key"] == "auto-kimi-key"

    def test_runtime_copilot_uses_gh_cli_token(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "gho_cli_secret")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="copilot")
        assert result["provider"] == "copilot"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "gho_cli_secret"
        assert result["base_url"] == "https://api.githubcopilot.com"

    def test_runtime_copilot_uses_responses_for_gpt_5_4(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "gho_cli_secret")
        monkeypatch.setattr(
            "hermes_cli.runtime_provider._get_model_config",
            lambda: {"provider": "copilot", "default": "gpt-5.4"},
        )
        monkeypatch.setattr(
            "hermes_cli.models.fetch_github_model_catalog",
            lambda api_key=None, timeout=5.0: [
                {
                    "id": "gpt-5.4",
                    "supported_endpoints": ["/responses"],
                    "capabilities": {"type": "chat"},
                }
            ],
        )
        from hermes_cli.runtime_provider import resolve_runtime_provider

        result = resolve_runtime_provider(requested="copilot")

        assert result["provider"] == "copilot"
        assert result["api_mode"] == "codex_responses"

    def test_runtime_copilot_acp_uses_process_runtime(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/local/bin/{command}")
        monkeypatch.setenv("HERMES_COPILOT_ACP_ARGS", "--acp --stdio --debug")

        from hermes_cli.runtime_provider import resolve_runtime_provider

        result = resolve_runtime_provider(requested="copilot-acp")

        assert result["provider"] == "copilot-acp"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "copilot-acp"
        assert result["base_url"] == "acp://copilot"
        assert result["command"] == "/usr/local/bin/copilot"
        assert result["args"] == ["--acp", "--stdio", "--debug"]


# =============================================================================
# _has_any_provider_configured tests
# =============================================================================

class TestHasAnyProviderConfigured:

    def test_glm_key_counts(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_module
        monkeypatch.setenv("GLM_API_KEY", "test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_minimax_key_counts(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_module
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_gh_cli_token_counts(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_module
        monkeypatch.setattr("hermes_cli.copilot_auth._try_gh_cli_token", lambda: "gho_cli_secret")
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_claude_code_creds_ignored_on_fresh_install(self, monkeypatch, tmp_path):
        """Claude Code credentials should NOT skip the wizard when Hermes is unconfigured."""
        from hermes_cli import config as config_module
        from hermes_cli.auth import PROVIDER_REGISTRY
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setattr("hermes_cli.copilot_auth.resolve_copilot_token", lambda: ("", ""))
        # Clear all provider env vars so earlier checks don't short-circuit
        _all_vars = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                      "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"}
        for pconfig in PROVIDER_REGISTRY.values():
            if pconfig.auth_type == "api_key":
                _all_vars.update(pconfig.api_key_env_vars)
        for var in _all_vars:
            monkeypatch.delenv(var, raising=False)
        # Prevent gh-cli / copilot auth fallback from leaking in
        monkeypatch.setattr("hermes_cli.auth.get_auth_status", lambda _pid: {})
        # Simulate valid Claude Code credentials
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {"accessToken": "sk-ant-test", "refreshToken": "ref-tok"},
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid",
            lambda creds: True,
        )
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is False

    def test_config_provider_counts(self, monkeypatch, tmp_path):
        """config.yaml with model.provider set should count as configured."""
        import yaml
        from hermes_cli import config as config_module
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text(yaml.dump({
            "model": {"default": "anthropic/claude-opus-4.6", "provider": "openrouter"},
        }))
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Clear all provider env vars
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_config_base_url_counts(self, monkeypatch, tmp_path):
        """config.yaml with model.base_url set (custom endpoint) should count."""
        import yaml
        from hermes_cli import config as config_module
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text(yaml.dump({
            "model": {"default": "my-model", "base_url": "http://localhost:11434/v1"},
        }))
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_config_api_key_counts(self, monkeypatch, tmp_path):
        """config.yaml with model.api_key set should count."""
        import yaml
        from hermes_cli import config as config_module
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text(yaml.dump({
            "model": {"default": "my-model", "api_key": "sk-test-key"},
        }))
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True

    def test_config_dict_no_provider_no_creds_still_false(self, monkeypatch, tmp_path):
        """config.yaml model dict with empty default and no creds stays false."""
        import yaml
        from hermes_cli import config as config_module
        from hermes_cli.auth import PROVIDER_REGISTRY
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text(yaml.dump({
            "model": {"default": ""},
        }))
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setattr("hermes_cli.copilot_auth.resolve_copilot_token", lambda: ("", ""))
        _all_vars = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                      "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"}
        for pconfig in PROVIDER_REGISTRY.values():
            if pconfig.auth_type == "api_key":
                _all_vars.update(pconfig.api_key_env_vars)
        for var in _all_vars:
            monkeypatch.delenv(var, raising=False)
        # Prevent gh-cli / copilot auth fallback from leaking in
        monkeypatch.setattr("hermes_cli.auth.get_auth_status", lambda _pid: {})
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is False

    def test_claude_code_creds_counted_when_hermes_configured(self, monkeypatch, tmp_path):
        """Claude Code credentials should count when Hermes has been explicitly configured."""
        import yaml
        from hermes_cli import config as config_module
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        # Write a config with a non-default model to simulate explicit configuration
        config_file = hermes_home / "config.yaml"
        config_file.write_text(yaml.dump({"model": {"default": "my-local-model"}}))
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        # Clear all provider env vars
        for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        # Simulate valid Claude Code credentials
        monkeypatch.setattr(
            "agent.anthropic_adapter.read_claude_code_credentials",
            lambda: {"accessToken": "sk-ant-test", "refreshToken": "ref-tok"},
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.is_claude_code_token_valid",
            lambda creds: True,
        )
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True


# =============================================================================
# Kimi Code auto-detection tests
# =============================================================================

MOONSHOT_DEFAULT_URL = "https://api.moonshot.ai/v1"


class TestResolveKimiBaseUrl:
    """Test _resolve_kimi_base_url() helper for key-prefix auto-detection."""

    def test_sk_kimi_prefix_routes_to_kimi_code(self):
        url = _resolve_kimi_base_url("sk-kimi-abc123", MOONSHOT_DEFAULT_URL, "")
        assert url == KIMI_CODE_BASE_URL

    def test_legacy_key_uses_default(self):
        url = _resolve_kimi_base_url("sk-abc123", MOONSHOT_DEFAULT_URL, "")
        assert url == MOONSHOT_DEFAULT_URL

    def test_empty_key_uses_default(self):
        url = _resolve_kimi_base_url("", MOONSHOT_DEFAULT_URL, "")
        assert url == MOONSHOT_DEFAULT_URL

    def test_env_override_wins_over_sk_kimi(self):
        """KIMI_BASE_URL env var should always take priority."""
        custom = "https://custom.example.com/v1"
        url = _resolve_kimi_base_url("sk-kimi-abc123", MOONSHOT_DEFAULT_URL, custom)
        assert url == custom

    def test_env_override_wins_over_legacy(self):
        custom = "https://custom.example.com/v1"
        url = _resolve_kimi_base_url("sk-abc123", MOONSHOT_DEFAULT_URL, custom)
        assert url == custom


class TestKimiCodeStatusAutoDetect:
    """Test that get_api_key_provider_status auto-detects sk-kimi- keys."""

    def test_sk_kimi_key_gets_kimi_code_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key-123")
        status = get_api_key_provider_status("kimi-coding")
        assert status["configured"] is True
        assert status["base_url"] == KIMI_CODE_BASE_URL

    def test_legacy_key_gets_moonshot_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-test-key")
        status = get_api_key_provider_status("kimi-coding")
        assert status["configured"] is True
        assert status["base_url"] == MOONSHOT_DEFAULT_URL

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://override.example/v1")
        status = get_api_key_provider_status("kimi-coding")
        assert status["base_url"] == "https://override.example/v1"


class TestKimiCodeCredentialAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects sk-kimi- keys."""

    def test_sk_kimi_key_gets_kimi_code_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["api_key"] == "sk-kimi-secret-key"
        assert creds["base_url"] == KIMI_CODE_BASE_URL

    def test_legacy_key_gets_moonshot_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["api_key"] == "sk-legacy-secret-key"
        assert creds["base_url"] == MOONSHOT_DEFAULT_URL

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-secret-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://override.example/v1")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["base_url"] == "https://override.example/v1"

    def test_non_kimi_providers_unaffected(self, monkeypatch):
        """Ensure the auto-detect logic doesn't leak to other providers."""
        monkeypatch.setenv("GLM_API_KEY", "sk-kim...isnt")
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"


class TestZaiEndpointAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects Z.AI endpoints."""

    def test_probe_success_returns_detected_url(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-coding-key")
        monkeypatch.setattr(
            "hermes_cli.auth.detect_zai_endpoint",
            lambda *a, **kw: {
                "id": "coding-global",
                "base_url": "https://api.z.ai/api/coding/paas/v4",
                "model": "glm-4.7",
                "label": "Global (Coding Plan)",
            },
        )
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://api.z.ai/api/coding/paas/v4"

    def test_probe_failure_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"

    def test_env_override_skips_probe(self, monkeypatch):
        """GLM_BASE_URL should always win without probing."""
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        monkeypatch.setenv("GLM_BASE_URL", "https://custom.example/v4")
        probe_called = False

        def _never_called(*a, **kw):
            nonlocal probe_called
            probe_called = True
            return None

        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", _never_called)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://custom.example/v4"
        assert not probe_called

    def test_no_key_skips_probe(self, monkeypatch):
        """Without an API key, no probe should occur."""
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["api_key"] == ""


# =============================================================================
# Kimi / Moonshot model list isolation tests
# =============================================================================

class TestKimiMoonshotModelListIsolation:
    """Moonshot (legacy) users must not see Coding Plan-only models."""

    def test_moonshot_list_excludes_coding_plan_only_models(self):
        from hermes_cli.main import _PROVIDER_MODELS
        moonshot_models = _PROVIDER_MODELS["moonshot"]
        coding_plan_only = {"kimi-for-coding", "kimi-k2-thinking-turbo"}
        leaked = set(moonshot_models) & coding_plan_only
        assert not leaked, f"Moonshot list contains Coding Plan-only models: {leaked}"

    def test_moonshot_list_non_empty(self):
        from hermes_cli.main import _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["moonshot"]) >= 1

    def test_coding_plan_list_non_empty(self):
        from hermes_cli.main import _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["kimi-coding"]) >= 1


# =============================================================================
# Hugging Face provider model list tests
# =============================================================================

class TestHuggingFaceModels:
    """Verify Hugging Face model lists are consistent across all locations."""

    def test_main_provider_models_has_huggingface(self):
        from hermes_cli.main import _PROVIDER_MODELS
        assert "huggingface" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["huggingface"]) >= 1

    def test_models_py_has_huggingface(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "huggingface" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["huggingface"]) >= 1

    def test_model_lists_match(self):
        """Model lists in main.py and models.py should be identical."""
        from hermes_cli.main import _PROVIDER_MODELS as main_models
        from hermes_cli.models import _PROVIDER_MODELS as models_models
        assert main_models["huggingface"] == models_models["huggingface"]

    def test_model_metadata_has_context_lengths(self):
        """Every HF model should have a context length entry."""
        from hermes_cli.models import _PROVIDER_MODELS
        from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS
        lower_keys = {k.lower() for k in DEFAULT_CONTEXT_LENGTHS}
        hf_models = _PROVIDER_MODELS["huggingface"]
        for model in hf_models:
            assert model.lower() in lower_keys, (
                f"HF model {model!r} missing from DEFAULT_CONTEXT_LENGTHS"
            )

    def test_models_use_org_name_format(self):
        """HF models should use org/name format (e.g. Qwen/Qwen3-235B)."""
        from hermes_cli.models import _PROVIDER_MODELS
        for model in _PROVIDER_MODELS["huggingface"]:
            assert "/" in model, f"HF model {model!r} missing org/ prefix"

    def test_provider_aliases_in_models_py(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("hf") == "huggingface"
        assert _PROVIDER_ALIASES.get("hugging-face") == "huggingface"

    def test_provider_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert "huggingface" in _PROVIDER_LABELS
        assert _PROVIDER_LABELS["huggingface"] == "Hugging Face"


# =============================================================================
# MiniMax OAuth provider tests (added by feat/minimax-oauth-provider)
# =============================================================================

class TestMinimaxOAuthProvider:
    """Tests for the minimax-oauth OAuth provider."""

    def test_minimax_oauth_in_provider_registry(self):
        assert "minimax-oauth" in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["minimax-oauth"]
        assert pconfig.auth_type == "oauth_minimax"
        assert pconfig.id == "minimax-oauth"

    def test_minimax_oauth_has_correct_endpoints(self):
        from hermes_cli.auth import (
            MINIMAX_OAUTH_GLOBAL_BASE,
            MINIMAX_OAUTH_GLOBAL_INFERENCE,
            MINIMAX_OAUTH_CN_BASE,
            MINIMAX_OAUTH_CN_INFERENCE,
        )
        pconfig = PROVIDER_REGISTRY["minimax-oauth"]
        assert pconfig.portal_base_url == MINIMAX_OAUTH_GLOBAL_BASE
        assert pconfig.inference_base_url == MINIMAX_OAUTH_GLOBAL_INFERENCE
        assert pconfig.extra["cn_portal_base_url"] == MINIMAX_OAUTH_CN_BASE
        assert pconfig.extra["cn_inference_base_url"] == MINIMAX_OAUTH_CN_INFERENCE

    def test_minimax_oauth_alias_resolves_portal(self):
        result = resolve_provider("minimax-portal")
        assert result == "minimax-oauth"

    def test_minimax_oauth_alias_resolves_global(self):
        result = resolve_provider("minimax-global")
        assert result == "minimax-oauth"

    def test_minimax_oauth_alias_resolves_underscore(self):
        result = resolve_provider("minimax_oauth")
        assert result == "minimax-oauth"

    def test_minimax_oauth_listed_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "minimax-oauth" in slugs

    def test_minimax_oauth_models_alias_in_models_py(self):
        from hermes_cli.models import _PROVIDER_ALIASES
        assert _PROVIDER_ALIASES.get("minimax-portal") == "minimax-oauth"
        assert _PROVIDER_ALIASES.get("minimax-global") == "minimax-oauth"
        assert _PROVIDER_ALIASES.get("minimax_oauth") == "minimax-oauth"

    def test_minimax_oauth_has_models(self):
        from hermes_cli.models import _PROVIDER_MODELS
        models = _PROVIDER_MODELS.get("minimax-oauth", [])
        assert len(models) >= 1

    def test_minimax_oauth_aux_model_registered(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "minimax-oauth" in _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS["minimax-oauth"]  # non-empty
