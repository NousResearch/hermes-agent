"""Tests for API-key provider support (z.ai/GLM, Kimi, MiniMax, AI Gateway)."""

import json

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    AuthError,
    KIMI_CODE_BASE_URL,
    STEPFUN_STEP_PLAN_INTL_BASE_URL,
    _resolve_kimi_base_url,
)
from hermes_cli.copilot_auth import _try_gh_cli_token


# =============================================================================
# Provider Registry tests
# =============================================================================


# =============================================================================
# Provider Resolution tests
# =============================================================================

# Derived from the live PROVIDER_REGISTRY so the list can never drift when a
# new provider (and its env var) is added — a hand-maintained tuple here was
# missing HF_TOKEN/DEEPINFRA_API_KEY, which made the auto-detection tests
# env-dependent (they failed on any machine with HF_TOKEN exported).
from hermes_cli.auth import PROVIDER_REGISTRY as _REGISTRY

_EXTRA_ENV_VARS = (
    # Checked directly in resolve_provider("auto"), not via the registry.
    "OPENROUTER_API_KEY", "NOUS_API_KEY",
    # Base URLs / paths that influence detection but aren't api_key_env_vars.
    "LM_BASE_URL", "KIMI_BASE_URL", "STEPFUN_BASE_URL", "KILOCODE_BASE_URL",
    "GMI_BASE_URL", "OPENAI_BASE_URL",
    "HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH",
    "HERMES_COPILOT_ACP_ARGS", "COPILOT_ACP_BASE_URL",
)

PROVIDER_ENV_VARS = tuple(
    dict.fromkeys(
        [var for cfg in _REGISTRY.values() for var in cfg.api_key_env_vars]
        + list(_EXTRA_ENV_VARS)
    )
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in PROVIDER_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("hermes_cli.auth._load_auth_store", lambda: {})


class TestResolveProvider:
    """Test resolve_provider() with new providers."""


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


    def test_alias_vercel(self):
        assert resolve_provider("vercel") == "ai-gateway"


    def test_alias_kilo(self):
        assert resolve_provider("kilo") == "kilocode"


    def test_alias_case_insensitive(self):
        assert resolve_provider("GLM") == "zai"
        assert resolve_provider("Z-AI") == "zai"
        assert resolve_provider("Kimi") == "kimi-coding"

    def test_alias_chatgpt(self):
        """Issue #95794: ``--provider chatgpt`` selects the ChatGPT-backed Codex OAuth provider."""
        assert resolve_provider("chatgpt") == "openai-codex"
        assert resolve_provider("chatgpt-codex") == "openai-codex"

    def test_alias_chatgpt_every_alias_table(self):
        """Issue #95794: the runtime (providers.py), the /model parser (models_catalog_static via
        parse_model_input) and ``hermes auth login`` all resolve the ChatGPT alias, not just auth."""
        from hermes_cli.providers import normalize_provider
        from hermes_cli.models import parse_model_input
        from hermes_cli.auth_commands import _normalize_provider

        assert normalize_provider("chatgpt") == "openai-codex"
        assert normalize_provider("chatgpt-codex") == "openai-codex"
        assert parse_model_input("chatgpt:gpt-5.5", "openrouter") == ("openai-codex", "gpt-5.5")
        assert parse_model_input("chatgpt-codex:gpt-5.5", "openrouter") == ("openai-codex", "gpt-5.5")
        assert _normalize_provider("chatgpt") == "openai-codex"

    def test_alias_github_copilot(self):
        assert resolve_provider("github-copilot") == "copilot"


    def test_alias_github_copilot_acp(self):
        assert resolve_provider("github-copilot-acp") == "copilot-acp"
        assert resolve_provider("copilot-acp-agent") == "copilot-acp"


    def test_alias_hf(self):
        assert resolve_provider("hf") == "huggingface"


    def test_unknown_provider_raises(self):
        with pytest.raises(AuthError):
            resolve_provider("nonexistent-provider-xyz")

    def test_auto_detects_glm_key(self, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "test-glm-key")
        assert resolve_provider("auto") == "zai"


    def test_auto_detects_kimi_key(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
        assert resolve_provider("auto") == "kimi-coding"


    def test_auto_detects_minimax_cn_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-mm-cn-key")
        assert resolve_provider("auto") == "minimax-cn"


    def test_openrouter_takes_priority_over_glm(self, monkeypatch):
        """OpenRouter API key should win over GLM in auto-detection."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("GLM_API_KEY", "glm-key")
        assert resolve_provider("auto") == "openrouter"

    def test_auto_does_not_select_copilot_from_github_token(self, monkeypatch):
        # AWS Bedrock auto-detection (via boto3's credential chain) runs at
        # the tail of resolve_provider("auto") and will silently pick up
        # ~/.aws/credentials on developer machines that aren't blanked by
        # the hermetic conftest. Force-disable it so this test exercises
        # the specific "GitHub token alone shouldn't auto-pick copilot"
        # behavior, not the Bedrock fallback.
        monkeypatch.setattr(
            "agent.bedrock_adapter.has_aws_credentials",
            lambda env=None: False,
        )
        monkeypatch.setenv("GITHUB_TOKEN", "gh-test-token")
        with pytest.raises(AuthError, match="not connected to any AI provider"):
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


# =============================================================================
# Credential Resolution tests
# =============================================================================

class TestResolveApiKeyProviderCredentials:


    def test_try_gh_cli_token_uses_homebrew_path_when_not_on_path(self, monkeypatch, tmp_path):
        from hermes_cli.copilot_auth import _invalidate_gh_cli_token_cache
        from hermes_platform.resolver import known_dirs

        _invalidate_gh_cli_token_cache()
        brew = tmp_path / "homebrew" / "bin"
        brew.mkdir(parents=True)
        gh = brew / "gh"
        gh.write_text("#!/bin/sh\n", encoding="utf-8")
        gh.chmod(0o755)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setattr(known_dirs, "homebrew_dirs", lambda: (str(brew),))
        monkeypatch.setattr(known_dirs, "user_local_bin", lambda: ())

        calls = []

        class _Result:
            returncode = 0
            stdout = "gh-cli-secret\n"

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            return _Result()

        monkeypatch.setattr("hermes_cli.copilot_auth.subprocess.run", _fake_run)

        assert _try_gh_cli_token() == "gh-cli-secret"
        assert calls == [[str(gh), "auth", "token"]]


    def test_resolve_stepfun_with_key(self, monkeypatch):
        monkeypatch.setenv("STEPFUN_API_KEY", "stepfun-secret-key")
        creds = resolve_api_key_provider_credentials("stepfun")
        assert creds["provider"] == "stepfun"
        assert creds["api_key"] == "stepfun-secret-key"
        assert creds["base_url"] == STEPFUN_STEP_PLAN_INTL_BASE_URL


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


    def test_runtime_minimax(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="minimax")
        assert result["provider"] == "minimax"
        assert result["api_key"] == "mm-key"


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


    def test_claude_code_creds_ignored_on_fresh_install(self, monkeypatch, tmp_path):
        """Claude Code credentials should NOT skip the wizard when Hermes is unconfigured."""
        from hermes_cli import config as config_module
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
            "agent.anthropic_credentials.read_claude_code_credentials",
            lambda: {"accessToken": "sk-ant-test", "refreshToken": "ref-tok"},
        )
        monkeypatch.setattr(
            "agent.anthropic_credentials.is_claude_code_token_valid",
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

    @staticmethod
    def _clear_provider_env(monkeypatch):
        """Clear every provider env var so early checks can't short-circuit."""
        _all_vars = {"OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "ANTHROPIC_TOKEN", "OPENAI_BASE_URL"}
        for pconfig in PROVIDER_REGISTRY.values():
            if pconfig.auth_type == "api_key":
                _all_vars.update(pconfig.api_key_env_vars)
        for var in _all_vars:
            monkeypatch.delenv(var, raising=False)

    def _setup_home(self, monkeypatch, tmp_path):
        from hermes_cli import config as config_module
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setattr(config_module, "get_env_path", lambda: hermes_home / ".env")
        monkeypatch.setattr(config_module, "get_hermes_home", lambda: hermes_home)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        self._clear_provider_env(monkeypatch)
        return hermes_home

    def test_config_provider_skips_registry_sweep(self, monkeypatch, tmp_path):
        """model.provider in config.yaml must short-circuit BEFORE the slow
        provider-registry sweep (gh subprocess etc.) is ever invoked.

        Regression test for the auth-first ordering: get_auth_status is
        booby-trapped to fail loudly if the sweep runs. The sweep wraps its
        loop in ``except Exception``, so we also record every call — any
        recorded call proves the sweep ran even if the raise was swallowed.
        """
        import yaml
        hermes_home = self._setup_home(monkeypatch, tmp_path)
        (hermes_home / "config.yaml").write_text(yaml.dump({
            "model": {"default": "anthropic/claude-opus-4.6", "provider": "openrouter"},
        }))
        sweep_calls = []

        def _trap(provider_id):
            sweep_calls.append(provider_id)
            raise AssertionError("sweep must be skipped")

        monkeypatch.setattr("hermes_cli.auth.get_auth_status", _trap)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True
        assert sweep_calls == [], (
            f"provider registry sweep ran before config short-circuit: {sweep_calls}"
        )

    def test_config_base_url_api_key_skips_registry_sweep(self, monkeypatch, tmp_path):
        """Custom endpoint (base_url/api_key in config, no provider) must also
        short-circuit before the registry sweep."""
        import yaml
        hermes_home = self._setup_home(monkeypatch, tmp_path)
        (hermes_home / "config.yaml").write_text(yaml.dump({
            "model": {
                "default": "local/custom-model",
                "base_url": "http://localhost:8000/v1",
                "api_key": "sk-local-test",
            },
        }))
        sweep_calls = []

        def _trap(provider_id):
            sweep_calls.append(provider_id)
            raise AssertionError("sweep must be skipped")

        monkeypatch.setattr("hermes_cli.auth.get_auth_status", _trap)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True
        assert sweep_calls == [], (
            f"provider registry sweep ran before config short-circuit: {sweep_calls}"
        )

    def test_auth_json_skips_registry_sweep(self, monkeypatch, tmp_path):
        """auth.json with a logged-in active provider must short-circuit before
        the registry sweep. get_auth_status may be called ONLY for the active
        provider from auth.json — any other provider id means the sweep ran.
        """
        import json
        hermes_home = self._setup_home(monkeypatch, tmp_path)
        (hermes_home / "auth.json").write_text(json.dumps({
            "active_provider": "nous",
        }))
        calls = []

        def _guarded_status(provider_id):
            calls.append(provider_id)
            assert provider_id == "nous", "sweep must be skipped"
            return {"logged_in": True}

        monkeypatch.setattr("hermes_cli.auth.get_auth_status", _guarded_status)
        from hermes_cli.main import _has_any_provider_configured
        assert _has_any_provider_configured() is True
        assert calls == ["nous"], (
            f"provider registry sweep ran before auth.json short-circuit: {calls}"
        )


# =============================================================================
# Kimi Code auto-detection tests
# =============================================================================

MOONSHOT_DEFAULT_URL = "https://api.moonshot.ai/v1"


class TestResolveKimiBaseUrl:
    """Test _resolve_kimi_base_url() helper for key-prefix auto-detection."""

    def test_sk_kimi_prefix_routes_to_kimi_code(self):
        url = _resolve_kimi_base_url("sk-kimi-abc123", MOONSHOT_DEFAULT_URL, "")
        assert url == KIMI_CODE_BASE_URL


    def test_env_override_wins_over_legacy(self):
        custom = "https://custom.example.com/v1"
        url = _resolve_kimi_base_url("sk-abc123", MOONSHOT_DEFAULT_URL, custom)
        assert url == custom


class TestKimiCodeStatusAutoDetect:
    """Test that get_api_key_provider_status auto-detects sk-kimi- keys."""


    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-kimi-test-key")
        monkeypatch.setenv("KIMI_BASE_URL", "https://override.example/v1")
        status = get_api_key_provider_status("kimi-coding")
        assert status["base_url"] == "https://override.example/v1"


class TestKimiCodeCredentialAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects sk-kimi- keys."""


    def test_legacy_key_gets_moonshot_url(self, monkeypatch):
        monkeypatch.setenv("KIMI_API_KEY", "sk-legacy-secret-key")
        creds = resolve_api_key_provider_credentials("kimi-coding")
        assert creds["api_key"] == "sk-legacy-secret-key"
        assert creds["base_url"] == MOONSHOT_DEFAULT_URL


    def test_non_kimi_providers_unaffected(self, monkeypatch):
        """Ensure the auto-detect logic doesn't leak to other providers."""
        monkeypatch.setenv("GLM_API_KEY", "sk-kim...isnt")
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: None)
        creds = resolve_api_key_provider_credentials("zai")
        assert creds["base_url"] == "https://api.z.ai/api/paas/v4"


class TestZaiEndpointAutoDetect:
    """Test that resolve_api_key_provider_credentials auto-detects Z.AI endpoints."""


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

    def test_failed_probe_is_not_repeated_within_ttl(self, monkeypatch):
        """A key whose detection fails (429 on every endpoint) is probed once, not on every
        credential resolution — the picker resolves Z.AI dozens of times per open (#114215)."""
        from hermes_cli import auth_zai_kimi
        monkeypatch.setenv("GLM_API_KEY", "glm-key-that-429s")
        monkeypatch.setattr(auth_zai_kimi, "_zai_probe_failed_until", {})
        calls = []
        monkeypatch.setattr("hermes_cli.auth.detect_zai_endpoint", lambda *a, **kw: calls.append(1))
        for _ in range(3):
            assert resolve_api_key_provider_credentials("zai")["base_url"] == "https://api.z.ai/api/paas/v4"
        assert len(calls) == 1


class TestZaiParallelProbe:
    """detect_zai_endpoint probes endpoints in parallel workers.

    Contract under test: (1) each endpoint worker preserves the per-endpoint
    candidate-model fallback loop, (2) when several endpoints succeed the
    winner is chosen by ZAI_ENDPOINTS priority order (not completion order).
    """

    def _mock_post(self, ok):
        """Return an httpx.post replacement; `ok` maps (base_url, model) -> bool."""
        import httpx as _httpx

        def _post(url, headers=None, json=None, timeout=None):
            base = url.rsplit("/chat/completions", 1)[0]
            code = 200 if ok.get((base, json["model"])) else 401
            request = _httpx.Request("POST", url)
            return _httpx.Response(code, request=request, json={})

        return _post

    def test_candidate_model_fallback_within_endpoint(self, monkeypatch):
        """A worker must try its endpoint's later candidate models when the
        first ones fail — the fallback the scalar-model version dropped."""
        from hermes_cli.auth import ZAI_ENDPOINTS, detect_zai_endpoint

        coding_global = next(ep for ep in ZAI_ENDPOINTS if ep[0] == "coding-global")
        base = coding_global[1]
        last_model = coding_global[2][-1]
        # Only the LAST candidate model of coding-global succeeds.
        monkeypatch.setattr(
            "hermes_cli.auth.httpx.post",
            self._mock_post({(base, last_model): True}),
        )
        result = detect_zai_endpoint("test-key", timeout=1.0)
        assert result is not None
        assert result["id"] == "coding-global"
        assert result["model"] == last_model

    def test_priority_order_wins_over_completion_order(self, monkeypatch):
        """When multiple endpoints accept the key, the FIRST in
        ZAI_ENDPOINTS order must win, even if another finishes earlier."""
        import time as _time

        from hermes_cli.auth import ZAI_ENDPOINTS, detect_zai_endpoint

        first = ZAI_ENDPOINTS[0]
        last = ZAI_ENDPOINTS[-1]
        ok = {
            (first[1], first[2][0]): True,
            (last[1], last[2][0]): True,
        }
        inner = self._mock_post(ok)

        def _slow_first(url, headers=None, json=None, timeout=None):
            if url.startswith(first[1]):
                _time.sleep(0.15)  # first-priority endpoint finishes LAST
            return inner(url, headers=headers, json=json, timeout=timeout)

        monkeypatch.setattr("hermes_cli.auth.httpx.post", _slow_first)
        result = detect_zai_endpoint("test-key", timeout=1.0)
        assert result is not None
        assert result["id"] == first[0]

    def test_all_fail_returns_none(self, monkeypatch):
        from hermes_cli.auth import detect_zai_endpoint

        monkeypatch.setattr("hermes_cli.auth.httpx.post", self._mock_post({}))
        assert detect_zai_endpoint("bad-key", timeout=1.0) is None

    def test_early_exit_does_not_wait_for_slow_losers(self, monkeypatch):
        """When the highest-priority endpoint succeeds fast, the caller must
        return without waiting for slow lower-priority probes to finish."""
        import time as _time

        from hermes_cli.auth import ZAI_ENDPOINTS, detect_zai_endpoint

        first = ZAI_ENDPOINTS[0]
        inner = self._mock_post({(first[1], first[2][0]): True})

        def _slow_losers(url, headers=None, json=None, timeout=None):
            if not url.startswith(first[1]):
                _time.sleep(2.0)  # slow lower-priority endpoints
            return inner(url, headers=headers, json=json, timeout=timeout)

        monkeypatch.setattr("hermes_cli.auth.httpx.post", _slow_losers)
        t0 = _time.perf_counter()
        result = detect_zai_endpoint("test-key", timeout=5.0)
        elapsed = _time.perf_counter() - t0
        assert result is not None and result["id"] == first[0]
        assert elapsed < 1.5, f"early exit failed: waited {elapsed:.2f}s for losers"


# =============================================================================
# Kimi / Moonshot model list isolation tests
# =============================================================================

class TestKimiMoonshotModelListIsolation:
    """Moonshot (legacy) users must not see Coding Plan-only models."""

    def test_moonshot_list_excludes_coding_plan_only_models(self):
        from hermes_cli.models import _PROVIDER_MODELS
        moonshot_models = _PROVIDER_MODELS["moonshot"]
        coding_plan_only = {"kimi-for-coding", "kimi-k2-thinking-turbo"}
        leaked = set(moonshot_models) & coding_plan_only
        assert not leaked, f"Moonshot list contains Coding Plan-only models: {leaked}"


# =============================================================================
# Hugging Face provider model list tests
# =============================================================================

class TestHuggingFaceModels:
    """Verify Hugging Face model lists are consistent across all locations."""


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


# =============================================================================
# NovitaAI provider tests (added by feat/add-novita-provider)
# =============================================================================

class TestNovitaProvider:
    """Tests for NovitaAI — an OpenAI-compatible multi-model aggregator."""


    def test_novita_pricing_cache(self, monkeypatch):
        """_fetch_novita_pricing should cache results in _pricing_cache."""
        from hermes_cli import models as models_mod
        from hermes_cli import models_pricing
        monkeypatch.setenv("NOVITA_API_KEY", "sk-test-key")
        monkeypatch.setenv("NOVITA_BASE_URL", "https://api.novita.ai/openai/v1")
        models_pricing._pricing_cache.pop("https://api.novita.ai/openai/v1", None)

        call_count = {"n": 0}
        fake_payload = {
            "data": [
                {
                    "id": "x/y",
                    "input_token_price_per_m": 1000,
                    "output_token_price_per_m": 2000,
                }
            ]
        }

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                import json as _json
                return _json.dumps(fake_payload).encode()

        def fake_urlopen(req, timeout=None):
            call_count["n"] += 1
            return _FakeResp()

        monkeypatch.setattr(
            models_mod, "_urlopen_model_catalog_request", fake_urlopen
        )

        # First call hits the network.
        first = models_pricing._fetch_novita_pricing()
        assert "x/y" in first
        assert call_count["n"] == 1

        # Second call returns cached result without re-hitting the network.
        second = models_pricing._fetch_novita_pricing()
        assert second == first
        assert call_count["n"] == 1

        # force_refresh bypasses the cache.
        models_pricing._fetch_novita_pricing(force_refresh=True)
        assert call_count["n"] == 2


# =============================================================================
# MiniMax OAuth provider tests (added by feat/minimax-oauth-provider)
# =============================================================================

class TestMinimaxOAuthProvider:
    """Tests for the minimax-oauth OAuth provider."""


    def test_minimax_oauth_aux_model_registered(self):
        # Aux model for the minimax-oauth provider now lives on the
        # ProviderProfile (plugins/model-providers/minimax/__init__.py),
        # not the legacy _API_KEY_PROVIDER_AUX_MODELS dict in
        # agent/auxiliary_client.py. The profile layer is the source
        # of truth; _get_aux_model_for_provider() reads from it first
        # and only falls back to the dict when no profile is registered.
        import model_tools  # noqa: F401  -- triggers plugin discovery
        import providers

        profile = providers.get_provider_profile("minimax-oauth")
        assert profile is not None, "minimax-oauth provider profile must be registered"
        assert profile.default_aux_model, (
            "minimax-oauth profile must advertise a non-empty default_aux_model "
            "so the auxiliary client (compression / vision / session-search) "
            "doesn't fire the 'No auxiliary LLM provider configured' warning "
            "for every minimax-oauth session."
        )


# =============================================================================
# DeepInfra provider tests
# =============================================================================
# Registration / alias / env-var invariants are asserted in
# TestProviderRegistry + TestResolveProvider above. The classes below
# cover the catalog/tag/pricing/profile machinery added on top of the
# baseline provider wiring.


@pytest.fixture
def _deepinfra_cache_isolation(monkeypatch):
    """Reset the module-level catalog cache around each DeepInfra test.

    The cache is keyed by base URL and would otherwise leak fixture data
    from one test into the next in the same session. The negative cache is
    reset too, so a test that simulates an unreachable catalog can't suppress
    a later test's fetch within the failure TTL.
    """
    import hermes_cli.models as _models_mod
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_cache", {})
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_neg_cache", {})
    yield


@pytest.mark.usefixtures("_deepinfra_cache_isolation")
class TestFetchDeepInfraModels:
    """Tests for _fetch_deepinfra_models() live model discovery."""

    def test_returns_filtered_models_on_success(self, monkeypatch):
        monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps({"data": [
                    {"id": "meta-llama/Llama-3-70B-Instruct", "metadata": {}},
                    {"id": "mistralai/Mistral-Nemo-Instruct-2407", "metadata": {}},
                    {"id": "BAAI/bge-large-en-v1.5-embed", "metadata": {}},
                    {"id": "stabilityai/stable-diffusion-xl-base-1.0", "metadata": {}},
                ]}).encode()

        import hermes_cli.models as models
        monkeypatch.setattr(
            models, "_urlopen_model_catalog_request", lambda *a, **kw: _Resp()
        )
        from hermes_cli.models import _fetch_deepinfra_models
        result = _fetch_deepinfra_models()

        assert result is not None
        assert "meta-llama/Llama-3-70B-Instruct" in result
        assert "mistralai/Mistral-Nemo-Instruct-2407" in result
        # Embedding and image models should be excluded
        assert not any("embed" in m.lower() for m in result)
        assert not any("stable-diffusion" in m.lower() for m in result)


    def test_catalog_uses_credential_safe_opener(self, monkeypatch):
        import hermes_cli.models as models

        seen = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"data": []}).encode()

        def _safe_open(request, *, timeout):
            seen["authorization"] = request.get_header("Authorization")
            seen["timeout"] = timeout
            return _Resp()

        monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
        monkeypatch.setattr(models, "_urlopen_model_catalog_request", _safe_open)

        assert models._fetch_deepinfra_catalog(force_refresh=True) == []
        assert seen == {"authorization": "Bearer test-key", "timeout": 5.0}


def _make_urlopen_returning(payload):
    """Helper: build a urlopen() shim returning a fixed JSON payload."""
    import json as _json

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return _json.dumps(payload).encode()

    return lambda *a, **kw: _Resp()


@pytest.mark.usefixtures("_deepinfra_cache_isolation")
class TestDeepInfraTagFiltering:
    """Contract tests for the shared _fetch_deepinfra_models_by_tag helper."""

    def test_filters_by_surface_tag_and_handles_rollout_states(self, monkeypatch):
        # One payload, several invariants in one test:
        #  - explicit surface tags are honored (chat / image-gen / tts / stt / embed)
        #  - capability-tags-only items fall through to the regex fallback
        #    (used during the surface-tag rollout)
        #  - the regex excludes id-name matches (whisper, embed, …)
        #  - a surface tag takes priority over the regex
        #  - ``metadata: None`` stubs are dropped
        payload = {"data": [
            {"id": "vendor/chat-tagged", "metadata": {"tags": ["chat"]}},
            {"id": "vendor/image-tagged", "metadata": {"tags": ["image-gen"]}},
            {"id": "vendor/tts-tagged", "metadata": {"tags": ["tts"]}},
            {"id": "vendor/stt-tagged", "metadata": {"tags": ["stt"]}},
            {"id": "vendor/embed-tagged", "metadata": {"tags": ["embed"]}},
            # capability-only — rolls through regex fallback
            {"id": "Qwen/Qwen3-30B", "metadata": {"tags": ["reasoning", "vision"]}},
            {"id": "openai/whisper-large", "metadata": {"tags": ["reasoning"]}},
            # surface tag overrides legacy regex exclusion
            {"id": "some-org/whisper-finetune-chat", "metadata": {"tags": ["chat"]}},
            # null metadata — stub model, must be skipped
            {"id": "stub-model", "metadata": None},
        ]}
        from hermes_cli.models import _fetch_deepinfra_models_by_tag
        import hermes_cli.models as _m

        for surface in ("chat", "image-gen", "tts", "stt", "embed"):
            monkeypatch.setattr(
                _m,
                "_urlopen_model_catalog_request",
                _make_urlopen_returning(payload),
            )
            # Reset cache between iterations so each surface re-parses the payload.
            _m._deepinfra_catalog_cache.clear()
            got = _fetch_deepinfra_models_by_tag(surface)
            assert got is not None
            ids = {item["id"] for item in got}
            assert "stub-model" not in ids  # null-metadata always skipped
            if surface == "chat":
                # explicit chat + capability-only (Qwen) + surface-tag-over-regex
                assert "vendor/chat-tagged" in ids
                assert "Qwen/Qwen3-30B" in ids
                assert "some-org/whisper-finetune-chat" in ids
                # regex still excludes capability-only items that match the excluder
                assert "openai/whisper-large" not in ids
            else:
                # non-chat surfaces only see explicit surface-tagged items
                for item in got:
                    assert surface in item["metadata"]["tags"]


@pytest.mark.usefixtures("_deepinfra_cache_isolation")
class TestDeepInfraPricingFetcher:
    """_fetch_deepinfra_pricing reshapes $/MTok values into per-token strings
    and is wired into the get_pricing_for_provider dispatch."""

    def test_pricing_shape_and_dispatch(self, monkeypatch):
        payload = {"data": [
            {
                "id": "vendor/model-a",
                "metadata": {
                    "tags": ["chat", "prompt_cache"],
                    "pricing": {
                        "input_tokens": 0.1,
                        "output_tokens": 0.3,
                        "cache_read_tokens": 0.02,
                    },
                },
            },
            {
                "id": "vendor/model-b",
                "metadata": {"tags": ["chat"], "pricing": {"input_tokens": 1.0, "output_tokens": 5.0}},
            },
            # non-chat — must not appear
            {"id": "vendor/model-image", "metadata": {"tags": ["image-gen"], "pricing": {"per_image_unit": 0.05}}},
        ]}
        import hermes_cli.models as models
        monkeypatch.setattr(
            models,
            "_urlopen_model_catalog_request",
            _make_urlopen_returning(payload),
        )
        from hermes_cli.models_pricing import get_pricing_for_provider

        # get_pricing_for_provider → _fetch_deepinfra_pricing dispatch path
        result = get_pricing_for_provider("deepinfra")
        assert set(result) == {"vendor/model-a", "vendor/model-b"}
        # Picker-shape: per-token strings under prompt/completion (+ cache_read when source had it)
        assert float(result["vendor/model-a"]["prompt"]) == pytest.approx(0.1 / 1_000_000)
        assert float(result["vendor/model-a"]["completion"]) == pytest.approx(0.3 / 1_000_000)
        assert "input_cache_read" in result["vendor/model-a"]
        assert "input_cache_read" not in result["vendor/model-b"]


class TestDeepInfraProviderProfile:
    """plugins/model-providers/deepinfra registration + aux resolution."""

    def test_profile_registered_with_alias_and_aux(self):
        from providers import get_provider_profile
        from agent.auxiliary_client import _get_aux_model_for_provider
        from hermes_cli.auth import resolve_provider
        from hermes_cli.config import OPTIONAL_ENV_VARS
        from hermes_cli.models import CANONICAL_PROVIDERS

        profile = get_provider_profile("deepinfra")
        assert profile is not None
        assert profile.name == "deepinfra"
        assert profile.auth_type == "api_key"
        # Alias resolves to the same profile.
        assert get_provider_profile("deep-infra") is profile
        assert resolve_provider("deep-infra") == "deepinfra"
        assert PROVIDER_REGISTRY["deepinfra"].inference_base_url == profile.base_url
        assert any(entry.slug == "deepinfra" for entry in CANONICAL_PROVIDERS)
        assert OPTIONAL_ENV_VARS["DEEPINFRA_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["DEEPINFRA_BASE_URL"]["password"] is False
        # Aux model is resolved via the profile (not via the legacy
        # _API_KEY_PROVIDER_AUX_MODELS_FALLBACK dict, which has no
        # deepinfra entry).
        assert _get_aux_model_for_provider("deepinfra")
        # Fallback list intentionally empty — live catalog is the source
        # of truth. Pin the shape only, not contents.
        assert isinstance(profile.fallback_models, tuple)


class TestRuntimeAlibabaRegionalAndTokenPlan:
    """#73265: the catalog-advertised Alibaba China/Token Plan variants must
    resolve on the shared execution path (resolve_runtime_provider), not just
    in the profile registry — provider, key, api mode, and base URL."""

    def test_runtime_alibaba_cn(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "ds-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="alibaba-cn")
        assert result["provider"] == "alibaba-cn"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "ds-key"
        assert result["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def test_runtime_alibaba_coding_plan_cn(self, monkeypatch):
        monkeypatch.setenv("ALIBABA_CODING_PLAN_API_KEY", "acp-key")
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="alibaba-coding-plan-cn")
        assert result["provider"] == "alibaba-coding-plan-cn"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "acp-key"
        assert result["base_url"] == "https://coding.dashscope.aliyuncs.com/v1"

    def test_runtime_alibaba_token_plan(self, monkeypatch):
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "atp-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="alibaba-token-plan")
        assert result["provider"] == "alibaba-token-plan"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "atp-key"
        assert result["base_url"] == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

    def test_runtime_alibaba_token_plan_cn(self, monkeypatch):
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "atp-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="alibaba-token-plan-cn")
        assert result["provider"] == "alibaba-token-plan-cn"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "atp-key"
        assert result["base_url"] == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
