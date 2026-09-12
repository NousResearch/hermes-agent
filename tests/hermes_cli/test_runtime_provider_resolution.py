import base64
import json
import time
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from hermes_cli import runtime_provider as rp


def test_configured_api_key_provider_without_key_fails_closed(monkeypatch):
    """A saved provider must not resolve as another authenticated provider."""
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "deepseek", "default": "deepseek-v4-pro"},
    )
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "deepseek",
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "source": "default",
        },
    )

    with pytest.raises(rp.AuthError, match="No usable credentials.*deepseek"):
        rp.resolve_runtime_provider()


def test_builtin_provider_config_api_key_precedes_environment(monkeypatch):
    """An explicit built-in provider key in config.yaml wins over its env key."""
    (get_hermes_home() / "config.yaml").write_text(
        "providers:\n"
        "  deepseek:\n"
        "    api_key: configured-deepseek-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-deepseek-key")

    from hermes_cli.config import load_config

    assert load_config()["providers"]["deepseek"]["api_key"] == "configured-deepseek-key"
    resolved = rp.resolve_runtime_provider(requested="deepseek")

    assert resolved["provider"] == "deepseek"
    assert resolved["base_url"] == "https://api.deepseek.com/v1"
    assert resolved["api_key"] == "configured-deepseek-key"
    assert resolved["source"] == "config:providers.deepseek.api_key"


def test_copilot_config_api_key_does_not_bypass_token_exchange(monkeypatch):
    """Copilot config values must not be used as direct inference tokens."""
    (get_hermes_home() / "config.yaml").write_text(
        "providers:\n"
        "  copilot:\n"
        "    api_key: configured-github-token\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rp, "load_pool", lambda _provider: None)
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("gho-raw-github-token", "GH_TOKEN"),
    )
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.get_copilot_api_token",
        lambda _token: ("exchanged-copilot-token", None),
    )

    resolved = rp.resolve_runtime_provider(requested="copilot")

    assert resolved["api_key"] == "exchanged-copilot-token"
    assert resolved["source"] == "GH_TOKEN"


def test_noauth_lmstudio_still_resolves(monkeypatch):
    """The fail-closed key guard preserves LM Studio's no-auth contract."""
    monkeypatch.setattr(rp, "load_pool", lambda _provider: SimpleNamespace(has_credentials=lambda: False))
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        lambda _provider: {
            "provider": "lmstudio",
            "api_key": "lmstudio-noauth",
            "base_url": "http://localhost:1234/v1",
            "source": "default",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_key"]


def _fake_invoke_jwt(ttl_seconds=3600):
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {
                "scope": "inference:invoke",
                "exp": int(time.time() + ttl_seconds),
            }
        ).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def test_resolve_runtime_provider_uses_credential_pool(monkeypatch):
    class _Entry:
        access_token = "pool-token"
        source = "manual"
        base_url = "https://chatgpt.com/backend-api/codex"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openai-codex")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())

    resolved = rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["provider"] == "openai-codex"
    assert resolved["api_key"] == "pool-token"
    assert resolved["credential_pool"] is not None
    assert resolved["source"] == "manual"


class TestCustomProviderPoolLoopbackNoKeyExemption:
    """Regression for issue #86864: legacy custom_providers configs often
    used short/placeholder api_keys ('123', 'm') for local no-auth
    services like Ollama -- fine for the endpoint itself, but
    has_usable_secret's 4-char floor now rejects them with a misleading
    "No usable credentials found" error and no migration path. Every
    OTHER resolution path in this file already substitutes
    "no-key-required" for a loopback endpoint with no usable secret; the
    credential-pool path was the one gap.
    """

    @staticmethod
    def _pool_with(api_key: str):
        entry = SimpleNamespace(runtime_api_key=api_key, access_token="")

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return entry

        return _Pool()

    def test_short_placeholder_key_exempted_for_loopback_endpoint(self, monkeypatch):
        """The exact reported repro: a 3-char legacy placeholder key
        ('123') for a local Ollama endpoint must resolve to the same
        "no-key-required" placeholder every other local no-auth path uses,
        not the raw unusable value."""
        monkeypatch.setattr(rp, "custom_provider_pool_key_candidates", lambda base_url, provider_name=None: ["custom:local-ollama"])
        monkeypatch.setattr(rp, "load_pool", lambda pool_key: self._pool_with("123"))

        result = rp._try_resolve_from_custom_pool("http://localhost:11434/v1", "custom", None)

        assert result is not None
        assert result["api_key"] == "no-key-required"

    def test_single_char_placeholder_key_also_exempted(self, monkeypatch):
        monkeypatch.setattr(rp, "custom_provider_pool_key_candidates", lambda base_url, provider_name=None: ["custom:local"])
        monkeypatch.setattr(rp, "load_pool", lambda pool_key: self._pool_with("m"))

        result = rp._try_resolve_from_custom_pool("http://127.0.0.1:11434/v1", "custom", None)

        assert result["api_key"] == "no-key-required"

    def test_short_key_not_exempted_for_non_loopback_endpoint(self, monkeypatch):
        """Sanity: the exemption is scoped to loopback hosts only -- a
        remote endpoint with a genuinely-too-short key must NOT get a
        free pass. The short value passes through unchanged, so the
        downstream has_usable_secret() gate still catches it."""
        monkeypatch.setattr(rp, "custom_provider_pool_key_candidates", lambda base_url, provider_name=None: ["custom:remote"])
        monkeypatch.setattr(rp, "load_pool", lambda pool_key: self._pool_with("xy"))

        result = rp._try_resolve_from_custom_pool("https://api.remote-vendor.example/v1", "custom", None)

        assert result["api_key"] == "xy"

    def test_usable_loopback_key_passes_through_unchanged(self, monkeypatch):
        """Sanity: a genuinely usable key for a loopback endpoint (a real
        API key happens to be configured for a local proxy, say) must not
        be silently overwritten."""
        monkeypatch.setattr(rp, "custom_provider_pool_key_candidates", lambda base_url, provider_name=None: ["custom:local"])
        monkeypatch.setattr(rp, "load_pool", lambda pool_key: self._pool_with("sk-genuinely-long-real-key-12345"))

        result = rp._try_resolve_from_custom_pool("http://localhost:11434/v1", "custom", None)

        assert result["api_key"] == "sk-genuinely-long-real-key-12345"


def test_qwen_oauth_auto_fallthrough_on_auth_failure(monkeypatch):
    """When requested_provider is 'auto' and Qwen creds fail, fall through."""
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "qwen-oauth")
    monkeypatch.setattr(
        rp,
        "resolve_qwen_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(AuthError("stale", provider="qwen-oauth", code="qwen_auth_missing")),
    )
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")

    # Should NOT raise — falls through to OpenRouter
    resolved = rp.resolve_runtime_provider(requested="auto")
    # The fallthrough means it won't be qwen-oauth
    assert resolved["provider"] != "qwen-oauth"


def test_resolve_runtime_provider_ai_gateway(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "ai-gateway")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-ai-gw-key")

    resolved = rp.resolve_runtime_provider(requested="ai-gateway")

    assert resolved["provider"] == "ai-gateway"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://ai-gateway.vercel.sh/v1"
    assert resolved["api_key"] == "test-ai-gw-key"
    assert resolved["requested_provider"] == "ai-gateway"


def test_resolve_runtime_provider_lmstudio_uses_token_when_present(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "lmstudio")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "default": "publisher/model-a",
        },
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: type("Pool", (), {"has_credentials": lambda self: False})(),
    )
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda provider: {
            "provider": "lmstudio",
            "api_key": "lm-token",
            "base_url": "http://127.0.0.1:1234/v1",
            "source": "LM_API_KEY",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_key"] == "lm-token"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "http://127.0.0.1:1234/v1"


def test_resolve_runtime_provider_lmstudio_honors_saved_base_url(monkeypatch):
    """Pre-existing configs with `provider: lmstudio` + custom base_url must keep working.

    Before this PR, `lmstudio` aliased to `custom`, so a user with a remote
    LM Studio (e.g. lab box) could write `provider: "lmstudio"` plus
    `base_url: "http://192.168.1.10:1234/v1"` and the custom path honored it.
    Now that `lmstudio` is first-class with `inference_base_url=127.0.0.1`,
    the saved `base_url` from `model_cfg` must still win — otherwise this
    PR is a silent breaking change for those users.
    """
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.delenv("LM_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "lmstudio")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "lmstudio",
            "base_url": "http://192.168.1.10:1234/v1",
            "default": "qwen/qwen3-coder-30b",
        },
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: type("Pool", (), {"has_credentials": lambda self: False})(),
    )
    # Don't mock resolve_api_key_provider_credentials — exercise the real
    # function so we test the end-to-end precedence between model_cfg and
    # the pconfig default.

    resolved = rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_mode"] == "chat_completions"
    # The saved base_url must NOT be shadowed by the 127.0.0.1 default.
    assert resolved["base_url"] == "http://192.168.1.10:1234/v1"
    # No-auth LM Studio: missing LM_API_KEY substitutes the placeholder.
    assert resolved["api_key"] == "dummy-lm-api-key"


def test_resolve_runtime_provider_lmstudio_saved_base_url_wins_over_env(monkeypatch):
    """Saved model.base_url takes precedence over LM_BASE_URL env var.

    This matches the established contract for all api_key providers: the
    explicit config value (model.base_url) wins over the env-derived
    default.  Users who saved a remote LM Studio URL must not have it
    silently overridden by a stale shell variable.
    """
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.setenv("LM_BASE_URL", "http://override.local:9999/v1")
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "lmstudio")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "lmstudio",
            "base_url": "http://192.168.1.10:1234/v1",
            "default": "qwen/qwen3-coder-30b",
        },
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: type("Pool", (), {"has_credentials": lambda self: False})(),
    )

    resolved = rp.resolve_runtime_provider(requested="lmstudio")

    assert resolved["provider"] == "lmstudio"
    assert resolved["api_mode"] == "chat_completions"
    # Saved config base_url wins over env var (standard contract).
    assert resolved["base_url"] == "http://192.168.1.10:1234/v1"
    assert resolved["api_key"] == "dummy-lm-api-key"


def test_resolve_runtime_provider_ai_gateway_explicit_override_skips_pool(monkeypatch):
    def _unexpected_pool(provider):
        raise AssertionError(f"load_pool should not be called for {provider}")

    def _unexpected_provider_resolution(provider):
        raise AssertionError(f"resolve_api_key_provider_credentials should not be called for {provider}")

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "ai-gateway")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setattr(rp, "load_pool", _unexpected_pool)
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        _unexpected_provider_resolution,
    )

    resolved = rp.resolve_runtime_provider(
        requested="ai-gateway",
        explicit_api_key="ai-gateway-explicit-token",
        explicit_base_url="https://proxy.example.com/v1/",
    )

    assert resolved["provider"] == "ai-gateway"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "ai-gateway-explicit-token"
    assert resolved["base_url"] == "https://proxy.example.com/v1"
    assert resolved["source"] == "explicit"
    assert resolved.get("credential_pool") is None


def test_resolve_runtime_provider_openrouter_explicit(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(
        requested="openrouter",
        explicit_api_key="test-key",
        explicit_base_url="https://example.com/v1/",
    )

    assert resolved["provider"] == "openrouter"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["api_key"] == "test-key"
    assert resolved["base_url"] == "https://example.com/v1"
    assert resolved["source"] == "explicit"


@pytest.mark.parametrize(
    "case_name, model_cfg, expected_api_mode",
    [
        (
            "stale_anthropic_api_mode_is_ignored_on_provider_switch",
            {
                "provider": "anthropic",
                "api_mode": "anthropic_messages",
                "default": "claude-3-5-sonnet",
            },
            "chat_completions",
        ),
        (
            "matching_provider_still_honors_persisted_api_mode",
            {
                "provider": "gemini",
                "api_mode": "anthropic_messages",
                "default": "gemini-2.5-pro",
            },
            "anthropic_messages",
        ),
        (
            "no_persisted_provider_still_honors_api_mode",
            {
                "api_mode": "anthropic_messages",
                "default": "gemini-2.5-pro",
            },
            "anthropic_messages",
        ),
    ],
)
def test_resolve_runtime_provider_gemini_explicit_api_mode_provider_guard(
    monkeypatch, case_name, model_cfg, expected_api_mode
):
    """A persisted model.api_mode from a different provider must not leak
    into an explicitly-resolved provider after a switch (issue #74318).

    Only when ``model_cfg["provider"]`` matches the provider actually being
    resolved (or is unset) should the persisted api_mode be honoured.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "gemini")
    monkeypatch.setattr(rp, "_get_model_config", lambda: model_cfg)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(
        requested="gemini",
        explicit_api_key="fake-gemini-key",
        explicit_base_url="https://fake-gemini-endpoint.example.com/v1beta/",
    )

    assert resolved["provider"] == "gemini", case_name
    assert resolved["api_mode"] == expected_api_mode, case_name
    assert resolved["api_key"] == "fake-gemini-key", case_name
    assert resolved["base_url"] == "https://fake-gemini-endpoint.example.com/v1beta", case_name
    assert resolved["source"] == "explicit", case_name


def test_resolve_runtime_provider_auto_uses_openrouter_pool(monkeypatch):
    class _Entry:
        access_token = "pool-key"
        source = "manual"
        base_url = "https://openrouter.ai/api/v1"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="auto")

    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "pool-key"
    assert resolved["base_url"] == "https://openrouter.ai/api/v1"
    assert resolved["source"] == "manual"
    assert resolved.get("credential_pool") is not None


def test_resolve_runtime_provider_openrouter_explicit_api_key_skips_pool(monkeypatch):
    class _Entry:
        access_token = "pool-key"
        source = "manual"
        base_url = "https://openrouter.ai/api/v1"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(
        requested="openrouter",
        explicit_api_key="explicit-key",
    )

    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "explicit-key"
    assert resolved["base_url"] == rp.OPENROUTER_BASE_URL
    assert resolved["source"] == "explicit"
    assert resolved.get("credential_pool") is None


def test_resolve_runtime_provider_openrouter_ignores_codex_config_base_url(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"] == rp.OPENROUTER_BASE_URL


def test_resolve_runtime_provider_auto_uses_custom_config_base_url(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "auto",
            "base_url": "https://custom.example/v1/",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="auto")

    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"] == "https://custom.example/v1"


def test_openrouter_key_takes_priority_over_openai_key(monkeypatch):
    """OPENROUTER_API_KEY should be used over OPENAI_API_KEY when both are set.

    Regression test for #289: users with OPENAI_API_KEY in .bashrc had it
    sent to OpenRouter instead of their OPENROUTER_API_KEY.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-lose")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-should-win")

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["api_key"] == "sk-or-should-win"


def test_openai_key_used_when_no_openrouter_key(monkeypatch):
    """OPENAI_API_KEY is used as fallback when OPENROUTER_API_KEY is not set."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["api_key"] == "sk-openai-fallback"


def test_custom_endpoint_uses_saved_config_base_url_when_env_missing(monkeypatch):
    """Persisted custom endpoints in config.yaml must still resolve when
    OPENAI_BASE_URL is absent from the current environment.
    OPENAI_API_KEY / OPENROUTER_API_KEY must NOT leak to a non-OpenAI host
    (issue #28660) — local LLM servers get no-key-required instead."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "http://127.0.0.1:1234/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "http://127.0.0.1:1234/v1"
    # OPENAI_API_KEY must not leak to an unrelated host — local servers get
    # the no-key-required placeholder so the OpenAI SDK stays happy.
    assert resolved["api_key"] == "no-key-required"


def test_custom_endpoint_explicit_custom_prefers_config_key(monkeypatch):
    """Explicit 'custom' provider with config base_url+api_key should use them.

    Updated for #4165: config.yaml is the source of truth, not OPENAI_BASE_URL.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": "https://my-vllm-server.example.com/v1",
        "api_key": "sk-vllm-key",
    })
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-...leak")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://my-vllm-server.example.com/v1"
    assert resolved["api_key"] == "sk-vllm-key"


def test_bare_custom_uses_loopback_model_base_url_when_provider_not_custom(monkeypatch):
    """Regression for #14676: /model can select Custom while YAML still lists another provider."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "openrouter",
            "base_url": "http://127.0.0.1:8082/v1",
            "default": "my-local-model",
        },
    )
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "http://127.0.0.1:8082/v1"
    # 127.0.0.1 is not openai.com — OPENAI_API_KEY must not leak here
    assert resolved["api_key"] == "no-key-required"


def test_named_custom_provider_uses_saved_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "Local",
                    "base_url": "http://1.2.3.4:1234/v1",
                    "api_key": "local-provider-key",
                    "model": "gpt-5.6",
                    "capabilities": {"openai_native_compaction": True},
                }
            ]
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="local")

    assert resolved["provider"] == "custom"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "http://1.2.3.4:1234/v1"
    assert resolved["api_key"] == "local-provider-key"
    assert resolved["requested_provider"] == "local"
    assert resolved["capabilities"] == {"openai_native_compaction": True}
    assert resolved["source"] == "custom_provider:Local"


def test_named_custom_provider_filters_capabilities_at_lookup_boundary(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "local": {
                    "name": "Local",
                    "base_url": "http://1.2.3.4:1234/v1",
                    "capabilities": {
                        "openai_native_compaction": True,
                        "invalid-value": "yes",
                        42: True,
                    },
                }
            }
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    provider = rp._get_named_custom_provider("local")

    assert provider["capabilities"] == {"openai_native_compaction": True}


def test_bare_custom_resolves_providers_dict_entry_named_custom(monkeypatch):
    """A request for bare ``provider="custom"`` must resolve a literal
    ``providers.custom`` entry (e.g. a cliproxy endpoint) instead of falling
    through to the global default. Regression for cron jobs stored with
    ``provider: "custom"`` failing with ``auth_unavailable: providers=codex``.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "custom": {
                    "api": "https://cliproxy.example.com/v1",
                    "api_key": "cliproxy-key",
                    "default_model": "gpt-5.4",
                    "name": "CLIProxy",
                }
            }
        },
    )
    # Reaching resolve_provider for bare custom with a matching entry means the
    # named-custom path was bypassed — that is the bug we are fixing.
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider must not be called; providers.custom should match"
            )
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://cliproxy.example.com/v1"
    assert resolved["api_key"] == "cliproxy-key"
    assert resolved["requested_provider"] == "custom"




def test_named_custom_provider_same_url_uses_matching_key_env_and_api_mode(monkeypatch):
    """Named custom providers on one gateway must keep their own credentials and protocol."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("GPT_KEY", "gpt-secret")
    monkeypatch.setenv("CLAUDE_KEY", "claude-secret")
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "gpt",
                    "base_url": "https://gateway.example.com",
                    "key_env": "GPT_KEY",
                    "api_mode": "codex_responses",
                    "model": "gpt-5.5",
                },
                {
                    "name": "claude",
                    "base_url": "https://gateway.example.com",
                    "key_env": "CLAUDE_KEY",
                    "api_mode": "anthropic_messages",
                    "model": "claude-opus-4-8",
                },
            ],
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="custom:claude")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "https://gateway.example.com"
    assert resolved["api_key"] == "claude-secret"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["requested_provider"] == "custom:claude"
    assert resolved["model"] == "claude-opus-4-8"


def test_named_custom_provider_falls_back_to_openai_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "Local LLM",
                    "base_url": "http://localhost:1234/v1",
                }
            ]
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_provider",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError(
                "resolve_provider should not be called for named custom providers"
            )
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="custom:local-llm")

    assert resolved["base_url"] == "http://localhost:1234/v1"
    # localhost is not openai.com — OPENAI_API_KEY must not leak to local endpoints (#28660)
    assert resolved["api_key"] == "no-key-required"
    assert resolved["requested_provider"] == "custom:local-llm"






def test_named_custom_provider_wins_over_builtin_alias(monkeypatch):
    """A custom_providers entry named after a built-in *alias* (not a canonical
    provider name) must win over the built-in.  Regression guard for #15743:
    when users define ``custom_providers: [{name: kimi, ...}]`` and reference
    ``provider: kimi``, the built-in alias rewriting (``kimi`` → ``kimi-coding``)
    would otherwise hijack the request and send it to the wrong endpoint.
    """
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "kimi",
                    "base_url": "https://my-custom-kimi.example.com/v1",
                    "api_key": "my-kimi-key",
                }
            ]
        },
    )

    entry = rp._get_named_custom_provider("kimi")

    assert entry is not None
    assert entry["base_url"] == "https://my-custom-kimi.example.com/v1"
    assert entry["api_key"] == "my-kimi-key"


def test_explicit_openrouter_skips_openai_base_url(monkeypatch):
    """When the user explicitly requests openrouter, OPENAI_BASE_URL
    (which may point to a custom endpoint) must not override the
    OpenRouter base URL.  Regression test for #874."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-custom-llm.example.com/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["provider"] == "openrouter"
    assert "openrouter.ai" in resolved["base_url"]
    assert "my-custom-llm" not in resolved["base_url"]
    assert resolved["api_key"] == "or-test-key"






# ── api_mode config override tests ──────────────────────────────────────




def test_minimax_config_base_url_overrides_hardcoded_default(monkeypatch):
    """model.base_url in config.yaml should override the hardcoded default (#6039)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "minimax",
        "base_url": "https://api.minimaxi.com/anthropic",
    })
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["base_url"] == "https://api.minimaxi.com/anthropic"
    assert resolved["api_mode"] == "anthropic_messages"


def test_opencode_go_model_derivation_beats_stale_persisted_api_mode(monkeypatch):
    """opencode-zen/go re-derive api_mode from the effective model on every
    resolve, ignoring any persisted ``api_mode`` in config. Refs #16878 /
    PR #16888: the persisted mode from the previous default model must not
    leak across /model switches (a stale ``anthropic_messages`` on a
    chat_completions target would strip /v1 from base_url and 404).

    minimax-m2.5 is an Anthropic-routed model on opencode-go, so even when
    the config claims ``api_mode: chat_completions`` the runtime must pick
    ``anthropic_messages`` — the model dictates the mode, not the stale
    persisted setting.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-go")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "opencode-go",
            "default": "minimax-m2.5",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="opencode-go")

    assert resolved["provider"] == "opencode-go"
    assert resolved["api_mode"] == "anthropic_messages"


# ------------------------------------------------------------------
# fix #2562 — resolve_provider("custom") must not remap to "openrouter"
# ------------------------------------------------------------------






def test_auto_detected_nous_auth_failure_falls_through_to_openrouter(monkeypatch):
    """When auto-detect picks Nous but credentials are revoked, fall through to OpenRouter."""
    from hermes_cli.auth import AuthError

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "load_config", lambda: {})

    # resolve_provider returns "nous" (stale active_provider in auth.json)
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "nous")
    # load_pool returns empty pool so we hit the direct credential resolution
    monkeypatch.setattr(rp, "load_pool", lambda p: type("P", (), {
        "has_credentials": lambda self: False,
    })())
    # Nous credential resolution fails with revoked token
    monkeypatch.setattr(
        rp, "resolve_nous_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(
            AuthError("Refresh session has been revoked",
                      provider="nous", code="invalid_grant", relogin_required=True)
        ),
    )

    # With requested="auto", should fall through to OpenRouter
    resolved = rp.resolve_runtime_provider(requested="auto")
    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "test-or-key"


# ------------------------------------------------------------------
# fix #7828 — custom_providers model field must propagate to runtime
# ------------------------------------------------------------------




# ---------------------------------------------------------------------------
# GHSA-76xc-57q6-vm5m — Ollama URL substring leak
#
# Same bug class as the previously-fixed GHSA-xf8p-v2cg-h7h5 (OpenRouter).
# _resolve_openrouter_runtime's custom-endpoint branch selects OLLAMA_API_KEY
# when the base_url "looks like" ollama.com. Previous implementation used
# raw substring match; a custom base_url whose PATH or look-alike host
# merely contained "ollama.com" leaked OLLAMA_API_KEY to that endpoint.
# Fix: use base_url_host_matches (same helper as the OpenRouter sweep).
# ---------------------------------------------------------------------------

class TestOllamaUrlSubstringLeak:
    """Call-site regression tests for the fix in _resolve_openrouter_runtime."""

    def _make_cfg(self, base_url):
        return {"base_url": base_url, "api_key": "", "provider": "custom"}

    def test_ollama_key_not_leaked_to_path_injection(self, monkeypatch):
        """http://127.0.0.1:9000/ollama.com/v1 — attacker endpoint with
        ollama.com in PATH. Must resolve to OPENAI_API_KEY, not OLLAMA_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-SECRET-should-not-leak")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "http://127.0.0.1:9000/ollama.com/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

        resolved = rp.resolve_runtime_provider(requested="custom")

        assert "ol-SECRET" not in resolved["api_key"], (
            "OLLAMA_API_KEY must not be sent to an endpoint whose "
            "hostname is not ollama.com (GHSA-76xc-57q6-vm5m)"
        )
        # OPENAI_API_KEY must also not leak to non-openai.com hosts (#28660)
        assert resolved["api_key"] == "no-key-required"

    def test_ollama_key_not_leaked_to_lookalike_host(self, monkeypatch):
        """ollama.com.attacker.test — look-alike host. OLLAMA_API_KEY
        must not be sent."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-SECRET-should-not-leak")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "http://ollama.com.attacker.test:9000/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

        resolved = rp.resolve_runtime_provider(requested="custom")

        assert "ol-SECRET" not in resolved["api_key"]
        # OPENAI_API_KEY must also not leak to non-openai.com hosts (#28660)
        assert resolved["api_key"] == "no-key-required"

    def test_ollama_key_sent_to_genuine_ollama_com(self, monkeypatch):
        """https://ollama.com/v1 — legit Ollama Cloud. OLLAMA_API_KEY
        should be used."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-legit-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "https://ollama.com/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)
        monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

        resolved = rp.resolve_runtime_provider(requested="custom")

        assert resolved["api_key"] == "ol-legit-key"



# =============================================================================
# Azure Foundry — both OpenAI-style and Anthropic-style endpoints
# =============================================================================

class TestAzureFoundryResolution:
    """Verify Azure Foundry resolves correctly for both API modes."""

    def _make_cfg(self, base_url: str, api_mode: str = "chat_completions"):
        return {
            "provider": "azure-foundry",
            "base_url": base_url,
            "api_mode": api_mode,
            # GPT-4 speaks chat completions on Azure, so this test's assertion
            # about chat_completions stays valid across the Apr 2026 fix that
            # upgrades GPT-5.x / codex deployments to codex_responses.
            "default": "gpt-4.1",
        }


    def test_azure_foundry_missing_base_url_raises(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        with pytest.raises(rp.AuthError, match="base URL"):
            rp.resolve_runtime_provider(requested="azure-foundry")


    # -- Model-family api_mode inference -------------------------------------
    # Azure rejects /chat/completions on GPT-5.x / codex / o-series with
    # ``400 "The requested operation is unsupported."`` — the resolver must
    # upgrade api_mode to ``codex_responses`` for those models even when the
    # config was persisted as ``chat_completions`` (the default the setup
    # wizard writes when the user didn't pick explicitly).

    def _make_cfg_with_model(self, model: str, api_mode: str = "chat_completions"):
        return {
            "provider": "azure-foundry",
            "base_url": "https://synopsisse.openai.azure.com/openai/v1",
            "api_mode": api_mode,
            "default": model,
        }

    def test_gpt5_codex_upgrades_chat_completions_to_responses(self, monkeypatch):
        """Reproduces Bob's April 2026 bug: gpt-5.3-codex on chat_completions."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._make_cfg_with_model("gpt-5.3-codex", "chat_completions"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["api_mode"] == "codex_responses"
        assert resolved["base_url"] == "https://synopsisse.openai.azure.com/openai/v1"


    def test_o3_mini_upgrades(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._make_cfg_with_model("o3-mini", "chat_completions"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["api_mode"] == "codex_responses"


# ──────────────────────────────────────────────────────────────────────────
# Azure Anthropic — honor user-specified env var hints (key_env / api_key_env)
#
# When the user points provider=anthropic at an Azure Foundry base URL, the
# runtime resolver previously hardcoded `AZURE_ANTHROPIC_KEY` and
# `ANTHROPIC_API_KEY` as the only env var sources.  This meant
# `key_env: MY_CUSTOM_VAR` on the model config was silently ignored — and
# the Azure Foundry docs that showed `api_key_env:` were broken as a result.
#
# These tests lock in the priority chain:
#   1. model_cfg.key_env → os.getenv(value)
#   2. model_cfg.api_key_env → os.getenv(value) (docs alias)
#   3. model_cfg.api_key (inline value)
#   4. AZURE_ANTHROPIC_KEY env var
#   5. ANTHROPIC_API_KEY env var
# ──────────────────────────────────────────────────────────────────────────


class TestAzureAnthropicEnvVarHint:
    _AZURE_URL = "https://my-resource.services.ai.azure.com/anthropic"

    def _cfg(self, **overrides):
        base = {"provider": "anthropic", "base_url": self._AZURE_URL}
        base.update(overrides)
        return base

    def test_key_env_hint_picks_custom_var(self, monkeypatch):
        """model.key_env names a non-default env var → that var's value is used."""
        monkeypatch.delenv("AZURE_ANTHROPIC_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("MY_CUSTOM_AZURE_KEY", "from-custom-var")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._cfg(key_env="MY_CUSTOM_AZURE_KEY"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "from-custom-var"
        assert resolved["base_url"] == self._AZURE_URL


    def test_key_env_points_at_unset_var_falls_through(self, monkeypatch):
        """If key_env names an env var that isn't set, fall through to the
        historical fixed names rather than failing outright."""
        monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "fallback-works")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("UNSET_VAR", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._cfg(key_env="UNSET_VAR"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "fallback-works"


    def test_non_azure_anthropic_path_ignores_key_env(self, monkeypatch):
        """key_env is only consulted on Azure endpoints — non-Azure Anthropic
        still goes through the regular resolve_anthropic_token chain."""
        monkeypatch.setenv("MY_KEY", "custom-key-value")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",  # non-Azure
            "key_env": "MY_KEY",
        })
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)
        called = {"resolve_anthropic_token": False}
        def _fake_resolve():
            called["resolve_anthropic_token"] = True
            return "token-from-resolver"
        monkeypatch.setattr(
            "agent.anthropic_credentials.resolve_anthropic_token",
            _fake_resolve,
        )

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        # The normal chain runs — key_env is not consulted off-Azure.
        assert called["resolve_anthropic_token"] is True
        assert resolved["api_key"] == "token-from-resolver"


# ──────────────────────────────────────────────────────────────────────────
# custom_providers / providers normalizer — api_key_env alias for key_env
# ──────────────────────────────────────────────────────────────────────────


class TestProviderEntryApiKeyEnvAlias:
    """The `providers.<name>` and `custom_providers[i]` normalizer must accept
    `api_key_env` as an alias for `key_env` so configs written against the
    documented Azure Foundry YAML shape (or imported from other tools that
    use `api_key_env`) resolve correctly."""

    def test_snake_case_api_key_env_normalizes_to_key_env(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "vendor",
            "base_url": "https://api.vendor.example.com/v1",
            "api_key_env": "MY_VENDOR_KEY",
        }
        normalized = _normalize_custom_provider_entry(dict(entry), provider_key="vendor")
        assert normalized is not None
        assert normalized.get("key_env") == "MY_VENDOR_KEY"


    def test_valid_fields_set_lists_key_env(self):
        """The _VALID_CUSTOM_PROVIDER_FIELDS documentation set must include
        key_env so the set stays in sync with what the runtime actually reads."""
        from hermes_cli.config import _VALID_CUSTOM_PROVIDER_FIELDS
        assert "key_env" in _VALID_CUSTOM_PROVIDER_FIELDS

    def test_extra_body_is_supported_schema(self):
        from hermes_cli.config import (
            _VALID_CUSTOM_PROVIDER_FIELDS,
            _normalize_custom_provider_entry,
        )
        entry = {
            "name": "vendor",
            "base_url": "https://api.vendor.example.com/v1",
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "include_reasoning": True,
            },
        }
        normalized = _normalize_custom_provider_entry(dict(entry), provider_key="vendor")
        assert normalized is not None
        assert "extra_body" in _VALID_CUSTOM_PROVIDER_FIELDS
        assert normalized["extra_body"] == entry["extra_body"]
# =============================================================================
# Tencent TokenHub — API-key provider runtime resolution
# =============================================================================



# ---------------------------------------------------------------------------
# minimax-oauth runtime resolution tests (added by feat/minimax-oauth-provider)
# ---------------------------------------------------------------------------

def test_minimax_oauth_runtime_returns_anthropic_messages_mode(monkeypatch):
    """resolve_runtime_provider for minimax-oauth must return api_mode='anthropic_messages'."""
    from hermes_cli.auth import MINIMAX_OAUTH_GLOBAL_INFERENCE

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax-oauth")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "minimax-oauth"})
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)
    monkeypatch.setattr(
        rp,
        "_resolve_named_custom_runtime",
        lambda **k: None,
    )
    monkeypatch.setattr(
        rp,
        "_resolve_explicit_runtime",
        lambda **k: None,
    )

    fake_creds = {
        "provider": "minimax-oauth",
        "api_key": "mock-access-token",
        "base_url": MINIMAX_OAUTH_GLOBAL_INFERENCE.rstrip("/"),
        "source": "oauth",
    }

    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(auth_mod, "resolve_minimax_oauth_runtime_credentials",
                        lambda **k: fake_creds)

    resolved = rp.resolve_runtime_provider(requested="minimax-oauth")

    assert resolved["provider"] == "minimax-oauth"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["api_key"] == "mock-access-token"


def test_minimax_oauth_pool_forces_anthropic_messages_despite_stale_config(monkeypatch):
    """A pooled MiniMax OAuth token must not inherit stale chat_completions config."""

    class _Entry:
        access_token = "oauth-token"
        source = "manual:minimax_oauth"
        base_url = "https://api.minimax.io/anthropic"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax-oauth")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "minimax-oauth",
            "default": "MiniMax-M2.7",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())
    monkeypatch.setattr(rp, "_resolve_named_custom_runtime", lambda **k: None)
    monkeypatch.setattr(rp, "_resolve_explicit_runtime", lambda **k: None)

    resolved = rp.resolve_runtime_provider(requested="minimax-oauth")

    assert resolved["provider"] == "minimax-oauth"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://api.minimax.io/anthropic"


# ----------------------------------------------------------------------
# GitHub #27132 — provider aliases (ollama/vllm/llamacpp/llama-cpp) must
# follow the same base_url trust + routing rules as bare `provider: custom`.
# Without this, a YAML `provider: ollama` with a LAN/WireGuard `base_url`
# silently falls through to OpenRouter (HTTP 401).
# ----------------------------------------------------------------------






def test_openai_key_only_sent_to_openai_host(monkeypatch):
    """OPENAI_API_KEY must only be forwarded to api.openai.com, not to
    arbitrary custom endpoints (issue #28660)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://api.deepseek.com/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://api.deepseek.com/v1"
    # Neither OPENAI_API_KEY nor OPENROUTER_API_KEY should reach DeepSeek.
    assert resolved["api_key"] == "no-key-required"


def test_openai_key_reaches_openai_host(monkeypatch):
    """OPENAI_API_KEY must be forwarded when the base_url is api.openai.com."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://api.openai.com/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["api_key"] == "sk-openai-secret"


def test_openrouter_key_reaches_openrouter_host(monkeypatch):
    """OPENROUTER_API_KEY must be forwarded when the base_url is openrouter.ai."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["api_key"] == "or-secret"


# ----------------------------------------------------------------------
# Issue #28660 — bonus: `<VENDOR>_API_KEY` derivation from host.
# After the host-gating fix, users with a `DEEPSEEK_API_KEY` set and
# `base_url: https://api.deepseek.com/v1` should get the key picked up
# without needing to configure custom_providers.key_env first.
# ----------------------------------------------------------------------






def test_host_derived_key_does_not_leak_to_lookalike_host(monkeypatch):
    """DEEPSEEK_API_KEY must NOT be sent to an attacker-controlled lookalike
    host (e.g. api.deepseek.com.attacker.test). The host-derive helper uses
    proper hostname parsing so it picks the *attacker's* vendor label, not
    DEEPSEEK — and any real DEEPSEEK_API_KEY stays put."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://api.deepseek.com.attacker.test/v1",
        },
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-secret")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert "sk-deepseek-secret" not in (resolved["api_key"] or "")
    # No ATTACKER_API_KEY is set, so the chain falls through to no-key-required.
    assert resolved["api_key"] == "no-key-required"


def test_host_derived_key_skips_already_handled_vendors(monkeypatch):
    """The host-derive helper must not double-resolve OPENAI / OPENROUTER /
    OLLAMA env vars — those are owned by their explicit host-gated paths.
    Specifically, OPENAI_API_KEY must not leak to a non-openai host via the
    `openai` label in a path or subdomain."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            # Hosts like proxy.openai.evil should derive nothing — but even
            # if "openai" were the registrable label, the explicit
            # OPENAI/OPENROUTER/OLLAMA filter blocks it.
            "base_url": "https://api.example.com/v1",
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret")

    resolved = rp.resolve_runtime_provider(requested="custom")

    # example.com has no EXAMPLE_API_KEY set, and OPENAI/OPENROUTER are gated
    # on their own hosts — chain falls through to no-key-required.
    assert resolved["api_key"] == "no-key-required"




def _patch_bedrock(monkeypatch, config_default=""):
    """Stub the bedrock_adapter helpers resolve_runtime_provider imports.

    Leaves the real ``is_anthropic_bedrock_model`` in place so the dual-path
    routing decision is exercised for real. ``config_default`` is the persisted
    ``model.default`` — kept non-Claude here to prove ``target_model`` (not the
    stale config default) drives the api_mode decision.
    """
    import agent.bedrock_adapter as ba

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "bedrock")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"default": config_default})
    monkeypatch.setattr(rp, "load_config", lambda: {"bedrock": {}})
    monkeypatch.setattr(ba, "has_aws_credentials", lambda: True)
    monkeypatch.setattr(ba, "resolve_aws_auth_env_var", lambda: "AWS_PROFILE")
    monkeypatch.setattr(ba, "resolve_bedrock_region", lambda: "eu-north-1")


def test_resolve_runtime_provider_bedrock_claude_target_model_uses_anthropic_messages(monkeypatch):
    """Claude-on-Bedrock delegation must route through the AnthropicBedrock SDK.

    Regression for #49095: the bedrock branch derived api_mode from the stale
    persisted ``model.default`` instead of ``target_model``. When delegation
    targets a Claude model but the parent's default is non-Claude, the wrong
    branch (Converse) was picked, and the child silently fell back to
    openrouter/free. ``target_model`` must win so Claude keeps the
    anthropic_messages path (prompt caching, thinking budgets).
    """
    _patch_bedrock(monkeypatch, config_default="amazon.nova-pro-v1:0")

    resolved = rp.resolve_runtime_provider(
        requested="bedrock",
        target_model="global.anthropic.claude-sonnet-4-6",
    )

    assert resolved["provider"] == "bedrock"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved.get("bedrock_anthropic") is True


def test_auto_provider_with_local_base_url_bypasses_anthropic_key(monkeypatch):
    """provider:auto + base_url:localhost should NOT route to Anthropic even if
    ANTHROPIC_API_KEY is set in the environment. Regression test for #3846.

    Users running Ollama locally had requests silently sent to Anthropic's API
    because resolve_provider("auto") picked up ANTHROPIC_API_KEY before the
    config.yaml base_url was checked.
    """
    # ANTHROPIC_API_KEY is present in environment — should be ignored
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "default": "ollama/minimax-m2.7:cloud",
            "provider": "auto",
            "base_url": "http://localhost:11434",
        },
    )

    resolved = rp.resolve_runtime_provider()

    # Must NOT go to Anthropic's API
    assert "anthropic.com" not in resolved.get("base_url", ""), (
        f"Expected custom endpoint, got Anthropic: {resolved}"
    )
    assert resolved["base_url"] == "http://localhost:11434"
    # provider should be custom/openrouter (OpenAI-compat), not anthropic
    assert resolved["provider"] != "anthropic", (
        f"Should have routed to custom endpoint, not anthropic: {resolved}"
    )


def test_auto_provider_with_known_cloud_base_url_still_uses_anthropic(monkeypatch):
    """provider:auto + base_url pointing to Anthropic should still use Anthropic.

    The local-endpoint bypass only applies to non-cloud endpoints; when the
    configured base_url IS a cloud API root, resolve_provider() must run
    normally and pick up ANTHROPIC_API_KEY.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "auto",
            "base_url": "https://api.anthropic.com",
        },
    )

    resolved = rp.resolve_runtime_provider()

    # Cloud base_url → bypass does NOT fire → resolve_provider picks anthropic.
    assert resolved["provider"] == "anthropic", (
        f"Cloud base_url must not be diverted to the custom resolver: {resolved}"
    )


def test_auto_provider_lookalike_cloud_host_does_not_bypass_to_cloud(monkeypatch):
    """A look-alike host (api.anthropic.com.attacker.test) must be treated as a
    custom endpoint, NOT mistaken for the real Anthropic cloud.

    Guards the host-match (vs naive substring) check in the local-endpoint
    bypass: substring matching on "api.anthropic.com" would wrongly classify
    this attacker-controlled host as cloud and hand it the ANTHROPIC_API_KEY.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    lookalike = "http://api.anthropic.com.attacker.test/v1"
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "auto", "base_url": lookalike},
    )

    resolved = rp.resolve_runtime_provider()

    # Host doesn't actually match anthropic.com → bypass fires → custom route,
    # and the real Anthropic credential is NOT sent there.
    assert resolved["provider"] != "anthropic", (
        f"Look-alike host must not be classified as Anthropic cloud: {resolved}"
    )
    assert resolved["base_url"] == lookalike


# ---------------------------------------------------------------------------
# extra_headers support for named custom providers (#3526 salvage)
# ---------------------------------------------------------------------------






def test_resolve_named_custom_runtime_pool_result_includes_extra_headers(monkeypatch):
    """extra_headers must survive the credential-pool path too."""
    pool_return_value = {
        "provider": "custom",
        "api_mode": "chat_completions",
        "base_url": "https://lmstudio.example.com/v1",
        "api_key": "pooled-key",
        "source": "pool:lmstudio-pool",
        "credential_pool": "fake-pool",
    }
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: pool_return_value)
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda p: {
            "name": "lmstudio",
            "base_url": "https://lmstudio.example.com/v1",
            "api_key": "not-used-when-pooled",
            "extra_headers": {
                "CF-Access-Client-Id": "xxx.access",
                "CF-Access-Client-Secret": "yyy",
            },
        },
    )

    # Exercise the public resolver: it is responsible for preserving the
    # original named identity after the pool path canonicalizes to "custom".
    resolved = rp.resolve_runtime_provider(requested="custom:lmstudio")

    assert resolved is not None
    assert resolved["extra_headers"] == {
        "CF-Access-Client-Id": "xxx.access",
        "CF-Access-Client-Secret": "yyy",
    }
    assert resolved["api_key"] == "pooled-key"
    assert resolved["source"] == "pool:lmstudio-pool"
    assert resolved["provider"] == "custom"
    assert resolved["requested_provider"] == "custom:lmstudio"


def test_resolve_runtime_provider_opencode_free_keyless_despite_exhausted_pool(monkeypatch):
    """OpenCode Free is keyless: an exhausted credential pool must not raise
    a missing-credential error. The provider resolves with the keyless
    placeholder + empty-Authorization headers so the request goes out
    anonymously."""
    class _ExhaustedPool:
        def has_credentials(self):
            return True

        def select(self):
            return None

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-free")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "opencode-free", "default": "x-preview-f-free"},
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: _ExhaustedPool())

    resolved = rp.resolve_runtime_provider(
        requested="opencode-free", target_model="x-preview-f-free"
    )

    assert resolved["provider"] == "opencode-free"
    assert resolved["api_key"] == "opencode-zen-free-keyless"
    assert resolved["base_url"] == "https://opencode.ai/zen/v1"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["default_headers"]["Authorization"] == ""


def test_resolve_runtime_provider_opencode_free_missing_env_still_resolves(monkeypatch):
    """OpenCode Free resolves keylessly with no env var configured at all —
    the provider declares no credentials."""
    class _NoPool:
        def has_credentials(self):
            return False

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-free")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "opencode-free", "default": "x-preview-f-free"},
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: _NoPool())

    resolved = rp.resolve_runtime_provider(
        requested="opencode-free", target_model="x-preview-f-free"
    )

    assert resolved["provider"] == "opencode-free"
    assert resolved["api_key"] == "opencode-zen-free-keyless"
    assert resolved["base_url"] == "https://opencode.ai/zen/v1"


def test_custom_provider_explicit_target_model_wins(monkeypatch):
    """An explicit target_model must not be silently replaced by the custom
    provider's configured default model (regression: auxiliary slots such as
    background-review resolve a concrete model and got default_model instead)."""
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda p: {
            "name": "myproxy",
            "base_url": "http://127.0.0.1:10100/v1",
            "api_key": "no-key-required",
            "model": "default-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="myproxy", target_model="myproxy/gemini-flash")

    assert resolved is not None
    assert resolved["provider"] == "custom"
    assert resolved["model"] == "myproxy/gemini-flash"
    assert resolved["base_url"] == "http://127.0.0.1:10100/v1"


def test_custom_provider_without_target_model_keeps_default(monkeypatch):
    """No target_model -> the provider's configured model is preserved."""
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda p: {
            "name": "myproxy",
            "base_url": "http://127.0.0.1:10100/v1",
            "api_key": "no-key-required",
            "model": "default-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="myproxy")

    assert resolved is not None
    assert resolved["model"] == "default-model"


def test_custom_provider_pool_target_model_wins(monkeypatch):
    """Pooled-credentials path also honors target_model over the default."""
    monkeypatch.setattr(
        rp,
        "_try_resolve_from_custom_pool",
        lambda *a, **k: {
            "provider": "custom",
            "api_key": "pooled-key",
            "base_url": "http://127.0.0.1:10100/v1",
        },
    )
    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda p: {
            "name": "myproxy",
            "base_url": "http://127.0.0.1:10100/v1",
            "api_key": "no-key-required",
            "model": "default-model",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="myproxy", target_model="myproxy/gemini-flash")

    assert resolved is not None
    assert resolved["model"] == "myproxy/gemini-flash"
