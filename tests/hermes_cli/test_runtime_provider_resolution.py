import pytest

from hermes_cli import runtime_provider as rp


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


def test_resolve_runtime_provider_anthropic_pool_respects_config_base_url(monkeypatch):
    class _Entry:
        access_token = "pool-token"
        source = "manual"
        base_url = "https://api.anthropic.com"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "anthropic",
            "base_url": "https://proxy.example.com/anthropic",
        },
    )
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())

    resolved = rp.resolve_runtime_provider(requested="anthropic")

    assert resolved["provider"] == "anthropic"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["api_key"] == "pool-token"
    assert resolved["base_url"] == "https://proxy.example.com/anthropic"


def test_resolve_runtime_provider_anthropic_explicit_override_skips_pool(monkeypatch):
    def _unexpected_pool(provider):
        raise AssertionError(f"load_pool should not be called for {provider}")

    def _unexpected_anthropic_token():
        raise AssertionError("resolve_anthropic_token should not be called")

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "anthropic",
            "base_url": "https://config.example.com/anthropic",
        },
    )
    monkeypatch.setattr(rp, "load_pool", _unexpected_pool)
    monkeypatch.setattr(
        "agent.anthropic_adapter.resolve_anthropic_token",
        _unexpected_anthropic_token,
    )

    resolved = rp.resolve_runtime_provider(
        requested="anthropic",
        explicit_api_key="anthropic-explicit-token",
        explicit_base_url="https://proxy.example.com/anthropic/",
    )

    assert resolved["provider"] == "anthropic"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["api_key"] == "anthropic-explicit-token"
    assert resolved["base_url"] == "https://proxy.example.com/anthropic"
    assert resolved["source"] == "explicit"
    assert resolved.get("credential_pool") is None


def test_resolve_runtime_provider_falls_back_when_pool_empty(monkeypatch):
    class _Pool:
        def has_credentials(self):
            return False

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openai-codex")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())
    monkeypatch.setattr(
        rp,
        "resolve_codex_runtime_credentials",
        lambda: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
            "source": "hermes-auth-store",
            "last_refresh": "2026-02-26T00:00:00Z",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["api_key"] == "codex-token"
    assert resolved.get("credential_pool") is None


def test_resolve_runtime_provider_codex(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: type("P", (), {"has_credentials": lambda self: False})(),
    )
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openai-codex")
    monkeypatch.setattr(
        rp,
        "resolve_codex_runtime_credentials",
        lambda: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
            "source": "codex-auth-json",
            "auth_file": "/tmp/auth.json",
            "codex_home": "/tmp/codex",
            "last_refresh": "2026-02-26T00:00:00Z",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="openai-codex")

    assert resolved["provider"] == "openai-codex"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://chatgpt.com/backend-api/codex"
    assert resolved["api_key"] == "codex-token"
    assert resolved["requested_provider"] == "openai-codex"


def test_resolve_runtime_provider_qwen_oauth(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "qwen-oauth")
    monkeypatch.setattr(
        rp,
        "resolve_qwen_runtime_credentials",
        lambda: {
            "provider": "qwen-oauth",
            "base_url": "https://portal.qwen.ai/v1",
            "api_key": "qwen-token",
            "source": "qwen-cli",
            "expires_at_ms": 1775640710946,
        },
    )

    resolved = rp.resolve_runtime_provider(requested="qwen-oauth")

    assert resolved["provider"] == "qwen-oauth"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://portal.qwen.ai/v1"
    assert resolved["api_key"] == "qwen-token"
    assert resolved["requested_provider"] == "qwen-oauth"


def test_resolve_runtime_provider_uses_qwen_pool_entry(monkeypatch):
    class _Entry:
        access_token = "pool-qwen-token"
        source = "manual:qwen_cli"
        base_url = "https://portal.qwen.ai/v1"

    class _Pool:
        def has_credentials(self):
            return True

        def select(self):
            return _Entry()

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "qwen-oauth")
    monkeypatch.setattr(rp, "load_pool", lambda provider: _Pool())
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "qwen-oauth", "default": "coder-model"})

    resolved = rp.resolve_runtime_provider(requested="qwen-oauth")

    assert resolved["provider"] == "qwen-oauth"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://portal.qwen.ai/v1"
    assert resolved["api_key"] == "pool-qwen-token"
    assert resolved["source"] == "manual:qwen_cli"


def test_resolve_provider_alias_qwen(monkeypatch):
    monkeypatch.setattr(rp.auth_mod, "_load_auth_store", lambda: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert rp.resolve_provider("qwen-portal") == "qwen-oauth"
    assert rp.resolve_provider("qwen-cli") == "qwen-oauth"


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


def test_custom_endpoint_prefers_openai_key(monkeypatch):
    """Custom endpoint should use config api_key over OPENROUTER_API_KEY.

    Updated for #4165: config.yaml is now the source of truth for endpoint URLs,
    OPENAI_BASE_URL env var is no longer consulted.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": "https://api.z.ai/api/coding/paas/v4",
        "api_key": "zai-key",
    })
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert resolved["api_key"] == "zai-key"


def test_custom_endpoint_uses_saved_config_base_url_when_env_missing(monkeypatch):
    """Persisted custom endpoints in config.yaml must still resolve when
    OPENAI_BASE_URL is absent from the current environment."""
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
    assert resolved["api_key"] == "local-key"


def test_custom_endpoint_uses_config_api_key_over_env(monkeypatch):
    """provider: custom with base_url and api_key in config uses them (#1760)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://my-api.example.com/v1",
            "api_key": "config-api-key",
        },
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "https://other.example.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://my-api.example.com/v1"
    assert resolved["api_key"] == "config-api-key"


def test_custom_endpoint_uses_config_api_field_when_no_api_key(monkeypatch):
    """provider: custom with 'api' in config uses it as api_key (#1760)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "https://custom.example.com/v1",
            "api": "config-api-field",
        },
    )
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://custom.example.com/v1"
    assert resolved["api_key"] == "config-api-field"


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
    assert resolved["api_key"] == "openai-key"


def test_bare_custom_custom_base_url_env_overrides_remote_yaml(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "openrouter",
            "base_url": "https://api.openrouter.ai/api/v1",
        },
    )
    monkeypatch.setenv("CUSTOM_BASE_URL", "http://localhost:9999/v1")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["provider"] == "custom"
    assert resolved["base_url"] == "http://localhost:9999/v1"


def test_bare_custom_does_not_trust_non_loopback_when_provider_not_custom(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "openrouter",
            "base_url": "https://remote.example.com/v1",
        },
    )
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["provider"] == "custom"
    assert "openrouter.ai" in resolved["base_url"]
    assert "remote.example.com" not in resolved["base_url"]


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
    assert resolved["source"] == "custom_provider:Local"


def test_named_custom_provider_uses_providers_dict_when_list_missing(monkeypatch):
    """After v11→v12 migration deletes custom_providers, resolution should
    still find entries in the providers dict via get_compatible_custom_providers."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "openai-direct-primary": {
                    "api": "https://api.openai.com/v1",
                    "api_key": "dir-key",
                    "default_model": "gpt-5-mini",
                    "name": "OpenAI Direct (Primary)",
                    "transport": "codex_responses",
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

    resolved = rp.resolve_runtime_provider(requested="openai-direct-primary")

    assert resolved["provider"] == "custom"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://api.openai.com/v1"
    assert resolved["api_key"] == "dir-key"
    assert resolved["requested_provider"] == "openai-direct-primary"
    assert resolved["source"] == "custom_provider:OpenAI Direct (Primary)"
    assert resolved["model"] == "gpt-5-mini"


def test_named_custom_provider_uses_key_env_from_providers_dict(monkeypatch):
    """providers dict entries with key_env should resolve API key from env var."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MYCORP_API_KEY", "env-secret")
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "providers": {
                "mycorp-proxy": {
                    "base_url": "https://proxy.example.com/v1",
                    "default_model": "acme-large",
                    "key_env": "MYCORP_API_KEY",
                    "name": "MyCorp Proxy",
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

    resolved = rp.resolve_runtime_provider(requested="mycorp-proxy")

    assert resolved["provider"] == "custom"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://proxy.example.com/v1"
    assert resolved["api_key"] == "env-secret"
    assert resolved["requested_provider"] == "mycorp-proxy"
    assert resolved["source"] == "custom_provider:MyCorp Proxy"
    assert resolved["model"] == "acme-large"


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
    assert resolved["api_key"] == "env-openai-key"
    assert resolved["requested_provider"] == "custom:local-llm"


def test_named_custom_provider_does_not_shadow_builtin_provider(monkeypatch):
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "custom_providers": [
                {
                    "name": "nous",
                    "base_url": "http://localhost:1234/v1",
                    "api_key": "shadow-key",
                }
            ]
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_nous_runtime_credentials",
        lambda **kwargs: {
            "base_url": "https://inference-api.nousresearch.com/v1",
            "api_key": "nous-runtime-key",
            "source": "portal",
            "expires_at": None,
        },
    )

    resolved = rp.resolve_runtime_provider(requested="nous")

    assert resolved["provider"] == "nous"
    assert resolved["base_url"] == "https://inference-api.nousresearch.com/v1"
    assert resolved["api_key"] == "nous-runtime-key"
    assert resolved["requested_provider"] == "nous"


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


def test_explicit_openrouter_honors_openrouter_base_url_over_pool(monkeypatch):
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
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://mirror.example.com/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mirror-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="openrouter")

    assert resolved["provider"] == "openrouter"
    assert resolved["base_url"] == "https://mirror.example.com/v1"
    assert resolved["api_key"] == "mirror-key"
    assert resolved["source"] == "env/config"
    assert resolved.get("credential_pool") is None


def test_resolve_requested_provider_precedence(monkeypatch):
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "nous")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "openai-codex"})
    assert rp.resolve_requested_provider("openrouter") == "openrouter"
    assert rp.resolve_requested_provider() == "openai-codex"

    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    assert rp.resolve_requested_provider() == "nous"

    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    assert rp.resolve_requested_provider() == "auto"


# ── api_mode config override tests ──────────────────────────────────────


def test_model_config_api_mode(monkeypatch):
    """model.api_mode in config.yaml should override the default chat_completions."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(
        rp, "_get_model_config",
        lambda: {
            "provider": "custom",
            "base_url": "http://127.0.0.1:9208/v1",
            "api_mode": "codex_responses",
        },
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9208/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "http://127.0.0.1:9208/v1"


def test_model_config_api_mode_ignored_when_provider_differs(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "zai")
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {
            "provider": "opencode-go",
            "default": "minimax-m2.5",
            "api_mode": "anthropic_messages",
        },
    )
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda provider: {
            "provider": provider,
            "api_key": "test-key",
            "base_url": "https://api.z.ai/api/paas/v4",
            "source": "env",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="zai")

    assert resolved["provider"] == "zai"
    assert resolved["api_mode"] == "chat_completions"


def test_invalid_api_mode_ignored(monkeypatch):
    """Invalid api_mode values should fall back to chat_completions."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"api_mode": "bogus_mode"})
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9208/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["api_mode"] == "chat_completions"


def test_named_custom_provider_api_mode(monkeypatch):
    """custom_providers entries with api_mode should use it."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "my-server")
    monkeypatch.setattr(
        rp, "_get_named_custom_provider",
        lambda p: {
            "name": "my-server",
            "base_url": "http://localhost:8000/v1",
            "api_key": "sk-test",
            "api_mode": "codex_responses",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="my-server")

    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "http://localhost:8000/v1"


def test_named_custom_provider_without_api_mode_defaults(monkeypatch):
    """custom_providers entries without api_mode should default to chat_completions."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "my-server")
    monkeypatch.setattr(
        rp, "_get_named_custom_provider",
        lambda p: {
            "name": "my-server",
            "base_url": "http://localhost:8000/v1",
            "api_key": "***",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="my-server")

    assert resolved["api_mode"] == "chat_completions"


def test_anthropic_messages_in_valid_api_modes():
    """anthropic_messages should be accepted by _parse_api_mode."""
    assert rp._parse_api_mode("anthropic_messages") == "anthropic_messages"


def test_api_key_provider_anthropic_url_auto_detection(monkeypatch):
    """API-key providers with /anthropic base URL should auto-detect anthropic_messages mode."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic")

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://api.minimax.io/anthropic"


def test_api_key_provider_explicit_api_mode_config(monkeypatch):
    """API-key providers should respect api_mode from model config."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"api_mode": "anthropic_messages"})
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["api_mode"] == "anthropic_messages"


def test_minimax_default_url_uses_anthropic_messages(monkeypatch):
    """MiniMax with default /anthropic URL should auto-detect anthropic_messages mode."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://api.minimax.io/anthropic"


def test_minimax_v1_url_uses_chat_completions(monkeypatch):
    """MiniMax with /v1 base URL should use chat_completions (user override for regions where /anthropic 404s)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1")

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://api.minimax.chat/v1"


def test_minimax_cn_v1_url_uses_chat_completions(monkeypatch):
    """MiniMax-CN with /v1 base URL should use chat_completions (user override)."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax-cn")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("MINIMAX_CN_API_KEY", "test-minimax-cn-key")
    monkeypatch.setenv("MINIMAX_CN_BASE_URL", "https://api.minimaxi.com/v1")

    resolved = rp.resolve_runtime_provider(requested="minimax-cn")

    assert resolved["provider"] == "minimax-cn"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://api.minimaxi.com/v1"


def test_minimax_explicit_api_mode_respected(monkeypatch):
    """Explicit api_mode config should override MiniMax auto-detection."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"api_mode": "chat_completions"})
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="minimax")

    assert resolved["provider"] == "minimax"
    assert resolved["api_mode"] == "chat_completions"


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


def test_minimax_env_base_url_still_wins_over_config(monkeypatch):
    """MINIMAX_BASE_URL env var should take priority over config.yaml model.base_url."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "minimax",
        "base_url": "https://api.minimaxi.com/anthropic",
    })
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://custom.example.com/v1")

    resolved = rp.resolve_runtime_provider(requested="minimax")

    # Env var wins because resolve_api_key_provider_credentials prefers it
    assert resolved["base_url"] == "https://custom.example.com/v1"


def test_minimax_config_base_url_ignored_for_different_provider(monkeypatch):
    """model.base_url should NOT be used when model.provider doesn't match."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "openrouter",
        "base_url": "https://some-other-endpoint.com/v1",
    })
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="minimax")

    # Should use the default, NOT the config base_url from a different provider
    assert resolved["base_url"] == "https://api.minimax.io/anthropic"


def test_alibaba_default_coding_intl_endpoint_uses_chat_completions(monkeypatch):
    """Alibaba default coding-intl /v1 URL should use chat_completions mode."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "alibaba")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="alibaba")

    assert resolved["provider"] == "alibaba"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_alibaba_anthropic_endpoint_override_uses_anthropic_messages(monkeypatch):
    """Alibaba with /apps/anthropic URL override should auto-detect anthropic_messages mode."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "alibaba")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-dashscope-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic")

    resolved = rp.resolve_runtime_provider(requested="alibaba")

    assert resolved["provider"] == "alibaba"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic"


def test_opencode_zen_gpt_defaults_to_responses(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-zen")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"default": "gpt-5.4"})
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-opencode-zen-key")
    monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="opencode-zen")

    assert resolved["provider"] == "opencode-zen"
    assert resolved["api_mode"] == "codex_responses"
    assert resolved["base_url"] == "https://opencode.ai/zen/v1"


def test_opencode_zen_claude_defaults_to_messages(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-zen")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"default": "claude-sonnet-4-6"})
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "test-opencode-zen-key")
    monkeypatch.delenv("OPENCODE_ZEN_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="opencode-zen")

    assert resolved["provider"] == "opencode-zen"
    assert resolved["api_mode"] == "anthropic_messages"
    # Trailing /v1 stripped for anthropic_messages mode — the Anthropic SDK
    # appends its own /v1/messages to the base_url.
    assert resolved["base_url"] == "https://opencode.ai/zen"


def test_opencode_go_minimax_defaults_to_messages(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-go")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"default": "minimax-m2.5"})
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="opencode-go")

    assert resolved["provider"] == "opencode-go"
    assert resolved["api_mode"] == "anthropic_messages"
    # Trailing /v1 stripped — Anthropic SDK appends /v1/messages itself.
    assert resolved["base_url"] == "https://opencode.ai/zen/go"


def test_opencode_go_glm_defaults_to_chat_completions(monkeypatch):
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "opencode-go")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"default": "glm-5"})
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "test-opencode-go-key")
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)

    resolved = rp.resolve_runtime_provider(requested="opencode-go")

    assert resolved["provider"] == "opencode-go"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://opencode.ai/zen/go/v1"


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


def test_named_custom_provider_anthropic_api_mode(monkeypatch):
    """Custom providers should accept api_mode: anthropic_messages."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "my-anthropic-proxy")
    monkeypatch.setattr(
        rp, "_get_named_custom_provider",
        lambda p: {
            "name": "my-anthropic-proxy",
            "base_url": "https://proxy.example.com/anthropic",
            "api_key": "test-key",
            "api_mode": "anthropic_messages",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="my-anthropic-proxy")

    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["base_url"] == "https://proxy.example.com/anthropic"


# ------------------------------------------------------------------
# fix #2562 — resolve_provider("custom") must not remap to "openrouter"
# ------------------------------------------------------------------


def test_resolve_provider_custom_returns_custom():
    """resolve_provider('custom') must return 'custom', not 'openrouter'."""
    from hermes_cli.auth import resolve_provider
    assert resolve_provider("custom") == "custom"


def test_resolve_provider_openrouter_unchanged():
    """resolve_provider('openrouter') must still return 'openrouter'."""
    from hermes_cli.auth import resolve_provider
    assert resolve_provider("openrouter") == "openrouter"


def test_resolve_provider_lmstudio_returns_lmstudio(monkeypatch):
    """resolve_provider('lmstudio') must return 'lmstudio', not 'custom'.

    Regression for the alias-map bug where 'lmstudio' was rewritten to
    'custom' before the PROVIDER_REGISTRY lookup, bypassing the first-class
    LM Studio provider entirely at runtime.
    """
    from hermes_cli.auth import resolve_provider
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert resolve_provider("lmstudio") == "lmstudio"
    assert resolve_provider("lm-studio") == "lmstudio"
    assert resolve_provider("lm_studio") == "lmstudio"


def test_custom_provider_runtime_preserves_provider_name(monkeypatch):
    """resolve_runtime_provider with provider='custom' must return provider='custom'."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "model": {
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
                "api_key": "test-key-123",
            }
        },
    )

    resolved = rp.resolve_runtime_provider(requested="custom")
    assert resolved["provider"] == "custom", (
        f"Expected provider='custom', got provider='{resolved['provider']}'"
    )
    assert resolved["base_url"] == "http://localhost:8080/v1"
    assert resolved["api_key"] == "test-key-123"


def test_custom_provider_no_key_gets_placeholder(monkeypatch):
    """Local server with no API key should get 'no-key-required' placeholder."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: {
            "model": {
                "provider": "custom",
                "base_url": "http://localhost:8080/v1",
            }
        },
    )

    resolved = rp.resolve_runtime_provider(requested="custom")
    assert resolved["provider"] == "custom"
    assert resolved["api_key"] == "no-key-required"
    assert resolved["base_url"] == "http://localhost:8080/v1"


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


def test_auto_detected_codex_auth_failure_falls_through_to_openrouter(monkeypatch):
    """When auto-detect picks Codex but credentials are revoked, fall through to OpenRouter."""
    from hermes_cli.auth import AuthError

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setattr(rp, "load_config", lambda: {})

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openai-codex")
    monkeypatch.setattr(rp, "load_pool", lambda p: type("P", (), {
        "has_credentials": lambda self: False,
    })())
    monkeypatch.setattr(
        rp, "resolve_codex_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(
            AuthError("Codex token refresh failed: session revoked",
                      provider="openai-codex", code="invalid_grant", relogin_required=True)
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="auto")
    assert resolved["provider"] == "openrouter"
    assert resolved["api_key"] == "test-or-key"


def test_explicit_nous_auth_failure_still_raises(monkeypatch):
    """When user explicitly requests Nous and auth fails, the error should propagate."""
    from hermes_cli.auth import AuthError
    import pytest

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setattr(rp, "load_config", lambda: {})

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "nous")
    monkeypatch.setattr(rp, "load_pool", lambda p: type("P", (), {
        "has_credentials": lambda self: False,
    })())
    monkeypatch.setattr(
        rp, "resolve_nous_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(
            AuthError("Refresh session has been revoked",
                      provider="nous", code="invalid_grant", relogin_required=True)
        ),
    )

    # With explicit "nous", should raise — don't silently switch providers
    with pytest.raises(AuthError, match="Refresh session has been revoked"):
        rp.resolve_runtime_provider(requested="nous")


def test_openrouter_provider_not_affected_by_custom_fix(monkeypatch):
    """Fixing custom must not change openrouter behavior."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-or-key")
    monkeypatch.setattr(rp, "load_config", lambda: {})

    resolved = rp.resolve_runtime_provider(requested="openrouter")
    assert resolved["provider"] == "openrouter"


# ------------------------------------------------------------------
# fix #7828 — custom_providers model field must propagate to runtime
# ------------------------------------------------------------------


def test_get_named_custom_provider_includes_model(monkeypatch):
    """_get_named_custom_provider should include the model field from config."""
    monkeypatch.setattr(rp, "load_config", lambda: {
        "custom_providers": [{
            "name": "my-dashscope",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "test-key",
            "api_mode": "chat_completions",
            "model": "qwen3.6-plus",
        }],
    })

    result = rp._get_named_custom_provider("my-dashscope")
    assert result is not None
    assert result["model"] == "qwen3.6-plus"


def test_get_named_custom_provider_excludes_empty_model(monkeypatch):
    """Empty or whitespace-only model field should not appear in result."""
    for model_val in ["", "   ", None]:
        entry = {
            "name": "test-ep",
            "base_url": "https://example.com/v1",
            "api_key": "key",
        }
        if model_val is not None:
            entry["model"] = model_val

        monkeypatch.setattr(rp, "load_config", lambda e=entry: {
            "custom_providers": [e],
        })

        result = rp._get_named_custom_provider("test-ep")
        assert result is not None
        assert "model" not in result, (
            f"model field {model_val!r} should not be included in result"
        )


def test_named_custom_runtime_propagates_model_direct_path(monkeypatch):
    """Model should propagate through the direct (non-pool) resolution path."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "my-server")
    monkeypatch.setattr(
        rp, "_get_named_custom_provider",
        lambda p: {
            "name": "my-server",
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
            "model": "qwen3.6-plus",
        },
    )
    # Ensure pool doesn't intercept
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

    resolved = rp.resolve_runtime_provider(requested="my-server")
    assert resolved["model"] == "qwen3.6-plus"
    assert resolved["provider"] == "custom"


def test_named_custom_runtime_propagates_model_pool_path(monkeypatch):
    """Model should propagate even when credential pool handles credentials."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "my-server")
    monkeypatch.setattr(
        rp, "_get_named_custom_provider",
        lambda p: {
            "name": "my-server",
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
            "model": "qwen3.6-plus",
        },
    )
    # Pool returns a result (intercepting the normal path)
    monkeypatch.setattr(
        rp, "_try_resolve_from_custom_pool",
        lambda *a, **k: {
            "provider": "custom",
            "api_mode": "chat_completions",
            "base_url": "http://localhost:8000/v1",
            "api_key": "pool-key",
            "source": "pool:custom:my-server",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="my-server")
    assert resolved["model"] == "qwen3.6-plus", (
        "model must be injected into pool result"
    )
    assert resolved["api_key"] == "pool-key", "pool credentials should be used"


def test_named_custom_runtime_no_model_when_absent(monkeypatch):
    """When custom_providers entry has no model field, runtime should not either."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "my-server")
    monkeypatch.setattr(
        rp, "_get_named_custom_provider",
        lambda p: {
            "name": "my-server",
            "base_url": "http://localhost:8000/v1",
            "api_key": "test-key",
        },
    )
    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", lambda *a, **k: None)

    resolved = rp.resolve_runtime_provider(requested="my-server")
    assert "model" not in resolved


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
        assert resolved["api_key"] == "oa-secret"

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
        assert resolved["api_key"] == "oa-secret"

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

    def test_ollama_key_sent_to_ollama_subdomain(self, monkeypatch):
        """https://api.ollama.com/v1 — legit subdomain."""
        monkeypatch.setenv("OPENAI_API_KEY", "oa-secret")
        monkeypatch.setenv("OLLAMA_API_KEY", "ol-legit-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "custom")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "https://api.ollama.com/v1"
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

    def test_azure_foundry_openai_style_explicit(self, monkeypatch):
        """OpenAI-style Azure Foundry → chat_completions, keeps base_url as-is."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key-openai")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "https://my-resource.openai.azure.com/openai/v1",
            "chat_completions",
        ))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["provider"] == "azure-foundry"
        assert resolved["api_mode"] == "chat_completions"
        assert resolved["base_url"] == "https://my-resource.openai.azure.com/openai/v1"
        assert resolved["api_key"] == "az-key-openai"

    def test_azure_foundry_anthropic_style_strips_v1_suffix(self, monkeypatch):
        """Anthropic-style Azure Foundry → anthropic_messages, /v1 stripped
        because the Anthropic SDK appends /v1/messages itself."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key-ant")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "https://my-resource.services.ai.azure.com/anthropic/v1",
            "anthropic_messages",
        ))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["provider"] == "azure-foundry"
        assert resolved["api_mode"] == "anthropic_messages"
        # /v1 stripped so SDK can append /v1/messages cleanly
        assert resolved["base_url"] == "https://my-resource.services.ai.azure.com/anthropic"

    def test_azure_foundry_missing_base_url_raises(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.delenv("AZURE_FOUNDRY_BASE_URL", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        with pytest.raises(rp.AuthError, match="base URL"):
            rp.resolve_runtime_provider(requested="azure-foundry")

    def test_azure_foundry_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("AZURE_FOUNDRY_API_KEY", raising=False)
        # `get_env_value` reads from ~/.hermes/.env — mock it to return None
        # so the resolver can't find a key there either.
        import hermes_cli.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "get_env_value", lambda k: None)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._make_cfg(
            "https://my-resource.openai.azure.com/openai/v1"
        ))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        with pytest.raises(rp.AuthError, match="API key"):
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

    def test_gpt4o_stays_on_chat_completions(self, monkeypatch):
        """gpt-4o-pure worked on Bob's endpoint — must not get upgraded."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._make_cfg_with_model("gpt-4o-pure", "chat_completions"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["api_mode"] == "chat_completions"

    def test_anthropic_messages_not_downgraded(self, monkeypatch):
        """Anthropic-style endpoint: keep anthropic_messages even for gpt-5 names."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {
            "provider": "azure-foundry",
            "base_url": "https://my-resource.services.ai.azure.com/anthropic/v1",
            "api_mode": "anthropic_messages",
            "default": "gpt-5.3-codex",  # nonsensical on Anthropic but tests the guard
        })
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="azure-foundry")

        assert resolved["api_mode"] == "anthropic_messages"

    def test_target_model_overrides_stale_default(self, monkeypatch):
        """/model switch: target_model should drive api_mode, not the stale config default."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        # Config still pinned to gpt-4o, but user just ran /model gpt-5.3-codex
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._make_cfg_with_model("gpt-4o-pure", "chat_completions"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(
            requested="azure-foundry",
            target_model="gpt-5.3-codex",
        )

        assert resolved["api_mode"] == "codex_responses"

    def test_target_model_downgrade_path(self, monkeypatch):
        """/model switch gpt-5.3-codex → gpt-4o: api_mode follows new model."""
        monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "az-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "azure-foundry")
        # Config was upgraded to codex_responses for the previous model; user
        # now switches to gpt-4o which speaks chat completions.
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._make_cfg_with_model("gpt-5.3-codex", "codex_responses"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(
            requested="azure-foundry",
            target_model="gpt-4o-pure",
        )

        # codex_responses was persisted; we keep it because gpt-4o can speak
        # both protocols but the explicit persisted mode is the safer signal.
        # (gpt-4o returning None from the inference function means "don't
        # override" — the persisted codex_responses survives.)
        assert resolved["api_mode"] == "codex_responses"

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

    def test_api_key_env_alias_honored(self, monkeypatch):
        """The `api_key_env` alias (used in azure-foundry docs) also works."""
        monkeypatch.delenv("AZURE_ANTHROPIC_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("DOCS_VARIANT_KEY", "from-docs-alias")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._cfg(api_key_env="DOCS_VARIANT_KEY"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "from-docs-alias"

    def test_key_env_beats_fallback_chain(self, monkeypatch):
        """key_env takes priority over AZURE_ANTHROPIC_KEY / ANTHROPIC_API_KEY."""
        monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "should-not-win")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-win-either")
        monkeypatch.setenv("MY_PROVIDER_KEY", "winning-key")
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._cfg(key_env="MY_PROVIDER_KEY"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "winning-key"

    def test_inline_api_key_on_model_cfg(self, monkeypatch):
        """model.api_key (inline value) works for single-config setups."""
        monkeypatch.delenv("AZURE_ANTHROPIC_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config",
                            lambda: self._cfg(api_key="inline-azure-key"))
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "inline-azure-key"

    def test_azure_anthropic_key_still_works_as_fallback(self, monkeypatch):
        """Historical fixed-name env vars still resolve when no hint is set."""
        monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "historical-key")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._cfg())
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        resolved = rp.resolve_runtime_provider(requested="anthropic")

        assert resolved["api_key"] == "historical-key"

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


    def test_no_key_anywhere_raises_helpful_error(self, monkeypatch):
        """When nothing resolves, the error message mentions key_env as an option."""
        monkeypatch.delenv("AZURE_ANTHROPIC_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
        monkeypatch.setattr(rp, "_get_model_config", lambda: self._cfg())
        monkeypatch.setattr(rp, "load_pool", lambda provider: None)

        with pytest.raises(rp.AuthError, match="key_env"):
            rp.resolve_runtime_provider(requested="anthropic")

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
            "agent.anthropic_adapter.resolve_anthropic_token",
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

    def test_camel_case_api_key_env_normalizes_to_key_env(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "vendor",
            "base_url": "https://api.vendor.example.com/v1",
            "apiKeyEnv": "MY_VENDOR_KEY",
        }
        normalized = _normalize_custom_provider_entry(dict(entry), provider_key="vendor")
        assert normalized is not None
        assert normalized.get("key_env") == "MY_VENDOR_KEY"

    def test_key_env_wins_if_both_forms_present(self):
        """If both key_env and api_key_env are set, the canonical key_env wins."""
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "vendor",
            "base_url": "https://api.vendor.example.com/v1",
            "key_env": "CANONICAL",
            "api_key_env": "ALIAS",
        }
        normalized = _normalize_custom_provider_entry(dict(entry), provider_key="vendor")
        assert normalized is not None
        assert normalized.get("key_env") == "CANONICAL"

    def test_valid_fields_set_lists_key_env(self):
        """The _VALID_CUSTOM_PROVIDER_FIELDS documentation set must include
        key_env so the set stays in sync with what the runtime actually reads."""
        from hermes_cli.config import _VALID_CUSTOM_PROVIDER_FIELDS
        assert "key_env" in _VALID_CUSTOM_PROVIDER_FIELDS
# =============================================================================
# Tencent TokenHub — API-key provider runtime resolution
# =============================================================================

class TestTencentTokenhubRuntimeResolution:
    """Verify Tencent TokenHub resolves correctly through the generic
    API-key provider path in resolve_runtime_provider."""

    def test_resolves_with_env_key(self, monkeypatch):
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "tencent-tokenhub")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setenv("TOKENHUB_API_KEY", "test-tokenhub-key")
        monkeypatch.delenv("TOKENHUB_BASE_URL", raising=False)

        resolved = rp.resolve_runtime_provider(requested="tencent-tokenhub")

        assert resolved["provider"] == "tencent-tokenhub"
        assert resolved["api_mode"] == "chat_completions"
        assert resolved["base_url"] == "https://tokenhub.tencentmaas.com/v1"
        assert resolved["api_key"] == "test-tokenhub-key"
        assert resolved["requested_provider"] == "tencent-tokenhub"

    def test_custom_base_url_from_env(self, monkeypatch):
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "tencent-tokenhub")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setenv("TOKENHUB_API_KEY", "test-tokenhub-key")
        monkeypatch.setenv("TOKENHUB_BASE_URL", "https://custom-proxy.example.com/v1")

        resolved = rp.resolve_runtime_provider(requested="tencent-tokenhub")

        assert resolved["provider"] == "tencent-tokenhub"
        assert resolved["base_url"] == "https://custom-proxy.example.com/v1"
        assert resolved["api_key"] == "test-tokenhub-key"

    def test_config_base_url_honoured_when_provider_matches(self, monkeypatch):
        """model.base_url in config.yaml should override the hardcoded default
        when model.provider == tencent-tokenhub."""
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "tencent-tokenhub")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {
            "provider": "tencent-tokenhub",
            "base_url": "https://proxy.internal.com/v1",
        })
        monkeypatch.setenv("TOKENHUB_API_KEY", "test-tokenhub-key")
        monkeypatch.delenv("TOKENHUB_BASE_URL", raising=False)

        resolved = rp.resolve_runtime_provider(requested="tencent-tokenhub")

        assert resolved["base_url"] == "https://proxy.internal.com/v1"

    def test_config_base_url_ignored_for_different_provider(self, monkeypatch):
        """model.base_url should NOT be used when model.provider doesn't match."""
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "tencent-tokenhub")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {
            "provider": "openrouter",
            "base_url": "https://some-other-endpoint.com/v1",
        })
        monkeypatch.setenv("TOKENHUB_API_KEY", "test-tokenhub-key")
        monkeypatch.delenv("TOKENHUB_BASE_URL", raising=False)

        resolved = rp.resolve_runtime_provider(requested="tencent-tokenhub")

        # Should use the default, NOT the config base_url from a different provider
        assert resolved["base_url"] == "https://tokenhub.tencentmaas.com/v1"

    def test_explicit_override_skips_env(self, monkeypatch):
        monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "tencent-tokenhub")
        monkeypatch.setattr(rp, "_get_model_config", lambda: {})
        monkeypatch.setenv("TOKENHUB_API_KEY", "env-key-should-lose")
        monkeypatch.delenv("TOKENHUB_BASE_URL", raising=False)

        resolved = rp.resolve_runtime_provider(
            requested="tencent-tokenhub",
            explicit_api_key="explicit-tokenhub-key",
            explicit_base_url="https://explicit-proxy.example.com/v1/",
        )

        assert resolved["provider"] == "tencent-tokenhub"
        assert resolved["api_key"] == "explicit-tokenhub-key"
        assert resolved["base_url"] == "https://explicit-proxy.example.com/v1"
        assert resolved["source"] == "explicit"

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


def test_minimax_oauth_runtime_uses_inference_base_url(monkeypatch):
    """Base URL returned by resolve_runtime_provider should match the OAuth credentials."""
    from hermes_cli.auth import MINIMAX_OAUTH_CN_INFERENCE

    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "minimax-oauth")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "minimax-oauth"})
    monkeypatch.setattr(rp, "load_pool", lambda provider: None)
    monkeypatch.setattr(rp, "_resolve_named_custom_runtime", lambda **k: None)
    monkeypatch.setattr(rp, "_resolve_explicit_runtime", lambda **k: None)

    fake_creds = {
        "provider": "minimax-oauth",
        "api_key": "cn-token",
        "base_url": MINIMAX_OAUTH_CN_INFERENCE.rstrip("/"),
        "source": "oauth",
    }

    import hermes_cli.auth as auth_mod
    monkeypatch.setattr(auth_mod, "resolve_minimax_oauth_runtime_credentials",
                        lambda **k: fake_creds)

    resolved = rp.resolve_runtime_provider(requested="minimax-oauth")

    assert MINIMAX_OAUTH_CN_INFERENCE.rstrip("/") in resolved["base_url"]
