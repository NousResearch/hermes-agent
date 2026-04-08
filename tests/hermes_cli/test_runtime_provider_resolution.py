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


def test_opencode_go_configured_api_mode_still_overrides_default(monkeypatch):
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
    assert resolved["api_mode"] == "chat_completions"


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
