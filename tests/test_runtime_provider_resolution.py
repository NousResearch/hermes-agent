from hermes_cli import runtime_provider as rp


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
    """Custom endpoint should use OPENAI_API_KEY, not OPENROUTER_API_KEY.

    Regression test for #560: when base_url is a non-OpenRouter endpoint,
    OPENROUTER_API_KEY was being sent as the auth header instead of OPENAI_API_KEY.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-zai-correct-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-wrong-key-for-zai")

    resolved = rp.resolve_runtime_provider(requested="custom")

    assert resolved["base_url"] == "https://api.z.ai/api/coding/paas/v4"
    assert resolved["api_key"] == "sk-zai-correct-key"


def test_custom_endpoint_auto_provider_prefers_openai_key(monkeypatch):
    """Auto provider with non-OpenRouter base_url should prefer OPENAI_API_KEY.

    Same as #560 but via 'hermes model' flow which sets provider to 'auto'.
    """
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "openrouter")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://my-vllm-server.example.com/v1")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-vllm-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-...leak")

    resolved = rp.resolve_runtime_provider(requested="auto")

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

    assert resolved["provider"] == "openrouter"
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


def test_resolve_runtime_provider_nous_api(monkeypatch):
    """Nous Portal API key provider resolves via the api_key path."""
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "nous-api")
    monkeypatch.setattr(
        rp,
        "resolve_api_key_provider_credentials",
        lambda pid: {
            "provider": "nous-api",
            "api_key": "nous-test-key",
            "base_url": "https://inference-api.nousresearch.com/v1",
            "source": "NOUS_API_KEY",
        },
    )

    resolved = rp.resolve_runtime_provider(requested="nous-api")

    assert resolved["provider"] == "nous-api"
    assert resolved["api_mode"] == "chat_completions"
    assert resolved["base_url"] == "https://inference-api.nousresearch.com/v1"
    assert resolved["api_key"] == "nous-test-key"
    assert resolved["requested_provider"] == "nous-api"


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


def test_resolve_requested_provider_precedence(monkeypatch):
    monkeypatch.setenv("HERMES_INFERENCE_PROVIDER", "nous")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "openai-codex"})
    assert rp.resolve_requested_provider("openrouter") == "openrouter"
    assert rp.resolve_requested_provider() == "openai-codex"

    monkeypatch.setattr(rp, "_get_model_config", lambda: {})
    assert rp.resolve_requested_provider() == "nous"

    monkeypatch.delenv("HERMES_INFERENCE_PROVIDER", raising=False)
    assert rp.resolve_requested_provider() == "auto"
