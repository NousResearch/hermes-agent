import sys

import hermes_cli.model_switch as ms
from hermes_cli.model_switch import DirectAlias
from hermes_cli.runtime_provider import _resolve_named_custom_runtime

def test_ensure_direct_aliases_mutates_in_place(monkeypatch):
    """_ensure_direct_aliases mutates DIRECT_ALIASES in place (guards against rebinding regression)."""
    # Ensure we start with an empty but existing dict to check for mutation vs rebinding
    ms.DIRECT_ALIASES.clear()
    initial_id = id(ms.DIRECT_ALIASES)
    
    mock_data = {
        "my-custom-alias": DirectAlias("custom-model:v1", "custom", "https://example.com/v1")
    }
    monkeypatch.setattr(ms, "_load_direct_aliases", lambda: mock_data)
    
    ms._ensure_direct_aliases()
    
    assert id(ms.DIRECT_ALIASES) == initial_id, f"DIRECT_ALIASES was rebound (ID changed from {initial_id} to {id(ms.DIRECT_ALIASES)})"
    assert "my-custom-alias" in ms.DIRECT_ALIASES
    assert ms.DIRECT_ALIASES["my-custom-alias"].model == "custom-model:v1"

def test_chat_provider_argparse_acceptance(monkeypatch):
    """chat --provider <user-defined> is accepted by argparse (guards against restrictive choices)."""
    recorded: dict[str, str] = {}

    # Mock cmd_chat to record the provider passed to it
    def mock_cmd_chat(args):
        recorded["provider"] = args.provider

    monkeypatch.setattr("hermes_cli.main.cmd_chat", mock_cmd_chat)
    monkeypatch.setattr(sys, "argv", ["hermes", "chat", "--provider", "my-custom-key"])

    from hermes_cli.main import main
    main()

    assert recorded["provider"] == "my-custom-key"

def test_resolve_named_custom_runtime_honors_explicit_base_url(monkeypatch):
    """_resolve_named_custom_runtime honors (provider='custom', explicit_base_url=...)."""
    # Mock has_usable_secret to recognize our test key
    monkeypatch.setattr("hermes_cli.runtime_provider.has_usable_secret", lambda x: x == "test-api-key")
    
    result = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_api_key="test-api-key",
        explicit_base_url="http://example.test:1234/v1"
    )
    
    assert result is not None
    assert result["base_url"] == "http://example.test:1234/v1"
    assert result["provider"] == "custom"
    assert result["api_key"] == "test-api-key"
    assert result["source"] == "direct-alias"


# --- Regression tests for #43586 ---
# The direct-alias branch (provider=custom + explicit_base_url) silently ignored
# key_env and api_key from the model: config block, falling through to
# "no-key-required".  The named-provider branch honored them.  These tests
# verify parity between the two paths.

def test_direct_alias_honors_key_env_from_model_config(monkeypatch):
    """model: block with provider: custom + key_env must resolve the env var.

    Reproducer for #43586: bare `provider: custom` with `key_env: MY_TOKEN`
    in the model: block sent "no-key-required" instead of the env var value.
    """
    import hermes_cli.runtime_provider as rp

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MY_GATEWAY_TOKEN", "env-secret-123")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": "https://my-gateway.example.com/v1",
        "key_env": "MY_GATEWAY_TOKEN",
    })
    monkeypatch.setattr(rp, "has_usable_secret", lambda x: bool(x and x != "no-key-required"))

    result = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_base_url="https://my-gateway.example.com/v1",
    )

    assert result is not None
    assert result["api_key"] == "env-secret-123"
    assert result["source"] == "direct-alias"


def test_direct_alias_honors_inline_api_key_from_model_config(monkeypatch):
    """model: block with provider: custom + api_key must use the inline value.

    Verifies that the direct-alias branch reads api_key from model config,
    matching the named-provider branch behavior.
    """
    import hermes_cli.runtime_provider as rp

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": "https://my-gateway.example.com/v1",
        "api_key": "inline-key-456",
    })
    monkeypatch.setattr(rp, "has_usable_secret", lambda x: bool(x and x != "no-key-required"))

    result = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_base_url="https://my-gateway.example.com/v1",
    )

    assert result is not None
    assert result["api_key"] == "inline-key-456"


def test_direct_alias_key_env_falls_through_when_env_var_unset(monkeypatch):
    """key_env in model config that doesn't match any env var falls through.

    When key_env is set but the env var doesn't exist, the next candidate
    (e.g. _host_derived_api_key or no-key-required) should be used.
    """
    import hermes_cli.runtime_provider as rp

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": "https://my-gateway.example.com/v1",
        "key_env": "NONEXISTENT_KEY",
    })
    monkeypatch.setattr(rp, "has_usable_secret", lambda x: bool(x and x != "no-key-required"))
    monkeypatch.setattr(rp, "_host_derived_api_key", lambda url: "host-derived-key")

    result = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_base_url="https://my-gateway.example.com/v1",
    )

    assert result is not None
    # Should fall through to host-derived key, not get stuck on empty string
    assert result["api_key"] == "host-derived-key"


def test_direct_alias_explicit_api_key_takes_precedence_over_key_env(monkeypatch):
    """explicit_api_key parameter takes precedence over model config key_env.

    When the caller passes explicit_api_key (e.g. from credential pool),
    it should win over model config's key_env.
    """
    import hermes_cli.runtime_provider as rp

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("MY_TOKEN", "env-value")
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "custom",
        "base_url": "https://my-gateway.example.com/v1",
        "key_env": "MY_TOKEN",
    })
    monkeypatch.setattr(rp, "has_usable_secret", lambda x: bool(x and x != "no-key-required"))

    result = _resolve_named_custom_runtime(
        requested_provider="custom",
        explicit_api_key="pool-key-789",
        explicit_base_url="https://my-gateway.example.com/v1",
    )

    assert result is not None
    # explicit_api_key (first candidate) wins over key_env (second candidate)
    assert result["api_key"] == "pool-key-789"
