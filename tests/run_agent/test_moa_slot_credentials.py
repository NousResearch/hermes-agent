"""Test that _slot_runtime passes slot credentials to resolve_runtime_provider.

Regression for #60064: MoA custom_provider credentials not passed to reference
models (HTTP 401). When a slot contains api_key/base_url/api_mode, those must
be passed as explicit_* args to resolve_runtime_provider so custom_providers
get their configured credentials.
"""


def test_moa_slot_runtime_preserves_slot_credentials(monkeypatch):
    """Slot api_key/base_url/api_mode must be passed through to the resolver.

    Without this fix, a custom_provider slot configured with an inline api_key
    would lose that key and fail with HTTP 401 (Missing Authentication header).
    The fix extracts slot credentials and passes them as explicit_api_key/
    explicit_base_url to resolve_runtime_provider.
    """
    from agent.moa_loop import _slot_runtime

    def fake_resolve_runtime_provider(*, requested, target_model=None, explicit_api_key=None, explicit_base_url=None):
        """Fake resolver that echoes back what it received."""
        return {
            "provider": requested,
            "model": target_model or "default",
            "api_key": explicit_api_key or "no-key",
            "base_url": explicit_base_url or "https://default.api/v1",
            "api_mode": "chat_completions",
            "source": "test-fake",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    # Slot with inline credentials (the issue case)
    rt = _slot_runtime(
        {
            "provider": "custom:doubao",
            "model": "ep-202412",
            "api_key": "***",
            "base_url": "https://doubao.example.com/v1",
        }
    )

    assert rt["provider"] == "custom:doubao"
    assert rt["model"] == "ep-202412"
    assert rt["api_key"] == "***"  # Preserved from slot
    assert rt["base_url"] == "https://doubao.example.com/v1"  # Preserved from slot
    assert rt["api_mode"] == "chat_completions"


def test_moa_slot_runtime_preserves_slot_api_mode(monkeypatch):
    """Slot api_mode (when present) must be forwarded to call_llm.

    Some custom endpoints require a specific api_mode (e.g., anthropic_messages
    for certain providers). If the slot specifies it, it should be preserved
    rather than being overridden by automatic detection.
    """
    from agent.moa_loop import _slot_runtime

    def fake_resolve_runtime_provider(*, requested, target_model=None, explicit_api_key=None, explicit_base_url=None):
        return {
            "provider": requested,
            "model": target_model or "default",
            "api_key": explicit_api_key or "no-key",
            "base_url": explicit_base_url or "https://default.api/v1",
            "api_mode": "chat_completions",  # Default from resolver
            "source": "test-fake",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    # Slot with explicit api_mode
    rt = _slot_runtime(
        {
            "provider": "custom:vendor-x",
            "model": "x-1",
            "api_mode": "anthropic_messages",
        }
    )

    # Slot's api_mode overrides resolver's default
    assert rt["api_mode"] == "anthropic_messages"


def test_moa_slot_runtime_ignores_empty_slot_credentials(monkeypatch):
    """Empty string credentials are treated as "not provided" (None).

    When a slot has api_key="" or base_url="", we should not pass an explicit
    empty string to resolve_runtime_provider — pass None instead so the resolver
    can apply its normal fallback logic (env vars, credential pools, etc.).
    """
    from agent.moa_loop import _slot_runtime

    called_with = {}

    def tracking_resolver(*, requested, target_model=None, explicit_api_key=None, explicit_base_url=None):
        called_with["api_key"] = explicit_api_key
        called_with["base_url"] = explicit_base_url
        return {
            "provider": requested,
            "model": target_model or "default",
            "api_key": explicit_api_key or "pooled-key",
            "base_url": explicit_base_url or "https://pooled.api/v1",
            "api_mode": "chat_completions",
            "source": "test-fake",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        tracking_resolver,
    )

    # Slot with empty credentials
    _slot_runtime(
        {
            "provider": "openai",
            "model": "gpt-5.5",
            "api_key": "",
            "base_url": "",
        }
    )

    # Empty strings are converted to None (not passed as explicit args)
    assert called_with["api_key"] is None
    assert called_with["base_url"] is None