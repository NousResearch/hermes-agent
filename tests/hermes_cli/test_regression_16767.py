import pytest
import sys
from unittest.mock import patch
from pathlib import Path

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


# ---------------------------------------------------------------------------
# `hermes chat -m <direct-alias>` resolves to the underlying runtime
# Covers #16767 (chat -m <alias>) and #18954 (custom-provider alias resolution)
# ---------------------------------------------------------------------------


class TestChatDispatchDirectAliasResolution:
    """Before this fix, `hermes chat -m <alias-listed-in-DIRECT_ALIASES>`
    handed the raw alias to AIAgent and the agent emitted "unknown
    provider/model" because the alias was never translated to its
    underlying (provider, base_url, model_id, api_mode) triple."""

    def test_alias_resolution_swaps_in_runtime(self, monkeypatch):
        """Resolving an alias must update provider, model, base_url, and
        api_mode in the runtime dict — without mutating the caller's dict."""
        from cli import _resolve_direct_alias_for_chat, _apply_resolved_alias

        # Stub out resolve_alias to return a known direct hit
        import hermes_cli.model_switch as ms

        monkeypatch.setitem(
            ms.DIRECT_ALIASES,
            "fake-alias",
            ms.DirectAlias(
                model="vendor/full-model-id:v2",
                provider="custom",
                base_url="http://192.0.2.10:8080/v1",
            ),
        )
        monkeypatch.setattr(
            ms,
            "resolve_alias",
            lambda raw, prov: (
                "custom",
                "vendor/full-model-id:v2",
                "fake-alias",
            ),
        )

        resolved = _resolve_direct_alias_for_chat("fake-alias", "")
        assert resolved == {
            "provider": "custom",
            "model": "vendor/full-model-id:v2",
            "base_url": "http://192.0.2.10:8080/v1",
        }

        original_runtime = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": None,
            "api_mode": "responses",
        }
        effective_model, new_runtime = _apply_resolved_alias(
            resolved, original_runtime
        )

        # Caller's dict must not be mutated:
        assert original_runtime["provider"] == "openrouter"
        assert original_runtime["base_url"] == "https://openrouter.ai/api/v1"

        # Resolved runtime points at the alias target:
        assert effective_model == "vendor/full-model-id:v2"
        assert new_runtime["provider"] == "custom"
        assert new_runtime["base_url"] == "http://192.0.2.10:8080/v1"
        # Local/local-style targets get a placeholder key so the OpenAI SDK
        # doesn't refuse to send the request:
        assert new_runtime["api_key"] == "no-key-required"
        # api_mode is recomputed from the resolved provider/base_url:
        assert new_runtime["api_mode"]

    def test_unknown_alias_returns_none(self, monkeypatch):
        """Non-alias inputs must fall through to the generic dispatch path."""
        from cli import _resolve_direct_alias_for_chat
        import hermes_cli.model_switch as ms

        monkeypatch.setattr(ms, "resolve_alias", lambda raw, prov: None)
        assert _resolve_direct_alias_for_chat("not-an-alias", "") is None

    def test_existing_api_key_is_preserved(self, monkeypatch):
        """If the caller already passed an api_key (e.g. custom provider
        with credentials), the resolver must not clobber it with the
        ``no-key-required`` placeholder."""
        from cli import _apply_resolved_alias

        resolved = {
            "provider": "custom",
            "model": "vendor/x:v1",
            "base_url": "http://192.0.2.10:8080/v1",
        }
        original = {
            "provider": "custom",
            "base_url": "http://old/",
            "api_key": "real-key",
            "api_mode": "",
        }
        _, new_runtime = _apply_resolved_alias(resolved, original)
        assert new_runtime["api_key"] == "real-key"

    def test_alias_without_base_url_does_not_set_one(self, monkeypatch):
        """Catalog-only aliases (no DirectAlias.base_url) must not blank
        out an existing base_url."""
        from cli import _apply_resolved_alias

        resolved = {
            "provider": "openrouter",
            "model": "vendor/full-id",
            "base_url": "",
        }
        original = {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-...",
            "api_mode": "responses",
        }
        _, new_runtime = _apply_resolved_alias(resolved, original)
        assert new_runtime["base_url"] == "https://openrouter.ai/api/v1"

    def test_missing_model_switch_module_returns_none(self, monkeypatch):
        """Trimmed embedded installs may omit hermes_cli.model_switch;
        the resolver must degrade gracefully so chat dispatch falls
        through to the generic provider/base_url path."""
        from cli import _resolve_direct_alias_for_chat
        import importlib
        import sys

        # Pretend the module isn't importable — the resolver catches
        # ImportError specifically (not all exceptions).
        sys.modules.pop("hermes_cli.model_switch", None)
        real_import = importlib.__import__

        def fake_import(name, *args, **kwargs):
            if name == "hermes_cli.model_switch":
                raise ImportError("trimmed install")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        assert _resolve_direct_alias_for_chat("any-alias", "") is None
