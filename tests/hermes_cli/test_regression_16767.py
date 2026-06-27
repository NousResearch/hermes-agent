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
