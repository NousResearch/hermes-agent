"""Regression coverage for issue #16767."""

import sys

from hermes_cli.runtime_provider import _resolve_named_custom_runtime


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
