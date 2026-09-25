"""Tests for acp_adapter.auth — provider detection."""

from acp_adapter.auth import (
    detect_provider,
)

class TestDetectProviderPresence:

    def test_has_provider_false_without_credentials(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "openrouter", "api_key": ""},
        )
        assert detect_provider() is None

class TestDetectProvider:
    def test_detect_openrouter(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda: {"provider": "openrouter", "api_key": "sk-or-test"},
        )
        assert detect_provider() == "openrouter"
