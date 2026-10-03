"""``determine_api_mode`` honors user-config ``custom_providers`` transports.

Without this, a config-defined gateway that declares ``transport: anthropic_messages``
is misrouted to ``chat_completions`` because the built-in catalog has no entry for
``custom:<name>``. Upstream then rejects every request with HTTP 500
(``advanced custom channel does not support request path /v1/chat/completions for
model claude-opus-5-5``). Regression locked here: a custom provider's declared
transport wins over the chat-completions fallback.
"""

from __future__ import annotations

import pytest

from hermes_cli.providers import determine_api_mode


class TestCustomProviderTransportLookup:
    """``custom:<name>`` providers must honor their declared ``transport``."""

    @pytest.fixture
    def custom_providers_config(self):
        """Stand in for ``load_config_readonly()`` — only ``providers`` and
        ``custom_providers`` are read by ``determine_api_mode``."""
        return {
            "providers": {},
            "custom_providers": [
                {
                    "provider_key": "mafia",
                    "name": "Mafia",
                    "base_url": "https://api.appintheloop.com/v1",
                    "key_env": "MAFIA_API_KEY",
                    "transport": "anthropic_messages",
                },
                {
                    "provider_key": "openai-relay",
                    "name": "OpenAIRelay",
                    "base_url": "https://relay.example.test/v1",
                    "key_env": "OPENAI_RELAY_KEY",
                    "transport": "openai_chat",
                },
            ],
        }

    @pytest.fixture
    def patched_custom_providers(self, monkeypatch, custom_providers_config):
        """Mock ``hermes_cli.config.load_config_readonly`` so the test runs without
        a real ``config.yaml``."""
        from hermes_cli import config

        monkeypatch.setattr(
            config,
            "load_config_readonly",
            lambda: custom_providers_config,
        )

    def test_anthropic_messages_transport_is_honored(self, patched_custom_providers):
        # The legacy code path returned ``chat_completions`` for ``custom:mafia``
        # because the built-in catalog has no entry for it, and the URL is not in
        # ``host_mandated_api_mode``. The config-declared ``anthropic_messages``
        # transport must win.
        assert (
            determine_api_mode("custom:mafia", "https://api.appintheloop.com/v1")
            == "anthropic_messages"
        )

    def test_openai_chat_transport_is_honored(self, patched_custom_providers):
        # A custom provider that declares ``openai_chat`` keeps chat_completions.
        assert (
            determine_api_mode("custom:openai-relay", "https://relay.example.test/v1")
            == "chat_completions"
        )

    def test_unknown_custom_provider_falls_through_to_chat_completions(
        self, patched_custom_providers
    ):
        # A name that does not appear in the user config still defaults to
        # chat_completions so we don't regress unknown providers.
        assert (
            determine_api_mode("custom:does-not-exist", "https://example.test/v1")
            == "chat_completions"
        )


class TestHostMandatedApiModeBeatsUserConfig:
    """URL-detected protocols keep priority over user-config transport."""

    @pytest.fixture
    def custom_providers_config(self):
        return {
            "providers": {},
            "custom_providers": [
                {
                    "provider_key": "wrong-transport",
                    "name": "WrongTransport",
                    "base_url": "https://example.test/v1",
                    "key_env": "KEY",
                    # Wrong for this host; the URL lane must still win.
                    "transport": "chat_completions",
                },
            ],
        }

    @pytest.fixture
    def patched_custom_providers(self, monkeypatch, custom_providers_config):
        from hermes_cli import config

        monkeypatch.setattr(
            config,
            "load_config_readonly",
            lambda: custom_providers_config,
        )

    def test_official_anthropic_endpoint_routes_to_anthropic_messages(
        self, patched_custom_providers
    ):
        assert (
            determine_api_mode("custom:wrong-transport", "https://api.anthropic.com")
            == "anthropic_messages"
        )


class TestBuiltInCatalogStillWins:
    """The built-in catalog takes precedence over the user-config fallback."""

    def test_builtin_provider_keeps_its_transport(self):
        # No fixture needed: anthropic is in the built-in catalog and resolves
        # before any user-config lookup is attempted.
        assert determine_api_mode("anthropic", "") == "anthropic_messages"