"""OpenCode Zen must reach its Anthropic-wire models on two specific settings.

opencode.ai serves ``space-bunny-free`` (and the rest of its Anthropic-wire
lineup) on ``/v1/messages`` with ``Authorization: Bearer``. It rejects both of
Hermes' fallbacks:

* ``x-api-key`` -- the relay 401s, so an anthropic_messages client built through
  the third-party branch can never authenticate.
* ``/v1/chat/completions`` -- the model is not served on that wire at all.

So a working ``opencode-zen`` + ``space-bunny-free`` combination needs
``_requires_bearer_auth`` to recognise the host AND the prefix table to route
the model id to ``anthropic_messages``. Either half alone leaves the model
unreachable, which is why both are asserted here.

Measured against ``https://opencode.ai/zen/v1/messages`` with one key and
model, varying only the header: ``Authorization: Bearer`` -> 200,
``x-api-key`` -> 401. ``/v1/chat/completions`` for the same model/key -> 401.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.anthropic_adapter import build_anthropic_client
from agent.anthropic_endpoints import _requires_bearer_auth
from hermes_cli.models import opencode_model_api_mode


class TestOpenCodeZenAnthropicWire:
    """The two tables that decide how the opencode.ai relay is reached."""

    def test_opencode_zen_builds_with_bearer_not_x_api_key(self):
        """The relay 401s on x-api-key, so the client must carry auth_token.

        Asserts on the kwargs the SDK is constructed with -- the exact line
        where ``_requires_bearer_auth`` manifests -- rather than on the helper
        alone, so a regression that skips the auth-style branch is caught too.
        """
        with patch("agent.anthropic_adapter._anthropic_sdk") as mock_sdk:
            build_anthropic_client("zen-key", base_url="https://opencode.ai/zen")

        kwargs = mock_sdk.Anthropic.call_args[1]
        assert kwargs["auth_token"] == "zen-key"
        assert "api_key" not in kwargs

    def test_opencode_host_is_classified_as_bearer(self):
        """Both Zen and Go relays are Bearer-only on their Anthropic routes."""
        assert _requires_bearer_auth("https://opencode.ai/zen") is True
        assert _requires_bearer_auth("https://opencode.ai/zen/go/v1") is True

    def test_space_bunny_routes_to_anthropic_messages(self):
        """Without the prefix, space-bunny-free falls through to chat_completions."""
        assert opencode_model_api_mode("opencode-zen", "space-bunny-free") == "anthropic_messages"
        # Sibling Anthropic-wire models on the same relay stay put.
        assert opencode_model_api_mode("opencode-zen", "claude-opus-4-6") == "anthropic_messages"
        # A model the relay serves on the OpenAI wire is unaffected.
        assert opencode_model_api_mode("opencode-zen", "deepseek-v4-flash") == "chat_completions"
