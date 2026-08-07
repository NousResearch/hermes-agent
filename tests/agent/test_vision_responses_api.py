"""Regression tests: auxiliary vision supports Responses-API-only backends.

Some OpenAI-compatible vision backends (e.g. opencode.ai's ``/zen/go/v1``
Responses API, and any Responses-only gateway) only deliver image/vision
content when requests go to ``/v1/responses`` with ``input_image`` blocks —
the Chat Completions endpoint strips/ignores image parts and returns empty
content.

Hermes already has the Responses transport built in (``CodexAuxiliaryClient``
wraps ``chat.completions.create()`` so it is translated to ``/v1/responses``
with ``input_image`` via ``_chat_messages_to_responses_input``). The knob that
exposes it to a user is ``auxiliary.vision.api_mode = codex_responses`` (or an
``api_mode`` on a named custom provider). These tests lock that wiring in:

  1. ``resolve_vision_provider_client`` returns a Responses-capable
     (``CodexAuxiliaryClient``) for a named custom provider declared with
     ``api_mode: codex_responses``.
  2. Image content in a chat-style message is lowered to ``input_image`` in the
     Responses request body (so the backend actually receives the pixels).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")
    # Clear the module-level aux client cache so a client built by a prior
    # test (keyed on provider/base_url/api_mode) can't leak into this one.
    import agent.auxiliary_client as _ac
    with _ac._client_cache_lock:
        _ac._client_cache.clear()
    yield


def _write_config(tmp_path, config_dict):
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


class TestVisionResponsesApiMode:
    """vision path should route through the Responses API when configured."""

    def test_named_custom_provider_codex_responses_returns_responses_client(self, tmp_path, monkeypatch):
        """api_mode=codex_responses on a named custom provider → CodexAuxiliaryClient."""
        monkeypatch.setenv("LUNA_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "custom_providers": [
                {
                    "name": "luna-proxy",
                    "base_url": "http://luna.local/v1",
                    "key_env": "LUNA_API_KEY",
                    "api_mode": "codex_responses",
                    "default_model": "opencode-gpt-5.6-luna",
                },
            ],
        })
        from agent.auxiliary_client import resolve_vision_provider_client, CodexAuxiliaryClient

        provider, client, model = resolve_vision_provider_client(
            provider="custom:luna-proxy",
            model="opencode-gpt-5.6-luna",
            async_mode=False,
        )

        assert provider == "luna-proxy"
        assert isinstance(client, CodexAuxiliaryClient), (
            f"expected CodexAuxiliaryClient (Responses), got {type(client).__name__}"
        )

    def test_named_custom_provider_chat_completions_returns_openai_client(self, tmp_path, monkeypatch):
        """Without codex_responses, vision stays on a plain OpenAI chat client."""
        monkeypatch.setenv("LUNA_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "custom_providers": [
                {
                    "name": "luna-proxy",
                    "base_url": "http://luna.local/v1",
                    "key_env": "LUNA_API_KEY",
                    "api_mode": "chat_completions",
                    "default_model": "opencode-gpt-5.6-luna",
                },
            ],
        })
        from agent.auxiliary_client import resolve_vision_provider_client

        provider, client, model = resolve_vision_provider_client(
            provider="custom:luna-proxy",
            model="opencode-gpt-5.6-luna",
            async_mode=False,
        )

        # ``OpenAI`` is exported as a lazy proxy class; compare on the
        # resolved type name instead to keep the assertion honest.
        assert type(client).__name__ == "OpenAI", (
            f"expected an OpenAI chat client, got {type(client).__name__}"
        )

    def test_aux_vision_api_mode_is_a_recognized_config_key(self, tmp_path, monkeypatch):
        """auxiliary.vision.api_mode is a valid schema key (via set_config_value).

        This guards the new DEFAULT_CONFIG leaf directly: if the ``api_mode``
        entry under the ``auxiliary.vision`` block in config_defaults.py is
        deleted, ``_validate_config_key`` reports it unknown and this test
        fails — even though the named-custom-provider route (tested above)
        would keep passing on its own.
        """
        from hermes_cli.config import set_config_value, _validate_config_key

        set_config_value("auxiliary.vision.api_mode", "codex_responses")

        # No unknown-key warning → the leaf is recognized in DEFAULT_CONFIG.
        known, suggestion = _validate_config_key("auxiliary.vision.api_mode")
        assert known is True, (
            f"auxiliary.vision.api_mode should be a recognized key, got "
            f"suggestion={suggestion!r}"
        )
        # The value actually landed in the user's config.yaml.
        config_yaml = (tmp_path / ".hermes" / "config.yaml").read_text()
        assert "api_mode: codex_responses" in config_yaml

    def test_aux_vision_api_mode_routes_vision_resolver_to_responses(self, tmp_path, monkeypatch):
        """auxiliary.vision.api_mode=codex_responses → CodexAuxiliaryClient.

        End-to-end through the real config write path (set_config_value) and
        the auxiliary vision resolver, matching the reviewer's ask: deleting
        the DEFAULT_CONFIG leaf must break this, not just the custom-provider
        route.
        """
        monkeypatch.setenv("LUNA_API_KEY", "sk-test")
        from hermes_cli.config import set_config_value

        # Configure the vision aux block exactly as a user would, then persist
        # api_mode through the real set_config_value path.
        _write_config(tmp_path, {
            "auxiliary": {
                "vision": {
                    "provider": "custom:luna-proxy",
                    "model": "opencode-gpt-5.6-luna",
                    "base_url": "http://luna.local/v1",
                    "api_key": "sk-test",
                },
            },
        })
        set_config_value("auxiliary.vision.api_mode", "codex_responses")

        from agent.auxiliary_client import resolve_vision_provider_client, CodexAuxiliaryClient

        provider, client, model = resolve_vision_provider_client(
            provider="custom:luna-proxy",
            model="opencode-gpt-5.6-luna",
            async_mode=False,
        )

        assert isinstance(client, CodexAuxiliaryClient), (
            f"auxiliary.vision.api_mode=codex_responses should route vision to the "
            f"Responses client, got {type(client).__name__}"
        )


class TestImageInputToResponses:
    """image_url chat content must lower to input_image in the Responses body."""

    def test_image_url_content_lowers_to_input_image(self):
        from agent.codex_responses_adapter import _chat_messages_to_responses_input

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this screenshot?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}},
                ],
            }
        ]

        items = _chat_messages_to_responses_input(messages)

        parts = items[0]["content"]
        image_parts = [p for p in parts if p.get("type") == "input_image"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"] == "data:image/jpeg;base64,AAAA"
