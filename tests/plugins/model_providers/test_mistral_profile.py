"""Unit tests for the Mistral provider profile.

Mistral is a plain OpenAI-compatible provider: chat and tool calling ride the
default ``openai_chat`` transport. These tests pin the profile's identity,
endpoint, aliases, auxiliary model, and curated fallback catalog so the
first-class wiring stays intact.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def mistral_profile():
    """Resolve the registered Mistral profile through the public registry."""
    # Importing ``model_tools`` triggers plugin discovery, which registers the
    # Mistral profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("mistral")
    assert profile is not None, "mistral provider profile must be registered"
    return profile


class TestMistralProfileMetadata:
    """Identity, endpoint, and aux model are wired as a first-class provider."""

    def test_identity_and_endpoint(self, mistral_profile):
        assert mistral_profile.name == "mistral"
        assert mistral_profile.base_url == "https://api.mistral.ai/v1"
        assert mistral_profile.auth_type == "api_key"
        assert "MISTRAL_API_KEY" in mistral_profile.env_vars
        assert mistral_profile.display_name == "Mistral AI"

    def test_aliases_resolve(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("mistral-ai").name == "mistral"
        assert providers.get_provider_profile("mistralai").name == "mistral"

    def test_default_aux_model(self, mistral_profile):
        assert mistral_profile.default_aux_model == "mistral-small-latest"

    def test_consumer_api_returns_aux_model(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("mistral") == "mistral-small-latest"

    def test_fallback_catalog_is_latest_aliases(self, mistral_profile):
        models = mistral_profile.fallback_models
        assert "mistral-large-latest" in models
        assert "codestral-latest" in models
        # Curated fallbacks use stable -latest aliases so they don't go stale.
        assert all(m.endswith("-latest") for m in models)

    def test_no_reasoning_kwargs_emitted(self, mistral_profile):
        """The plain profile sends no provider-specific reasoning kwargs."""
        extra_body, top_level = mistral_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="mistral-small-latest",
        )
        assert extra_body == {}
        assert top_level == {}


class TestMistralConfigResolution:
    """Selecting Mistral via ``config.yaml`` reaches the first-class path.

    ``hermes chat`` with no ``--provider`` flag resolves the provider from
    ``model.provider`` in ``config.yaml`` through the same ``normalize_provider``
    and api-key dispatch the flag uses, so the config-driven flow lands on the
    first-class provider, not a custom-endpoint fallback.
    """

    @pytest.mark.parametrize("configured", ["mistral", "mistral-ai", "mistralai"])
    def test_config_provider_value_normalizes(self, configured):
        from hermes_cli.models import normalize_provider

        assert normalize_provider(configured) == "mistral"

    def test_auto_resolves_provider_from_config_yaml(self, monkeypatch):
        # The exact flow from the reviewer's example: ``config.yaml`` sets
        # ``model.provider: mistral`` and ``hermes chat`` runs without
        # ``--provider`` (requested == "auto"). resolve_provider() must read the
        # config provider and land on the first-class ``mistral``, not fall
        # through to an OpenRouter/env default.
        import model_tools  # noqa: F401
        import hermes_cli.config as config_mod
        from hermes_cli.auth import resolve_provider

        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda: {"model": {"provider": "mistral", "default": "mistral-large-latest"}},
        )
        assert resolve_provider("auto") == "mistral"

    def test_config_provider_takes_api_key_path(self):
        # The agent treats this provider as a profile-backed api-key provider,
        # which is the branch a config-set ``provider: mistral`` flows through.
        import model_tools  # noqa: F401
        from hermes_cli.main import _is_profile_api_key_provider

        assert _is_profile_api_key_provider("mistral") is True


class TestMistralStrictTransport:
    """Mistral relies on the generic transport sanitization, not its own.

    ``api.mistral.ai`` is a strict OpenAI-compatible endpoint that rejects
    unknown message fields with HTTP 400. The Mistral profile carries no
    provider-specific message cleanup (``prepare_messages`` is pass-through);
    the shared ``chat_completions`` transport strips the Hermes-internal carriers
    (``timestamp``, ``tool_name``, ``_``-scaffolding, Gemini ``extra_content``)
    before the request leaves. These tests pin that contract so a future change
    that reintroduces a leak fails here.
    """

    def test_profile_prepare_messages_is_passthrough(self, mistral_profile):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        assert mistral_profile.prepare_messages(messages) == messages

    def test_transport_strips_fields_mistral_rejects(self):
        from agent.transports import get_transport
        import agent.transports.chat_completions  # noqa: F401

        transport = get_transport("chat_completions")
        messages = [
            {
                "role": "assistant",
                "content": "ok",
                "timestamp": "2026-06-21T00:00:00Z",
                "tool_name": "terminal",
                "_empty_recovery_synthetic": True,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "extra_content": {"google": {"thought_signature": "SIG"}},
                        "function": {"name": "t", "arguments": "{}"},
                    }
                ],
            }
        ]
        result = transport.convert_messages(messages, model="mistral-large-latest")
        sent = result[0]
        # Internal carriers that api.mistral.ai rejects are gone.
        assert "timestamp" not in sent
        assert "tool_name" not in sent
        assert "_empty_recovery_synthetic" not in sent
        assert "extra_content" not in sent["tool_calls"][0]
        # Real chat-completions fields survive untouched.
        assert sent["role"] == "assistant"
        assert sent["content"] == "ok"
        assert sent["tool_calls"][0]["id"] == "call_1"
        assert sent["tool_calls"][0]["function"]["name"] == "t"
        # The caller's list is not mutated (sanitization deep-copies on demand).
        assert "timestamp" in messages[0]
