"""Unit tests for the ClinePass provider profile.

ClinePass serves curated open-weight coding models over an OpenAI-compatible
Chat Completions API, so chat and tool calling ride the default ``openai_chat``
transport. Model IDs are namespaced (``cline-pass/<model>``) and pass through to
the endpoint unchanged. These tests pin the profile's identity, endpoint,
aliases, auxiliary model, curated fallback catalog, and the config-driven
resolution path so the first-class wiring stays intact.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def clinepass_profile():
    """Resolve the registered ClinePass profile through the public registry."""
    # Importing ``model_tools`` triggers plugin discovery, which registers the
    # ClinePass profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("clinepass")
    assert profile is not None, "clinepass provider profile must be registered"
    return profile


class TestClinePassProfileMetadata:
    """Identity, endpoint, and aux model are wired as a first-class provider."""

    def test_identity_and_endpoint(self, clinepass_profile):
        assert clinepass_profile.name == "clinepass"
        assert clinepass_profile.base_url == "https://api.cline.bot/api/v1"
        assert clinepass_profile.auth_type == "api_key"
        assert "CLINE_API_KEY" in clinepass_profile.env_vars
        assert clinepass_profile.display_name == "ClinePass"

    def test_aliases_resolve(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("cline-pass").name == "clinepass"
        assert providers.get_provider_profile("cline").name == "clinepass"

    def test_default_aux_model(self, clinepass_profile):
        assert clinepass_profile.default_aux_model == "cline-pass/deepseek-v4-flash"

    def test_consumer_api_returns_aux_model(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("clinepass") == "cline-pass/deepseek-v4-flash"

    def test_fallback_catalog_is_namespaced(self, clinepass_profile):
        models = clinepass_profile.fallback_models
        assert "cline-pass/glm-5.2" in models
        assert "cline-pass/qwen3.7-max" in models
        # Every ClinePass model ID carries the cline-pass/ namespace and is sent
        # to the endpoint unchanged.
        assert all(m.startswith("cline-pass/") for m in models)

    def test_no_reasoning_kwargs_emitted(self, clinepass_profile):
        """The plain profile sends no provider-specific reasoning kwargs."""
        extra_body, top_level = clinepass_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="cline-pass/glm-5.2",
        )
        assert extra_body == {}
        assert top_level == {}


class TestClinePassConfigResolution:
    """Selecting ClinePass via ``config.yaml`` reaches the first-class path.

    ``hermes chat`` with no ``--provider`` flag resolves the provider from
    ``model.provider`` in ``config.yaml`` through the same ``normalize_provider``
    and api-key dispatch the flag uses, so the config-driven flow lands on the
    first-class provider, not a custom-endpoint fallback.
    """

    @pytest.mark.parametrize("configured", ["clinepass", "cline-pass", "cline"])
    def test_config_provider_value_normalizes(self, configured):
        from hermes_cli.models import normalize_provider

        assert normalize_provider(configured) == "clinepass"

    def test_auto_resolves_provider_from_config_yaml(self, monkeypatch):
        # config.yaml sets ``model.provider: clinepass`` and ``hermes chat`` runs
        # without ``--provider`` (requested == "auto"). resolve_provider() must
        # read the config provider and land on the first-class ``clinepass``, not
        # fall through to an env default.
        import model_tools  # noqa: F401
        import hermes_cli.config as config_mod
        from hermes_cli.auth import resolve_provider

        monkeypatch.setattr(
            config_mod,
            "load_config",
            lambda: {"model": {"provider": "clinepass", "default": "cline-pass/glm-5.2"}},
        )
        assert resolve_provider("auto") == "clinepass"

    def test_config_provider_takes_api_key_path(self):
        # The agent treats this provider as a profile-backed api-key provider,
        # which is the branch a config-set ``provider: clinepass`` flows through.
        import model_tools  # noqa: F401
        from hermes_cli.main import _is_profile_api_key_provider

        assert _is_profile_api_key_provider("clinepass") is True


class TestClinePassTransport:
    """ClinePass relies on the shared transport sanitization, not its own.

    Like the other OpenAI-compatible providers, the ClinePass profile carries no
    provider-specific message cleanup (``prepare_messages`` is pass-through); the
    shared ``chat_completions`` transport strips the Hermes-internal carriers
    (``timestamp``, ``tool_name``, ``_``-scaffolding, Gemini ``extra_content``)
    before the request leaves. These tests pin that contract so a future change
    that reintroduces a leak fails here.
    """

    def test_profile_prepare_messages_is_passthrough(self, clinepass_profile):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "ok"},
        ]
        assert clinepass_profile.prepare_messages(messages) == messages

    def test_transport_strips_internal_fields(self):
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
        result = transport.convert_messages(messages, model="cline-pass/glm-5.2")
        sent = result[0]
        # Internal carriers a strict OpenAI-compatible endpoint would reject are gone.
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
