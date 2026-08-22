"""Unit tests for the MindsHub provider profile.

MindsHub (https://mindshub.ai) is a fully OpenAI-compatible inference
gateway addressed via short catalog aliases (``sonnet``, ``kimi``,
``deepseek``, ...). These tests pin:

- registration / identity (name, aliases, base URL, auth)
- the model catalog exposed as ``fallback_models``
- the ``reasoning_effort`` top-level passthrough contract in
  ``build_api_kwargs_extras``
- end-to-end kwargs shape through the shared chat_completions transport
"""

from __future__ import annotations

import pytest


@pytest.fixture
def mindshub_profile():
    """Resolve the registered MindsHub profile.

    Going through ``providers.get_provider_profile`` (rather than importing
    the plugin module directly) keeps the test honest about discovery too.
    """
    import model_tools  # noqa: F401  (triggers plugin discovery on import)
    import providers

    profile = providers.get_provider_profile("mindshub")
    assert profile is not None, "mindshub provider profile must be registered"
    return profile


class TestMindsHubIdentity:
    def test_name(self, mindshub_profile):
        assert mindshub_profile.name == "mindshub"

    def test_alias_resolves(self):
        import providers

        assert providers.get_provider_profile("mindshub-ai").name == "mindshub"

    def test_base_url(self, mindshub_profile):
        assert mindshub_profile.base_url == "https://api.mindshub.ai/v1"

    def test_auth_type(self, mindshub_profile):
        assert mindshub_profile.auth_type == "api_key"

    def test_env_vars(self, mindshub_profile):
        assert mindshub_profile.env_vars == ("MINDSHUB_API_KEY", "MINDSHUB_BASE_URL")

    def test_no_fixed_temperature(self, mindshub_profile):
        assert mindshub_profile.fixed_temperature is None

    def test_default_aux_model_set(self, mindshub_profile):
        assert mindshub_profile.default_aux_model == "haiku"

    def test_signup_url(self, mindshub_profile):
        assert mindshub_profile.signup_url == "https://console.mindshub.ai"


class TestMindsHubCatalog:
    def test_fallback_models_are_catalog_aliases(self, mindshub_profile):
        expected = {
            "sonnet",
            "opus",
            "fable",
            "haiku",
            "gpt",
            "gpt-codex",
            "gpt-mini",
            "gemini",
            "gemini-flash",
            "kimi",
            "deepseek",
            "qwen",
            "glm",
            "grok",
        }
        assert set(mindshub_profile.fallback_models) == expected

    def test_embedding_only_alias_excluded(self, mindshub_profile):
        """embed-small is embeddings-only and must never appear in a chat
        model picker fallback list."""
        assert "embed-small" not in mindshub_profile.fallback_models

    def test_raw_provider_model_ids_not_used(self, mindshub_profile):
        """MindsHub only accepts catalog aliases on chat completions — raw
        upstream model IDs like claude-sonnet-5 404. Guard against ever
        listing one by accident."""
        for model in mindshub_profile.fallback_models:
            assert "claude-" not in model
            assert "gpt-5" not in model


class TestMindsHubAuxModel:
    def test_consumer_api_returns_haiku(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("mindshub") == "haiku"

    def test_consumer_api_returns_non_empty(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("mindshub") != ""


class TestMindsHubReasoningEffortPassthrough:
    """``build_api_kwargs_extras`` forwards ``reasoning_effort`` verbatim.

    MindsHub's own API applies the graceful degrade documented in
    /models#reasoning-effort (clamp or drop server-side, never a hard
    failure), so the profile does no per-model gating — it just forwards
    whatever effort level the caller has.
    """

    def test_no_reasoning_config_emits_nothing(self, mindshub_profile):
        eb, tl = mindshub_profile.build_api_kwargs_extras(reasoning_config=None)
        assert eb == {}
        assert tl == {}

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "max", "xhigh"])
    def test_effort_passes_through_top_level(self, mindshub_profile, effort):
        eb, tl = mindshub_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}
        )
        assert tl == {"reasoning_effort": effort}
        assert eb == {}

    def test_effort_is_lowercased(self, mindshub_profile):
        _, tl = mindshub_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "  HIGH  "}
        )
        assert tl == {"reasoning_effort": "high"}

    def test_explicitly_disabled_emits_nothing(self, mindshub_profile):
        eb, tl = mindshub_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}
        )
        assert eb == {}
        assert tl == {}

    def test_none_effort_omitted(self, mindshub_profile):
        _, tl = mindshub_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}
        )
        assert tl == {}

    def test_empty_effort_omitted(self, mindshub_profile):
        _, tl = mindshub_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""}
        )
        assert tl == {}

    def test_effort_without_explicit_enabled_key_passes_through(self, mindshub_profile):
        """``enabled`` defaults to True when absent."""
        _, tl = mindshub_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "medium"}
        )
        assert tl == {"reasoning_effort": "medium"}


class TestMindsHubFullKwargsIntegration:
    """End-to-end: the transport produces MindsHub's plain Chat Completions
    wire shape — a top-level ``reasoning_effort`` string, no extra_body."""

    def test_full_kwargs_include_reasoning_effort(self, mindshub_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="sonnet",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=mindshub_profile,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://api.mindshub.ai/v1",
            provider_name="mindshub",
        )
        assert kwargs["model"] == "sonnet"
        assert kwargs["reasoning_effort"] == "high"
        assert "extra_body" not in kwargs

    def test_full_kwargs_omit_reasoning_effort_when_disabled(self, mindshub_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="sonnet",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=mindshub_profile,
            reasoning_config={"enabled": False},
            base_url="https://api.mindshub.ai/v1",
            provider_name="mindshub",
        )
        assert "reasoning_effort" not in kwargs

    def test_tools_pass_through_unmodified(self, mindshub_profile):
        """MindsHub's chat/completions is standard OpenAI tool-calling
        shape — tools should pass through the transport untouched."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="kimi",
            messages=[{"role": "user", "content": "read foo.txt"}],
            tools=tools,
            provider_profile=mindshub_profile,
            base_url="https://api.mindshub.ai/v1",
            provider_name="mindshub",
        )
        assert kwargs["tools"] == tools

    def test_image_content_parts_pass_through_unmodified(self, mindshub_profile):
        """Images are accepted on every MindsHub chat model as standard
        OpenAI ``image_url`` content parts — no per-provider conversion."""
        from agent.transports.chat_completions import ChatCompletionsTransport

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBOR..."},
                    },
                ],
            }
        ]
        kwargs = ChatCompletionsTransport().build_kwargs(
            model="sonnet",
            messages=messages,
            tools=None,
            provider_profile=mindshub_profile,
            base_url="https://api.mindshub.ai/v1",
            provider_name="mindshub",
        )
        assert kwargs["messages"] == messages
