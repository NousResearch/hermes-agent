"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
  - disabled            → extra_body.think = False
  - enabled + effort    → top-level reasoning_effort (native OpenAI-compat
                          format GLM/ARK expect), passed through verbatim
                          including ``max``/``xhigh``
  - enabled + no effort → nothing emitted (endpoint's server default applies)
  - ollama_num_ctx      → extra_body.options.num_ctx, orthogonal to reasoning
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered custom profile via the global registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    ``custom`` profile. Going through ``get_provider_profile`` keeps the test
    honest — if the registered class is ever downgraded to a plain
    ``ProviderProfile``, the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomReasoningWireShape:
    """``build_api_kwargs_extras`` produces the correct wire format."""

    def test_no_reasoning_config_emits_nothing(self, custom_profile):
        """Unset reasoning → omit everything so the endpoint's default applies."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

    def test_disabled_sends_think_false(self, custom_profile):
        """enabled=False → reasoning_effort='none' top-level + think=False.

        Both fields are required: Ollama's /v1/chat/completions silently
        ignores extra_body.think (only /api/chat honours it — ollama#14820)
        but respects top-level reasoning_effort (#25758). think=False stays
        for proxies and the native /api/chat path.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → same dual emission."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "effort", ["minimal", "low", "medium", "high", "xhigh", "max"]
    )
    def test_enabled_effort_goes_top_level(self, custom_profile, effort):
        """enabled + effort → TOP-LEVEL reasoning_effort, passed through verbatim.

        GLM-5.2/ARK and OpenAI-compatible reasoning APIs read reasoning_effort
        as a top-level string, not nested in extra_body. ``max`` is GLM's
        native deep-reasoning level and must survive.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": effort}
        assert "reasoning_effort" not in eb
        assert "think" not in eb


    def test_does_not_force_think_true_on_enable(self, custom_profile):
        """We must never send think=True on enable — it's Ollama-only and
        would 400 on GLM/vLLM endpoints that don't recognize it."""
        eb, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert eb.get("think") is not True


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}


class TestQwen38EffortMapping:
    """qwen3.8's renderer accepts low/medium/xhigh; clamp never escalates."""

    def test_high_clamps_to_medium_for_qwen38(self, custom_profile):
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="qwen3.8:latest",
        )
        assert tl == {"reasoning_effort": "medium"}

    def test_max_clamps_to_xhigh_for_qwen38(self, custom_profile):
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="qwen3.8",
        )
        assert tl == {"reasoning_effort": "xhigh"}

    def test_keeps_medium_for_qwen38(self, custom_profile):
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="qwen3.8",
        )
        assert tl == {"reasoning_effort": "medium"}

    def test_does_not_remap_qwen3_colon_8b(self, custom_profile):
        _, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="qwen3:8b",
        )
        assert tl == {"reasoning_effort": "high"}


class TestOllamaUserQueryInjection:
    """Ollama qwen3.8 500s when the payload has no plain user query (#17778)."""

    def test_injects_user_after_system_in_tool_loop(self, custom_profile):
        original = [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "results"},
        ]
        prepared = custom_profile.prepare_messages(original)
        assert original[1]["role"] == "assistant"
        assert prepared[0] == original[0]
        assert prepared[1] == {
            "role": "user",
            "content": "Continue with the current task using the latest tool results.",
        }
        assert prepared[2:] == original[1:]

    def test_leaves_real_user_message_alone(self, custom_profile):
        original = [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": "Search for local qwen3.8"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "results"},
        ]
        prepared = custom_profile.prepare_messages(original)
        assert prepared is original

    def test_treats_tool_response_wrapper_as_not_a_query(self, custom_profile):
        original = [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "user",
                "content": "<tool_response>compressed tool output</tool_response>",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "results"},
        ]
        prepared = custom_profile.prepare_messages(original)
        assert prepared[1]["role"] == "user"
        assert prepared[1]["content"].startswith("Continue with the current task")
        assert prepared[2]["content"] == original[1]["content"]

    def test_does_not_inject_without_a_tool_loop(self, custom_profile):
        original = [{"role": "system", "content": "You are Hermes."}]
        prepared = custom_profile.prepare_messages(original)
        assert prepared is original

