"""Unit tests for the custom provider profile's reasoning wiring.

``provider=custom`` covers any OpenAI-compatible endpoint the user points
Hermes at — local Ollama, vLLM, llama.cpp, and hosted reasoning APIs like
GLM-5.2 on Volcengine ARK. Before #57601's salvage, ``CustomProfile`` emitted
nothing when reasoning was *enabled*, so a configured ``reasoning_effort``
was silently dropped for every custom endpoint.

These tests pin the wire-shape contract:
    - disabled on Ollama  → extra_body.think = False + reasoning_effort=none
    - disabled elsewhere  → reasoning_effort=none, no think (strict APIs 422)
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
        """enabled=False on an Ollama URL → reasoning_effort='none' + think=False.

        Both fields are required on Ollama: /v1/chat/completions silently
        ignores extra_body.think (only /api/chat honours it — ollama#14820)
        but respects top-level reasoning_effort (#25758). think=False stays
        for proxies and the native /api/chat path.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url="http://127.0.0.1:11434/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → same dual emission on Ollama."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="qwen3",
            base_url="http://localhost:11434/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_disabled_omits_think_on_mistral(self, custom_profile):
        """Strict OpenAI-compat hosts forbid extra ``think`` (HTTP 422)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="mistral-small-latest",
            base_url="https://api.mistral.ai/v1",
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    def test_disabled_omits_think_without_base_url(self, custom_profile):
        """Unknown custom endpoint — do not send the Ollama-only flag."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://127.0.0.1:8080/v1",
            "http://localhost:1234/v1",
            "https://api.groq.com/openai/v1",
        ],
    )
    def test_disabled_omits_think_on_non_ollama_relays(self, custom_profile, base_url):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"},
            model="llama3",
            base_url=base_url,
        )
        assert "think" not in eb
        assert tl == {"reasoning_effort": "none"}

    def test_disabled_sends_think_false_on_ollama_cloud_host(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url="https://ollama.com/v1",
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://myhost:99999/v1",  # out-of-range port: OpenAI client accepts it
            "http://localhost:80a/v1",  # non-integer port
            "http://localhost:11434./v1",  # trailing-dot port
        ],
    )
    def test_malformed_port_does_not_raise(self, custom_profile, base_url):
        """Malformed ports must not raise — urlparse's ``port`` is ValueError-happy.

        The OpenAI client accepts ``http://myhost:99999/v1`` at construction
        (only httpx fails later), so these URLs reach ``build_api_kwargs_extras``
        in production. The heuristic must treat them as non-Ollama rather than
        killing the kwargs build.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3",
            base_url=base_url,
        )
        assert "think" not in eb
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

    @pytest.mark.parametrize(
        "reasoning_config, expected",
        [({"enabled": True, "effort": "high"}, "default"), ({"enabled": False, "effort": "medium"}, "none")],
    )
    def test_groq_host_clamps_effort_to_groq_vocabulary(self, custom_profile, reasoning_config, expected):
        """api.groq.com accepts top-level reasoning_effort only as 'none' / 'default' (#75089).

        Drives the main transport so the clamp is proven where production reads it.
        """
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="qwen/qwen3.6-27b", messages=[{"role": "user", "content": "ping"}], tools=None,
            provider_profile=custom_profile, reasoning_config=reasoning_config,
            base_url="https://api.groq.com/openai/v1", provider_name="custom",
        )
        assert kwargs["reasoning_effort"] == expected
        assert "think" not in kwargs.get("extra_body", {}) and "reasoning" not in kwargs.get("extra_body", {})

    @pytest.mark.parametrize(
        "model, effort, expected",
        [
            ("openai/gpt-oss-120b", "medium", "medium"),
            ("openai/gpt-oss-20b", "high", "high"),
            ("gpt-oss-120b", "low", "low"),
            # Outside gpt-oss's own low/medium/high ladder, clamp_effort finds the
            # nearest weaker supported level rather than passing the raw value through.
            ("openai/gpt-oss-120b", "xhigh", "high"),
        ],
    )
    def test_groq_gpt_oss_keeps_graded_effort(self, custom_profile, model, effort, expected):
        """Groq's GPT-OSS models are the one exception to the blanket none/default clamp

        (#75089's fix was too broad) — they have their own graded low/medium/high knob
        per Groq's own docs and 400 on 'default' itself.
        """
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model=model, messages=[{"role": "user", "content": "ping"}], tools=None,
            provider_profile=custom_profile, reasoning_config={"enabled": True, "effort": effort},
            base_url="https://api.groq.com/openai/v1", provider_name="custom",
        )
        assert kwargs["reasoning_effort"] == expected
        assert "think" not in kwargs.get("extra_body", {}) and "reasoning" not in kwargs.get("extra_body", {})

    def test_groq_gpt_oss_disabled_still_clamps_to_none(self, custom_profile):
        """Disabling reasoning takes the same 'none' path as every other endpoint —

        gpt-oss's own vocabulary only governs the graded-effort branch.
        """
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="openai/gpt-oss-120b",
            base_url="https://api.groq.com/openai/v1",
        )
        assert tl == {"reasoning_effort": "none"}
        assert "think" not in eb


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}


class TestCustomResponsesEffortVocabulary:
    """The Responses transport honours a custom endpoint's configured ``max`` (#114249).

    A custom relay's models are not on the OpenAI per-model ladder; without the
    profile declaration the Responses transport clamped a configured ``max`` to
    ``xhigh`` while chat-completions on the same provider forwarded ``max``.
    """

    def test_named_relay_keeps_configured_max_where_chat_completions_does(self, custom_profile):
        from agent.transports.codex import _resolve_reasoning

        reasoning = {"enabled": True, "effort": "max"}
        _, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config=reasoning, base_url="https://relay.example/v1"
        )
        effort, enabled = _resolve_reasoning(
            "deepseek-flash",
            {"provider": "custom:relay", "base_url": "https://relay.example/v1", "reasoning_config": reasoning},
        )
        assert (effort, enabled) == (top_level["reasoning_effort"], True) == ("max", True)

    def test_custom_entry_at_the_official_openai_origin_keeps_the_per_model_ladder(self):
        """api.openai.com's vocabulary is known per model; the custom declaration must not
        put ``max`` on the wire for a slug OpenAI rejects it on."""
        from agent.transports.codex import _resolve_reasoning

        effort, enabled = _resolve_reasoning(
            "gpt-5.2",
            {
                "provider": "custom:oai",
                "base_url": "https://api.openai.com/v1",
                "reasoning_config": {"enabled": True, "effort": "max"},
            },
        )
        assert (effort, enabled) == ("xhigh", True)
