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


class TestCustomReasoningWithNumCtx:
    """Ollama num_ctx and reasoning are independent and compose."""

    def test_num_ctx_alone(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config=None, ollama_num_ctx=8192, model="qwen3"
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {}


@pytest.fixture
def clear_thinking_probe_cache(custom_profile):
    """Isolate the probe cache (held on the registry-singleton profile instance)
    so tests never see each other's probes."""
    cache = custom_profile._THINKING_PROBE_CACHE
    cache.clear()
    yield cache
    cache.clear()


class TestThinkingCapabilityProbe:
    """Ollama /v1 400s ("does not support thinking") when ``reasoning_effort`` targets
    a model without the ``thinking`` capability (#granite4). Before the probe, the
    profile forwarded any configured effort verbatim, so a global
    ``agent.reasoning_effort: high`` broke every non-thinking local model.
    ``ollama_model_supports_thinking`` (native /api/show capabilities) already existed
    for Ollama Cloud — custom now reuses it for Ollama-shaped localhost endpoints."""

    @pytest.mark.parametrize(
        "base_url",
        ["http://localhost:11434/v1", "http://127.0.0.1:11434/v1", "https://ollama.com/v1"],
    )
    def test_non_thinking_model_omits_effort(self, custom_profile, clear_thinking_probe_cache,
                                             monkeypatch, base_url):
        """Probed OK without ``thinking`` → effort omitted; request cannot 400."""
        import hermes_cli.models_local as ml

        probes = []
        monkeypatch.setattr(
            ml, "ollama_model_supports_thinking",
            lambda model, base_url_, api_key=None, timeout=5.0: probes.append(model) or False)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="granite4:3b", base_url=base_url)
        assert probes == ["granite4:3b"]
        assert eb == {}
        assert tl == {}

    def test_thinking_model_sends_effort(self, custom_profile, clear_thinking_probe_cache,
                                         monkeypatch):
        """Probed OK with ``thinking`` → effort forwarded verbatim (qwen3, deepseek-r1…)."""
        import hermes_cli.models_local as ml

        monkeypatch.setattr(
            ml, "ollama_model_supports_thinking",
            lambda model, base_url_, api_key=None, timeout=5.0: True)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="qwen3:8b", base_url="http://localhost:11434/v1")
        assert eb == {}
        assert tl == {"reasoning_effort": "high"}

    def test_probe_failure_fails_open(self, custom_profile, clear_thinking_probe_cache,
                                      monkeypatch):
        """Unreachable /api/show (server down, ollama removed) → send the effort.
        A transient probe failure must not silently drop the user's reasoning config;
        the worst case is the pre-probe behaviour (a 400 from the endpoint itself)."""
        import hermes_cli.models_local as ml

        monkeypatch.setattr(
            ml, "ollama_model_supports_thinking",
            lambda model, base_url_, api_key=None, timeout=5.0: None)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="granite4:3b", base_url="http://localhost:11434/v1")
        assert tl == {"reasoning_effort": "high"}

        # ...and a raising probe is equally fail-open, not fatal to kwargs building.
        def _boom(model, base_url_, api_key=None, timeout=5.0):
            raise OSError("connection refused")

        clear_thinking_probe_cache.clear()
        monkeypatch.setattr(ml, "ollama_model_supports_thinking", _boom)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="granite4:3b", base_url="http://localhost:11434/v1")
        assert tl == {"reasoning_effort": "high"}

    def test_definitive_probe_cached(self, custom_profile, clear_thinking_probe_cache,
                                     monkeypatch):
        """A definitive True/False is cached for the process lifetime — one /api/show
        per (model, base_url), not one per request build."""
        import hermes_cli.models_local as ml

        calls = []
        monkeypatch.setattr(
            ml, "ollama_model_supports_thinking",
            lambda model, base_url_, api_key=None, timeout=5.0: calls.append(model) or False)
        rc = {"enabled": True, "effort": "high"}
        for _ in range(3):
            custom_profile.build_api_kwargs_extras(
                reasoning_config=rc, model="granite4:3b", base_url="http://localhost:11434/v1")
        assert calls == ["granite4:3b"]

        # A different model is a different cache key → probed separately.
        custom_profile.build_api_kwargs_extras(
            reasoning_config=rc, model="ministral-3:8b", base_url="http://localhost:11434/v1")
        assert calls == ["granite4:3b", "ministral-3:8b"]

    @pytest.mark.parametrize(
        "base_url",
        ["https://ark.cn-beijing.volces.com/api/v3", "https://api.groq.com/openai/v1"],
    )
    def test_probe_skipped_on_non_ollama_endpoints(self, custom_profile,
                                                   clear_thinking_probe_cache, monkeypatch,
                                                   base_url):
        """GLM/ARK, Groq, llama.cpp, vLLM… keep the verbatim passthrough — no probe,
        no latency, no behaviour change (#57601 contract)."""
        import hermes_cli.models_local as ml

        probes = []
        monkeypatch.setattr(
            ml, "ollama_model_supports_thinking",
            lambda model, base_url_, api_key=None, timeout=5.0: probes.append(model) or False)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2", base_url=base_url)
        assert probes == []
        assert tl == {"reasoning_effort": "high"}

    def test_disabled_still_disables_on_thinking_model(self, custom_profile,
                                                       clear_thinking_probe_cache,
                                                       monkeypatch):
        """The explicit disable path (reasoning_effort="none" + think=False) is probed
        never and always honoured — #14820/#25758 contract unchanged."""
        import hermes_cli.models_local as ml

        probes = []
        monkeypatch.setattr(
            ml, "ollama_model_supports_thinking",
            lambda model, base_url_, api_key=None, timeout=5.0: probes.append(model) or True)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="qwen3:8b", base_url="http://localhost:11434/v1")
        assert probes == []
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

