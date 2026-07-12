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


@pytest.fixture
def custom_mod():
    """Resolve the dynamically-loaded ``custom`` provider module.

    ``plugins/model-providers/custom`` is a hyphenated directory loaded via
    importlib.util.spec_from_file_location, not a real importable package —
    ``import plugins.model_providers.custom`` fails even after discovery has
    run. Pull it out of sys.modules by the profile class's ``__module__``
    instead, which is how the interpreter actually knows it.
    """
    import sys

    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    mod = sys.modules.get(type(profile).__module__)
    assert mod is not None, f"module {type(profile).__module__!r} not found in sys.modules"
    return mod


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
        """enabled=False → extra_body.think = False (Ollama thinking-off flag)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

    def test_effort_none_sends_think_false(self, custom_profile):
        """effort='none' is the disable alias → think=False, no effort."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, model="glm-5.2"
        )
        assert eb == {"think": False}
        assert tl == {}

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

    def test_enabled_without_effort_emits_nothing(self, custom_profile):
        """enabled but no effort → omit; do NOT force a level the user didn't pick."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="glm-5.2"
        )
        assert eb == {}
        assert tl == {}

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

    def test_num_ctx_with_effort(self, custom_profile):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            ollama_num_ctx=8192,
            model="qwen3",
        )
        assert eb == {"options": {"num_ctx": 8192}}
        assert tl == {"reasoning_effort": "high"}


class TestOllamaThinkingCapabilityGate:
    """Ollama backends must probe /api/show and skip think/reasoning_effort
    entirely for models that don't declare "thinking" support (#57601 follow-up).

    Non-Ollama custom backends (no base_url, or a base_url that doesn't look
    like Ollama) are untouched — they keep unconditionally emitting the field,
    matching every test above.
    """

    def test_non_thinking_model_emits_nothing(self, custom_profile, custom_mod, monkeypatch):
        """llama3.3:70b-style capabilities (no "thinking") → omit both fields."""

        monkeypatch.setattr(
            custom_mod, "_ollama_model_capabilities", lambda model, base_url, timeout=3.0: ["completion", "tools"]
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="llama3.3:70b",
            base_url="http://chrysalis:11434/v1",
        )
        assert eb == {}
        assert tl == {}

    def test_thinking_model_still_emits(self, custom_profile, monkeypatch, custom_mod):
        """Model that DOES declare "thinking" → behaves like before (top-level effort)."""

        monkeypatch.setattr(
            custom_mod, "_ollama_model_capabilities", lambda model, base_url, timeout=3.0: ["completion", "tools", "thinking"]
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="qwen3:32b",
            base_url="http://chrysalis:11434/v1",
        )
        assert tl == {"reasoning_effort": "medium"}
        assert eb == {}

    def test_non_thinking_model_disabled_also_emits_nothing(self, custom_profile, monkeypatch, custom_mod):
        """Even the disable path (think=False) is gated — a non-thinking model
        doesn't recognize the think field at all, so sending False also 400s."""

        monkeypatch.setattr(
            custom_mod, "_ollama_model_capabilities", lambda model, base_url, timeout=3.0: ["completion", "tools"]
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="llama3.3:70b",
            base_url="http://chrysalis:11434/v1",
        )
        assert eb == {}
        assert tl == {}

    def test_non_ollama_base_url_bypasses_probe(self, custom_profile, monkeypatch, custom_mod):
        """A non-Ollama custom base_url (e.g. GLM/ARK) never triggers the probe
        and keeps the original unconditional emit behavior."""

        def _fail(*a, **kw):
            raise AssertionError("should not probe non-Ollama base_url")

        monkeypatch.setattr(custom_mod, "_ollama_model_capabilities", _fail)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model="glm-5.2",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        assert tl == {"reasoning_effort": "max"}

    def test_no_base_url_bypasses_probe(self, custom_profile, monkeypatch, custom_mod):
        """No base_url passed at all (matches every pre-existing test above) →
        original unconditional behavior, no probe attempted."""

        def _fail(*a, **kw):
            raise AssertionError("should not probe when base_url is unset")

        monkeypatch.setattr(custom_mod, "_ollama_model_capabilities", _fail)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2"
        )
        assert tl == {"reasoning_effort": "high"}


class TestOllamaCapabilityProbe:
    """Unit tests for the /api/show capability fetch + cache helper itself."""

    def test_fetches_and_parses_capabilities(self, monkeypatch, custom_mod):
        import json as _json

        custom_mod._OLLAMA_CAPS_CACHE.clear()

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return _json.dumps({"capabilities": ["completion", "tools"]}).encode()

        monkeypatch.setattr(custom_mod.urllib.request, "urlopen", lambda req, timeout=3.0: _FakeResp())
        caps = custom_mod._ollama_model_capabilities("llama3.3:70b", "http://chrysalis:11434/v1")
        assert caps == ["completion", "tools"]

    def test_strips_v1_suffix_from_base_url(self, monkeypatch, custom_mod):
        import json as _json

        custom_mod._OLLAMA_CAPS_CACHE.clear()
        seen = {}

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return _json.dumps({"capabilities": []}).encode()

        def _fake_urlopen(req, timeout=3.0):
            seen["url"] = req.full_url
            return _FakeResp()

        monkeypatch.setattr(custom_mod.urllib.request, "urlopen", _fake_urlopen)
        custom_mod._ollama_model_capabilities("m", "http://chrysalis:11434/v1")
        assert seen["url"] == "http://chrysalis:11434/api/show"

    def test_failed_probe_returns_empty_and_is_cached_short_ttl(self, monkeypatch, custom_mod):

        custom_mod._OLLAMA_CAPS_CACHE.clear()

        def _boom(req, timeout=3.0):
            raise OSError("connection refused")

        monkeypatch.setattr(custom_mod.urllib.request, "urlopen", _boom)
        caps = custom_mod._ollama_model_capabilities("m", "http://chrysalis:11434/v1")
        assert caps == []
        # cached negative result present
        assert ("m", "http://chrysalis:11434/v1") in custom_mod._OLLAMA_CAPS_CACHE

    def test_missing_model_or_base_url_short_circuits(self, custom_profile, custom_mod):

        assert custom_mod._ollama_model_capabilities(None, "http://chrysalis:11434/v1") == []
        assert custom_mod._ollama_model_capabilities("m", None) == []


class TestIsOllamaBaseUrl:
    def test_matches_port_11434(self, custom_mod):

        assert custom_mod._is_ollama_base_url("http://chrysalis:11434/v1") is True
        assert custom_mod._is_ollama_base_url("http://localhost:11434") is True

    def test_matches_ollama_in_host(self, custom_mod):

        assert custom_mod._is_ollama_base_url("https://my-ollama-box.local/v1") is True

    def test_does_not_match_other_hosts(self, custom_mod):

        assert custom_mod._is_ollama_base_url("https://ark.cn-beijing.volces.com/api/v3") is False
        assert custom_mod._is_ollama_base_url(None) is False
