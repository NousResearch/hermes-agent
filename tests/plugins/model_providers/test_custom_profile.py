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
  - providers.<name>.reasoning_format overrides the dialect per endpoint
    (#72649): "reasoning_object" nests extra_body.reasoning={enabled, effort},
    "none" emits no reasoning fields, "top_level"/unset keeps the above.
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


class TestCustomReasoningFormatOverride:
    """``providers.<name>.reasoning_format`` switches the reasoning wire dialect.

    Custom endpoints disagree on how reasoning is spelled on the wire
    (#72649): GLM/ARK read a top-level ``reasoning_effort`` string,
    OpenRouter-style gateways expect a nested ``reasoning`` object, and some
    proxies 400 on either unknown field. The config lookup is monkeypatched
    at its import site so these tests pin only the profile's dispatch; the
    classes above run with no lookup result at all, which is the
    byte-for-byte-unchanged guarantee for existing configs.
    """

    BASE_URL = "http://127.0.0.1:8317/v1"

    def _patch_format(self, monkeypatch, fmt):
        """Route the profile's lazy config lookup to a canned answer."""
        calls: list = []

        def fake_lookup(base_url, custom_providers=None, config=None):
            calls.append(base_url)
            return fmt

        monkeypatch.setattr(
            "hermes_cli.config.get_custom_provider_reasoning_format",
            fake_lookup,
        )
        return calls

    def test_reasoning_object_enabled_nests_in_extra_body(
        self, custom_profile, monkeypatch
    ):
        """reasoning_object + enabled → extra_body.reasoning object, nothing top-level.

        This is the OpenRouter-style dialect. Efforts pass through verbatim —
        gateways that accept the object shape validate effort themselves
        (e.g. accept "ultra" that they reject as top-level reasoning_effort).
        """
        self._patch_format(monkeypatch, "reasoning_object")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "ultra"},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {"reasoning": {"enabled": True, "effort": "ultra"}}
        assert tl == {}

    def test_reasoning_object_disabled_sends_enabled_false(
        self, custom_profile, monkeypatch
    ):
        """reasoning_object + disabled → {"enabled": False}, no think/effort fields."""
        self._patch_format(monkeypatch, "reasoning_object")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {"reasoning": {"enabled": False}}
        assert tl == {}

    def test_reasoning_object_effort_none_alias_disables(
        self, custom_profile, monkeypatch
    ):
        """effort='none' is the disable alias in every dialect."""
        self._patch_format(monkeypatch, "reasoning_object")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {"reasoning": {"enabled": False}}
        assert tl == {}

    def test_format_none_enabled_emits_nothing(self, custom_profile, monkeypatch):
        """reasoning_format='none' suppresses reasoning fields on enable.

        For proxies (LiteLLM-style) that 400 on any unrecognized reasoning
        field — the endpoint's server-side default applies.
        """
        self._patch_format(monkeypatch, "none")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {}
        assert tl == {}

    def test_format_none_disabled_emits_nothing_and_warns(
        self, custom_profile, monkeypatch, caplog
    ):
        """reasoning_format='none' also suppresses the disable fields.

        Fixes the second failure mode in #72649: `/reasoning none` used to
        400 on strict proxies because the disable path still sent
        reasoning_effort + think.
        """
        self._patch_format(monkeypatch, "none")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {}
        assert tl == {}
        assert "suppresses the explicit reasoning disable request" in caplog.text

    def test_explicit_top_level_matches_default_enabled(
        self, custom_profile, monkeypatch
    ):
        """reasoning_format='top_level' is the spelled-out default — identical wire."""
        self._patch_format(monkeypatch, "top_level")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {}
        assert tl == {"reasoning_effort": "high"}

    def test_explicit_top_level_matches_default_disabled(
        self, custom_profile, monkeypatch
    ):
        self._patch_format(monkeypatch, "top_level")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {"think": False}
        assert tl == {"reasoning_effort": "none"}

    def test_no_configured_format_falls_back_to_top_level(
        self, custom_profile, monkeypatch
    ):
        """Lookup returning None (no providers entry / no key) → default dialect."""
        calls = self._patch_format(monkeypatch, None)
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {}
        assert tl == {"reasoning_effort": "high"}
        assert calls == [self.BASE_URL]

    def test_lookup_failure_falls_back_to_top_level(
        self, custom_profile, monkeypatch
    ):
        """A crashing config lookup must never break the request build."""

        def boom(base_url, custom_providers=None, config=None):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(
            "hermes_cli.config.get_custom_provider_reasoning_format", boom
        )
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            base_url=self.BASE_URL,
        )
        assert eb == {}
        assert tl == {"reasoning_effort": "high"}

    def test_reasoning_object_composes_with_num_ctx(
        self, custom_profile, monkeypatch
    ):
        """num_ctx wiring is orthogonal to the reasoning dialect."""
        self._patch_format(monkeypatch, "reasoning_object")
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            ollama_num_ctx=8192,
            model="qwen3",
            base_url=self.BASE_URL,
        )
        assert eb == {
            "options": {"num_ctx": 8192},
            "reasoning": {"enabled": True, "effort": "medium"},
        }
        assert tl == {}

