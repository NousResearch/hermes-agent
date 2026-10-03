"""Unit tests for the Z.AI / GLM provider profile's reasoning wiring.

Z.AI's GLM-4.5-and-later chat models default to thinking-mode ON when the
request omits ``thinking``.  Before the profile emitted the parameter,
``reasoning_config = {"enabled": False}`` was a silent no-op on the direct
Z.AI route — users who turned thinking off kept burning thinking tokens on
every turn (the desktop "thinking reverts to medium" report).

GLM-5.2 additionally exposes a native ``reasoning_effort`` knob with two
enabled levels (high / max) on the OpenAI-compatible ``/api/paas/v4``
endpoint; the Hermes effort scale is collapsed onto those.

These tests pin the profile's wire-shape contract so Z.AI requests stay
correctly shaped without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def zai_profile():
    """Resolve the registered Z.AI profile through the real discovery path."""
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Z.AI profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("zai")
    assert profile is not None, "zai provider profile must be registered"
    return profile


class TestZaiThinkingWireShape:
    """``build_api_kwargs_extras`` produces Z.AI's exact wire format."""

    def test_no_preference_omits_thinking(self, zai_profile):
        """No reasoning_config → omit ``thinking`` so the server default
        applies (matches prior behavior for users with no preference)."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config=None, model="glm-5"
        )
        assert extra_body == {}
        assert top_level == {}

    def test_enabled_sends_enabled_marker(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"}, model="glm-5"
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {}

    def test_explicitly_disabled_sends_disabled_marker(self, zai_profile):
        """``reasoning_config.enabled=False`` → ``thinking.type=disabled``.

        The crucial bit is that the parameter is *sent* at all — GLM defaults
        to thinking-on when ``thinking`` is absent, so an unsent disable
        burns thinking tokens forever.
        """
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model="glm-5"
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}


class TestZaiGLM52ReasoningEffort:
    """GLM-5.2's native ``reasoning_effort`` knob (two enabled levels)."""

    def test_high_maps_to_high(self, zai_profile):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("effort", ["low", "medium", "minimal"])
    def test_lower_efforts_clamp_up_to_high(self, zai_profile, effort):
        """GLM-5.2's minimum thinking level is high — lower Hermes levels
        clamp onto it."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("effort", ["xhigh", "max"])
    def test_strong_efforts_map_to_max(self, zai_profile, effort):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": "max"}

    def test_disabled_sends_no_effort(self, zai_profile):
        """Disabled reasoning still sends the thinking-off marker but never
        an effort level."""
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"},
            model="glm-5.2",
        )
        assert extra_body == {"thinking": {"type": "disabled"}}
        assert top_level == {}

    @pytest.mark.parametrize(
        "model",
        [
            "z-ai/glm-5.2",
            "glm-5-2",
            "glm-5p2",
            "accounts/fireworks/models/glm-5p2",
            "zai-org-glm-5-2",
        ],
    )
    def test_alias_spellings_recognized(self, zai_profile, model):
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "max"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "max"}

    @pytest.mark.parametrize(
        "model",
        ["glm-5.1", "glm-5", "glm-4.7", "glm-4-9b", "", None],
    )
    def test_non_glm_5_2_models_get_no_effort(self, zai_profile, model):
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model=model,
        )
        assert top_level == {}


class TestZaiGLM53ReasoningEffort:
    """GLM-5.3's graded low/medium/high/max effort scale (issue #91789).

    Verified live on api.z.ai/api/coding/paas/v4: all four levels accepted
    with monotonic reasoning-token scaling. Unlike 5.2, low and medium must
    reach the wire instead of clamping up to high.
    """

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("max", "max"),
            ("xhigh", "max"),
            ("minimal", "low"),
        ],
    )
    def test_graded_efforts_pass_through(self, zai_profile, effort, expected):
        extra_body, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.3",
        )
        assert extra_body == {"thinking": {"type": "enabled"}}
        assert top_level == {"reasoning_effort": expected}

    @pytest.mark.parametrize(
        "model",
        ["z-ai/glm-5.3", "glm-5-3", "glm-5p3", "zai-org-glm-5-3"],
    )
    def test_alias_spellings_get_graded_scale(self, zai_profile, model):
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"},
            model=model,
        )
        assert top_level == {"reasoning_effort": "low"}

    def test_glm_5_2_still_clamps_low_to_high(self, zai_profile):
        """The 5.3 widening must not leak into 5.2's two-level wire."""
        _, top_level = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"},
            model="glm-5.2",
        )
        assert top_level == {"reasoning_effort": "high"}


class TestZaiModelGating:
    """GLM 4.5+ get thinking; earlier GLM models are left untouched."""

    @pytest.mark.parametrize(
        "model",
        [
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-flash",
            "glm-4.6",
            "glm-5",
            "glm-5.2",
            "GLM-5",  # case-insensitive
        ],
    )
    def test_thinking_capable_models_emit_thinking(self, zai_profile, model):
        extra_body, _ = zai_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}, model=model
        )
        assert extra_body == {"thinking": {"type": "disabled"}}


class TestZaiFullKwargsIntegration:
    """End-to-end: the transport's full kwargs carry the reasoning wiring."""


    def test_glm_5_2_effort_reaches_top_level(self, zai_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="glm-5.2",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=zai_profile,
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://api.z.ai/api/paas/v4",
            provider_name="zai",
        )
        assert kwargs["reasoning_effort"] == "max"
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}


class TestZaiFetchModels:
    """``fetch_models`` appends universally-free models and forwards base_url.

    Z.AI's ``/v1/models`` omits a handful of Flash models that nonetheless
    accept real API calls.  China API keys are rejected by the international
    endpoint, so the free-model append must survive an empty/None live
    result, and the resolved China ``base_url`` must reach the base class.
    """

    def test_appends_free_models_to_live(self, zai_profile, monkeypatch):
        from providers.base import ProviderProfile
        from plugins.model_providers.zai import ZAI_FREE_MODELS

        def fake_fetch(self, *, api_key=None, base_url=None, timeout=8.0):
            return ["glm-5", "glm-4-9b"]

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch)
        models = zai_profile.fetch_models(api_key="k")
        assert "glm-5" in models
        assert "glm-4v-flash" in models
        assert "glm-4.5-flash" in models
        # Every verified free model is present.
        assert set(ZAI_FREE_MODELS) <= set(models)

    def test_dedup_when_live_already_lists_free_model(self, zai_profile, monkeypatch):
        from providers.base import ProviderProfile

        def fake_fetch(self, *, api_key=None, base_url=None, timeout=8.0):
            return ["glm-5", "glm-4v-flash"]

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch)
        models = zai_profile.fetch_models(api_key="k")
        assert models.count("glm-4v-flash") == 1

    def test_empty_or_none_live_still_returns_free_models(self, zai_profile, monkeypatch):
        from providers.base import ProviderProfile
        from plugins.model_providers.zai import ZAI_FREE_MODELS

        expected = sorted(ZAI_FREE_MODELS)

        def fake_fetch_empty(self, *, api_key=None, base_url=None, timeout=8.0):
            return []

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch_empty)
        assert sorted(zai_profile.fetch_models(api_key="k")) == expected

        def fake_fetch_none(self, *, api_key=None, base_url=None, timeout=8.0):
            return None

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch_none)
        # China keys: live fetch rejects → None; free models still surface.
        assert sorted(zai_profile.fetch_models(api_key="china-key")) == expected

    def test_forwards_base_url_to_super(self, zai_profile, monkeypatch):
        """The resolved China endpoint must reach the base implementation."""
        from providers.base import ProviderProfile

        seen = {}

        def fake_fetch(self, *, api_key=None, base_url=None, timeout=8.0):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            return ["glm-5"]

        monkeypatch.setattr(ProviderProfile, "fetch_models", fake_fetch)
        zai_profile.fetch_models(
            api_key="china-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        assert seen["api_key"] == "china-key"
        assert seen["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
