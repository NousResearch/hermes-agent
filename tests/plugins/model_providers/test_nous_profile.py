"""Unit tests for the Nous Portal profile's reasoning wiring.

The Portal honors ``reasoning: {enabled: false}`` — it is the only wire shape
that does, and ``extra_body.thinking`` is not forwarded upstream. The profile
used to drop a disable for every model, which on a thinking-first route like
``deepseek/deepseek-v4-pro`` (catalog: ``default_effort: high``) meant the
upstream default applied and "thinking off" burned reasoning tokens anyway.

A disable is still dropped for reasoning-mandatory routes, which answer
``reasoning: {enabled: false}`` with HTTP 400, and for models the catalog
can't speak to — an unknown model errs toward the old behavior rather than
risking a 400 on a cold first turn.

These tests pin that contract without going live.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def nous_profile():
    """Resolve the registered Nous profile through the real discovery path."""
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the Nous profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("nous")
    assert profile is not None, "nous provider profile must be registered"
    return profile


@pytest.fixture
def portal_catalog(monkeypatch):
    """Prime the Portal reasoning-capability cache with known entries."""
    import models.metadata.reasoning as models_mod

    monkeypatch.setattr(models_mod, "_nous_reasoning_caps_failed_at", None)
    monkeypatch.setattr(models_mod, "_nous_reasoning_caps_cache", {
        "deepseek/deepseek-v4-pro": {
            "supports_reasoning": True,
            "supported_efforts": ["xhigh", "high"],
            "mandatory": False,
        },
        "arcee-ai/trinity-large-thinking": {
            "supports_reasoning": True,
            "supported_efforts": None,
            "mandatory": True,
        },
        # Catalogued, and it takes no reasoning parameter at all.
        "moonshotai/kimi-k3-instruct": {"supports_reasoning": False},
    })


class TestNousReasoningWireShape:
    """``build_api_kwargs_extras`` produces the Portal's wire format."""

    def test_disable_reaches_optional_reasoning_model(self, nous_profile, portal_catalog):
        """The knob the user set is the knob the Portal receives."""
        extra_body, top_level = nous_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
            model="deepseek/deepseek-v4-pro",
        )
        assert extra_body == {"reasoning": {"enabled": False}}
        assert top_level == {}

    def test_disable_dropped_for_mandatory_reasoning_model(self, nous_profile, portal_catalog):
        """Mandatory routes 400 on a disable — send nothing instead."""
        extra_body, _ = nous_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
            model="arcee-ai/trinity-large-thinking",
        )
        assert "reasoning" not in extra_body

    def test_disable_dropped_for_unknown_model(self, nous_profile, portal_catalog):
        """Unlisted / cold catalog → fail safe, never risk the 400."""
        extra_body, _ = nous_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
            model="private/unlisted-route",
        )
        assert "reasoning" not in extra_body

    def test_disable_dropped_for_non_reasoning_route(self, nous_profile, portal_catalog):
        """A route the catalog says takes no reasoning parameter gets none.

        Hermes' own ``supports_reasoning`` can disagree with the Portal about a
        given route; when it does, the catalog of the service actually serving
        the model wins, and we don't send it a parameter it doesn't accept.
        """
        extra_body, _ = nous_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
            model="moonshotai/kimi-k3-instruct",
        )
        assert "reasoning" not in extra_body

    @pytest.mark.parametrize(
        "model",
        ["deepseek/deepseek-v4-pro", "arcee-ai/trinity-large-thinking", "private/unlisted-route"],
    )
    def test_enabled_config_always_forwarded(self, nous_profile, portal_catalog, model):
        """Mandatory-ness only gates the disable; an enable always ships."""
        extra_body, _ = nous_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
            model=model,
        )
        assert extra_body["reasoning"] == {"enabled": True, "effort": "high"}

    def test_no_config_defaults_to_medium(self, nous_profile, portal_catalog):
        extra_body, _ = nous_profile.build_api_kwargs_extras(
            reasoning_config=None,
            supports_reasoning=True,
            model="deepseek/deepseek-v4-pro",
        )
        assert extra_body["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_nothing_emitted_without_reasoning_support(self, nous_profile, portal_catalog):
        extra_body, top_level = nous_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=False,
            model="deepseek/deepseek-v4-pro",
        )
        assert extra_body == {}
        assert top_level == {}

    def test_caller_config_not_mutated(self, nous_profile, portal_catalog):
        cfg = {"enabled": False}
        nous_profile.build_api_kwargs_extras(
            reasoning_config=cfg,
            supports_reasoning=True,
            model="deepseek/deepseek-v4-pro",
        )
        assert cfg == {"enabled": False}


class TestNousRoutePolicy:
    def test_anthropic_models_default_to_chat_wire(self, nous_profile):
        assert nous_profile.resolve_route_policy("anthropic/claude-opus-4.8") == "chat_completions"

    def test_native_policy_selects_messages_wire(self, nous_profile):
        assert nous_profile.resolve_route_policy(
            "anthropic/claude-opus-4.8", options={"anthropic_wire": "native"}
        ) == "anthropic_messages"

    def test_non_anthropic_models_leave_the_profile_default_in_charge(self, nous_profile):
        assert nous_profile.resolve_route_policy("hermes-4-405b") is None
