"""Unit tests for the Upstage Solar provider profile.

Upstage Solar is a plain OpenAI-compatible api-key provider, so this verifies
the profile is registered correctly and wires the expected identity, endpoint,
auth, and catalog fields — the contract every downstream layer (auth, models,
doctor, runtime_provider, transport) reads from.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def upstage_profile():
    """Resolve the registered Upstage profile via the provider registry.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Upstage profile. Going through ``get_provider_profile`` keeps the test
    honest about the actual registration path (name + alias resolution).
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("upstage")
    assert profile is not None, "upstage provider profile must be registered"
    return profile


class TestUpstageProfile:
    def test_identity_and_endpoint(self, upstage_profile):
        assert upstage_profile.name == "upstage"
        assert upstage_profile.api_mode == "chat_completions"
        assert upstage_profile.auth_type == "api_key"
        assert upstage_profile.base_url == "https://api.upstage.ai/v1"
        assert upstage_profile.get_hostname() == "api.upstage.ai"

    def test_solar_alias_resolves(self):
        import model_tools  # noqa: F401
        import providers

        assert providers.get_provider_profile("solar") is upstage_profile_singleton()

    def test_env_vars(self, upstage_profile):
        # API key first, optional base-url override second (priority order).
        assert upstage_profile.env_vars == ("UPSTAGE_API_KEY", "UPSTAGE_BASE_URL")

    def test_fallback_models_are_agentic_pro_only(self, upstage_profile):
        # Only the agentic, tool-calling Solar Pro models belong in the offline
        # catalog — Mini is capable but not agentic, so it's never promoted as a
        # default. Live /v1/models still surfaces everything when a key is set.
        # Behavior contract (not a frozen list): non-empty, no denied families.
        assert upstage_profile.fallback_models
        for denied in ("solar-mini", "syn-pro"):
            assert not any(
                denied in m for m in upstage_profile.fallback_models
            ), f"non-agentic family {denied!r} must not be a fallback default"

    def test_default_model_is_solar_pro3(self, upstage_profile):
        # Entry [0] is the setup default (get_default_model_for_provider).
        assert upstage_profile.fallback_models[0] == "solar-pro3"

    def test_aux_model_left_empty(self, upstage_profile):
        # Unset → auxiliary side tasks fall back to the user's main model.
        assert upstage_profile.default_aux_model == ""


class TestUpstageReasoning:
    """``build_api_kwargs_extras`` wires Solar's top-level ``reasoning_effort``.

    Solar Pro accepts ``reasoning_effort`` (minimal|low|medium|high, default
    minimal=off) and never requires echoing ``reasoning_content`` back, so only
    the request field is emitted — always top-level, never in extra_body.
    """

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_pro_explicit_effort_passes_through(self, upstage_profile, effort):
        extra_body, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="solar-pro3"
        )
        assert extra_body == {}
        assert top_level == {"reasoning_effort": effort}

    @pytest.mark.parametrize("effort", ["xhigh", "max", "ultra"])
    def test_pro_strong_efforts_collapse_to_high(self, upstage_profile, effort):
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort}, model="solar-pro2"
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_unknown_future_effort_collapses_to_high(self, upstage_profile):
        # Guard against the #62650 recurrence: a future effort level Hermes
        # adds above "high" must collapse to Solar's strongest, not silently
        # downgrade to the medium default.
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "hyperthink"},
            model="solar-pro3",
        )
        assert top_level == {"reasoning_effort": "high"}

    def test_pro_enabled_without_effort_defaults_on(self, upstage_profile):
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, model="solar-pro3"
        )
        assert top_level == {"reasoning_effort": "medium"}

    def test_pro_minimal_effort_is_omitted(self, upstage_profile):
        # Explicit minimal == reasoning off → omit so Solar applies its default.
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "minimal"}, model="solar-pro3"
        )
        assert top_level == {}

    def test_disabled_omits_field(self, upstage_profile):
        # `/reasoning none` → enabled False → explicitly off.
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}, model="solar-pro3"
        )
        assert top_level == {}

    @pytest.mark.parametrize("model", ["solar-pro3", "solar-pro", "solar-open2"])
    def test_no_config_defaults_reasoning_on(self, upstage_profile, model):
        # Unset reasoning_config → default ON at medium (matches the /reasoning
        # "medium (default)" label), not Solar's server default of minimal/off.
        _, top_level = upstage_profile.build_api_kwargs_extras(model=model)
        assert top_level == {"reasoning_effort": "medium"}

    @pytest.mark.parametrize("model", ["solar-mini", "solar-mini-202610", "syn-pro"])
    def test_no_config_deny_listed_still_omits(self, upstage_profile, model):
        # Default-on must not leak to the deny-listed non-reasoning models.
        _, top_level = upstage_profile.build_api_kwargs_extras(model=model)
        assert top_level == {}

    @pytest.mark.parametrize(
        "model",
        [
            "solar-pro3-250127",
            "solar-open",
            "solar-open-250127",
            "solar-open2",
            "solar-open2-260528",
        ],
    )
    def test_pro_and_open_variants_support_reasoning(self, upstage_profile, model):
        # Both the Solar Pro and Solar Open families (incl. dated variants)
        # accept reasoning_effort.
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert top_level == {"reasoning_effort": "high"}

    @pytest.mark.parametrize("model", ["solar-mini", "solar-mini-202610", "syn-pro"])
    def test_deny_listed_models_never_send_reasoning(self, upstage_profile, model):
        # solar-mini / syn-pro ignore reasoning_effort, so never send it —
        # even when the user explicitly enables reasoning.
        extra_body, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert extra_body == {}
        assert top_level == {}

    @pytest.mark.parametrize("model", ["solar-future", "solar-future-260601"])
    def test_unknown_future_models_default_to_reasoning(self, upstage_profile, model):
        # Deny-list semantics: a future Solar model we've never heard of is
        # assumed reasoning-capable, so reasoning_effort is sent instead of
        # being silently dropped (the old allow-list failure mode).
        _, top_level = upstage_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}, model=model
        )
        assert top_level == {"reasoning_effort": "high"}

        # And the unset-config default-on path applies to it too.
        _, top_level = upstage_profile.build_api_kwargs_extras(model=model)
        assert top_level == {"reasoning_effort": "medium"}

    def test_none_model_defaults_to_reasoning(self, upstage_profile):
        # No model in context → treated as reasoning-capable, consistent with
        # the provider default (fallback_models[0] == "solar-pro3").
        _, top_level = upstage_profile.build_api_kwargs_extras(model=None)
        assert top_level == {"reasoning_effort": "medium"}


def upstage_profile_singleton():
    import providers

    return providers.get_provider_profile("upstage")
