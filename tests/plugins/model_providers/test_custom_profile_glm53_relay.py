"""Unit tests: glm-5.3 behind b.ai-style MaaS relays rejects reasoning_effort levels
outside [low, high, max] with an opaque HTTP 400 (code 400001, "rejected by an internal
MaaS component"). Live-verified against api.b.ai/v1 on 2026-09-27:

  reasoning_effort=medium  → HTTP 400 (3/3)
  reasoning_effort=none    → HTTP 400
  reasoning_effort=minimal → HTTP 400
  reasoning_effort=low     → HTTP 200
  reasoning_effort=high    → HTTP 200
  reasoning_effort=max     → HTTP 200

The native z.ai endpoint accepts the full graded scale (#91789), so the clamp applies
only to glm-5.3* models on non-z.ai hosts. Mirrors the Kimi K3 / Ox Alpha vocabulary
precedents.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestGlm53RelayEffortClamp:
    """glm-5.3 on a relay: effort clamped onto [low, high, max]."""

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("low", "low"),
            ("high", "high"),
            ("max", "max"),
            ("medium", "high"),     # b.ai rejects medium outright → round UP to high (server default middle on K3-style wires)
            ("xhigh", "max"),
            ("minimal", "low"),
        ],
    )
    def test_relay_clamps_rejected_levels(self, custom_profile, effort, expected):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": effort},
            model="glm-5.3-flash",
            base_url="https://api.b.ai/v1",
        )
        assert eb == {}
        assert tl == {"reasoning_effort": expected}

    def test_relay_disable_keeps_none_contract(self, custom_profile):
        """enabled=True + effort='none' → the disable contract emits 'none' verbatim
        (Ollama #25758 semantics). b.ai-style relays reject 'none' too; that path is
        the vocabulary rejection classified upstream (see #118627), not this clamp's
        scope — this test pins that the disable contract is unchanged here."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"},
            model="glm-5.3-flash",
            base_url="https://api.b.ai/v1",
        )
        assert tl == {"reasoning_effort": "none"}

    def test_enabled_false_still_disables(self, custom_profile):
        """enabled=False is an explicit disable — 'none' is emitted verbatim (unchanged
        contract; b.ai's rejection of 'none' is a relay-quirk handled at retry, not here)."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            model="glm-5.3-flash",
            base_url="https://api.b.ai/v1",
        )
        assert tl == {"reasoning_effort": "none"}

    @pytest.mark.parametrize("base_url", ["https://api.z.ai/api/coding/paas/v4", "https://api.z.ai/api/paas/v4"])
    def test_native_zai_unclamped(self, custom_profile, base_url):
        """The native z.ai endpoint accepts the full graded scale (#91789) — no clamp."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="glm-5.3-flash",
            base_url=base_url,
        )
        assert tl == {"reasoning_effort": "medium"}

    @pytest.mark.parametrize("model", ["glm-5.3-flashx", "zai-org/glm-5.3", "GLM-5.3-Flash"])
    def test_glm53_family_variants_clamped(self, custom_profile, model):
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model=model,
            base_url="https://relay.example/v1",
        )
        assert tl == {"reasoning_effort": "high"}

    def test_glm52_out_of_scope(self, custom_profile):
        """glm-5.2 has its own native vocabulary (high/max) — not the 5.3 relay case."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="glm-5.2",
            base_url="https://relay.example/v1",
        )
        assert tl == {"reasoning_effort": "medium"}

    def test_non_glm_models_unclamped(self, custom_profile):
        """Only the glm-5.3 family gets the relay vocabulary."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="deepseek-v4.1-flash",
            base_url="https://api.b.ai/v1",
        )
        assert tl == {"reasoning_effort": "medium"}

    def test_glm52_never_matched(self, custom_profile):
        """glm-5.2 has its own vocabulary (high/max native) — must not match the 5.3 slug."""
        eb, tl = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            model="glm-5.2",
            base_url="https://api.b.ai/v1",
        )
        assert tl == {"reasoning_effort": "medium"}
