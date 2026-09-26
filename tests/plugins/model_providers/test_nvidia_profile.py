"""Unit tests for the NVIDIA NIM provider profile's reasoning contract.

NIM cloud (integrate.api.nvidia.com) rejects the ``reasoning`` parameter
outright — HTTP 400 "Unsupported parameter(s): `reasoning`" — for every
model and every value, including ``{"enabled": false}``.  The generic
reasoning fallback in ``agent/auxiliary_client._build_call_kwargs`` makes
any auxiliary call (MoA aggregator/advisor, compression, …) on that route
fail whenever a reasoning_config resolves, e.g. from a global
``agent.reasoning_effort``.

These tests pin the profile's contract: omit ``reasoning`` on the cloud
route, preserve the generic-fallback wire shape everywhere else.
"""

from __future__ import annotations

import pytest

from providers.base import ProviderProfile

CLOUD_BASE = "https://integrate.api.nvidia.com/v1"
LOCAL_BASE = "http://localhost:8000/v1"


@pytest.fixture
def nvidia_profile():
    """Resolve the registered NVIDIA profile.

    Going through ``providers.get_provider_profile`` keeps the test honest —
    if someone later replaces the registered class with a plain
    ``ProviderProfile``, every assertion below collapses.
    """
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the NVIDIA profile in the global provider registry.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("nvidia")
    assert profile is not None, "nvidia provider profile must be registered"
    return profile


class TestNvidiaReasoningWireShape:
    """``build_api_kwargs_extras`` produces NIM's exact wire format."""

    def test_profile_owns_reasoning_handling(self, nvidia_profile):
        """The override must exist on the class — this is what flips
        ``profile_handles_reasoning`` in _build_call_kwargs and suppresses
        the generic extra_body.reasoning fallback."""
        assert (
            type(nvidia_profile).build_api_kwargs_extras
            is not ProviderProfile.build_api_kwargs_extras
        )

    def test_cloud_route_omits_reasoning_when_enabled(self, nvidia_profile):
        extra_body, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            supports_reasoning=True,
            model="nvidia/llama-3.3-70b-instruct",
            base_url=CLOUD_BASE,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_cloud_route_omits_reasoning_even_when_disabled(self, nvidia_profile):
        """NIM cloud 400s on ``{"enabled": false}`` too — never emit it."""
        extra_body, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
            model="nvidia/llama-3.3-70b-instruct",
            base_url=CLOUD_BASE,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_no_reasoning_config_emits_nothing(self, nvidia_profile):
        extra_body, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config=None,
            model="nvidia/llama-3.3-70b-instruct",
            base_url=CLOUD_BASE,
        )
        assert extra_body == {}
        assert top_level == {}

    def test_local_nim_keeps_generic_fallback_enabled(self, nvidia_profile):
        """Local NIM endpoints (NVIDIA_BASE_URL) keep prior behavior."""
        extra_body, top_level = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
            model="some-local-model",
            base_url=LOCAL_BASE,
        )
        assert extra_body == {"reasoning": {"enabled": True, "effort": "high"}}
        assert top_level == {}

    def test_local_nim_keeps_generic_fallback_disabled(self, nvidia_profile):
        extra_body, _ = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
            model="some-local-model",
            base_url=LOCAL_BASE,
        )
        assert extra_body == {"reasoning": {"enabled": False}}

    def test_local_nim_effort_defaults_to_medium(self, nvidia_profile):
        """Mirrors the generic fallback's ``effort or "medium"`` default."""
        extra_body, _ = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True},
            supports_reasoning=True,
            model="some-local-model",
            base_url=LOCAL_BASE,
        )
        assert extra_body == {"reasoning": {"enabled": True, "effort": "medium"}}

    def test_missing_base_url_keeps_generic_fallback(self, nvidia_profile):
        """No base_url (profile default resolution failure) must not change
        behavior — fail toward the pre-fix wire shape, not toward omission."""
        extra_body, _ = nvidia_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "low"},
            supports_reasoning=True,
            model="some-model",
            base_url=None,
        )
        assert extra_body == {"reasoning": {"enabled": True, "effort": "low"}}
