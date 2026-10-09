"""Unit tests for the Fireworks AI provider profile.

Pins the profile's contract without going live: identity, alias registration,
and the pay-as-you-go model defaults (direct catalog ``/models/``
IDs, not the router-only tier).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fireworks_profile():
    """Resolve the registered Fireworks profile through the real discovery path."""
    # Importing model_tools triggers plugin discovery, registering the profile.
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("fireworks")
    assert profile is not None, "fireworks provider profile must be registered"
    return profile




class TestFireworksHeaders:
    def test_attribution_matches_canonical_hermes_values(self, fireworks_profile):
        """Fireworks requests carry the same attribution identity Hermes sends
        everywhere else.

        Asserted against the shared constant rather than the literals so a
        rebrand can't leave one provider on a stale referer/title.
        """
        from agent.auxiliary_client import _OR_HEADERS_BASE

        headers = fireworks_profile.default_headers
        assert headers["HTTP-Referer"] == _OR_HEADERS_BASE["HTTP-Referer"]
        assert headers["X-Title"] == _OR_HEADERS_BASE["X-Title"]

    def test_user_agent_identifies_hermes(self, fireworks_profile):
        from hermes_cli.version_info import get_version_info
        assert fireworks_profile.default_headers["User-Agent"] == (
            f"HermesAgent/{get_version_info().base_version}"
        )




class TestFireworksModelDefaults:
    """Defaults must be usable with a standard pay-as-you-go key.

    PAYG keys address ``accounts/fireworks/models/...`` directly; the bundled
    defaults target that (the BYOK motion) so a fresh key works out of the box,
    and use the standard tier rather than turbo as the out-of-box default.
    """

    def test_aux_model_is_payg_model_not_router(self, fireworks_profile):
        aux = fireworks_profile.default_aux_model
        assert aux.startswith("accounts/fireworks/models/"), aux
        assert "/routers/" not in aux
        assert "turbo" not in aux.lower()

    def test_fallback_models_are_payg_models_not_routers(self, fireworks_profile):
        assert fireworks_profile.fallback_models, "expected curated fallbacks"
        for model in fireworks_profile.fallback_models:
            assert model.startswith("accounts/fireworks/models/"), model
            assert "/routers/" not in model
            assert "turbo" not in model.lower(), model


class TestFireworksReasoning:
    @pytest.mark.parametrize(
        "provider, reasoning_config, expect_top_level, expect_generic",
        [
            # Fireworks: thinking-off goes out as the documented top-level control, never as the
            # nested ``extra_body.reasoning`` the API 400s on (#109774, salvaged from #109807).
            ("fireworks", {"enabled": False}, {"reasoning_effort": "none"}, False),
            ("fireworks", {"enabled": True, "effort": "low"}, {"reasoning_effort": "low"}, False),
            # Control: a route without a reasoning-aware profile keeps the generic fallback.
            ("unregistered-gateway", {"enabled": False}, {}, True),
        ],
    )
    def test_auxiliary_reasoning_wire_shape(
        self, fireworks_profile, provider, reasoning_config, expect_top_level, expect_generic
    ):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider, "accounts/fireworks/models/glm-5p2",
            [{"role": "user", "content": "Generate a title"}],
            reasoning_config=reasoning_config, base_url="https://api.fireworks.ai/inference/v1",
            task="title_generation",
        )

        assert {k: v for k, v in kwargs.items() if k == "reasoning_effort"} == expect_top_level
        assert ("reasoning" in kwargs.get("extra_body", {})) is expect_generic
