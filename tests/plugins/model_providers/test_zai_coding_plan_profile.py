"""zai-coding-plan profile: endpoint separation from the standard zai provider.

Coding-plan subscriptions authenticate on different endpoints than the
standard /api/paas/v4 route (which rejects coding-plan keys with HTTP 429,
code 1113). The profile defaults to z.ai's Anthropic wire
(api.z.ai/api/anthropic) — the endpoint where preserved thinking actually
reaches the model for agent tool loops (the OpenAI-compat routes accept
replayed reasoning_content but silently drop it from model attention;
probed 2026-08-15). Mirrors alibaba-coding-plan / kimi-coding so
coding-plan users get a working default without hand-editing GLM_BASE_URL.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def coding_profile():
    import model_tools  # noqa: F401  — triggers plugin discovery
    import providers

    p = providers.get_provider_profile("zai-coding-plan")
    assert p is not None, "zai-coding-plan must be registered"
    return p


class TestZaiCodingPlanProfile:
    def test_anthropic_endpoint_is_default(self, coding_profile):
        import providers

        # Anthropic wire: preserved thinking reaches the model here (the
        # OpenAI-compat /api/coding/paas/v4 route accepts replayed
        # reasoning_content but silently drops it — probed 2026-08-15).
        assert coding_profile.base_url == "https://api.z.ai/api/anthropic"
        std = providers.get_provider_profile("zai")
        assert std.base_url != coding_profile.base_url, (
            "coding-plan profile must not share the standard endpoint"
        )

    def test_base_url_triggers_anthropic_adapter(self, coding_profile):
        """The /anthropic suffix is what agent_init.py keys on to select
        the Anthropic Messages adapter — pin the contract the default
        depends on."""
        assert coding_profile.base_url.rstrip("/").endswith("/anthropic")

    def test_distinct_from_standard_zai(self, coding_profile):
        import providers

        assert coding_profile.name == "zai-coding-plan"
        assert coding_profile.name != "zai"

    def test_env_var_chain_includes_fallback(self, coding_profile):
        """Dedicated vars first, ZAI_API_KEY as fallback so users with one
        key don't need to duplicate it."""
        assert coding_profile.env_vars[0] == "ZAI_CODING_PLAN_API_KEY"
        assert "ZAI_API_KEY" in coding_profile.env_vars

    def test_shares_glm_reasoning_wiring(self, coding_profile):
        """Subclassing ZaiProfile keeps the GLM thinking / reasoning wiring.
        On the Anthropic default the transport builds the request shape
        (thinking blocks / budget_tokens) natively, so the profile wiring
        stays shared with the standard zai provider."""
        extra_body, top_level = coding_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            model="glm-5.2",
            base_url="https://api.z.ai/api/paas/v4",
        )
        assert top_level == {"reasoning_effort": "high"}
        assert extra_body.get("thinking") == {"type": "enabled"}

    def test_default_aux_model_in_curated_list(self, coding_profile):
        """The profile's default aux model must resolve against the
        provider's own curated list — aux-task resolution (curator,
        vision, title generation) validates against it, so a mismatch
        fails to resolve or shows an unlisted model in the picker
        (review point on PR #86560).  glm-4.5-air: the live Anthropic
        wire serves no glm-4.5-flash (models-list probe 2026-08-17:
        glm-4.5, glm-4.5-air, glm-4.6, glm-4.7, glm-5, glm-5-turbo,
        glm-5.1, glm-5.2, glm-5.3), and air is the cheapest tier there."""
        from hermes_cli.models import _PROVIDER_MODELS

        assert coding_profile.default_aux_model == "glm-4.5-air"
        assert coding_profile.default_aux_model in _PROVIDER_MODELS["zai-coding-plan"]

    def test_default_endpoint_builds_anthropic_request_shape(self, coding_profile):
        """Pin the docstring claim at the adapter level: the default
        ``/api/anthropic`` endpoint produces the Anthropic request shape —
        thinking as budget_tokens (not the OpenAI-compat
        reasoning_effort/top-level param the paas/v4 wire would use)
        (review point on PR #86560)."""
        from agent.anthropic_adapter import build_anthropic_kwargs

        kwargs = build_anthropic_kwargs(
            model="glm-5.3",
            messages=[{"role": "user", "content": "hello"}],
            tools=None,
            max_tokens=8192,
            reasoning_config={"enabled": True, "effort": "high"},
            base_url=coding_profile.base_url,
        )
        assert "reasoning_effort" not in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert isinstance(kwargs["thinking"]["budget_tokens"], int)
        assert kwargs["thinking"]["budget_tokens"] > 0

    def test_model_list_registered(self):
        from hermes_cli.models import _PROVIDER_MODELS

        assert "zai-coding-plan" in _PROVIDER_MODELS
        # glm-5.2 is the effort-dial model guaranteed on main; 5.3 arrives
        # with the GLM-5.3 support PR — assert the shared core only.
        assert "glm-5.2" in _PROVIDER_MODELS["zai-coding-plan"]

    def test_aliases_normalize_to_coding_plan(self):
        from hermes_cli.models import curated_models_for_provider, normalize_provider

        for alias in ("zai-coding", "glm-coding", "z-ai-coding"):
            assert normalize_provider(alias) == "zai-coding-plan", alias

        # The public curated-models path resolves aliases to the coding list.
        models = [m for m, _ in curated_models_for_provider("glm-coding")]
        assert any(m == "glm-5.2" for m in models), models

    def test_bundled_import_binds_bundled_zai_profile(self):
        """Documented limitation: the cross-plugin import pins this profile
        to the BUNDLED ZaiProfile class. A user-plugin override of ``zai``
        loads later under a _hermes_user_provider_* module name and does not
        affect this profile. Assert the binding is at least the bundled one
        (deterministic behavior rather than accidental)."""
        import sys

        import providers  # noqa: F401  — discovery

        assert "plugins.model_providers.zai" in sys.modules
        # get_provider_profile returns the registered instance whose class
        # comes from the bundled module.
        p = providers.get_provider_profile("zai-coding-plan")
        assert type(p).__module__ == "plugins.model_providers.zai"
