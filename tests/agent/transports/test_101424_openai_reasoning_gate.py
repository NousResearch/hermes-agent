"""Tests for #101424: non-reasoning OpenAI models must not receive a
``reasoning`` field on the codex/Responses transport.

The Responses API 400s with "Unsupported parameter: 'reasoning.effort' is
not supported with this model" when a non-reasoning model (gpt-4o*,
gpt-4.1*, ...) receives any reasoning field.  Previously the transport's
no-profile fallback sent the legacy effort vocabulary to every model;
now the family allowlist suppresses the dial entirely for models that
reject it, mirroring the grok effort-dial conservatism.
"""

from __future__ import annotations

import pytest

from agent.transports import get_transport
from agent.model_metadata import openai_responses_supports_reasoning_effort


@pytest.fixture
def transport():
    import agent.transports.codex  # noqa: F401

    return get_transport("codex_responses")


class TestOpenaiFamilyAllowlist:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5.6",
            "gpt-5-mini",
            "o4-mini",
            "o3",
            "openai/gpt-5.6",  # aggregator-prefixed
        ],
    )
    def test_reasoning_families_accept_effort(self, model):
        assert openai_responses_supports_reasoning_effort(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.5",
            "gpt-4o",
            "chatgpt-4o-latest",
            "openai/gpt-4o-mini",
            "",  # unknown/blank -> conservative False
            "some-byok-route",
        ],
    )
    def test_non_reasoning_models_reject_effort(self, model):
        assert openai_responses_supports_reasoning_effort(model) is False


class TestNoProfileFallbackSuppressesNonReasoningModels:
    def _kwargs(self, transport, model, provider="openai-api",
                base_url="https://api.openai.com/v1", reasoning_config=None):
        return transport.build_kwargs(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            tools=[],
            base_url=base_url,
            session_id="sid",
            provider=provider,
            reasoning_config=reasoning_config,
        )

    def test_openai_api_gpt_4o_mini_gets_no_reasoning_field(self, transport):
        # openai-api registers no ProviderProfile -> no-profile fallback.
        # gpt-4o-mini must NOT receive reasoning (it 400s on any such field).
        kw = self._kwargs(transport, "gpt-4o-mini")
        assert "reasoning" not in kw

    def test_openai_codex_gpt_4o_mini_gets_no_reasoning_field(self, transport):
        kw = self._kwargs(
            transport, "gpt-4o-mini",
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
        )
        assert "reasoning" not in kw

    def test_openai_api_gpt_5_family_still_gets_effort_dial(self, transport):
        kw = self._kwargs(transport, "gpt-5.6", reasoning_config={"effort": "medium"})
        assert kw["reasoning"]["effort"] == "medium"

    def test_openai_compatible_byok_keeps_legacy_efforts(self, transport):
        # A non-OpenAI provider riding the codex transport (BYOK endpoint,
        # proxy, router without a catalog declaration) still sends the legacy
        # vocabulary — the family gate is scoped to OpenAI's own surfaces.
        kw = self._kwargs(
            transport, "gpt-4o-mini",
            provider="some-byok-route",
            base_url="https://generic.example.com/v1",
            reasoning_config={"effort": "medium"},
        )
        assert kw["reasoning"]["effort"] == "medium"

    def test_router_cold_cache_keeps_legacy_default(self, transport):
        # Regression mirror of test_router_codex_efforts.py: no catalog ->
        # no declaration -> legacy vocabulary for non-OpenAI providers.
        kw = self._kwargs(
            transport, "grok-4.6",
            provider="router",
            base_url="https://api.router.com/v1",
            reasoning_config={"effort": "xhigh"},
        )
        assert kw["reasoning"]["effort"] == "xhigh"
