"""Unit tests for the OpenRouter profile's reported-usage-cost wiring.

OpenRouter bills the *endpoint* that serves a request (per-endpoint rates
differ from the model-level headline price by up to ~13x), and reports the
actual per-request bill in ``usage.cost`` / ``usage.cost_details`` when the
request asks for it (``"usage": {"include": true}``). These tests pin:

* every OpenRouter request carries the ``usage.include`` flag (and only
  OpenRouter requests do),
* a reported cost is surfaced as a ``CostResult`` with ``status="actual"``
  and ``source="provider_generation_api"``,
* responses without a reported cost (older servers, routes that omit it)
  fall back to ``None`` so the model-level estimate path still applies.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from agent.usage_pricing import CostResult, format_cost_label, normalize_usage
from providers.base import ProviderProfile


@pytest.fixture
def openrouter_profile():
    """Resolve the registered OpenRouter profile through the real discovery path."""
    # ``model_tools`` triggers plugin discovery on import, which is what
    # registers the OpenRouter profile in the global provider registry.
    import model_tools  # side effect: plugin discovery registers the profile
    import providers

    profile = providers.get_provider_profile("openrouter")
    assert profile is not None, "openrouter provider profile must be registered"
    return profile


class TestUsageIncludeFlag:
    def test_extra_body_requests_usage_report(self, openrouter_profile):
        assert openrouter_profile.build_extra_body() == {"usage": {"include": True}}

    def test_flag_coexists_with_sticky_session_and_prefs(self, openrouter_profile):
        body = openrouter_profile.build_extra_body(
            session_id="conv-123", provider_preferences={"only": ["inference-net"]}
        )
        assert body["usage"] == {"include": True}
        assert body["session_id"] == "conv-123"
        assert body["provider"] == {"only": ["inference-net"]}

    def test_flag_is_openrouter_only(self, openrouter_profile):
        """The base profile (and every other provider) adds no usage flag."""
        other = ProviderProfile(name="other")
        assert "usage" not in other.build_extra_body()


class TestReportedCost:
    def test_reported_cost_is_actual(self, openrouter_profile):
        usage = _usage_with_raw({"cost": 1.1106e-05})
        result = openrouter_profile.get_usage_cost("deepseek/deepseek-v4.1-flash-1017", usage)
        assert isinstance(result, CostResult)
        assert result.amount_usd == Decimal(str(1.1106e-05))
        assert result.status == "actual"
        assert result.source == "provider_generation_api"
        assert result.label == format_cost_label(Decimal(str(1.1106e-05)))

    def test_reported_cost_decimal_string(self, openrouter_profile):
        usage = _usage_with_raw({"cost": "0.0000345"})
        result = openrouter_profile.get_usage_cost("m", usage)
        assert result is not None and result.amount_usd == Decimal("0.0000345")

    def test_zero_cost_is_reported(self, openrouter_profile):
        """A $0.00 bill (e.g. a free endpoint) is still an actual bill."""
        usage = _usage_with_raw({"cost": 0})
        result = openrouter_profile.get_usage_cost("m", usage)
        assert result is not None and result.amount_usd == Decimal(0)

    def test_cost_details_lands_in_notes(self, openrouter_profile):
        usage = _usage_with_raw({
            "cost": 1.1106e-05,
            "cost_details": {
                "upstream_inference_prompt_cost": 0.00000072,
                "upstream_inference_completions_cost": 0.000009986,
                "upstream_inference_total_cost": 1.1106e-05,
            },
        })
        result = openrouter_profile.get_usage_cost("m", usage)
        assert any("upstream_inference_prompt_cost=7.2e-07" in note for note in result.notes)

    def test_absent_cost_returns_none(self, openrouter_profile):
        usage = _usage_with_raw({"prompt_tokens": 9, "completion_tokens": 18})
        assert openrouter_profile.get_usage_cost("m", usage) is None

    def test_missing_raw_usage_returns_none(self, openrouter_profile):
        usage = normalize_usage(
            {"prompt_tokens": 1, "completion_tokens": 1},
            provider="openrouter", api_mode="chat_completions",
        )
        assert openrouter_profile.get_usage_cost("m", usage) is None

    def test_non_numeric_cost_returns_none(self, openrouter_profile):
        assert openrouter_profile.get_usage_cost("m", _usage_with_raw({"cost": "n/a"})) is None
        assert openrouter_profile.get_usage_cost("m", _usage_with_raw({"cost": None})) is None

    def test_negative_cost_returns_none(self, openrouter_profile):
        assert openrouter_profile.get_usage_cost("m", _usage_with_raw({"cost": -0.5})) is None


def _usage_with_raw(raw: dict):
    """A CanonicalUsage carrying OpenRouter's raw response usage (cost included)."""
    from agent.usage_pricing import CanonicalUsage

    return CanonicalUsage(
        input_tokens=9, output_tokens=18, raw_usage=dict(raw),
    )
