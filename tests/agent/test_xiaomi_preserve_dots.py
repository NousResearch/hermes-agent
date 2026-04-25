"""Tests for Xiaomi/MiMo dot-preservation in model name normalization.

Xiaomi's MiMo models use dots in their version strings (e.g. mimo-v2.5-pro).
The Anthropic-compatible SDK layer converts dots to hyphens by default, producing
``mimo-v2-5-pro`` — a name the Xiaomi API rejects with HTTP 400. The fix adds
"xiaomi" to the _anthropic_preserve_dots() allowlist so dots survive intact.

Issue: #15619
"""
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# _anthropic_preserve_dots — flag tests
# ---------------------------------------------------------------------------


class TestXiaomiPreserveDots:
    """AIAgent._anthropic_preserve_dots returns True for Xiaomi endpoints."""

    def test_xiaomi_provider_name_preserves_dots(self):
        """Explicit ``provider="xiaomi"`` activates dot-preservation."""
        from run_agent import AIAgent
        agent = SimpleNamespace(provider="xiaomi", base_url="")
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_xiaomi_provider_case_insensitive(self):
        """Provider matching is case-insensitive."""
        from run_agent import AIAgent
        for variant in ("Xiaomi", "XIAOMI", "XiaoMi"):
            agent = SimpleNamespace(provider=variant, base_url="")
            assert AIAgent._anthropic_preserve_dots(agent) is True, variant

    def test_xiaomimimo_base_url_preserves_dots(self):
        """Defense-in-depth: a xiaomimimo.com base URL triggers preservation
        even if provider is not explicitly set to ``xiaomi``."""
        from run_agent import AIAgent
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://token-plan-cn.xiaomimimo.com/anthropic",
        )
        assert AIAgent._anthropic_preserve_dots(agent) is True

    def test_unrelated_provider_does_not_preserve_dots(self):
        """Canary: adding ``xiaomi`` must not accidentally enable dots for
        unrelated providers — ``claude-sonnet-4.6`` still normalises on the
        native Anthropic API."""
        from run_agent import AIAgent
        agent = SimpleNamespace(provider="anthropic", base_url="https://api.anthropic.com")
        assert AIAgent._anthropic_preserve_dots(agent) is False

    def test_unrelated_base_url_does_not_preserve_dots(self):
        """The xiaomimimo.com heuristic must not trigger for other .com URLs."""
        from run_agent import AIAgent
        agent = SimpleNamespace(
            provider="custom",
            base_url="https://api.openai.com/v1",
        )
        assert AIAgent._anthropic_preserve_dots(agent) is False


# ---------------------------------------------------------------------------
# normalize_model_name — end-to-end model-name tests
# ---------------------------------------------------------------------------


class TestXiaomiModelNameNormalization:
    """normalize_model_name preserves dots for MiMo model names when
    preserve_dots=True, confirming the fix resolves the HTTP 400 in #15619."""

    def test_mimo_v25_pro_preserved(self):
        """The exact model from the bug report survives normalization."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name("mimo-v2.5-pro", preserve_dots=True) == "mimo-v2.5-pro"

    def test_mimo_v25_pro_mangled_without_flag(self):
        """Without preserve_dots the dot is still converted (existing behaviour
        for providers that need it, e.g. raw Anthropic)."""
        from agent.anthropic_adapter import normalize_model_name
        assert normalize_model_name("mimo-v2.5-pro", preserve_dots=False) == "mimo-v2-5-pro"

    def test_mimo_variant_names_preserved(self):
        """Other MiMo variant names with dots are also preserved."""
        from agent.anthropic_adapter import normalize_model_name
        for model in ("mimo-v2.0-standard", "mimo-v3.1-ultra"):
            result = normalize_model_name(model, preserve_dots=True)
            assert result == model, f"expected {model!r}, got {result!r}"
