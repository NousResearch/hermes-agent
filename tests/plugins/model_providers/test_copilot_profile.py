"""Unit tests for the Copilot provider profile's reasoning-effort wiring.

GitHub Copilot serves different models with different supported reasoning-effort
sets (the live ``/models`` catalog reports them per model). The profile must
forward the requested effort when the catalog lists it as supported, and only
downgrade to the nearest weaker supported level when it does not, rather than
unconditionally collapsing ``xhigh`` to ``high`` (which silently capped models
that actually support the higher level).

These tests pin that contract without going live, by stubbing the catalog
lookup ``github_model_reasoning_efforts`` or passing it an explicit ``catalog=``.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def copilot_profile():
    """Resolve the registered Copilot profile.

    Importing ``model_tools`` triggers plugin discovery, which registers the
    Copilot profile. Going through ``get_provider_profile`` keeps the test
    honest: if the registered class is ever swapped for a plain
    ``ProviderProfile`` the assertions below collapse.
    """
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("copilot")
    assert profile is not None, "copilot provider profile must be registered"
    return profile


def _patch_efforts(monkeypatch, efforts):
    """Stub the catalog lookup the profile and main-agent path call for supported efforts."""
    import hermes_cli.models as models_mod
    monkeypatch.setattr(
        models_mod, "github_model_reasoning_efforts", lambda model: list(efforts)
    )


class TestCopilotReasoningEffortClamp:
    def test_supported_effort_forwarded_verbatim(self, copilot_profile, monkeypatch):
        """xhigh is forwarded unchanged when the catalog lists it."""
        _patch_efforts(monkeypatch, ["minimal", "low", "medium", "high", "xhigh"])
        extra_body, _ = copilot_profile.build_api_kwargs_extras(
            model="gpt-5.5",
            reasoning_config={"effort": "xhigh"},
            supports_reasoning=True,
        )
        assert extra_body["reasoning"] == {"effort": "xhigh"}

    def test_xhigh_downgrades_to_high_when_unsupported(self, copilot_profile, monkeypatch):
        """A model whose catalog lacks xhigh gets the nearest weaker level."""
        _patch_efforts(monkeypatch, ["low", "medium", "high"])
        extra_body, _ = copilot_profile.build_api_kwargs_extras(
            model="o-series-model",
            reasoning_config={"effort": "xhigh"},
            supports_reasoning=True,
        )
        assert extra_body["reasoning"] == {"effort": "high"}

    def test_minimal_downgrades_to_low_when_unsupported(self, copilot_profile, monkeypatch):
        _patch_efforts(monkeypatch, ["low", "medium", "high"])
        extra_body, _ = copilot_profile.build_api_kwargs_extras(
            model="o-series-model",
            reasoning_config={"effort": "minimal"},
            supports_reasoning=True,
        )
        assert extra_body["reasoning"] == {"effort": "low"}

    def test_unsupported_effort_falls_back_to_medium(self, copilot_profile, monkeypatch):
        """An effort not in the set, with no specific rule, falls to medium."""
        _patch_efforts(monkeypatch, ["low", "medium", "high"])
        extra_body, _ = copilot_profile.build_api_kwargs_extras(
            model="some-model",
            reasoning_config={"effort": "garbage"},
            supports_reasoning=True,
        )
        assert extra_body["reasoning"] == {"effort": "medium"}

    def test_falls_back_to_first_supported_when_no_medium(self, copilot_profile, monkeypatch):
        """If medium isn't supported either, pick the first supported level."""
        _patch_efforts(monkeypatch, ["low", "high"])
        extra_body, _ = copilot_profile.build_api_kwargs_extras(
            model="weird-model",
            reasoning_config={"effort": "xhigh"},
            supports_reasoning=True,
        )
        # xhigh not supported, high IS supported → high wins via the xhigh rule.
        assert extra_body["reasoning"] == {"effort": "high"}

    def test_first_supported_when_no_rule_matches(self, copilot_profile, monkeypatch):
        _patch_efforts(monkeypatch, ["low", "high"])
        extra_body, _ = copilot_profile.build_api_kwargs_extras(
            model="weird-model",
            reasoning_config={"effort": "garbage"},
            supports_reasoning=True,
        )
        assert extra_body["reasoning"] == {"effort": "low"}


class TestCopilotOfflineAstraEfforts:
    """With no live catalog entry, Astra still advertises its own ladder (exact slugs only),
    and a structured catalog entry keeps precedence over the offline fallback."""

    @pytest.mark.parametrize("model, expected", [
        ("gpt-6-astra", ["low", "medium", "high", "xhigh", "max"]),
        ("openai/gpt-6-astra", ["low", "medium", "high", "xhigh", "max"]),
        ("gpt-6-astra-pro", []),  # speed-tier / unknown suffixes stay off the Astra ladder
    ])
    def test_offline_fallback_uses_exact_astra_slugs(self, model, expected):
        from hermes_cli.models import github_model_reasoning_efforts

        assert github_model_reasoning_efforts(model, catalog=[]) == expected

    def test_structured_catalog_beats_offline_fallback(self):
        from hermes_cli.models import github_model_reasoning_efforts

        catalog = [{"id": "gpt-6-astra", "capabilities": {
            "type": "chat", "supports": {"reasoning_effort": ["low", "high"]}}}]
        assert github_model_reasoning_efforts("gpt-6-astra", catalog=catalog) == ["low", "high"]


@pytest.mark.parametrize("effort, expected", [
    ("xhigh", "high"), ("max", "high"), ("ultra", "high"), ("minimal", "minimal"),
    ("garbage", "medium"), (" HIGH ", "high"),
])
def test_main_agent_github_clamp_never_escalates(monkeypatch, effort, expected):
    """The main-agent GitHub path shares the profile's clamp: unsupported levels step down to
    the nearest weaker listed one (max/ultra no longer fall all the way to medium)."""
    from types import SimpleNamespace

    from agent.reasoning_params import ReasoningParamsMixin

    _patch_efforts(monkeypatch, ["minimal", "low", "medium", "high"])
    agent = SimpleNamespace(model="gpt-5.4", reasoning_config={"enabled": True, "effort": effort})
    assert ReasoningParamsMixin._github_models_reasoning_extra_body(agent) == {"effort": expected}
