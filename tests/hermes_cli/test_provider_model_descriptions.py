"""Provider-owned picker metadata is validated before it reaches clients."""

import pytest

import providers
from hermes_cli.inventory import _apply_capabilities, _profile_model_descriptions
from providers.base import ProviderProfile


def test_explicit_model_descriptions_override_heuristics(monkeypatch):
    low, high = "fixture-low", "fixture-high"
    shared = {
        "display_name": "Fixture Family",
        "family_id": low,
        "fast": False,
        "reasoning": True,
        "reasoning_control": "adjustable",
        "can_disable_reasoning": False,
        "reasoning_efforts": ["low", "high"],
        "private_data": "must-not-escape",
    }

    class FixtureProfile(ProviderProfile):
        def describe_models(self, *, model_ids):
            assert model_ids == [low, high]
            return {
                low: {**shared, "default_reasoning_effort": "low"},
                high: {**shared, "default_reasoning_effort": "high"},
            }

    monkeypatch.setattr(providers, "get_provider_profile", lambda _name: FixtureProfile(name="fixture"))
    monkeypatch.setattr("hermes_cli.models.model_supports_fast_mode", lambda _model: True)
    monkeypatch.setattr("agent.models_dev.get_model_capabilities", lambda *_args: None)
    row = {"slug": "fixture", "models": [low, high]}

    _apply_capabilities([row])

    for model, effort in ((low, "low"), (high, "high")):
        assert row["capabilities"][model] == {
            "fast": False,
            "reasoning": True,
            "display_name": "Fixture Family",
            "family_id": low,
            "reasoning_control": "adjustable",
            "can_disable_reasoning": False,
            "reasoning_efforts": ["low", "high"],
            "default_reasoning_effort": effort,
        }


def test_base_profile_metadata_hook_is_opt_in():
    assert ProviderProfile(name="legacy").describe_models(model_ids=["old-model"]) == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"display_name": 42},
        {"family_id": "outside-catalog"},
        {"reasoning": "true"},
        {"reasoning_control": "invented"},
        {"fast": 1},
        {"can_disable_reasoning": "false"},
        {"reasoning_efforts": "high"},
        {"reasoning_efforts": ["unrecognized"]},
        {"default_reasoning_effort": "high"},
    ],
)
def test_invalid_descriptor_values_do_not_escape_inventory(monkeypatch, bad):
    class FixtureProfile(ProviderProfile):
        def describe_models(self, *, model_ids):
            return {"fixture": bad}

    monkeypatch.setattr(providers, "get_provider_profile", lambda _name: FixtureProfile(name="fixture"))
    assert _profile_model_descriptions("fixture", ["fixture"]) == {"fixture": {}}
