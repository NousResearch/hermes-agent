"""A persisted MoA selection must never execute a different preset (#76191)."""

from types import SimpleNamespace

import pytest

from agent.error_classifier import classify_api_error
from agent.errors import MoAPresetNotFoundError
from agent.moa_loop import build_moa_facade
from hermes_cli.moa_config import normalize_moa_config


@pytest.mark.parametrize("slots", [{}, {"reference_models": [], "aggregator": {}},
                                   {"reference_models": [{"provider": "moa", "model": "recursive"}]}])
def test_named_incomplete_preset_does_not_acquire_factory_routes(slots):
    cfg = normalize_moa_config({"presets": {"my-preset": slots}})
    assert cfg["presets"]["my-preset"]["reference_models"] == []
    assert cfg["presets"]["my-preset"]["aggregator"] == {}
    assert normalize_moa_config({})["reference_models"]


def test_missing_preset_fails_without_fallback_with_real_profile_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("moa:\n  default_preset: remaining\n  presets:\n    remaining: {}\n")
    agent = SimpleNamespace(provider="moa", model="deleted-preset", tool_progress_callback=None)
    with pytest.raises(MoAPresetNotFoundError) as exc:
        build_moa_facade(agent, agent.model)
    assert agent.model == "deleted-preset"
    result = classify_api_error(exc.value, provider="moa", model=agent.model)
    assert not result.retryable
    assert not result.should_fallback
