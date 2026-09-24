"""Regression for #88681: an explicitly activated MoA preset must be the one that runs.

``moa.active_preset`` was persisted (config_defaults), validated and preserved (normalize_moa_config),
printed by ``hermes moa list`` and round-tripped through the dashboard API — while every runtime
preset pick read ``default_preset`` only. The rule is now: **active (when it names an existing
preset) > default**, and an ``active_preset`` naming nothing is ignored rather than leaving the
runtime without a preset.
"""

from types import SimpleNamespace

import pytest

from agent.errors import MoAPresetNotFoundError
from hermes_cli.moa_config import (
    effective_moa_preset_name,
    normalize_moa_config,
    resolve_moa_preset,
)


def _config():
    """default=alpha, active=beta, two presets with distinguishable aggregators."""
    return {
        "default_preset": "alpha",
        "active_preset": "beta",
        "presets": {
            "alpha": {
                "reference_models": [{"provider": "openrouter", "model": "alpha-ref"}],
                "aggregator": {"provider": "openrouter", "model": "alpha-agg"},
            },
            "beta": {
                "reference_models": [{"provider": "openrouter", "model": "beta-ref"}],
                "aggregator": {"provider": "openrouter", "model": "beta-agg"},
            },
        },
    }


def test_effective_preset_name_prefers_active_over_default():
    assert effective_moa_preset_name(_config()) == "beta"


@pytest.mark.parametrize("active", ["", None, "   "])
def test_effective_preset_name_falls_back_to_default(active):
    cfg = _config()
    cfg["active_preset"] = active
    assert effective_moa_preset_name(cfg) == "alpha"


def test_effective_preset_name_without_any_preset_field_is_default():
    assert effective_moa_preset_name({}) == "default"


def test_unknown_active_preset_is_dropped_not_fatal():
    """A stale activation must not run *no* preset — it falls back to default_preset."""
    cfg = _config()
    cfg["active_preset"] = "ghost"
    assert normalize_moa_config(cfg)["active_preset"] == ""
    assert effective_moa_preset_name(cfg) == "alpha"


def test_resolve_without_a_name_runs_the_active_preset():
    assert resolve_moa_preset(_config())["aggregator"]["model"] == "beta-agg"


def test_resolve_explicit_name_still_outranks_active():
    assert resolve_moa_preset(_config(), "alpha")["aggregator"]["model"] == "alpha-agg"


def test_resolve_unknown_explicit_name_still_raises():
    with pytest.raises(MoAPresetNotFoundError):
        resolve_moa_preset(_config(), "nope")


def test_moa_list_marks_and_reports_the_running_preset(capsys):
    from hermes_cli import moa_cmd

    moa_cmd._print_config({"moa": _config()})
    out = capsys.readouterr().out
    assert "Active in config: beta" in out
    assert "Runs when no preset is named: beta" in out
    assert "* beta" in out
    assert "* alpha" not in out


def test_model_switch_to_moa_provider_uses_active_preset(monkeypatch):
    """`/model --provider moa` with no model names: preset comes from active, else default."""
    from hermes_cli import model_switch

    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {"moa": _config()})
    assert model_switch._moa_effective_preset() == "beta"


def test_model_switch_to_moa_provider_falls_back_to_default(monkeypatch):
    from hermes_cli import model_switch

    cfg = _config()
    cfg["active_preset"] = ""
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {"moa": cfg})
    assert model_switch._moa_effective_preset() == "alpha"


def test_moa_picker_preselects_the_active_preset(monkeypatch):
    from hermes_cli import model_setup_flows

    seen = {}

    def fake_choice(title, rows, default_idx):
        seen["default_idx"] = default_idx
        seen["rows"] = rows
        return -1  # cancel: no config write, we only assert the preselection

    monkeypatch.setattr(model_setup_flows, "_curses_choice", fake_choice)
    model_setup_flows._model_flow_moa({"moa": _config()})

    assert seen["default_idx"] == 1  # beta
    assert "← active" in seen["rows"][1]


def test_moa_facade_fallback_uses_active_preset(monkeypatch):
    """A session whose preset name no longer exists falls back to the effective preset."""
    from agent import moa_loop

    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: {"moa": _config()})
    monkeypatch.setattr(moa_loop, "MoAClient", lambda name, **kwargs: name)
    agent = SimpleNamespace(provider="moa", model="gone", tool_progress_callback=None)

    assert moa_loop.build_moa_facade(agent) == "beta"
