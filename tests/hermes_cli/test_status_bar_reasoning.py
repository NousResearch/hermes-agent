"""Reasoning is an opt-in status field, independent of model visibility."""
from datetime import datetime
from copy import deepcopy
from types import SimpleNamespace

import cli as cli_mod
from cli import HermesCLI


def test_reasoning_field_uses_persisted_visibility_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli_mod, "_hermes_home", tmp_path)
    assert cli_mod.save_config_value("display.status_bar.fields", ["reasoning"])
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", cli_mod.load_cli_config())
    cli = HermesCLI.__new__(HermesCLI)
    cli.model = "test-model"
    cli.session_start = datetime.now()
    cli.agent = None
    cli.reasoning_config = {"enabled": True, "effort": "medium"}

    for width in (40, 65, 120):
        assert cli._build_status_bar_text(width=width).strip() == "∴ medium"

    for modern in (False, True):
        cli._modern_layout, cli._status_bar_visible = modern, True
        monkeypatch.setattr(HermesCLI, "_get_tui_terminal_width", lambda self: 120)
        fragments = cli._get_status_bar_fragments()
        text = "".join(text for _, text in fragments).strip()
        assert text.startswith("∴ medium")
        assert "test-model" not in text

    for fields in ([], ["model"]):
        assert cli_mod.save_config_value("display.status_bar.fields", fields)
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", cli_mod.load_cli_config())
        del cli._status_bar_field_set_cache
        assert "∴" not in cli._build_status_bar_text(width=120)

    cli._status_bar_field_set_cache = frozenset({"model", "reasoning"})
    assert cli._build_status_bar_text(width=120).strip() == "☤ test-model │ ∴ medium"


def test_reasoning_field_tracks_the_live_agent_without_changing_its_config():
    cli = HermesCLI.__new__(HermesCLI)
    cli.model, cli.provider = "test-model", "test-provider"
    cli.session_start = datetime.now()
    cli._status_bar_field_set_cache = frozenset({"reasoning"})
    cli.reasoning_config = {"enabled": True, "effort": "low"}
    agent = SimpleNamespace(provider="test-provider", reasoning_config=None)
    cli.agent = agent

    for config, expected in (
        ({"enabled": True, "effort": "high"}, "high"),
        ({"enabled": True, "effort": "medium"}, "medium"),
        ({"enabled": True, "effort": "ultra"}, "ultra"),
        ({"enabled": False, "effort": "high"}, "none"),
        (None, "default"),
    ):
        agent.reasoning_config = config
        before = deepcopy(config)
        assert cli._build_status_bar_text(width=120).strip() == f"∴ {expected}"
        assert agent.reasoning_config is config
        assert agent.reasoning_config == before

    agent.provider = "moa"
    assert cli._status_bar_reasoning_level() == ""
    # GatewayAgent._apply_info adopts the session route but retains the MoA preset.
    agent.provider = "test-provider"
    agent._moa_preset = "moa:quality"
    agent.reasoning_config = {"enabled": True, "effort": "high"}
    assert cli._status_bar_reasoning_level() == ""
    cli.agent = None
    assert cli._build_status_bar_text(width=120).strip() == "∴ low"
