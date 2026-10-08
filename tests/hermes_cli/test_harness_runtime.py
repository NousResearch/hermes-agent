"""Harness commands reach all loaders; malformed layers cannot drop valid siblings."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import config, config_effective, harness_manifest as harness


def _homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = (tmp_path / ".hermes", tmp_path / ".hermes" / "profiles" / "work")
    for home in homes:
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(json.dumps({"agent": {"max_turns": 42},
            "compression": {"threshold": 0.5}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[0]))
    monkeypatch.setattr(harness, "stock_revision", lambda: "revision-a")
    config._LOAD_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
    return homes


def _read_limits(home, monkeypatch, explicit=None):
    import cli
    from hermes_cli.cli_config_load import load_cli_config
    from hermes_cli.cli_init_mixin import CLIInitMixin
    monkeypatch.setenv("HERMES_HOME", str(home))
    # A CLI launched under this profile, without provider calls or the interactive loop.
    monkeypatch.setattr(cli, "_hermes_home", home)
    loaded = load_cli_config()
    monkeypatch.setattr(cli, "CLI_CONFIG", loaded)
    instance = SimpleNamespace()
    CLIInitMixin._init_turn_limits(instance, explicit, None)
    return (config.load_config()["agent"]["max_turns"],
            config_effective.load_user_config_effective()["agent"]["max_turns"], instance.max_turns)


@pytest.mark.parametrize("transition", ["profiles", "revision"])
def test_cli_mutations_reach_runtime_across_profile_and_revision_transitions(tmp_path, monkeypatch, transition):
    default, work = _homes(tmp_path, monkeypatch)
    from hermes_cli import main
    parser, _ = main._build_cli_parser()
    def command(*argv):
        args = parser.parse_args(["harness", *argv])
        assert args.func(args) == 0
    command("set", "agent.max_turns", "7", "--overlay", "trial", "--reason", "test")
    assert _read_limits(default, monkeypatch) == (7, 7, 7)
    assert _read_limits(default, monkeypatch, explicit=9) == (7, 7, 9)
    assert json.loads((default / "config.yaml").read_text())["agent"]["max_turns"] == 42
    if transition == "profiles":
        assert _read_limits(work, monkeypatch) == (42, 42, 42)
        command("set", "agent.max_turns", "13", "--overlay", "trial", "--reason", "work")
        assert _read_limits(work, monkeypatch) == (13, 13, 13)
        assert _read_limits(default, monkeypatch) == (7, 7, 7)
    else:
        (default / "config.yaml").write_text("agent: [broken\n", encoding="utf-8")
        assert config.load_config()["agent"]["max_turns"] == 7
        assert config_effective.load_user_config_effective()["agent"]["max_turns"] == 7
        monkeypatch.setattr(harness, "stock_revision", lambda: "revision-b")
        assert config.load_config()["agent"]["max_turns"] == 42
        assert config_effective.load_user_config_effective()["agent"]["max_turns"] == 42
        (default / "config.yaml").write_text("agent:\n  max_turns: 42\n", encoding="utf-8")
        monkeypatch.setattr(harness, "stock_revision", lambda: "revision-a")
        assert _read_limits(default, monkeypatch) == (7, 7, 7)
    command("set", "agent.max_turns", "null", "--overlay", "trial", "--reason", "restore default")
    assert config.load_config()["agent"]["max_turns"] is None
    assert config_effective.load_user_config_effective()["agent"]["max_turns"] is None
    (default / "config.yaml").write_text("agent:\n  max_turns: 222\n", encoding="utf-8")
    command("revert", "trial", "--yes")
    assert _read_limits(default, monkeypatch) == (222, 222, 222)


@pytest.mark.parametrize("bad", ["value", "shape", "duplicate"])
def test_runtime_skips_bad_layer_and_keeps_last_valid_winner(tmp_path, monkeypatch, caplog, bad):
    default, _ = _homes(tmp_path, monkeypatch)
    def layer(name, values):
        return {"id": name, "authored_against": "revision-a", "values": values}
    first = layer("first", {"agent.max_turns": 7})
    invalid = {"value": layer("bad", {"compression.threshold": "not-a-number"}),
               "shape": "not a mapping", "duplicate": layer("first", {"agent.max_turns": 11})}[bad]
    raw = {"schema_version": 1, "stock_revision": "revision-a",
           "overlays": [first, invalid, layer("last", {"agent.max_turns": 13})]}
    (default / "harness.yaml").write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(harness.HarnessManifestError):
        harness.load_manifest()
    with caplog.at_level("WARNING"):
        assert _read_limits(default, monkeypatch) == (13, 13, 13)
    assert "layer" in caplog.text.lower()
    raw["schema_version"] = 999
    (default / "harness.yaml").write_text(json.dumps(raw), encoding="utf-8")
    assert _read_limits(default, monkeypatch) == (42, 42, 42)
