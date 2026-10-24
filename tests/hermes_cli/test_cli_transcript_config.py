"""``display.tool_output_preview`` / ``display.cli_reasoning_style`` reach the classic CLI through
the CLI loader and are registered in DEFAULT_CONFIG (every key has a reader and a registry entry)."""
import hermes_yaml as yaml

from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_new_display_keys_are_registered():
    assert {"tool_output_preview", "cli_reasoning_style"} <= set(DEFAULT_CONFIG["display"])


def test_cli_reads_the_keys_from_config_yaml(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"display": {"tool_output_preview": 7, "cli_reasoning_style": "collapsed"}}), encoding="utf-8")
    import cli
    monkeypatch.setattr(cli, "_hermes_home", home)
    loaded = cli.load_cli_config()["display"]
    assert (loaded["tool_output_preview"], loaded["cli_reasoning_style"]) == (7, "collapsed")


def test_unknown_values_fall_back_to_safe_defaults():
    from tests.hermes_cli.test_cli_init import _make_cli as make_cli
    cli = make_cli(config_overrides={"display": {"tool_output_preview": "lots", "cli_reasoning_style": "sideways"}})
    assert cli._tool_output_preview == DEFAULT_CONFIG["display"]["tool_output_preview"]
    assert cli.cli_reasoning_style == DEFAULT_CONFIG["display"]["cli_reasoning_style"]
    tuned = make_cli(config_overrides={"display": {"tool_output_preview": 0, "cli_reasoning_style": "collapsed"}})
    assert (tuned._tool_output_preview, tuned.cli_reasoning_style) == (0, "collapsed")
