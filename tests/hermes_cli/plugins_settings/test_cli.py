"""``hermes plugins config``: a plugin's manifest alone gives it a terminal settings editor.

Drives the real discovery, config writer and ``.env`` credential route against a temp HERMES_HOME.
"""

import pytest

from hermes_cli.plugins_settings import cli

MANIFEST = """\
name: demo-plugin
version: 1.0.0
requires_env:
  - {name: DEMO_SERVICE_URL, description: "Where the service runs"}
  - {name: DEMO_SERVICE_TOKEN, secret: true}
config_schema:
  retries: {type: int, default: 3, label: Retries}
  mode: {type: str, choices: [fast, careful], default: fast}
  api_key: {type: secret, env: DEMO_API_KEY}
"""


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "plugins" / "demo-plugin").mkdir(parents=True)
    (home / "plugins" / "demo-plugin" / "plugin.yaml").write_text(MANIFEST, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("DEMO_SERVICE_URL", "DEMO_SERVICE_TOKEN", "DEMO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return home


def test_edits_land_where_the_plugin_reads_them_and_secrets_never_reach_config(plugin_home):
    plugin_dir = plugin_home / "plugins" / "demo-plugin"
    fields = {f["key"]: f for f in cli.collect_fields("demo-plugin", plugin_dir, {"requires_env": [
        {"name": "DEMO_SERVICE_URL"}, {"name": "DEMO_SERVICE_TOKEN", "secret": True}]})}
    assert set(fields) == {"retries", "mode", "api_key", "DEMO_SERVICE_URL", "DEMO_SERVICE_TOKEN"}

    cli.save_field("demo-plugin", plugin_dir, fields["retries"], 7)
    cli.save_field("demo-plugin", plugin_dir, fields["api_key"], "sk-demo-not-real")
    cli.save_field("demo-plugin", plugin_dir, fields["DEMO_SERVICE_TOKEN"], "tok-demo-not-real")
    with pytest.raises(ValueError):
        cli.save_field("demo-plugin", plugin_dir, fields["mode"], "reckless")

    from hermes_cli.config import load_config_readonly, load_env
    assert load_config_readonly()["plugins"]["entries"]["demo-plugin"]["settings"] == {"retries": 7}
    env = load_env()
    assert env["DEMO_API_KEY"] == "sk-demo-not-real" and env["DEMO_SERVICE_TOKEN"] == "tok-demo-not-real"
    config_text = (plugin_home / "config.yaml").read_text(encoding="utf-8")
    assert "not-real" not in config_text


def test_without_a_terminal_it_lists_settings_and_never_prints_a_secret(plugin_home, monkeypatch, capsys):
    monkeypatch.setenv("DEMO_API_KEY", "sk-demo-not-real")
    monkeypatch.setenv("DEMO_SERVICE_TOKEN", "tok-demo-not-real")  # masked by name, like a platform's BOT_TOKEN
    (plugin_home / "plugins" / "demo-plugin" / "plugin.yaml").write_text(
        MANIFEST.replace("{name: DEMO_SERVICE_TOKEN, secret: true}", "{name: DEMO_SERVICE_TOKEN}"), encoding="utf-8")
    cli.cmd_config("demo-plugin")
    out = capsys.readouterr().out

    assert "Retries: 3" in out and "plugins.entries.demo-plugin.settings.retries" in out
    assert "Mode: fast" in out and ".env DEMO_API_KEY" in out and "DEMO_SERVICE_URL: not set" in out
    assert "API key: set" in out and "DEMO_SERVICE_TOKEN: set" in out and "not-real" not in out


def test_list_values_are_json_safe_and_bad_input_is_a_readable_refusal():
    assert cli.parse_json_value("[a, 2, {k: true}]") == ["a", 2, {"k": True}]
    for raw in ("[2026-11-20]", "[a, b", "plain text"):
        with pytest.raises(ValueError):
            cli.parse_json_value(raw)


REQUIRED_KEY = MANIFEST.replace(
    "api_key: {type: secret, env: DEMO_API_KEY}", "api_key: {type: secret, env: DEMO_API_KEY, required: true}")


@pytest.mark.parametrize("key_value, answer, opens", [
    ("sk-demo-not-real", "y", True), ("sk-demo-not-real", "", False),  # nothing missing: default No
    (None, "", True), ("", "", True), (None, "n", False),               # required key empty or blank: default Yes
    (None, EOFError, False),                                            # Ctrl-D / Ctrl-C declines
])
def test_install_offer_defaults_to_no_unless_a_required_setting_is_empty(
        plugin_home, monkeypatch, capsys, key_value, answer, opens):
    from rich.console import Console
    from hermes_cli import plugins_cmd
    plugin_dir = plugin_home / "plugins" / "demo-plugin"
    (plugin_dir / "plugin.yaml").write_text(REQUIRED_KEY, encoding="utf-8")
    if key_value is not None:
        monkeypatch.setenv("DEMO_API_KEY", key_value)
    monkeypatch.setattr(plugins_cmd, "_is_tty", lambda: True)
    opened = []
    monkeypatch.setattr(cli, "run_editor", lambda *a, **k: opened.append(a[0]))

    def reply(_prompt):
        if answer is EOFError:
            raise EOFError
        return answer
    monkeypatch.setattr("builtins.input", reply)
    cli.offer_settings_after_install(Console(), "demo-plugin", plugin_dir)
    out = capsys.readouterr().out
    assert opened == (["demo-plugin"] if opens else [])
    assert ("Still needed: API key" in out) == (not key_value)


@pytest.mark.parametrize("brief, noninteractive, tty", [(False, False, False), (True, False, True), (False, True, True)])
def test_install_offer_never_reads_input_without_a_person(plugin_home, monkeypatch, capsys, brief, noninteractive, tty):
    from rich.console import Console
    from hermes_cli import plugins_cmd
    if noninteractive:
        monkeypatch.setenv("HERMES_NONINTERACTIVE", "1")
    monkeypatch.setattr(plugins_cmd, "_is_tty", lambda: tty)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("a non-interactive install must not prompt"))
    cli.offer_settings_after_install(Console(), "demo-plugin", plugin_home / "plugins" / "demo-plugin", brief=brief)
    assert "hermes plugins config demo-plugin" in capsys.readouterr().out
