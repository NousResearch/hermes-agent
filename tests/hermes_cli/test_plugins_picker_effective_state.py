"""The bare ``hermes plugins`` picker writes only the rows the user flipped.

Bundled platforms are active without a ``plugins.enabled`` entry; the picker used to open them unticked
and save every unticked row into ``plugins.disabled`` on any exit, so opening and closing it silenced
Telegram and Slack on the next gateway restart. Drives the real ``cmd_toggle`` (text fallback) through
real admission.
"""
import pytest
import hermes_yaml as yaml

from tests.hermes_cli.plugin_worker_support import (
    plugin_world as plugin_world,
    isolated_python as isolated_python,
)


@pytest.fixture
def run_picker(plugin_world, monkeypatch):
    from hermes_cli import plugins_cmd

    world = plugin_world
    for leaf in ("telegram", "slack"):
        platform = world.core / "plugins" / "platforms" / leaf
        platform.mkdir(parents=True)
        (platform / "plugin.yaml").write_text(f"name: {leaf}-platform\nkind: platform\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.plugins.get_bundled_plugins_dir", lambda: world.core / "plugins")
    monkeypatch.setattr(plugins_cmd, "_provider_categories", lambda: [])
    monkeypatch.setattr(plugins_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setitem(plugins_cmd.sys.modules, "curses", None)
    config_path = world.home / "config.yaml"

    def _run(plugins: dict, toggle_keys=()):
        config_path.write_text(yaml.safe_dump({"plugins": plugins}), encoding="utf-8")
        keys = [entry[5] for entry in plugins_cmd._discover_all_plugins()]
        answers = iter([str(keys.index(k) + 1) for k in toggle_keys] + [""])
        monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
        plugins_cmd.cmd_toggle()
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))["plugins"]

    return _run


def test_open_and_exit_without_changes_writes_nothing(run_picker):
    assert run_picker({"enabled": [], "disabled": []}) == {"enabled": [], "disabled": []}


def test_only_flipped_rows_are_written(run_picker):
    # Telegram is disabled under its canonical key AND its legacy manifest name; ticking it must clear
    # both. Slack is unticked. The enabled entry for a plugin the picker never shows must survive.
    plugins = run_picker(
        {"enabled": ["some/unshown-plugin"], "disabled": ["platforms/telegram", "telegram-platform"]},
        toggle_keys=("platforms/telegram", "platforms/slack"),
    )
    assert plugins["disabled"] == ["platforms/slack"]
    assert sorted(plugins["enabled"]) == ["platforms/telegram", "some/unshown-plugin"]
