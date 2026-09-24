"""The bare ``hermes plugins`` picker must not disable plugins nobody unticked.

Regression for the fleet report "bare `hermes plugins` picker writes every bundled default-on plugin
into plugins.disabled": bundled platforms/backends/model providers are active without a
``plugins.enabled`` entry, the picker preselected only listed plugins, and save-on-exit wrote every
unticked row into ``plugins.disabled``. Opening and closing the picker silenced Telegram and Slack on
the next gateway restart. Runs against the real bundled tree in a temp HERMES_HOME, no mocks of the
functions under test.
"""
import logging

import pytest

from hermes_cli import plugins_cmd


@pytest.fixture
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "plugins").mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("plugins:\n  enabled: []\n  disabled: []\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _picker_state():
    entries = plugins_cmd._discover_all_plugins()
    enabled, disabled = plugins_cmd._get_enabled_set(), plugins_cmd._get_disabled_set()
    keys = [e[5] for e in entries]
    selected = plugins_cmd._effective_plugin_selection(entries, enabled, disabled)
    return entries, keys, selected, disabled


def test_bundled_default_on_plugins_open_ticked(home):
    entries, keys, selected, _ = _picker_state()
    assert "platforms/telegram" in keys and "platforms/slack" in keys
    for key in ("platforms/telegram", "platforms/slack"):
        assert keys.index(key) in selected, f"{key} is active without a list entry and must open ticked"
    # Nothing is listed, so every ticked row is a bundled default-on kind; nothing user-installed shows ticked.
    assert all(entries[i][3] == "bundled" for i in selected)


def test_open_and_exit_without_changes_writes_nothing(home):
    _, keys, selected, disabled = _picker_state()
    changed, _ = plugins_cmd._persist_plugin_selection(keys, set(selected), disabled, selected)
    assert changed is False
    assert plugins_cmd._get_disabled_set() == set()
    assert plugins_cmd._get_enabled_set() == set()


def test_unticking_one_row_disables_exactly_that_plugin(home, caplog):
    _, keys, selected, disabled = _picker_state()
    target = keys.index("platforms/slack")
    chosen = set(selected) - {target}
    with caplog.at_level(logging.INFO, logger="hermes_cli.plugins_cmd"):
        changed, _ = plugins_cmd._persist_plugin_selection(keys, chosen, disabled, selected)
    assert changed is True
    assert plugins_cmd._get_disabled_set() == {"platforms/slack"}
    assert plugins_cmd._get_enabled_set() == set()  # untouched rows are not re-persisted
    assert any("platforms/slack" in r.getMessage() for r in caplog.records)  # attributable from the log


def test_ticking_one_row_enables_it_and_clears_a_stale_disable(home):
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [some/unshown-plugin]\n  disabled: [platforms/slack, slack]\n", encoding="utf-8")
    _, keys, selected, disabled = _picker_state()
    slack = keys.index("platforms/slack")
    assert slack not in selected  # explicit disable wins on open
    changed, _ = plugins_cmd._persist_plugin_selection(keys, set(selected) | {slack}, disabled, selected)
    assert changed is True
    assert plugins_cmd._get_disabled_set() == set()  # canonical key and legacy leaf both cleared
    # The enabled entry for a plugin the picker never showed survives; the re-ticked key is added.
    assert plugins_cmd._get_enabled_set() == {"some/unshown-plugin", "platforms/slack"}


def test_unticked_bundled_platform_is_gated_off_at_load(home):
    """The persisted disable is what gate_manifest honours: unticking a platform really turns it off."""
    _, keys, selected, disabled = _picker_state()
    target = keys.index("platforms/telegram")
    plugins_cmd._persist_plugin_selection(keys, set(selected) - {target}, disabled, selected)
    entries, keys2, selected2, _ = _picker_state()
    assert keys2.index("platforms/telegram") not in selected2
    assert keys2.index("platforms/slack") in selected2
