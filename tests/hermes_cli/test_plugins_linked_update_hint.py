"""`hermes plugins update` gives a linked plugin a next step that works.

A plugin dir that is a link to a checkout elsewhere was told to run `hermes plugins adopt`, which
refuses it ("resolves outside the plugins directory") so that Hermes never updates or removes
files outside the plugins directory.
"""
import os

import pytest

from hermes_cli.plugins_provenance import plugins_provenance
from hermes_cli.plugins_updates import check_local_provenance


def _checkout(path):
    (path / ".git").mkdir(parents=True)
    (path / ".git" / "config").write_text('[remote "origin"]\n\turl = https://example.invalid/o/plug.git\n')
    return path


def _reason(plugins_dir):
    (prov,) = plugins_provenance(plugins_dir)
    return check_local_provenance(prov).reason


@pytest.mark.platforms("posix")
def test_linked_checkout_is_told_to_pull_where_it_lives(tmp_path):
    checkout = _checkout(tmp_path / "src" / "plug")
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    os.symlink(checkout, plugins / "plug")

    reason = _reason(plugins)

    assert "adopt" not in reason
    assert str(checkout) in reason


@pytest.mark.platforms("posix")
def test_named_verb_on_a_linked_plugin_says_where_to_update_it(tmp_path, monkeypatch, capsys):
    from hermes_cli.plugins_cmd_update import cmd_adopt

    home = tmp_path / "home"
    checkout = _checkout(tmp_path / "src[bold]" / "plug")  # brackets in a path are text, not markup
    (home / "plugins").mkdir(parents=True)
    os.symlink(checkout, home / "plugins" / "plug")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("COLUMNS", "1000")  # one line, so the path is not wrapped

    with pytest.raises(SystemExit):
        cmd_adopt("plug")

    shown = capsys.readouterr().out
    assert str(checkout) in shown and "Pull updates in that checkout" in shown
