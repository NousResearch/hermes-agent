"""Unattended plugin updates decide at the mutation boundary: a ``--ref`` hold recorded while the check
ran, and a Git update that widens the plugin, are both left for the user (real Git histories, real
catalog installs, real PM publication)."""
from __future__ import annotations

import json

import pytest

from tests.hermes_cli import test_plugin_update_transaction as _tx
from tests.pm import test_plugin_survival_contract as _survival

# Shared fixtures, bound by assignment so the test parameters that request them are not import shadows.
installed, admission_env = _tx.installed, _survival.admission_env
_commit, _head, _version = _tx._commit, _tx._head, _tx._version


def _auto_apply_on(_section, key):
    return True if key == "auto_apply" else 24


@pytest.mark.parametrize("installed", ["catalog"], indirect=True)
@pytest.mark.parametrize("path", ["batch", "tick"])
def test_hold_recorded_during_the_check_survives_the_unattended_apply(installed, monkeypatch, path):
    """The user finishes `install --force --ref <older>` after the check snapshotted the install records;
    neither Update all nor the auto-apply tick may then re-pin it to the catalog."""
    from hermes_cli import plugin_catalog, plugins_cadence, plugins_updates
    from hermes_cli.plugins_auto_update import apply_unattended, update_all_unattended
    from hermes_cli.plugins_cmd_catalog import install_catalog_entry

    _root, _home, repo, target, state = installed
    first = state["sha"]
    state["sha"] = _version(repo, "2.0.0")
    real_checks = plugins_updates.run_checks

    def check_then_rollback(plugins_dir, **kwargs):
        results = real_checks(plugins_dir, **kwargs)
        assert any(r.name == "transactional" and r.update_available is True for r in results), results
        entry = plugin_catalog.load_catalog()[0]
        install_catalog_entry(entry, force=True, ref=first, assume_deps_consent=True)  # the hold lands between check and apply
        return results

    if path == "batch":
        monkeypatch.setattr(plugins_updates, "run_checks", check_then_rollback)
        outcome = update_all_unattended()
        assert outcome["results"] == [], outcome
        assert any(s["name"] == "transactional" and "--ref" in s["reason"] for s in outcome["skipped"]), outcome
    else:
        plugins_cadence.run_scheduled_check(run_checks_fn=check_then_rollback, plugins_dir=target.parent,
                                            apply_updates_fn=apply_unattended, config_get=_auto_apply_on)
    assert _head(target) == first, "an unattended update undid a --ref hold recorded during the check"
    record = json.loads((target.parent / ".install-metadata.json").read_text(encoding="utf-8"))["transactional"]
    assert record["catalog"]["pin"] == "" and record["pinned"] is True


@pytest.mark.parametrize("installed", ["custom"], indirect=True)
@pytest.mark.parametrize("path", ["batch", "tick"])
def test_unattended_git_update_that_adds_a_tool_waits_for_review(installed, monkeypatch, path):
    """A tracked Git update that declares a new tool is a widened plugin: Update all reports it as
    needs_review and the tick logs a pending review, with the installed tree unchanged. A named update
    still moves it."""
    from hermes_cli import plugins_cadence
    from hermes_cli.plugins_auto_update import apply_unattended, update_all_unattended
    from hermes_cli.plugins_cmd_update import dashboard_update_user_plugin

    _root, _home, repo, target, _state = installed
    old_head = _head(target)
    _version(repo, "2.0.0")
    manifest = repo / "plugin.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "provides_tools: [new_tool]\n", encoding="utf-8")
    widened = _commit(repo, "adds new_tool")

    if path == "batch":
        outcome = update_all_unattended()
        assert [(r["name"], r["status"]) for r in outcome["results"]] == [("transactional", "needs_review")], outcome
        assert "new_tool" in outcome["results"][0]["detail"]
    else:
        plugins_cadence.run_scheduled_check(run_checks_fn=_real_checks(), plugins_dir=target.parent,
                                            apply_updates_fn=apply_unattended, config_get=_auto_apply_on)
    assert _head(target) == old_head, "an unattended update published a plugin with a newly added tool"

    assert dashboard_update_user_plugin("transactional")["ok"] is True
    assert _head(target) == widened


def _real_checks():
    from hermes_cli.plugins_updates import run_checks
    return run_checks


def test_slash_update_with_an_argument_does_not_update_every_plugin(monkeypatch):
    """`/plugins update <name>` mirrors `hermes plugins update <name>`; it must not run Update all on every
    plugin (in the TUI it would also run inside the 45 s slash worker the batch was moved out of)."""
    from hermes_cli import plugins_auto_update

    calls = []
    monkeypatch.setattr(plugins_auto_update, "update_all_unattended",
                        lambda: calls.append(1) or {"results": [], "skipped": [], "current": 0})
    for parts in (["update", "my-plugin"], ["update", "all"], ["update", "--yes"]):
        assert plugins_auto_update.plugins_slash_action(parts)[0].startswith("Usage:")
    assert calls == []
    plugins_auto_update.plugins_slash_action(["update"])
    assert calls == [1]


def test_cli_update_all_skips_a_catalog_install_held_with_ref(tmp_path, monkeypatch, capsys):
    """`hermes plugins update` (the command `hermes update` recommends) leaves a `--ref` hold where the user
    put it, like every other Update all; naming the plugin is what moves it."""
    from hermes_cli import plugins_cmd, plugins_cmd_update, plugins_updates
    from hermes_cli.plugins_cmd_update_all import cmd_update_all
    from tests.hermes_cli.test_plugins_auto_update import _Row, _held_install_metadata

    home = tmp_path / "hermes-home"
    (home / "plugins" / "held").mkdir(parents=True)
    (home / "plugins" / "behind").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _held_install_metadata(home)
    monkeypatch.setattr(plugins_updates, "run_checks", lambda _dir: [_Row("held", "catalog"), _Row("behind", "catalog")])
    moved = []
    monkeypatch.setattr(plugins_cmd_update, "update_one", lambda name, *_a, **_k: moved.append(name) or True)
    monkeypatch.setattr(plugins_cmd, "_is_tty", lambda: False)
    cmd_update_all(yes=True)
    assert moved == ["behind"]
    assert "skipped held" in capsys.readouterr().out
