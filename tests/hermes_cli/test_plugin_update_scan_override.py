"""Updating a git-installed plugin scans the new code on every surface, and a caution verdict can be
overridden: ``hermes plugins update NAME`` asks ``Update anyway? [y/N]`` (``--force`` skips the question),
and Desktop/dashboard get the findings back and retry with ``force``. Before this, update blocked a caution
verdict with no way past it and told the user to pass a ``--force`` flag the command did not have."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.hermes_cli import test_plugin_update_transaction as _tx
from tests.pm import test_plugin_survival_contract as _survival

# Shared fixtures, bound by assignment so the test parameters that request them are not import shadows.
installed, admission_env = _tx.installed, _survival.admission_env
_commit, _head, _version = _tx._commit, _tx._head, _tx._version


def _caution_release(repo) -> str:
    """A new upstream commit the real scanner rates caution (not dangerous)."""
    _version(repo, "2.0.0")
    (repo / "helper.py").write_text("eval('1 + 1')\n", encoding="utf-8")
    return _commit(repo, "caution release")


@pytest.mark.parametrize("installed", ["custom"], indirect=True)
def test_desktop_update_returns_findings_then_force_applies(installed, monkeypatch):
    from hermes_cli import plugins_cmd as pc
    from tui_gateway import server

    _, _home, repo, target, _state = installed
    old_head = _head(target)
    new_head = _caution_release(repo)
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: True)

    def update(**params):
        resp = server.handle_request({"id": "1", "method": "plugins.manage",
                                      "params": {"action": "update", "name": "transactional", **params}})
        assert "error" not in resp, resp
        return resp["result"]

    blocked = update()
    assert blocked["ok"] is False and blocked["scan_blocked"] is True
    assert blocked["scan_verdict"] == "caution"
    assert any(f["file"] == "helper.py" for f in blocked["scan_findings"])
    assert _head(target) == old_head, "a blocked update changed the installed tree"

    applied = update(force=True)
    assert applied["ok"] is True and applied["unchanged"] is False, applied
    assert _head(target) == new_head


@pytest.mark.parametrize("installed", ["custom"], indirect=True)
@pytest.mark.parametrize("argv_force, answer, published", [
    (False, "y", True), (False, "n", False), (True, None, True)])
def test_cli_update_asks_on_caution_and_force_skips_the_question(installed, monkeypatch, argv_force, answer,
                                                                  published):
    from hermes_cli import plugins_cmd as pc

    _, _home, repo, target, _state = installed
    old_head = _head(target)
    new_head = _caution_release(repo)
    monkeypatch.setattr(pc, "_scan_on_install_enabled", lambda: True)
    monkeypatch.setattr(pc, "_is_tty", lambda: True)
    asked: list[str] = []
    monkeypatch.setattr(pc, "_ask_yes", lambda prompt, reader=None: asked.append(prompt) or answer == "y")

    args = SimpleNamespace(plugins_action="update", name="transactional", yes=False, force=argv_force)
    if published:
        pc.plugins_command(args)
    else:
        with pytest.raises(SystemExit):
            pc.plugins_command(args)

    scan_prompts = [p for p in asked if "Update anyway?" in p]
    assert bool(scan_prompts) is (not argv_force), asked
    assert _head(target) == (new_head if published else old_head)


def test_desktop_rows_offer_a_git_update_only_while_the_check_still_matches_the_install():
    """The Plugins tab shows a git plugin's Update button from the last update check, with no network call
    per listing; once the plugin moved past the revision the check compared, the stale result is ignored."""
    from hermes_cli.plugins_cadence import cached_git_updates, record_check_results
    from hermes_cli.plugins_updates import CheckResult

    record_check_results([
        CheckResult(name="tern", klass="git", current="a" * 40, latest="b" * 40, update_available=True),
        CheckResult(name="spotify", klass="catalog", current="c" * 40, latest="d" * 40, update_available=True),
    ])

    assert cached_git_updates({"tern": {"revision": "a" * 40}}) == {"tern": "b" * 40}
    assert cached_git_updates({"tern": {"revision": "b" * 40}}) == {}
