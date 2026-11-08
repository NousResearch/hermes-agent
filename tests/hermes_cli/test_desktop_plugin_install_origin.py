"""A standalone desktop plugin installed from a catalog entry records that entry on the Webapp host.

Without it the catalog card never matched the installed plugin (it kept its Install button) and the
next Install stopped at "already exists" (Accent Picker, reported on X 2026-11-07). Same real-Git
fixtures as ``test_desktop_plugin_install.py``.
"""

import json
from pathlib import Path

from hermes_cli import desktop_plugin_halves as halves
from hermes_cli import desktop_plugin_install as install
from tests.hermes_cli.test_desktop_plugin_install import URL, _catalog, _commit, _git

# The sibling's real-Git fixtures: `repo`, a local repository the https identifier clones from, and `home`.
pytest_plugins = ["tests.hermes_cli.test_desktop_plugin_install"]

DESKTOP_ONLY = {"plugin.yaml": "name: gizmo\n", "desktop/plugin.js": "export default { id: 'gizmo' }\n"}


def _folder(home: Path) -> Path:
    return home / "desktop-plugins" / "gizmo"


def test_a_catalog_install_records_its_entry_and_stays_standalone(home, repo, monkeypatch):
    sha = _commit(repo, DESKTOP_ONLY)
    _catalog(monkeypatch, sha)

    result = install.install_desktop_plugin(URL, ref=sha, catalog_name="gizmo")

    assert result["ok"] is True, result
    assert result["paired_with_agent"] is False
    origin = json.loads((_folder(home) / halves.INSTALL_ORIGIN).read_text(encoding="utf-8"))
    assert origin == {"catalogName": "gizmo", "repo": URL, "sha": sha}
    assert not (_folder(home) / halves.PACKAGE_MARKER).exists()


def test_a_git_url_install_records_no_entry_and_drops_one_the_repo_ships(home, repo):
    forged = json.dumps({"catalogName": "trusted", "repo": "https://x", "sha": "a" * 40})
    sha = _commit(repo, {**DESKTOP_ONLY, f"desktop/{halves.INSTALL_ORIGIN}": forged})

    result = install.install_desktop_plugin(URL, ref=sha)

    assert result["ok"] is True, result
    assert not (_folder(home) / halves.INSTALL_ORIGIN).exists()


def test_a_catalog_install_adopts_an_earlier_install_of_the_same_tree(home, repo, monkeypatch):
    sha = _commit(repo, DESKTOP_ONLY)
    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True
    (_folder(home) / "state.json").write_text("kept\n", encoding="utf-8")
    _catalog(monkeypatch, sha)

    result = install.install_desktop_plugin(URL, ref=sha, catalog_name="gizmo")

    assert result["ok"] is True, result
    assert json.loads((_folder(home) / halves.INSTALL_ORIGIN).read_text(encoding="utf-8"))["catalogName"] == "gizmo"
    assert (_folder(home) / "state.json").read_text(encoding="utf-8") == "kept\n"


def test_a_different_plugin_of_that_name_still_needs_force(home, repo, monkeypatch):
    sha = _commit(repo, DESKTOP_ONLY)
    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True
    (_folder(home) / "plugin.js").write_text("export default { id: 'edited' }\n", encoding="utf-8")
    _catalog(monkeypatch, sha)

    result = install.install_desktop_plugin(URL, ref=sha, catalog_name="gizmo")

    assert result["ok"] is False
    assert result["exists"] is True
    assert "already exists" in result["error"]
    assert not (_folder(home) / halves.INSTALL_ORIGIN).exists()
    assert "edited" in (_folder(home) / "plugin.js").read_text(encoding="utf-8")


def test_a_package_marker_link_never_reads_as_a_standalone_folder(home, repo, monkeypatch):
    # Electron checks the marker with lstat: a dangling marker link is still a marker, so not adoptable.
    sha = _commit(repo, DESKTOP_ONLY)
    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True
    (_folder(home) / halves.PACKAGE_MARKER).symlink_to(home / "missing.json")
    _catalog(monkeypatch, sha)

    result = install.install_desktop_plugin(URL, ref=sha, catalog_name="gizmo")

    assert result["ok"] is False and result["exists"] is True, result
    assert not (_folder(home) / halves.INSTALL_ORIGIN).exists()


def test_a_tag_pin_records_the_commit_and_the_tag_it_satisfies(home, repo, monkeypatch):
    commit = _commit(repo, DESKTOP_ONLY)
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "tag", "-a", "v1", "-m", "v1")
    tag = _git(repo, "rev-parse", "v1")
    assert tag != commit
    _catalog(monkeypatch, tag)

    result = install.install_desktop_plugin(URL, ref=commit, catalog_name="gizmo")

    assert result["ok"] is True, result
    origin = json.loads((_folder(home) / halves.INSTALL_ORIGIN).read_text(encoding="utf-8"))
    assert origin == {"catalogName": "gizmo", "repo": URL, "sha": commit, "pin": tag}


def test_the_gateway_marks_an_already_exists_refusal_so_the_dialog_offers_force():
    from tui_gateway.methods_tools import _refusal_data

    assert _refusal_data({"ok": False, "error": "x", "exists": True}) == {"reason": "exists"}
    assert _refusal_data({"ok": False, "error": "x", "source_moved": True}) == {"reason": "source_moved"}
    assert _refusal_data({"ok": False, "error": "x"}) is None
