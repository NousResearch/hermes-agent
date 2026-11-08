"""A standalone desktop plugin installed from a catalog entry records that entry on the Webapp host.

Without it the catalog card never matched the installed plugin (it kept its Install button) and the
next Install stopped at "already exists" (Accent Picker, reported on X 2026-11-07). Same real-Git
fixtures as ``test_desktop_plugin_install.py``.
"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import desktop_plugin_halves as halves
from hermes_cli import desktop_plugin_install as install

URL = "https://example.invalid/acme/gizmo.git"
DESKTOP_ONLY = {"plugin.yaml": "name: gizmo\n", "desktop/plugin.js": "export default { id: 'gizmo' }\n"}


def _git(path: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def _commit(path: Path, files: dict[str, str]) -> str:
    for rel, text in files.items():
        target = path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "c")
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture()
def home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import hermes_constants
    hermes_constants._default_hermes_root_memo = None
    return hermes_home


@pytest.fixture()
def repo(tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd
    path = tmp_path / "gizmo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    real = plugins_cmd._clone_plugin_repo
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo",
                        lambda dest, _url, revision, subdir=None: real(dest, path.as_uri(), revision, subdir))
    return path


def _catalog(monkeypatch, sha: str) -> None:
    from hermes_cli import plugins_cmd_catalog
    entry = SimpleNamespace(name="gizmo", install_identifier=URL, sha=sha)
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda name: entry if name == "gizmo" else None)


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


def test_the_gateway_marks_an_already_exists_refusal_so_the_dialog_offers_force():
    from tui_gateway.methods_tools import _refusal_data

    assert _refusal_data({"ok": False, "error": "x", "exists": True}) == {"reason": "exists"}
    assert _refusal_data({"ok": False, "error": "x", "source_moved": True}) == {"reason": "source_moved"}
    assert _refusal_data({"ok": False, "error": "x"}) is None
