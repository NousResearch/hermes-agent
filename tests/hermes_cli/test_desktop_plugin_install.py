"""``hermes_cli.desktop_plugin_install``: cloning a desktop plugin onto the host that serves the Webapp.

Real local Git repositories behind an https identifier (only the clone's network leg is pointed at the
local repository), a throwaway ``HERMES_HOME``: the folder lands where the Webapp's read routes serve
it, at exactly the commit the dialog reviewed; a package's half carries the marker that pairs it to its
agent row, a standalone plugin carries none, and nothing installs twice without ``force``.
"""

import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import desktop_plugin_install as install

URL = "https://example.invalid/acme/gizmo.git"


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
    """A local repository the https identifier clones from."""
    from hermes_cli import plugins_cmd
    path = tmp_path / "gizmo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    real = plugins_cmd._clone_plugin_repo
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo",
                        lambda dest, _url, revision, subdir=None: real(dest, path.as_uri(), revision, subdir))
    return path


def _root(home: Path) -> Path:
    return home / "desktop-plugins"


def test_desktop_only_repo_lands_under_its_own_name_without_a_marker(home, repo):
    # The repository ships its own package marker naming another package: only an installer writes one.
    sha = _commit(repo, {"plugin.js": "export default { id: 'gizmo' }\n", "panel.js": "export const x = 1\n",
                         ".hermes-package.json": json.dumps({"package": "victim", "source": "/x"})})

    result = install.install_desktop_plugin(URL, ref=sha)

    assert result["ok"] is True, result
    assert (result["plugin_name"], result["paired_with_agent"], result["sha"]) == ("gizmo", False, sha)
    folder = _root(home) / "gizmo"
    assert (folder / "plugin.js").read_text(encoding="utf-8").startswith("export default")
    assert (folder / "panel.js").is_file()
    assert not (folder / ".hermes-package.json").exists()


def test_package_lands_its_desktop_half_under_the_agent_name_with_a_marker(home, repo):
    sha = _commit(repo, {"plugin.yaml": "name: pkg\nversion: 1.0.0\n", "__init__.py": "",
                         "desktop/plugin.js": "export default { id: 'pkg' }\n"})

    result = install.install_desktop_plugin(URL, ref=sha)

    assert result["ok"] is True, result
    assert (result["plugin_name"], result["paired_with_agent"]) == ("pkg", True)
    folder = _root(home) / "pkg"
    assert (folder / "plugin.js").is_file()
    # Only the desktop half is copied: the agent payload stays in the repository.
    assert not (folder / "plugin.yaml").exists()
    marker = json.loads((folder / ".hermes-package.json").read_text(encoding="utf-8"))
    assert (marker["package"], marker["repo"], marker["source"]) == ("pkg", URL, str(folder))
    assert marker["sourceMtimeMs"] > 0


def test_the_install_is_the_reviewed_commit_and_nothing_else(home, repo):
    reviewed = _commit(repo, {"plugin.js": "export default { v: 1 }\n"})
    _commit(repo, {"plugin.js": "export default { v: 2 }\n"})

    assert install.install_desktop_plugin(URL, ref=None)["ok"] is False
    assert install.install_desktop_plugin(URL, ref="1" * 40)["ok"] is False
    assert not _root(home).joinpath("gizmo").exists()

    result = install.install_desktop_plugin(URL, ref=reviewed)

    assert result["ok"] is True, result
    assert "v: 1" in (_root(home) / "gizmo" / "plugin.js").read_text(encoding="utf-8")


def test_a_second_install_refuses_until_force_replaces_it(home, repo):
    first = _commit(repo, {"plugin.js": "export default { v: 1 }\n"})
    assert install.install_desktop_plugin(URL, ref=first)["ok"] is True

    again = install.install_desktop_plugin(URL, ref=first)
    assert again["ok"] is False
    assert "already exists" in again["error"] and "force" in again["error"]

    second = _commit(repo, {"plugin.js": "export default { v: 2 }\n"})
    assert install.install_desktop_plugin(URL, ref=second, force=True)["ok"] is True
    assert "v: 2" in (_root(home) / "gizmo" / "plugin.js").read_text(encoding="utf-8")


def test_only_an_https_source_is_installed(home, repo):
    sha = _commit(repo, {"plugin.js": "export default {}\n"})

    for source in (repo.as_uri(), "http://example.invalid/acme/gizmo.git", "git@example.invalid:acme/gizmo.git"):
        result = install.install_desktop_plugin(source, ref=sha)
        assert result["ok"] is False and "https://" in result["error"], (source, result)
    assert not _root(home).joinpath("gizmo").exists()


def test_a_link_out_of_the_clone_is_refused(home, repo, tmp_path):
    outside = tmp_path / "secrets"
    outside.mkdir()
    (outside / "plugin.js").write_text("// any folder of the host\n", encoding="utf-8")
    (repo / "desktop").symlink_to(outside, target_is_directory=True)
    sha = _commit(repo, {"README.md": "x\n"})

    result = install.install_desktop_plugin(URL, ref=sha)

    assert result["ok"] is False and "link" in result["error"], result
    assert not _root(home).joinpath("gizmo").exists()


def test_a_repository_with_no_desktop_half_is_refused(home, repo):
    sha = _commit(repo, {"plugin.yaml": "name: agently\n", "__init__.py": ""})

    result = install.install_desktop_plugin(URL, ref=sha)

    assert result["ok"] is False
    assert "no desktop plugin" in result["error"]
    assert not _root(home).joinpath("gizmo").exists()


def test_a_name_that_is_a_path_is_refused(home, repo):
    sha = _commit(repo, {"plugin.yaml": "name: ../escape\n", "__init__.py": "",
                         "desktop/plugin.js": "export default {}\n"})

    result = install.install_desktop_plugin(URL, ref=sha)

    assert result["ok"] is False
    assert "folder name" in result["error"]
    assert not (home / "escape").exists()


def test_remove_takes_a_standalone_plugin_away(home, repo):
    sha = _commit(repo, {"plugin.js": "export default {}\n"})
    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True

    result = install.remove_desktop_plugin("gizmo")

    assert result == {"ok": True, "plugin_name": "gizmo", "path": str(_root(home) / "gizmo"),
                      "unpaired_from": None}
    assert not (_root(home) / "gizmo").exists()


def test_remove_refuses_a_packages_own_half_until_forced(home, repo):
    sha = _commit(repo, {"plugin.yaml": "name: pkg\n", "__init__.py": "", "desktop/plugin.js": "export default {}\n"})
    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True

    refused = install.remove_desktop_plugin("pkg")
    assert refused["ok"] is False
    assert "uninstall that plugin instead" in refused["error"]
    assert (_root(home) / "pkg" / "plugin.js").is_file()

    forced = install.remove_desktop_plugin("pkg", force=True)
    assert (forced["ok"], forced["unpaired_from"]) == (True, "pkg")
    assert not (_root(home) / "pkg").exists()


def test_remove_leaves_a_linked_plugin_alone_and_names_nothing_outside_the_root(home, tmp_path):
    checkout = tmp_path / "dev"
    checkout.mkdir()
    (checkout / "plugin.js").write_text("export default {}\n", encoding="utf-8")
    root = _root(home)
    root.mkdir(parents=True)
    (root / "dev").symlink_to(checkout, target_is_directory=True)

    linked = install.remove_desktop_plugin("dev")
    assert linked["ok"] is False and "linked" in linked["error"]
    assert (root / "dev" / "plugin.js").is_file()

    assert "folder name" in install.remove_desktop_plugin("../config.yaml")["error"]
    assert "No desktop plugin named 'ghost'" in install.remove_desktop_plugin("ghost")["error"]


def _catalog(monkeypatch, sha: str) -> None:
    from hermes_cli import plugins_cmd_catalog
    entry = SimpleNamespace(install_identifier=URL, sha=sha)
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda name: entry if name == "gizmo" else None)


def test_a_catalog_pin_naming_an_annotated_tag_installs_the_commit_it_points_at(home, repo, monkeypatch):
    # The catalog publishes `git rev-parse v1.0` (the tag object); inspect reports the commit it peels to.
    reviewed = _commit(repo, {"plugin.js": "export default { v: 1 }\n"})
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "tag", "-a", "v1.0", reviewed, "-m", "v1.0")
    _commit(repo, {"plugin.js": "export default { v: 2 }\n"})
    tag_object = _git(repo, "rev-parse", "v1.0")
    assert tag_object != reviewed
    _catalog(monkeypatch, tag_object)

    result = install.install_desktop_plugin(URL, ref=reviewed, catalog_name="gizmo")

    assert result["ok"] is True, result
    assert "v: 1" in (_root(home) / "gizmo" / "plugin.js").read_text(encoding="utf-8")


def test_a_catalog_pin_that_moved_since_the_review_is_refused(home, repo, monkeypatch):
    reviewed = _commit(repo, {"plugin.js": "export default { v: 1 }\n"})
    _catalog(monkeypatch, _commit(repo, {"plugin.js": "export default { v: 2 }\n"}))

    result = install.install_desktop_plugin(URL, ref=reviewed, catalog_name="gizmo")

    assert result["ok"] is False and "inspect it again" in result["error"], result
    assert result.get("source_moved") is True  # the client re-reads the source instead of resending
    assert not (_root(home) / "gizmo").exists()


def test_the_published_plugin_carries_no_git_metadata(home, repo):
    sha = _commit(repo, {"plugin.js": "export default {}\n"})

    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True

    folder = _root(home) / "gizmo"
    assert (folder / "plugin.js").is_file()
    assert not (folder / ".git").exists()


def _read_only_git_objects(folder: Path) -> Path:
    """What a root-level plugin cloned by native Desktop (or by hand) carries: Git's loose object files
    are read-only, which a plain ``shutil.rmtree`` cannot delete on Windows. The folder holding them is
    made read-only too, which stops the same delete on POSIX (package-manager installs ship ``0555``)."""
    obj = folder / ".git" / "objects" / "ab" / "cdef"
    obj.parent.mkdir(parents=True)
    obj.write_bytes(b"x")
    os.chmod(obj, stat.S_IREAD)
    os.chmod(obj.parent, stat.S_IREAD | stat.S_IEXEC)
    return obj


def _replace_then_remove_a_plugin_holding_read_only_git_objects(home: Path, repo: Path) -> None:
    sha = _commit(repo, {"plugin.js": "export default { v: 1 }\n"})
    folder = _root(home) / "gizmo"
    assert install.install_desktop_plugin(URL, ref=sha)["ok"] is True
    _read_only_git_objects(folder)

    replaced = install.install_desktop_plugin(URL, ref=sha, force=True)

    assert replaced["ok"] is True, replaced
    assert not (folder / ".git").exists()

    _read_only_git_objects(folder)
    result = install.remove_desktop_plugin("gizmo")

    assert result["ok"] is True, result
    assert not folder.exists()


@pytest.mark.platforms("windows")
def test_windows_force_replaces_and_removes_a_plugin_holding_read_only_git_objects(home, repo):
    _replace_then_remove_a_plugin_holding_read_only_git_objects(home, repo)


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root deletes from read-only folders")
def test_posix_force_replaces_and_removes_a_plugin_holding_read_only_git_objects(home, repo):
    _replace_then_remove_a_plugin_holding_read_only_git_objects(home, repo)
