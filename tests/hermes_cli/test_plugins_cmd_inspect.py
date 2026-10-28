"""``plugins_cmd_inspect.inspect_plugin_source``: the install dialog's read-only repository check.

Every case clones a real local Git repository through the installer's clone path; nothing is mocked
but the catalog lookup and the kill list.
"""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import plugins_cmd_catalog
from hermes_cli.plugins_cmd import PluginOperationError
from hermes_cli.plugins_cmd_inspect import inspect_plugin_source


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout.strip()


def _repo(tmp_path: Path, files: dict[str, str], name: str = "source") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    _commit(repo, files)
    return repo


def _commit(repo: Path, files: dict[str, str]) -> str:
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def sentinel(tmp_path):
    return tmp_path / "imported.flag"


def _agent_files(sentinel: Path, name: str = "demo-agent") -> dict[str, str]:
    # Importing this package would leave the sentinel behind: inspection must never import it.
    return {"plugin.yaml": f"name: {name}\nversion: 1.0.0\n",
            "__init__.py": f"open({str(sentinel)!r}, 'w').write('ran')\n"}


def test_agent_only_package_is_read_without_running_its_code(tmp_path, sentinel):
    repo = _repo(tmp_path, _agent_files(sentinel))

    result = inspect_plugin_source(repo.as_uri())

    assert result == {"ok": True, "agent": True, "desktop": False, "agent_name": "demo-agent", "desktop_name": None,
                      "warnings": [result["warnings"][0]], "insecure": True}
    assert "insecure or local scheme" in result["warnings"][0]
    assert not sentinel.exists()


def test_mixed_package_in_a_subdirectory_reports_both_halves(tmp_path, sentinel):
    files = {f"pkgs/demo/{k}": v for k, v in _agent_files(sentinel).items()}
    files["pkgs/demo/desktop/plugin.js"] = "export default {}\n"
    files["other/huge.txt"] = "x"
    repo = _repo(tmp_path, files)

    result = inspect_plugin_source(f"{repo.as_uri()}#pkgs/demo")

    assert result["ok"] is True
    assert (result["agent"], result["desktop"]) == (True, True)
    assert (result["agent_name"], result["desktop_name"]) == ("demo-agent", "demo")
    assert not sentinel.exists()


def test_desktop_only_and_portable_packages(tmp_path):
    desktop = _repo(tmp_path, {"plugin.js": "export default {}\n"}, name="ui-plugin")
    portable = _repo(tmp_path, {"plugin.json": '{"name": "portable-one"}'}, name="portable")

    assert inspect_plugin_source(desktop.as_uri()) | {"warnings": []} == {
        "ok": True, "agent": False, "desktop": True, "agent_name": None, "desktop_name": "ui-plugin",
        "warnings": [], "insecure": True}
    assert inspect_plugin_source(portable.as_uri())["agent_name"] == "portable-one"


def test_repository_without_plugin_artifacts_is_a_failed_read(tmp_path):
    repo = _repo(tmp_path, {"README.md": "nothing here\n"})

    result = inspect_plugin_source(repo.as_uri())

    assert result["ok"] is False
    assert result["error"] == "No agent or desktop plugin artifacts found in this repository."


def test_unreachable_source_and_bad_identifier_fail_without_raising(tmp_path):
    missing = inspect_plugin_source((tmp_path / "nope").as_uri())
    bad = inspect_plugin_source("not-a-repo")

    assert missing["ok"] is False and missing["error"]
    assert bad["ok"] is False and bad["error"]


def test_subdirectory_escape_is_refused(tmp_path, sentinel):
    repo = _repo(tmp_path, _agent_files(sentinel))

    result = inspect_plugin_source(f"{repo.as_uri()}#../..")

    assert result["ok"] is False


def test_kill_listed_source_is_refused_before_cloning(tmp_path, sentinel, monkeypatch):
    repo = _repo(tmp_path, _agent_files(sentinel))
    seen = []

    def refuse(*candidates):
        seen.append(candidates)
        raise PluginOperationError("removed from the catalog")
    monkeypatch.setattr(plugins_cmd_catalog, "raise_if_removed", refuse)
    monkeypatch.setattr("hermes_cli.plugins_cmd._clone_plugin_repo", lambda *a, **k: pytest.fail("cloned"))

    result = inspect_plugin_source(repo.as_uri())

    assert result == {"ok": False, "agent": False, "desktop": False, "warnings": result["warnings"],
                      "insecure": True, "error": "removed from the catalog"}
    assert seen


def test_catalog_entry_is_inspected_at_its_reviewed_pin(tmp_path, sentinel, monkeypatch):
    repo = _repo(tmp_path, _agent_files(sentinel))
    pin = _git(repo, "rev-parse", "HEAD")
    _commit(repo, {"desktop/plugin.js": "export default {}\n"})  # newer than the pin
    entry = SimpleNamespace(name="demo", install_identifier=repo.as_uri(), sha=pin)
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda name: entry if name == "demo" else None)

    pinned = inspect_plugin_source("ignored/identifier", catalog_name="demo")
    unknown = inspect_plugin_source("", catalog_name="missing")

    assert (pinned["ok"], pinned["agent"], pinned["desktop"]) == (True, True, False)
    assert unknown == {"ok": False, "agent": False, "desktop": False, "warnings": [], "insecure": False,
                       "error": "'missing' is not in the Hermes plugin catalog."}


def test_inspection_writes_nothing_into_the_profile(tmp_path, sentinel, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = _repo(tmp_path, _agent_files(sentinel))

    assert inspect_plugin_source(repo.as_uri())["ok"] is True
    assert not (home / "plugins").exists() or not any((home / "plugins").iterdir())
