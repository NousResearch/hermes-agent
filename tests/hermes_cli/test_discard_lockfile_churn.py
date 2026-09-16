"""Behavioral tests for npm lockfile ownership cleanup (#112378)."""

import json
import subprocess
from pathlib import Path

from hermes_cli.update_cmd_git import _discard_lockfile_churn


def _repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["apps/*", "ui-tui", "web"]}))
    (tmp_path / "package-lock.json").write_text("root clean\n")
    (tmp_path / "apps" / "desktop").mkdir(parents=True)
    (tmp_path / "apps" / "desktop" / "package.json").write_text("desktop clean\n")
    (tmp_path / "ui-tui").mkdir()
    (tmp_path / "ui-tui" / "package.json").write_text("tui clean\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    return tmp_path


def _dirty(path, text="dirty\n"):
    path.write_text(text)


def _changed(repo):
    return subprocess.run(["git", "diff", "--name-only"], cwd=repo, text=True, capture_output=True, check=True).stdout.splitlines()


def test_dirty_workspace_manifest_preserves_root_lock(tmp_path):
    repo = _repo(tmp_path)
    _dirty(repo / "apps/desktop/package.json")
    _dirty(repo / "package-lock.json")
    _discard_lockfile_churn(["git"], repo)
    assert "package-lock.json" in _changed(repo)


def test_lock_only_churn_is_discarded(tmp_path):
    repo = _repo(tmp_path)
    _dirty(repo / "package-lock.json")
    _discard_lockfile_churn(["git"], repo)
    assert "package-lock.json" not in _changed(repo)


def test_dirty_root_manifest_preserves_root_lock(tmp_path):
    repo = _repo(tmp_path)
    _dirty(repo / "package.json")
    _dirty(repo / "package-lock.json")
    _discard_lockfile_churn(["git"], repo)
    assert set(_changed(repo)) == {"package-lock.json", "package.json"}


def test_sibling_workspace_manifest_preserves_root_lock(tmp_path):
    repo = _repo(tmp_path)
    _dirty(repo / "ui-tui/package.json")
    _dirty(repo / "package-lock.json")
    _discard_lockfile_churn(["git"], repo)
    assert "package-lock.json" in _changed(repo)


def test_unrelated_nested_lock_is_discarded(tmp_path):
    repo = _repo(tmp_path)
    nested = repo / "ui-tui/package-lock.json"
    _dirty(nested)
    _discard_lockfile_churn(["git"], repo)
    assert "ui-tui/package-lock.json" not in _changed(repo)


def test_non_workspace_manifest_does_not_protect_root_lock(tmp_path):
    repo = _repo(tmp_path)
    vendor = repo / "vendor/foo"
    vendor.mkdir(parents=True)
    _dirty(vendor / "package.json")
    _dirty(repo / "package-lock.json")
    _discard_lockfile_churn(["git"], repo)
    assert "package-lock.json" not in _changed(repo)
