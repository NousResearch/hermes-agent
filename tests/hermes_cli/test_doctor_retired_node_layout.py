"""Regression tests for #126934: the retired $HERMES_HOME/node layout must stop
being served to users — not on gateway service PATHs, and flagged by doctor when
its install-era PATH symlinks are still in place."""

from pathlib import Path
from unittest.mock import patch

from hermes_cli.doctor_tools import _retired_node_path_links
from hermes_cli.gateway import _build_service_path_dirs


def _retired_tree(home: Path) -> Path:
    """Reify a minimal old-layout tree: node/bin/node + npm + npx launchers."""
    bin_dir = home / "node" / "bin"
    bin_dir.mkdir(parents=True)
    for name in ("node", "npm", "npx"):
        (bin_dir / name).write_text("#!/bin/sh\n")
    return bin_dir


def test_service_path_excludes_retired_node_bin_even_when_present(tmp_path):
    """The whole point of the fix: the retired tree exists (pre-PM install) and
    its bin is still a real directory — it must STILL not enter service PATH.
    The current node_modules/.bin contract is unchanged."""
    home = tmp_path / ".hermes"
    retired_bin = _retired_tree(home)
    project_nm_bin = tmp_path / "node_modules" / ".bin"
    project_nm_bin.mkdir(parents=True)

    with patch("hermes_cli.gateway.get_hermes_home", return_value=home):
        dirs = _build_service_path_dirs(project_root=tmp_path)

    assert str(retired_bin) not in dirs
    assert str(project_nm_bin) in dirs


def test_retired_links_detected_when_symlinks_point_into_retired_tree(tmp_path, monkeypatch):
    """Install-era symlinks resolving into THIS home's retired node tree are ours."""
    home = tmp_path / ".hermes"
    retired_bin = _retired_tree(home)
    local_bin = tmp_path / "localbin"
    local_bin.mkdir()
    for name in ("node", "npm"):
        (local_bin / name).symlink_to(retired_bin / name)
    monkeypatch.setattr("hermes_cli.uninstall._node_symlink_candidate_dirs", lambda: [local_bin])

    assert _retired_node_path_links(home) == [local_bin / "node", local_bin / "npm"]


def test_repointed_symlinks_are_not_ours(tmp_path, monkeypatch):
    """Links the user repointed elsewhere (nvm, fnm) are left alone — the
    uninstall.py ownership rule: only targets inside OUR node tree count."""
    home = tmp_path / ".hermes"
    retired_bin = _retired_tree(home)
    nvm_bin = tmp_path / "nvm" / "versions" / "node" / "v20" / "bin"
    nvm_bin.mkdir(parents=True)
    (nvm_bin / "node").write_text("#!/bin/sh\n")
    local_bin = tmp_path / "localbin"
    local_bin.mkdir()
    (local_bin / "node").symlink_to(nvm_bin / "node")
    # decoy: a real file (not a symlink) must never be reported either
    (local_bin / "npm").write_text("#!/bin/sh\n")
    monkeypatch.setattr("hermes_cli.uninstall._node_symlink_candidate_dirs", lambda: [local_bin])

    assert _retired_node_path_links(home) == []


def test_no_link_reported_without_retired_tree(tmp_path, monkeypatch):
    """hermes_managed_node_tree_present gates the check: an install without the
    retired layout produces zero output (and zero filesystem probes beyond the gate)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    local_bin = tmp_path / "localbin"
    local_bin.mkdir()
    monkeypatch.setattr("hermes_cli.uninstall._node_symlink_candidate_dirs", lambda: [local_bin])

    assert _retired_node_path_links(home) == []
