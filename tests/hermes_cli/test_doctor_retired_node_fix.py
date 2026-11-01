"""Doctor coverage for the retired $HERMES_HOME/node layout (#126934).

Detection + ``--fix`` removal of install-era node/npm/npx symlinks that still
resolve into the retired tree. Complements the report-only direction of the
open #126953: this covers the issue's "offers removal" half — ``doctor --fix``
unlinks exactly the Hermes-owned links (the uninstaller's ownership rule), so
user-repointed links (nvm, fnm, ...) and real files are never touched.
"""

from pathlib import Path

import pytest

import hermes_cli.doctor as doctor
import hermes_cli.uninstall as uninstall
from hermes_cli.doctor_tools import _check_retired_node_layout, _retired_node_path_links


@pytest.fixture
def retired_install(tmp_path, monkeypatch):
    """Fake HOME + HERMES_HOME with a retired node tree and a candidate bin dir.

    Mirrors tests/hermes_cli/test_uninstall_node_symlinks.py: the candidate-dir
    seam is pointed at a temp dir; every symlink/resolve/unlink below is real.
    """
    home = tmp_path / "home"
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    hermes_home = home / ".hermes"
    node_bin = hermes_home / "node" / "bin"
    node_bin.mkdir(parents=True)
    # Bare names (POSIX) plus the Windows shims so the managed-tree presence
    # probe fires on every platform.
    for name in ("node", "npm", "npx", "node.exe", "npm.cmd", "npx.cmd"):
        (node_bin / name).write_text("fake", encoding="utf-8")
    monkeypatch.setattr(
        uninstall, "_node_symlink_candidate_dirs", lambda: [local_bin]
    )
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    try:
        for name in ("node", "npm", "npx"):
            (local_bin / name).symlink_to(node_bin / name)
    except OSError as exc:
        pytest.skip(f"native symlink creation unavailable: {exc}")
    return hermes_home, local_bin


def test_detects_hermes_owned_links(retired_install):
    hermes_home, local_bin = retired_install
    assert sorted(p.name for p in _retired_node_path_links(hermes_home)) == [
        "node",
        "npm",
        "npx",
    ]
    assert all((local_bin / name).is_symlink() for name in ("node", "npm", "npx"))


def test_report_only_leaves_links_in_place(retired_install, capsys):
    _check_retired_node_layout(False)
    _, local_bin = retired_install
    assert all((local_bin / name).is_symlink() for name in ("node", "npm", "npx"))
    out = capsys.readouterr().out
    assert "Retired node layout" in out


def test_fix_removes_owned_links(retired_install, capsys):
    _check_retired_node_layout(True)
    _, local_bin = retired_install
    assert not any(
        (local_bin / name).is_symlink() or (local_bin / name).exists()
        for name in ("node", "npm", "npx")
    )
    out = capsys.readouterr().out
    assert "Removed retired node symlink" in out
    # Second run is quiet: nothing left to flag.
    _check_retired_node_layout(False)
    assert "Retired node layout" not in capsys.readouterr().out


def test_user_repointed_links_survive_fix(retired_install, tmp_path):
    hermes_home, local_bin = retired_install
    elsewhere = tmp_path / "nvm" / "node"
    elsewhere.write_text("fake", encoding="utf-8")
    (local_bin / "node").unlink()
    (local_bin / "node").symlink_to(elsewhere)
    assert [p.name for p in _retired_node_path_links(hermes_home)] == ["npm", "npx"]
    _check_retired_node_layout(True)
    assert (local_bin / "node").is_symlink()
    assert (local_bin / "node").resolve() == elsewhere.resolve()
    assert not (local_bin / "npm").is_symlink()


def test_no_retired_tree_is_silent(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / "empty-hermes"
    hermes_home.mkdir()
    monkeypatch.setattr(doctor, "HERMES_HOME", hermes_home)
    assert _retired_node_path_links(hermes_home) == []
    _check_retired_node_layout(False)
    assert capsys.readouterr().out == ""
