"""Tests for hermes_cli.uninstall.remove_node_symlinks.

Regression for #34536: the POSIX installer drops node/npm/npx symlinks in
~/.local/bin pointing into $HERMES_HOME/node and prepends ~/.local/bin to
PATH, shadowing an existing nvm. Uninstall must remove those symlinks, but
only when they still resolve into the Hermes-managed node dir.
"""

import os
from pathlib import Path

import pytest

import hermes_cli.uninstall as uninstall


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() at the home both the installer-symlink target and
    the ~/.local/bin links live under the same temp dir."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    (home / ".local" / "bin").mkdir(parents=True)
    return home


def _make_hermes_node(hermes_home: Path) -> Path:
    """Create a fake $HERMES_HOME/node/bin/{node,npm,npx} tree."""
    node_bin = hermes_home / "node" / "bin"
    node_bin.mkdir(parents=True)
    for name in ("node", "npm", "npx"):
        (node_bin / name).write_text("#!/bin/sh\n")
        (node_bin / name).chmod(0o755)
    return node_bin




def test_leaves_unrelated_symlinks_untouched(fake_home):
    """A node symlink the user repointed at nvm must survive uninstall."""
    hermes_home = fake_home / ".hermes"
    _make_hermes_node(hermes_home)
    local_bin = fake_home / ".local" / "bin"

    # Simulate nvm's node living elsewhere; user's ~/.local/bin/node -> nvm.
    nvm_bin = fake_home / ".nvm" / "versions" / "node" / "v20.0.0" / "bin"
    nvm_bin.mkdir(parents=True)
    (nvm_bin / "node").write_text("#!/bin/sh\n")
    (local_bin / "node").symlink_to(nvm_bin / "node")

    removed = uninstall.remove_node_symlinks(hermes_home)

    assert removed == []
    assert (local_bin / "node").is_symlink()
    assert (local_bin / "node").resolve() == (nvm_bin / "node").resolve()










def test_removes_fhs_symlinks_in_usr_local_bin(fake_home, tmp_path, monkeypatch):
    """Root FHS installs place node symlinks in /usr/local/bin.

    We monkeypatch _node_symlink_candidate_dirs to return a temp dir standing
    in for /usr/local/bin so the test doesn't need real root privileges.
    """
    hermes_home = fake_home / ".hermes"
    node_bin = _make_hermes_node(hermes_home)

    # Fake /usr/local/bin as a temp dir with our symlinks.
    fhs_bin = tmp_path / "usr_local_bin"
    fhs_bin.mkdir()
    for name in ("node", "npm", "npx"):
        (fhs_bin / name).symlink_to(node_bin / name)

    # Ensure ~/.local/bin has NO symlinks (simulate pure FHS install).
    local_bin = fake_home / ".local" / "bin"
    for name in ("node", "npm", "npx"):
        p = local_bin / name
        if p.exists() or p.is_symlink():
            p.unlink()

    # Return only our fake FHS dir as a candidate.
    monkeypatch.setattr(
        uninstall, "_node_symlink_candidate_dirs", lambda: [fhs_bin]
    )

    removed = uninstall.remove_node_symlinks(hermes_home)

    assert sorted(p.name for p in removed) == ["node", "npm", "npx"]
    for name in ("node", "npm", "npx"):
        assert not (fhs_bin / name).is_symlink()
