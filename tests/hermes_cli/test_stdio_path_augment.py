"""Tests for hermes_cli.stdio._augment_path_with_known_tools.

On Windows, the installer (scripts/install.ps1) persists its managed tool
directories to the *User* PATH, but that broadcast does not reach the
already-running shell the user launches ``hermes`` from. To smooth over that
first-launch gap, ``_augment_path_with_known_tools`` prepends the known
Hermes-managed tool directories to the current process PATH so spawned
subprocesses resolve them immediately.

These tests force the Windows branch headlessly and assert the managed Node
directory (``%LOCALAPPDATA%\\hermes\\node``) is one of the mirrored entries —
it is persisted to the User PATH by Install-Node exactly like the Git dirs.
"""

import os

from hermes_cli import stdio


def test_augment_path_includes_managed_node_dir(tmp_path, monkeypatch):
    """The Hermes-managed portable Node dir must be prefilled onto PATH.

    Install-Node (scripts/install.ps1) unpacks portable Node to
    ``%LOCALAPPDATA%\\hermes\\node`` and persists it to the User PATH, the same
    way it persists the Git dirs. ``_augment_path_with_known_tools`` must mirror
    it so first-session ``node``/``npm``/``npx`` spawns resolve before the
    User-PATH broadcast reaches a fresh shell.
    """
    # Force the Windows branch without running on Windows.
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    # The managed Node dir exists on disk (the loop is os.path.isdir-guarded).
    node_dir = tmp_path / "hermes" / "node"
    node_dir.mkdir(parents=True)

    # A second mirrored dir, to prove the test exercises the real loop and is
    # not a tautology that would pass even if the Node entry were removed.
    git_cmd_dir = tmp_path / "hermes" / "git" / "cmd"
    git_cmd_dir.mkdir(parents=True)

    # Clear PATH so the managed dirs are genuinely absent before the call.
    monkeypatch.setenv("PATH", r"C:\Windows\System32")

    stdio._augment_path_with_known_tools()

    resulting = os.environ["PATH"].split(os.pathsep)
    assert str(node_dir) in resulting
    # The pre-existing mirrored entry still works — locks the contract.
    assert str(git_cmd_dir) in resulting


def test_augment_path_skips_managed_node_dir_when_absent(tmp_path, monkeypatch):
    """No-op for the Node dir when it does not exist on disk.

    The loop is guarded by ``os.path.isdir``; a missing managed Node dir must
    never be prepended (this is what keeps the prefill harmless on installs
    that did not use portable Node).
    """
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    # Deliberately do NOT create hermes/node on disk.
    monkeypatch.setenv("PATH", r"C:\Windows\System32")

    stdio._augment_path_with_known_tools()

    node_dir = tmp_path / "hermes" / "node"
    assert str(node_dir) not in os.environ["PATH"].split(os.pathsep)
