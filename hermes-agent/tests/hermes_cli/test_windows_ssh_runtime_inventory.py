"""Desktop's Windows SSH inventory reads through the canonical root and profile predicate."""
from __future__ import annotations

import os

import pytest

import hermes_constants
from hermes_cli.windows_ssh_runtime import dispatch


def _roster(root):
    profiles = root / "profiles"
    (profiles / ".deleted").mkdir(parents=True)
    (root / "config.yaml").write_text("")
    for name in ("bob", "win-research", "ghost", "tombstoned", ".rollback-1", "bad name"):
        (profiles / name).mkdir()
        if name != "ghost":
            (profiles / name / "config.yaml").write_text("")
    (profiles / ".deleted" / "tombstoned").write_text("")
    return profiles


@pytest.mark.parametrize("pinned", [False, True])
def test_inventory_lists_live_profiles_without_minting_an_identity(monkeypatch, tmp_path, pinned):
    root = tmp_path / "state-root"
    profiles = _roster(root)
    monkeypatch.setenv("HERMES_HOME", str(profiles / "bob" if pinned else root))

    result = dispatch(["inventory"])

    assert result == {"profiles": ["default", "bob", "win-research"], "installId": None}
    assert not (root / "install_id").exists()


@pytest.mark.platforms("posix")
def test_a_link_inside_a_profile_does_not_hide_the_roster(monkeypatch, tmp_path):
    root = tmp_path / "state-root"
    profiles = _roster(root)
    os.symlink(tmp_path, profiles / "bob" / "workspace")
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert dispatch(["inventory"])["profiles"] == ["default", "bob", "win-research"]


def test_a_nested_home_reads_the_identity_of_its_install_root(monkeypatch, tmp_path):
    native = tmp_path / "native"
    (native / "work").mkdir(parents=True)
    (native / "install_id").write_text("A" * 32 + "\n")
    monkeypatch.setattr(hermes_constants, "_get_platform_default_hermes_home", lambda: native)
    monkeypatch.setenv("HERMES_HOME", str(native / "work"))

    assert dispatch(["inventory"])["installId"] == "a" * 32
