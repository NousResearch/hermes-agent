"""Permission policy tests for transcript and diagnostic artifacts."""

import os
import stat
from pathlib import Path

import pytest

import hermes_cli.config as config


posix_only = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX permission bits are advisory on Windows",
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@posix_only
def test_secure_artifact_dir_creates_unmanaged_leaf_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    leaf = tmp_path / "artifacts"

    old_umask = os.umask(0o022)
    try:
        config.secure_artifact_dir(leaf)
    finally:
        os.umask(old_umask)

    assert _mode(leaf) == 0o700


@posix_only
def test_secure_artifact_dir_keeps_managed_group_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    parent = tmp_path / "managed"
    parent.mkdir()
    os.chmod(parent, 0o2770)
    leaf = parent / "artifacts"

    old_umask = os.umask(0o022)
    try:
        config.secure_artifact_dir(leaf)
    finally:
        os.umask(old_umask)

    assert _mode(leaf) == 0o770


@posix_only
def test_secure_artifact_dir_preserves_pre_existing_managed_leaf_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    leaf = tmp_path / "artifacts"
    leaf.mkdir()
    os.chmod(leaf, 0o750)

    config.secure_artifact_dir(leaf)

    assert _mode(leaf) == 0o750


def test_artifact_file_mode_is_owner_only_when_unmanaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    assert config.artifact_file_mode() == 0o600


def test_artifact_file_mode_is_group_writable_when_managed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_MANAGED", "nixos")

    assert config.artifact_file_mode() == 0o660
