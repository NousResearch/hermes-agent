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

    assert _mode(leaf) & 0o777 == 0o770


@posix_only
def test_secure_artifact_dir_preserves_inherited_setgid_on_managed_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A managed leaf must not lose setgid the kernel inherited for it.

    Linux propagates setgid from a 0o2770 parent to a fresh child so both
    UIDs sharing the hermes group keep write access to artifacts nested one
    level deeper. A bare chmod(0o770) silently clears it. macOS does not
    inherit setgid on directories, so the assertion only runs where the
    kernel actually granted the bit.
    """
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    parent = tmp_path / "managed"
    parent.mkdir()
    os.chmod(parent, 0o2770)

    probe = parent / "kernel-probe"
    probe.mkdir()
    if not _mode(probe) & stat.S_ISGID:
        pytest.skip("kernel does not inherit setgid on new directories")

    leaf = parent / "artifacts"
    old_umask = os.umask(0o022)
    try:
        config.secure_artifact_dir(leaf)
    finally:
        os.umask(old_umask)

    assert _mode(leaf) & stat.S_ISGID, "inherited setgid was cleared"
    assert _mode(leaf) & 0o777 == 0o770


@posix_only
def test_managed_leaf_passes_shared_group_to_artifacts_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifacts created in a managed leaf must land in the shared group.

    This is what makes ``artifact_file_mode()``'s group-write bit usable: the
    service UID can only append to a transcript the interactive UID created if
    that file's group is the shared hermes group, which on Linux follows from
    setgid on the containing dir. Verified against a real two-UID Linux setup
    (service could not append when setgid was cleared).
    """
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    parent = tmp_path / "managed"
    parent.mkdir()
    os.chmod(parent, 0o2770)

    probe = parent / "kernel-probe"
    probe.mkdir()
    if not _mode(probe) & stat.S_ISGID:
        pytest.skip("kernel does not inherit setgid on new directories")

    leaf = parent / "artifacts"
    old_umask = os.umask(0o022)
    try:
        config.secure_artifact_dir(leaf)
        artifact = leaf / "transcript.jsonl"
        artifact.write_text("x\n", encoding="utf-8")
    finally:
        os.umask(old_umask)

    assert artifact.stat().st_gid == leaf.stat().st_gid


@posix_only
def test_secure_artifact_dir_denies_other_access_under_permissive_umask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permissive umask must never leak artifacts to other local users.

    A plain mkdir under a 0o000 umask yields 0o777, which would expose
    private transcripts to every account on the box. Both the managed and
    unmanaged branches must deny other access regardless of umask.
    """
    managed_leaf = tmp_path / "managed" / "artifacts"
    unmanaged_leaf = tmp_path / "unmanaged" / "artifacts"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    old_umask = os.umask(0o000)
    try:
        monkeypatch.delenv("HERMES_MANAGED", raising=False)
        config.secure_artifact_dir(unmanaged_leaf)
        monkeypatch.setenv("HERMES_MANAGED", "nixos")
        config.secure_artifact_dir(managed_leaf)
    finally:
        os.umask(old_umask)

    assert not _mode(unmanaged_leaf) & 0o007
    assert not _mode(managed_leaf) & 0o007


@posix_only
@pytest.mark.require_symlinks
def test_secure_artifact_dir_never_widens_a_symlinked_leaf_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing symlinked artifact dir must not be chmodded through.

    Managed deployments symlink state into a git-tracked profile package, so a
    symlinked leaf is legitimate. Create-only means an existing leaf --
    symlink or not -- keeps its target mode instead of being widened.
    """
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    external = tmp_path / "external"
    external.mkdir()
    os.chmod(external, 0o750)
    leaf = tmp_path / "artifacts"
    leaf.symlink_to(external, target_is_directory=True)

    config.secure_artifact_dir(leaf)

    assert leaf.is_symlink(), "symlinked artifact dir must survive"
    assert _mode(external) == 0o750


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


@posix_only
def test_internal_legacy_directory_is_tightened_but_explicit_directory_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)
    for internal in (False, True):
        leaf = tmp_path / str(internal)
        leaf.mkdir(mode=0o755)
        config.secure_artifact_dir(leaf, tighten_existing=internal)
        assert _mode(leaf) == (0o700 if internal else 0o755)
