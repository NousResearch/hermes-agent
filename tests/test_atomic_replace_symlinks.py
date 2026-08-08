"""Regression tests for GitHub #16743 — atomic writes must preserve symlinks.

``os.replace(tmp, target)`` replaces whatever exists at ``target`` — including
symlinks, which it swaps for a regular file.  Managed deployments that
symlink ``~/.hermes/config.yaml`` (and other state files) to a git-tracked
profile package were silently detached on every config write.

The fix: a shared ``atomic_replace`` helper in ``utils.py`` that resolves the
target through ``os.path.realpath`` when it is a symlink, so the real file is
overwritten in-place while the symlink survives.  All atomic-write sites in
the codebase were migrated to the helper; these tests pin that invariant.
"""
from __future__ import annotations

import errno
import json
import os
import sys
from pathlib import Path

import pytest
import yaml

# Ensure the repo root is importable when running via `pytest tests/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import (
    atomic_json_write,
    atomic_replace,
    atomic_roundtrip_yaml_update,
    atomic_yaml_write,
)


# ─── Direct helper ────────────────────────────────────────────────────────────


def _write_tmp(dir_: Path, content: str) -> Path:
    tmp = dir_ / ".src.tmp"
    tmp.write_text(content, encoding="utf-8")
    return tmp


def test_atomic_replace_preserves_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("original\n", encoding="utf-8")
    link.symlink_to(real)

    tmp = _write_tmp(tmp_path, "updated\n")
    returned = atomic_replace(tmp, link)

    assert link.is_symlink(), "symlink must not be replaced with a regular file"
    assert real.read_text(encoding="utf-8") == "updated\n"
    assert Path(returned) == real
    # Follow the symlink — same content.
    assert link.read_text(encoding="utf-8") == "updated\n"


def test_atomic_replace_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "plain.yaml"
    target.write_text("old\n", encoding="utf-8")

    tmp = _write_tmp(tmp_path, "fresh\n")
    returned = atomic_replace(tmp, target)

    assert Path(returned) == target
    assert target.read_text(encoding="utf-8") == "fresh\n"
    assert not target.is_symlink()




def test_atomic_replace_accepts_pathlike_and_str(tmp_path: Path) -> None:
    target = tmp_path / "dual.json"
    target.write_text("{}", encoding="utf-8")

    # str inputs
    tmp1 = _write_tmp(tmp_path, "1")
    atomic_replace(str(tmp1), str(target))
    assert target.read_text(encoding="utf-8") == "1"

    # Path inputs
    tmp2 = _write_tmp(tmp_path, "2")
    atomic_replace(tmp2, target)
    assert target.read_text(encoding="utf-8") == "2"


# ─── atomic_json_write / atomic_yaml_write wiring ──────────────────────────


def test_atomic_json_write_preserves_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    link = tmp_path / "link.json"
    real.write_text("{}", encoding="utf-8")
    link.symlink_to(real)

    atomic_json_write(link, {"hello": "world"})

    assert link.is_symlink()
    loaded = json.loads(real.read_text(encoding="utf-8"))
    assert loaded == {"hello": "world"}


def test_atomic_yaml_write_preserves_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("placeholder: true\n", encoding="utf-8")
    link.symlink_to(real)

    atomic_yaml_write(link, {"model": {"provider": "openrouter"}})

    assert link.is_symlink()
    data = yaml.safe_load(real.read_text(encoding="utf-8"))
    assert data == {"model": {"provider": "openrouter"}}


def test_atomic_json_write_preserves_symlink_permissions(tmp_path: Path) -> None:
    """Symlinked targets keep the real file's permission bits."""
    if os.name != "posix":
        pytest.skip("POSIX-only")

    real = tmp_path / "real.json"
    link = tmp_path / "link.json"
    real.write_text("{}", encoding="utf-8")
    os.chmod(real, 0o644)
    link.symlink_to(real)

    atomic_json_write(link, {"x": 1})

    import stat as _stat
    mode = _stat.S_IMODE(real.stat().st_mode)
    assert mode == 0o644, f"permissions drifted after symlinked write: {oct(mode)}"


def test_atomic_yaml_write_restores_owner_on_real_symlink_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config writes through symlinks must restore the real file's owner.

    Docker support hit this when a root-run setup wizard rewrote a
    hermes-owned /opt/data/config.yaml via atomic replace, leaving the new file
    root-owned. The test forces a preserved uid/gid so it does not need root.
    """
    if os.name != "posix":
        pytest.skip("POSIX-only")

    real = tmp_path / "config.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("old: true\n", encoding="utf-8")
    link.symlink_to(real)

    chown_calls: list[tuple[Path, int, int]] = []
    monkeypatch.setattr("utils._preserve_file_owner", lambda _path: (123, 456))
    monkeypatch.setattr(
        "utils.os.chown",
        lambda path, uid, gid: chown_calls.append((Path(path), uid, gid)),
    )

    atomic_yaml_write(link, {"new": True})

    assert chown_calls == [(real, 123, 456)]






# ─── Broken-symlink edge case ─────────────────────────────────────────────


def test_atomic_replace_broken_symlink_creates_target(tmp_path: Path) -> None:
    """A symlink pointing at a missing file: the write should create the
    real target (resolving via realpath) rather than leaving the dangling
    link in place as a regular file.
    """
    missing = tmp_path / "does_not_exist_yet.yaml"
    link = tmp_path / "link.yaml"
    link.symlink_to(missing)
    assert link.is_symlink()
    assert not missing.exists()

    tmp = _write_tmp(tmp_path, "created-through-link\n")
    atomic_replace(tmp, link)

    assert link.is_symlink(), "symlink must be preserved"
    assert missing.exists(), "real target should now exist"
    assert missing.read_text(encoding="utf-8") == "created-through-link\n"


# ─── EXDEV / EBUSY copy fallback ───────────────────────────────────────────




def test_atomic_replace_copy_fallback_preserves_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("old\n", encoding="utf-8")
    link.symlink_to(real)
    tmp = _write_tmp(tmp_path, "new\n")

    def fail_replace(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)

    monkeypatch.setattr("utils.os.replace", fail_replace)

    assert Path(atomic_replace(tmp, link)) == real
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()






def test_atomic_replace_real_cross_device(tmp_path: Path) -> None:
    shm = Path("/dev/shm")
    if os.name != "posix" or not os.access(shm, os.W_OK):
        pytest.skip("requires writable /dev/shm")

    import shutil as _shutil
    import uuid as _uuid

    other_fs_dir = shm / f"hermes-exdev-test-{_uuid.uuid4().hex[:8]}"
    other_fs_dir.mkdir()
    try:
        real = other_fs_dir / "config.yaml"
        real.write_text("old\n", encoding="utf-8")
        if os.stat(real).st_dev == os.stat(tmp_path).st_dev:
            pytest.skip("/dev/shm is not a separate filesystem here")

        link = tmp_path / "config.yaml"
        link.symlink_to(real)
        tmp = _write_tmp(tmp_path, "new\n")

        assert Path(atomic_replace(tmp, link)) == real
        assert link.is_symlink()
        assert real.read_text(encoding="utf-8") == "new\n"
        assert not tmp.exists()
    finally:
        _shutil.rmtree(other_fs_dir, ignore_errors=True)


def test_atomic_replace_cross_device_never_truncates_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted cross-device write must leave the target untouched.

    ``shutil.copyfile`` opens its destination ``"wb"``, so copying straight
    onto the resolved target truncates the user's file before the first
    replacement byte exists.  A crash, ENOSPC or SIGKILL midway then leaves
    ``config.yaml`` / ``auth.json`` partial or empty — precisely the outcome
    this helper exists to prevent.
    """
    if os.name != "posix":
        pytest.skip("POSIX-only")

    real = tmp_path / "config.yaml"
    link = tmp_path / "link.yaml"
    original = "provider: openrouter\napi_key: keep-me\n"
    real.write_text(original, encoding="utf-8")
    os.chmod(real, 0o600)
    link.symlink_to(real)
    tmp = _write_tmp(tmp_path, "provider: replacement\n")

    genuine_replace = os.replace
    replace_calls = 0

    def exdev_once(src: str, dst: str) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 1:
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)
        genuine_replace(src, dst)

    def copyfileobj_runs_out_of_space(
        src_handle: object, dst_handle: object, *_args: object, **_kwargs: object
    ) -> None:
        # Land a few bytes, then fail the way a filling disk does.
        dst_handle.write(src_handle.read(8))  # type: ignore[attr-defined]
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr("utils.os.replace", exdev_once)
    monkeypatch.setattr("utils.shutil.copyfileobj", copyfileobj_runs_out_of_space)

    with pytest.raises(OSError) as excinfo:
        atomic_replace(tmp, link)
    assert excinfo.value.errno == errno.ENOSPC

    assert link.is_symlink(), "symlink must survive a failed cross-device write"
    assert (
        real.read_text(encoding="utf-8") == original
    ), "a failed cross-device write destroyed the target's contents"

    import stat as _stat

    assert _stat.S_IMODE(real.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".tmp_xdev_*")), "staged temp leaked after failure"


def test_atomic_replace_cross_device_uses_rename_not_inplace_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cross-device fallback must finish by rename, not by in-place copy."""
    if os.name != "posix":
        pytest.skip("POSIX-only")

    real = tmp_path / "config.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("old\n", encoding="utf-8")
    link.symlink_to(real)
    tmp = _write_tmp(tmp_path, "new\n")
    original_inode = real.stat().st_ino

    genuine_replace = os.replace
    destinations: list[str] = []

    def exdev_once(src: str, dst: str) -> None:
        destinations.append(str(dst))
        if len(destinations) == 1:
            raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)
        genuine_replace(src, dst)

    monkeypatch.setattr("utils.os.replace", exdev_once)

    assert Path(atomic_replace(tmp, link)) == real

    assert (
        len(destinations) >= 2
    ), "cross-device fallback copied in place instead of staging then renaming"
    assert destinations[-1] == str(real)
    assert (
        real.stat().st_ino != original_inode
    ), "target was overwritten in place rather than replaced by an atomic rename"
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()
    assert not list(tmp_path.glob(".tmp_xdev_*")), "staged temp leaked"


def test_atomic_replace_busy_target_falls_back_to_inplace_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A busy inode cannot be renamed onto, so the in-place copy must remain.

    Behaviour-preservation guard: green before and after the staging change.
    It pins the last-resort path and catches a staged temp leaking when the
    staged rename fails.
    """
    real = tmp_path / "config.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("old\n", encoding="utf-8")
    link.symlink_to(real)
    tmp = _write_tmp(tmp_path, "new\n")

    def always_busy(src: str, dst: str) -> None:
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), src, None, dst)

    monkeypatch.setattr("utils.os.replace", always_busy)

    assert Path(atomic_replace(tmp, link)) == real
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()
    assert not list(tmp_path.glob(".tmp_xdev_*")), "staged temp leaked on EBUSY"


def test_atomic_replace_cross_device_survives_unstageable_target_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target directory that rejects a staged temp still gets the write.

    Behaviour-preservation guard: green before and after the staging change.
    """
    real = tmp_path / "config.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("old\n", encoding="utf-8")
    link.symlink_to(real)
    tmp = _write_tmp(tmp_path, "new\n")

    def fail_replace(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)

    def refuse_staging(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EACCES, os.strerror(errno.EACCES))

    monkeypatch.setattr("utils.os.replace", fail_replace)
    monkeypatch.setattr("utils.tempfile.mkstemp", refuse_staging)

    assert Path(atomic_replace(tmp, link)) == real
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()
