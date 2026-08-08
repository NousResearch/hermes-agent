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

import utils

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


def _windows_lock_error(winerror: int) -> PermissionError:
    exc = PermissionError(errno.EACCES, "destination is temporarily locked")
    exc.winerror = winerror
    return exc


def test_windows_transient_replace_error_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils.os, "name", "nt")

    for winerror in (5, 32, 33):
        assert utils._is_transient_windows_replace_error(_windows_lock_error(winerror))

    assert not utils._is_transient_windows_replace_error(_windows_lock_error(2))


def test_atomic_replace_retries_transient_windows_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    real_replace = os.replace
    attempts = 0
    sleeps: list[float] = []

    def flaky_replace(src: str, dst: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise _windows_lock_error(5)
        real_replace(src, dst)

    monkeypatch.setattr(utils, "_is_transient_windows_replace_error", lambda _exc: True)
    monkeypatch.setattr(utils.os, "replace", flaky_replace)
    monkeypatch.setattr(utils.time, "sleep", sleeps.append)

    atomic_replace(tmp, target)

    assert attempts == 3
    assert sleeps == [0.05, 0.1]
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_replace_reraises_after_windows_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    attempts = 0

    def always_locked(_src: str, _dst: str) -> None:
        nonlocal attempts
        attempts += 1
        raise _windows_lock_error(32)

    monkeypatch.setattr(utils, "_WINDOWS_ATOMIC_REPLACE_RETRY_DELAYS", (0.0, 0.0))
    monkeypatch.setattr(utils, "_is_transient_windows_replace_error", lambda _exc: True)
    monkeypatch.setattr(utils.os, "replace", always_locked)
    monkeypatch.setattr(utils.time, "sleep", lambda _delay: None)

    with pytest.raises(PermissionError):
        atomic_replace(tmp, target)

    assert attempts == 3
    assert target.read_text(encoding="utf-8") == "old"
    assert tmp.exists(), "failed replacement must leave the temp file for caller cleanup"


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
