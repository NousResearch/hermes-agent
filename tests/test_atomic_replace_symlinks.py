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

from utils import atomic_json_write, atomic_replace, atomic_yaml_write


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


def test_atomic_replace_first_time_create(tmp_path: Path) -> None:
    target = tmp_path / "new.yaml"
    assert not target.exists()

    tmp = _write_tmp(tmp_path, "brand new\n")
    returned = atomic_replace(tmp, target)

    assert Path(returned) == target
    assert target.read_text(encoding="utf-8") == "brand new\n"


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


# ─── Cross-device fallback (EXDEV) ─────────────────────────────────────────


def test_atomic_replace_exdev_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.replace fails with EXDEV (cross-device link), the helper
    falls back to shutil.copy2 + os.unlink so WSL→Windows symlinked configs
    still update successfully.
    """
    import utils as utils_mod

    target = tmp_path / "target.yaml"
    target.write_text("old\n", encoding="utf-8")

    original_replace = os.replace

    def _raise_exdev(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)

    monkeypatch.setattr(os, "replace", _raise_exdev)

    tmp = _write_tmp(tmp_path, "cross-device\n")
    returned = atomic_replace(tmp, target)

    assert Path(returned) == target
    assert target.read_text(encoding="utf-8") == "cross-device\n"
    assert not tmp.exists(), "temp file must be cleaned up after EXDEV fallback"


def test_atomic_replace_exdev_fallback_via_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """EXDEV fallback works when target is a symlink pointing to another fs."""
    real = tmp_path / "real.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("original\n", encoding="utf-8")
    link.symlink_to(real)

    def _raise_exdev(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)

    monkeypatch.setattr(os, "replace", _raise_exdev)

    tmp = _write_tmp(tmp_path, "updated-via-symlink\n")
    returned = atomic_replace(tmp, link)

    assert link.is_symlink(), "symlink must be preserved even after EXDEV fallback"
    assert Path(returned) == real
    assert real.read_text(encoding="utf-8") == "updated-via-symlink\n"
    assert not tmp.exists(), "temp file must be cleaned up after EXDEV fallback"


def test_atomic_replace_exdev_does_not_swallow_other_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-EXDEV OSErrors from os.replace must propagate, not be silently caught."""
    target = tmp_path / "target.yaml"
    target.write_text("unchanged\n", encoding="utf-8")

    def _raise_eacces(src: str, dst: str) -> None:
        raise OSError(errno.EACCES, "Permission denied", dst)

    monkeypatch.setattr(os, "replace", _raise_eacces)

    tmp = _write_tmp(tmp_path, "should-not-land\n")
    with pytest.raises(OSError, match="Permission denied"):
        atomic_replace(tmp, target)

    # Target should be untouched.
    assert target.read_text(encoding="utf-8") == "unchanged\n"


def test_atomic_yaml_write_exdev_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: atomic_yaml_write recovers from EXDEV during write."""
    target = tmp_path / "config.yaml"
    target.write_text("placeholder: true\n", encoding="utf-8")

    def _raise_exdev(src: str, dst: str) -> None:
        raise OSError(errno.EXDEV, os.strerror(errno.EXDEV), src, None, dst)

    monkeypatch.setattr(os, "replace", _raise_exdev)

    atomic_yaml_write(target, {"display": {"skin": "auto"}})

    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data == {"display": {"skin": "auto"}}
