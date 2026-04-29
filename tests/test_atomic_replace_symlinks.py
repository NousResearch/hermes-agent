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


# ─── Cross-device fallback (GitHub #17313) ───────────────────────────────────


def _exdev_once(real_replace):
    """Return a replacement for os.replace that raises EXDEV exactly once.

    Simulates a cross-filesystem move on the first call (the original tmp →
    real target swap) and lets the fallback's same-filesystem replace
    succeed by delegating to the real ``os.replace`` thereafter.
    """
    state = {"raised": False}

    def _replace(src, dst):
        if not state["raised"]:
            state["raised"] = True
            raise OSError(errno.EXDEV, "Invalid cross-device link", src)
        return real_replace(src, dst)

    return _replace


def test_atomic_replace_recovers_from_exdev(tmp_path: Path, monkeypatch) -> None:
    """First os.replace raises EXDEV; helper must copy onto a sibling temp
    in the real target's directory and atomically replace there."""
    target = tmp_path / "real.yaml"
    target.write_text("old\n", encoding="utf-8")

    tmp = _write_tmp(tmp_path, "fresh-after-xdev\n")
    monkeypatch.setattr("utils.os.replace", _exdev_once(os.replace))

    returned = atomic_replace(tmp, target)

    assert Path(returned) == target
    assert target.read_text(encoding="utf-8") == "fresh-after-xdev\n"
    assert not tmp.exists(), "original temp must be cleaned up after fallback"
    # No leftover sibling temps from the fallback.
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".atomic_xdev_"))
    assert leftovers == [], f"unexpected sibling temps: {leftovers}"


def test_atomic_replace_exdev_with_symlink_preserves_link(tmp_path: Path, monkeypatch) -> None:
    """Cross-device swap through a symlink keeps the symlink and writes the
    real file. Mirrors the #17313 reporter's setup where ~/.hermes/memories/
    is symlinked into a vault on a separate mount."""
    real = tmp_path / "real.md"
    link = tmp_path / "link.md"
    real.write_text("old\n", encoding="utf-8")
    link.symlink_to(real)

    tmp = _write_tmp(tmp_path, "fresh-via-symlink\n")
    monkeypatch.setattr("utils.os.replace", _exdev_once(os.replace))

    returned = atomic_replace(tmp, link)

    assert link.is_symlink(), "symlink must survive cross-device swap"
    assert Path(returned) == real
    assert real.read_text(encoding="utf-8") == "fresh-via-symlink\n"
    assert link.read_text(encoding="utf-8") == "fresh-via-symlink\n"
    assert not tmp.exists()


def test_atomic_replace_propagates_non_exdev_oserror(tmp_path: Path, monkeypatch) -> None:
    """Only EXDEV triggers the fallback — every other OSError must propagate."""
    target = tmp_path / "real.yaml"
    target.write_text("untouched\n", encoding="utf-8")

    tmp = _write_tmp(tmp_path, "ignored\n")

    def _disk_full(_src, _dst):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("utils.os.replace", _disk_full)

    with pytest.raises(OSError) as excinfo:
        atomic_replace(tmp, target)
    assert excinfo.value.errno == errno.ENOSPC
    assert target.read_text(encoding="utf-8") == "untouched\n"


def test_atomic_replace_exdev_cleans_sibling_on_replace_failure(
    tmp_path: Path, monkeypatch
) -> None:
    """If the fallback's same-filesystem replace itself fails, the sibling
    temp must be cleaned up rather than leaked into the real directory."""
    target = tmp_path / "real.yaml"
    target.write_text("untouched\n", encoding="utf-8")

    tmp = _write_tmp(tmp_path, "ignored\n")

    state = {"calls": 0}

    def _replace(_src, _dst):
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr("utils.os.replace", _replace)

    with pytest.raises(OSError) as excinfo:
        atomic_replace(tmp, target)
    assert excinfo.value.errno == errno.EACCES
    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".atomic_xdev_"))
    assert leftovers == [], f"sibling temp not cleaned up: {leftovers}"
