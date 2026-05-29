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


# ─── #34252: Cross-filesystem rename (EXDEV) regression ────────────────────


def test_atomic_replace_falls_back_to_shutil_on_exdev(tmp_path: Path, monkeypatch) -> None:
    """#34252: When os.replace raises OSError(EXDEV) — which happens when
    ~/.hermes/ is symlinked to a different filesystem — the helper must
    fall back to shutil.move so the write still lands. Previously this
    raised, breaking ``hermes config set`` / model picks / dedup-state
    persistence for any deployment with a cross-mount hermes home.
    """
    import errno as _errno

    target = tmp_path / "target.json"
    target.write_text('{"old": true}', encoding="utf-8")

    src = _write_tmp(tmp_path, '{"new": true}')

    real_os_replace = os.replace
    call_state = {"count": 0}

    def _replace_raises_exdev(src_path, dest_path):
        call_state["count"] += 1
        # First call (the one inside atomic_replace) raises EXDEV.
        if call_state["count"] == 1:
            err = OSError("Invalid cross-device link")
            err.errno = _errno.EXDEV
            raise err
        return real_os_replace(src_path, dest_path)

    monkeypatch.setattr(os, "replace", _replace_raises_exdev)

    # Should NOT raise even though os.replace raised EXDEV.
    real_path = atomic_replace(src, target)

    # Content of target was updated via the shutil.move fallback.
    assert json.loads(Path(real_path).read_text(encoding="utf-8")) == {"new": True}
    # Source temp was consumed.
    assert not src.exists()
    # And os.replace was called once (the failed attempt before fallback).
    assert call_state["count"] == 1


def test_atomic_replace_reraises_non_exdev_errors(tmp_path: Path, monkeypatch) -> None:
    """Sanity check: ANY OSError other than EXDEV must still propagate.
    The fallback is intentionally narrow — we don't want to mask
    permission errors or read-only-filesystem errors as silent failures."""
    import errno as _errno

    target = tmp_path / "target.json"
    target.write_text('{"old": true}', encoding="utf-8")
    src = _write_tmp(tmp_path, '{"new": true}')

    def _replace_raises_eacces(src_path, dest_path):
        err = OSError("Permission denied")
        err.errno = _errno.EACCES
        raise err

    monkeypatch.setattr(os, "replace", _replace_raises_eacces)

    with pytest.raises(OSError) as exc_info:
        atomic_replace(src, target)
    assert exc_info.value.errno == _errno.EACCES


def test_atomic_replace_happy_path_still_uses_os_replace(tmp_path: Path) -> None:
    """When no EXDEV occurs (the common case), the helper uses os.replace
    directly — atomicity is preserved on same-filesystem writes, which
    is what 99% of users actually have. Cross-fs is the fallback path."""
    target = tmp_path / "target.json"
    target.write_text('{"old": true}', encoding="utf-8")
    src = _write_tmp(tmp_path, '{"new": true}')

    real_path = atomic_replace(src, target)

    assert json.loads(Path(real_path).read_text(encoding="utf-8")) == {"new": True}
    assert not src.exists()
