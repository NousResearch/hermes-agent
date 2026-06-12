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

from unittest import mock


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


# ─── Windows PermissionError retry (#43268) ──────────────────────────────


def test_atomic_replace_retries_on_windows_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.replace raising PermissionError (WinError 5 on Windows, from
    concurrent hermes processes briefly holding the target open) should
    be retried with jittered backoff. The mock simulates the Windows-only
    branch by forcing os.name == 'nt' and stubbing the inner os.replace
    to fail twice before succeeding on the third attempt.
    """
    target = tmp_path / "auth.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            # PermissionError maps to WinError 5 on Windows; raising it on
            # any platform exercises the same retry branch.
            raise PermissionError(5, "Access is denied", dst)
        real_replace(src, dst)

    monkeypatch.setattr("utils.os.name", "nt", raising=False)
    monkeypatch.setattr("utils.os.replace", flaky_replace)
    # Make time.sleep a no-op so the test doesn't actually wait.
    monkeypatch.setattr("utils.time.sleep", lambda _s: None)

    returned = atomic_replace(tmp, target)
    assert Path(returned) == target
    assert target.read_text(encoding="utf-8") == "new"
    assert calls["n"] == 3, f"expected 3 attempts, got {calls['n']}"


def test_atomic_replace_reraises_after_six_windows_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the file is still held open after the full retry budget, the
    PermissionError must propagate so the caller can fall back / surface
    a real error rather than silently leaving stale data.
    """
    target = tmp_path / "auth.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    def always_denied(src: str, dst: str) -> None:
        raise PermissionError(5, "Access is denied", dst)

    monkeypatch.setattr("utils.os.name", "nt", raising=False)
    monkeypatch.setattr("utils.os.replace", always_denied)
    monkeypatch.setattr("utils.time.sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        atomic_replace(tmp, target)
    # Target untouched — temp file replaced the original, retry never succeeded.
    assert target.read_text(encoding="utf-8") == "old"


def test_atomic_replace_no_retry_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On POSIX the retry branch must not engage — a PermissionError is
    surfaced immediately so existing callers' behavior is unchanged.
    """
    target = tmp_path / "auth.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    calls = {"n": 0}

    def denied_once(src: str, dst: str) -> None:
        calls["n"] += 1
        raise PermissionError(13, "Permission denied", dst)

    monkeypatch.setattr("utils.os.name", "posix", raising=False)
    monkeypatch.setattr("utils.os.replace", denied_once)
    monkeypatch.setattr("utils.time.sleep", lambda _s: None)

    with pytest.raises(PermissionError):
        atomic_replace(tmp, target)
    assert calls["n"] == 1, f"POSIX must not retry, got {calls['n']} attempts"
