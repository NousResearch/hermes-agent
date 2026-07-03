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


def test_atomic_json_write_restores_owner_with_explicit_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only")

    target = tmp_path / "state.json"
    target.write_text("{}", encoding="utf-8")

    chown_calls: list[tuple[Path, int, int]] = []
    monkeypatch.setattr("utils._preserve_file_owner", lambda _path: (234, 567))
    monkeypatch.setattr(
        "utils.os.chown",
        lambda path, uid, gid: chown_calls.append((Path(path), uid, gid)),
    )

    atomic_json_write(target, {"api_key": "secret"}, mode=0o600)

    assert chown_calls == [(target, 234, 567)]
    assert target.stat().st_mode & 0o777 == 0o600


def test_atomic_roundtrip_yaml_update_restores_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only")

    target = tmp_path / "config.yaml"
    target.write_text("model:\n  provider: openrouter\n", encoding="utf-8")

    chown_calls: list[tuple[Path, int, int]] = []
    monkeypatch.setattr("utils._preserve_file_owner", lambda _path: (345, 678))
    monkeypatch.setattr(
        "utils.os.chown",
        lambda path, uid, gid: chown_calls.append((Path(path), uid, gid)),
    )

    atomic_roundtrip_yaml_update(target, "model.provider", "nvidia")

    assert chown_calls == [(target, 345, 678)]
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["model"]["provider"] == "nvidia"


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


@pytest.mark.parametrize("fail_errno", [errno.EXDEV, errno.EBUSY])
def test_atomic_replace_copy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_errno: int
) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("old\n", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new\n")

    def fail_replace(src: str, dst: str) -> None:
        raise OSError(fail_errno, os.strerror(fail_errno), src, None, dst)

    monkeypatch.setattr("utils.os.replace", fail_replace)

    assert Path(atomic_replace(tmp, target)) == target
    assert target.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()


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


def test_atomic_replace_copy_fallback_preserves_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX-only")

    target = tmp_path / "config.yaml"
    target.write_text("old\n", encoding="utf-8")
    os.chmod(target, 0o600)
    tmp = _write_tmp(tmp_path, "new\n")
    os.chmod(tmp, 0o644)

    def fail_replace(src: str, dst: str) -> None:
        raise OSError(errno.EBUSY, os.strerror(errno.EBUSY), src, None, dst)

    monkeypatch.setattr("utils.os.replace", fail_replace)

    atomic_replace(tmp, target)
    assert target.read_text(encoding="utf-8") == "new\n"
    assert target.stat().st_mode & 0o777 == 0o644


def test_atomic_replace_other_oserror_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("old\n", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new\n")

    def fail_replace(src: str, dst: str) -> None:
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC), src, None, dst)

    monkeypatch.setattr("utils.os.replace", fail_replace)

    with pytest.raises(OSError) as excinfo:
        atomic_replace(tmp, target)
    assert excinfo.value.errno == errno.ENOSPC
    assert target.read_text(encoding="utf-8") == "old\n"
    assert tmp.exists()


def test_atomic_replace_eacces_propagates_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On POSIX, EACCES means directory permissions — the copy fallback
    would fail the same way, so the error must propagate unchanged.  Only
    Windows treats EACCES as a (retryable) sharing violation.
    """
    target = tmp_path / "config.yaml"
    target.write_text("old\n", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new\n")

    def fail_replace(src: str, dst: str) -> None:
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), src, None, dst)

    monkeypatch.setattr("utils.os.replace", fail_replace)
    monkeypatch.setattr("utils._IS_WINDOWS", False)

    with pytest.raises(PermissionError):
        atomic_replace(tmp, target)
    assert target.read_text(encoding="utf-8") == "old\n"
    assert tmp.exists()


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


# ─── Windows sharing violations (ERROR_SHARING_VIOLATION → EACCES) ─────────
#
# CPython opens files without FILE_SHARE_DELETE on Windows, so any process
# holding a plain read handle on the target blocks os.replace with a
# PermissionError.  gateway_state.json is rewritten at every turn boundary
# (gateway/run.py _persist_active_agents) while gateway/status.py readers
# poll it — before the fix, every collision silently dropped the status
# update and orphaned a .tmp file.


@pytest.fixture()
def fast_sharing_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils._SHARING_RETRY_DELAY_S", 0.005)


def test_atomic_replace_windows_held_read_handle_falls_back_to_copy(
    tmp_path: Path, fast_sharing_retries: None
) -> None:
    """A read handle held for the whole call: retries exhaust, copy fallback
    lands the new content anyway and cleans up the temp file."""
    if os.name != "nt":
        pytest.skip("Windows-only: exercises a real ERROR_SHARING_VIOLATION")

    target = tmp_path / "gateway_state.json"
    target.write_text('{"active_agents": 1}', encoding="utf-8")
    tmp = _write_tmp(tmp_path, '{"active_agents": 2}')

    with open(target, "r", encoding="utf-8") as reader:
        assert Path(atomic_replace(tmp, target)) == target
        # The reader's handle is still valid and the file was rewritten.
        assert reader is not None

    assert json.loads(target.read_text(encoding="utf-8")) == {"active_agents": 2}
    assert not tmp.exists()


def test_atomic_replace_windows_transient_reader_succeeds_via_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_sharing_retries: None
) -> None:
    """A reader that closes during the retry window: the atomic rename wins
    and the copy fallback is never reached.  The reader is released from the
    retry loop's own sleep hook, so the test is deterministic rather than
    racing wall-clock timers against Windows' coarse timer resolution."""
    if os.name != "nt":
        pytest.skip("Windows-only: exercises a real ERROR_SHARING_VIOLATION")

    def forbid_copy(src: str, dst: str) -> None:
        raise AssertionError("copy fallback must not run when a retry succeeds")

    monkeypatch.setattr("utils.shutil.copyfile", forbid_copy)

    target = tmp_path / "gateway_state.json"
    target.write_text('{"active_agents": 1}', encoding="utf-8")
    tmp = _write_tmp(tmp_path, '{"active_agents": 2}')

    reader = open(target, "r", encoding="utf-8")
    sleeps = []

    def release_reader_on_second_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2 and not reader.closed:
            reader.close()

    monkeypatch.setattr("utils.time.sleep", release_reader_on_second_sleep)
    try:
        assert Path(atomic_replace(tmp, target)) == target
    finally:
        if not reader.closed:
            reader.close()

    # First retry still hit the sharing violation; the second succeeded.
    assert len(sleeps) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"active_agents": 2}
    assert not tmp.exists()


def test_atomic_json_write_windows_concurrent_reader(
    tmp_path: Path, fast_sharing_retries: None
) -> None:
    """End-to-end gateway_state.json scenario: atomic_json_write must not
    raise (or silently orphan its .tmp) while a reader holds the target."""
    if os.name != "nt":
        pytest.skip("Windows-only: exercises a real ERROR_SHARING_VIOLATION")

    target = tmp_path / "gateway_state.json"
    atomic_json_write(target, {"active_agents": 1})

    with open(target, "r", encoding="utf-8"):
        atomic_json_write(target, {"active_agents": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"active_agents": 2}
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*.tmp"))
    assert leftovers == [], f"orphaned temp files: {leftovers}"


def test_atomic_replace_sharing_violation_simulated_retry_then_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_sharing_retries: None
) -> None:
    """Cross-platform pin of the retry-then-fallback sequence: the rename is
    attempted 1 + _SHARING_RETRY_ATTEMPTS times before the copy runs."""
    import utils as utils_mod

    target = tmp_path / "gateway_state.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    attempts = []

    def always_sharing_violation(src: str, dst: str) -> None:
        attempts.append(src)
        raise PermissionError(errno.EACCES, "sharing violation", src, None, dst)

    monkeypatch.setattr("utils.os.replace", always_sharing_violation)
    monkeypatch.setattr("utils._IS_WINDOWS", True)

    assert Path(atomic_replace(tmp, target)) == target
    assert len(attempts) == 1 + utils_mod._SHARING_RETRY_ATTEMPTS
    assert target.read_text(encoding="utf-8") == "new"
    assert not tmp.exists()
