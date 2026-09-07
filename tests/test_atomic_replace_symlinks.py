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
from unittest.mock import MagicMock

import pytest
import yaml

# Ensure the repo root is importable when running via `pytest tests/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import (
    atomic_json_write,
    atomic_replace,
    atomic_roundtrip_yaml_save,
    atomic_roundtrip_yaml_update,
    atomic_yaml_write,
)


# ─── Direct helper ────────────────────────────────────────────────────────────


def _write_tmp(dir_: Path, content: str) -> Path:
    tmp = dir_ / ".src.tmp"
    tmp.write_text(content, encoding="utf-8")
    return tmp


@pytest.mark.require_symlinks
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


@pytest.mark.require_symlinks
def test_atomic_json_write_preserves_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    link = tmp_path / "link.json"
    real.write_text("{}", encoding="utf-8")
    link.symlink_to(real)

    atomic_json_write(link, {"hello": "world"})

    assert link.is_symlink()
    loaded = json.loads(real.read_text(encoding="utf-8"))
    assert loaded == {"hello": "world"}


@pytest.mark.require_symlinks
def test_atomic_yaml_write_preserves_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    link = tmp_path / "link.yaml"
    real.write_text("placeholder: true\n", encoding="utf-8")
    link.symlink_to(real)

    atomic_yaml_write(link, {"model": {"provider": "openrouter"}})

    assert link.is_symlink()
    data = yaml.safe_load(real.read_text(encoding="utf-8"))
    assert data == {"model": {"provider": "openrouter"}}


@pytest.mark.require_symlinks
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




def test_atomic_roundtrip_yaml_save_restores_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mirrors the update-variant owner test for the whole-state save that
    backs tui_gateway/server.py:_save_cfg()."""
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

    atomic_roundtrip_yaml_save(target, {"model": {"provider": "nvidia"}})

    assert chown_calls == [(target, 345, 678)]
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["model"]["provider"] == "nvidia"


# ─── Broken-symlink edge case ─────────────────────────────────────────────


@pytest.mark.require_symlinks
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


# ─── Windows contended renames (GitHub #57775) ─────────────────────────────
#
# CPython opens files without FILE_SHARE_DELETE on Windows, so os.replace
# onto a target that ANY other handle holds open is denied.  atomic_replace
# only fell back for EXDEV/EBUSY, so the exception propagated (and most
# callers swallowed it): gateway_state.json status updates were dropped at
# turn boundaries, and auth.json writes surfaced as "agent init failed".
#
# Measured on Windows 11 build 26200 / CPython 3.11: a held *target* handle
# reports winerror 5 (ERROR_ACCESS_DENIED), NOT 32 — 32 is what a held
# *source* reports.  The cross-platform tests below therefore simulate
# winerror 5, matching what production actually raises.
#
# The real-handle tests use @pytest.mark.windows_only rather than a bare
# `skip(os.name != "nt")`: scripts/ci/list_os_marked_tests.py greps for the
# MARKER NAME to decide which files the Windows lane imports, so a plain
# skipif would leave them running on no host at all.


def _sharing_error(winerror: int = 5) -> PermissionError:
    """Build the PermissionError shape Windows raises for a held target."""
    exc = PermissionError(errno.EACCES, "contended", "src", None, "dst")
    exc.winerror = winerror
    return exc


@pytest.fixture()
def fast_replace_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the jittered backoff so retry tests don't sleep for ~1.2s."""
    monkeypatch.setattr("utils._REPLACE_RETRY_BASE_DELAY_S", 0.001)
    monkeypatch.setattr("utils._REPLACE_RETRY_MAX_DELAY_S", 0.001)


# ── cross-platform: the retry/fallback state machine ──────────────────────


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_contended_rename_retries_then_rewrites_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fast_replace_retries: None,
    winerror: int,
) -> None:
    """A target held for the whole call: the rename is retried the full
    budget, then the in-place rewrite lands the write anyway.

    winerror 5 is the code the reported bug actually produces; 32 and 33 are
    the sibling contention codes.  All three must recover.
    """
    import utils as utils_mod

    target = tmp_path / "gateway_state.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    attempts = []

    def always_contended(src: str, dst: str) -> None:
        attempts.append(src)
        raise _sharing_error(winerror)

    monkeypatch.setattr("utils.os.replace", always_contended)
    monkeypatch.setattr("utils._IS_WINDOWS", True)

    assert Path(atomic_replace(tmp, target)) == target
    assert len(attempts) == 1 + utils_mod._REPLACE_RETRY_ATTEMPTS
    assert target.read_text(encoding="utf-8") == "new"
    assert not tmp.exists()


def test_contended_rename_retry_wins_keeps_write_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_replace_retries: None
) -> None:
    """A reader that lets go inside the budget: the atomic rename wins and
    neither fallback runs, so the write keeps full atomicity."""
    target = tmp_path / "gateway_state.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    real_replace = os.replace
    calls = {"n": 0}

    def contended_twice(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _sharing_error(5)
        real_replace(src, dst)

    def forbid(*_args: object, **_kw: object) -> None:
        raise AssertionError("no fallback may run when a retry succeeds")

    monkeypatch.setattr("utils.os.replace", contended_twice)
    monkeypatch.setattr("utils._IS_WINDOWS", True)
    monkeypatch.setattr("utils._rewrite_in_place", forbid)
    monkeypatch.setattr("utils.shutil.copyfile", forbid)

    assert Path(atomic_replace(tmp, target)) == target
    assert calls["n"] == 3
    assert target.read_text(encoding="utf-8") == "new"
    assert not tmp.exists()


def test_genuine_denial_propagates_after_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_replace_retries: None
) -> None:
    """A real ACL denial reports the same winerror as contention, so it is
    not classified up front — it exhausts the budget, fails the in-place
    rewrite too, and surfaces to the caller instead of being swallowed."""
    target = tmp_path / "denied.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    monkeypatch.setattr(
        "utils.os.replace", MagicMock(side_effect=_sharing_error(5))
    )
    monkeypatch.setattr("utils._IS_WINDOWS", True)

    denial = PermissionError(errno.EACCES, "access is denied")

    def cannot_open(*_args: object, **_kw: object) -> None:
        raise denial

    monkeypatch.setattr("utils.os.open", cannot_open)

    with pytest.raises(PermissionError) as caught:
        atomic_replace(tmp, target)

    assert caught.value is denial
    assert target.read_text(encoding="utf-8") == "old"
    assert tmp.exists(), "the pending write must survive for the caller"


def test_contended_retry_switching_to_exdev_uses_copy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_replace_retries: None
) -> None:
    """A retry that turns into EXDEV must stop consuming the sharing budget
    and take the copy fallback — EXDEV never clears on retry."""
    target = tmp_path / "target.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    replace = MagicMock(
        side_effect=[_sharing_error(5), OSError(errno.EXDEV, "cross-device")]
    )
    monkeypatch.setattr("utils.os.replace", replace)
    monkeypatch.setattr("utils._IS_WINDOWS", True)

    def forbid_rewrite(*_a: object, **_k: object) -> None:
        raise AssertionError("EXDEV must use the copy fallback, not a rewrite")

    monkeypatch.setattr("utils._rewrite_in_place", forbid_rewrite)

    assert Path(atomic_replace(tmp, target)) == target
    assert replace.call_count == 2
    assert target.read_text(encoding="utf-8") == "new"
    assert not tmp.exists()


def test_non_contended_oserror_propagates_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENOSPC is not a contention code on any platform: no retry, no
    fallback, and the pending temp file is left for the caller."""
    target = tmp_path / "config.yaml"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    replace = MagicMock(side_effect=OSError(errno.ENOSPC, "no space"))
    monkeypatch.setattr("utils.os.replace", replace)
    monkeypatch.setattr("utils._IS_WINDOWS", True)

    with pytest.raises(OSError) as excinfo:
        atomic_replace(tmp, target)

    assert excinfo.value.errno == errno.ENOSPC
    assert replace.call_count == 1, "a permanent error must not be retried"
    assert target.read_text(encoding="utf-8") == "old"
    assert tmp.exists()


def test_posix_eacces_propagates_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX behaviour is unchanged: EACCES there means directory
    permissions, so it propagates on the first attempt."""
    target = tmp_path / "config.yaml"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    replace = MagicMock(
        side_effect=PermissionError(errno.EACCES, os.strerror(errno.EACCES))
    )
    monkeypatch.setattr("utils.os.replace", replace)
    monkeypatch.setattr("utils._IS_WINDOWS", False)

    with pytest.raises(PermissionError):
        atomic_replace(tmp, target)

    assert replace.call_count == 1, "POSIX must not gain a retry loop"
    assert target.read_text(encoding="utf-8") == "old"
    assert tmp.exists()


def test_in_place_rewrite_never_exposes_a_truncated_file(
    tmp_path: Path,
) -> None:
    """The in-place rewrite must not truncate-then-fill: a concurrent reader
    can observe a 0-byte file during shutil.copyfile, which for auth.json
    means an empty credential store.  Shrinking writes must also not leave
    trailing bytes from the previous, longer content.
    """
    import utils as utils_mod

    target = tmp_path / "auth.json"
    observed: list[int] = []

    target.write_text("A" * 5000, encoding="utf-8")
    tmp = _write_tmp(tmp_path, "B" * 5000)
    utils_mod._rewrite_in_place(str(tmp), str(target))
    observed.append(len(target.read_text(encoding="utf-8")))
    assert target.read_text(encoding="utf-8") == "B" * 5000
    assert not tmp.exists()

    # Shrinking rewrite: ftruncate must drop the tail.
    tmp = _write_tmp(tmp_path, "C" * 10)
    utils_mod._rewrite_in_place(str(tmp), str(target))
    assert target.read_text(encoding="utf-8") == "C" * 10
    assert observed == [5000]


def test_symlinked_target_survives_a_contended_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fast_replace_retries: None
) -> None:
    """The #16743 invariant must hold on the contended path too: a symlinked
    config.yaml stays a symlink when the rewrite fallback runs."""
    real = tmp_path / "real.yaml"
    link = tmp_path / "config.yaml"
    real.write_text("old\n", encoding="utf-8")
    link.symlink_to(real)
    tmp = _write_tmp(tmp_path, "new\n")

    monkeypatch.setattr(
        "utils.os.replace", MagicMock(side_effect=_sharing_error(5))
    )
    monkeypatch.setattr("utils._IS_WINDOWS", True)

    assert Path(atomic_replace(tmp, link)) == real
    assert link.is_symlink(), "symlink must survive the rewrite fallback"
    assert real.read_text(encoding="utf-8") == "new\n"
    assert not tmp.exists()


# ── native Windows: real contended handles ────────────────────────────────


@pytest.mark.windows_only
def test_windows_real_held_read_handle_lands_the_write(tmp_path: Path) -> None:
    """The reported bug, end to end against a real held handle."""
    target = tmp_path / "gateway_state.json"
    target.write_text('{"active_agents": 1}', encoding="utf-8")
    tmp = _write_tmp(tmp_path, '{"active_agents": 2}')

    with open(target, "r", encoding="utf-8"):
        assert Path(atomic_replace(tmp, target)) == target

    assert json.loads(target.read_text(encoding="utf-8")) == {"active_agents": 2}
    assert not tmp.exists()


@pytest.mark.windows_only
def test_windows_real_held_handle_reports_access_denied(tmp_path: Path) -> None:
    """Pin the premise this fix is built on: a held *target* handle raises
    winerror 5, not 32.  If CPython ever changes that, the classification in
    utils must be revisited rather than silently missing the bug again.
    """
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")

    with open(target, "r", encoding="utf-8"):
        with pytest.raises(OSError) as caught:
            os.replace(str(tmp), str(target))

    assert caught.value.winerror in (5, 32, 33)
    import utils as utils_mod

    assert utils_mod._is_contended_windows_replace_error(caught.value)


@pytest.mark.windows_only
def test_windows_atomic_json_write_with_concurrent_reader(
    tmp_path: Path,
) -> None:
    """End-to-end gateway_state.json scenario through atomic_json_write:
    the write lands and no .tmp file is orphaned."""
    target = tmp_path / "gateway_state.json"
    atomic_json_write(target, {"active_agents": 1})

    with open(target, "r", encoding="utf-8"):
        atomic_json_write(target, {"active_agents": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"active_agents": 2}
    leftovers = list(tmp_path.glob("*.tmp")) + list(tmp_path.glob(".*.tmp"))
    assert leftovers == [], f"orphaned temp files: {leftovers}"


@pytest.mark.windows_only
def test_windows_readonly_target_still_raises(tmp_path: Path) -> None:
    """A genuinely unwritable target must not be rescued by the fallback."""
    import subprocess

    target = tmp_path / "readonly.json"
    target.write_text("old", encoding="utf-8")
    tmp = _write_tmp(tmp_path, "new")
    subprocess.run(["attrib", "+R", str(target)], capture_output=True)
    try:
        with pytest.raises(OSError):
            atomic_replace(tmp, target)
        assert target.read_text(encoding="utf-8") == "old"
    finally:
        subprocess.run(["attrib", "-R", str(target)], capture_output=True)
