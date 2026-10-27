"""Behaviour contracts for profile-local cron export custody."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import threading

import pytest

from hermes_cron_export_custody import (
    COORDINATION_FILENAME,
    CronCustodyRefused,
    cron_capture_custody,
    cron_writer_custody,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _child(profile: Path, mode: str) -> subprocess.Popen[str]:
    function = "cron_capture_custody" if mode == "capture" else "cron_writer_custody"
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_REPO_ROOT)!r})
        from hermes_cron_export_custody import {function}

        with {function}(timeout_seconds=5.0) as lease:
            lease.assert_current()
            print("ready", flush=True)
            sys.stdin.readline()
        """
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env={**os.environ, "HERMES_HOME": str(profile)},
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_ready(process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    ready = threading.Event()
    lines: list[str] = []

    def read_line() -> None:
        lines.append(process.stdout.readline())
        ready.set()

    threading.Thread(target=read_line, daemon=True).start()
    assert ready.wait(timeout=10.0), "custody child did not report readiness"
    assert lines == ["ready\n"]


def _release_child(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("release\n")
    process.stdin.flush()
    process.wait(timeout=10.0)
    assert process.returncode == 0


def test_fresh_child_custody_does_not_initialize_business_stores(profile: Path) -> None:
    before_mode = profile.stat().st_mode
    child = _child(profile, "writer")
    try:
        _wait_ready(child)
        assert {p.relative_to(profile).as_posix() for p in profile.rglob("*")} == {
            COORDINATION_FILENAME
        }
        assert profile.stat().st_mode == before_mode
        _release_child(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10.0)


@pytest.mark.platforms("posix")
def test_existing_broad_coordination_permissions_refuse_without_repair(profile: Path) -> None:
    lock = profile / COORDINATION_FILENAME
    lock.write_bytes(b"")
    lock.chmod(0o666)
    before_mode = lock.stat().st_mode
    with pytest.raises(CronCustodyRefused):
        with cron_capture_custody(timeout_seconds=2.0):
            pass
    assert lock.stat().st_mode == before_mode


def test_shared_writer_custody_coexists_across_processes(profile: Path) -> None:
    child = _child(profile, "writer")
    try:
        with cron_writer_custody(timeout_seconds=2.0) as lease:
            lease.assert_current()
            _wait_ready(child)
        _release_child(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10.0)


def test_exclusive_capture_refuses_writer_traffic_and_recovers(profile: Path) -> None:
    with cron_writer_custody(timeout_seconds=2.0):
        with pytest.raises(CronCustodyRefused) as error:
            with cron_capture_custody(timeout_seconds=2.0):
                pass
        assert error.value.code == "cron_custody_busy"

    with cron_capture_custody(timeout_seconds=2.0) as lease:
        lease.assert_current()


def test_shared_child_enters_while_exclusive_contender_polls(profile: Path) -> None:
    contender_started = threading.Event()
    contender_done = threading.Event()
    contender_result: list[str] = []

    def contend() -> None:
        contender_started.set()
        try:
            with cron_capture_custody(timeout_seconds=5.0):
                contender_result.append("acquired")
        except CronCustodyRefused as error:
            contender_result.append(error.code)
        finally:
            contender_done.set()

    child: subprocess.Popen[str] | None = None
    with cron_writer_custody(timeout_seconds=2.0):
        thread = threading.Thread(target=contend)
        thread.start()
        assert contender_started.wait(timeout=5.0)
        child = _child(profile, "writer")
        try:
            _wait_ready(child)
            _release_child(child)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10.0)
    assert contender_done.wait(timeout=10.0)
    thread.join(timeout=1.0)
    assert contender_result == ["acquired"]


def test_writer_reentry_refuses_during_capture_and_recovers(profile: Path) -> None:
    started = threading.Event()
    result: list[str] = []

    def writer() -> None:
        started.set()
        try:
            with cron_writer_custody(timeout_seconds=2.0):
                result.append("acquired")
        except CronCustodyRefused as error:
            result.append(error.code)

    with cron_capture_custody(timeout_seconds=2.0) as lease:
        thread = threading.Thread(target=writer)
        thread.start()
        assert started.wait(timeout=5.0)
        thread.join(timeout=5.0)
        lease.assert_current()
    assert result == ["cron_custody_busy"]
    with cron_writer_custody(timeout_seconds=2.0):
        pass


@pytest.mark.parametrize(
    "bad_timeout",
    [True, False, 0, -1, "2", object(), float("nan"), float("inf")],
)
def test_timeout_budget_is_strict(profile: Path, bad_timeout: object) -> None:
    with pytest.raises(CronCustodyRefused) as error:
        with cron_writer_custody(timeout_seconds=bad_timeout):
            pass
    assert error.value.code == "cron_custody_timeout_invalid"


def test_profiles_use_independent_coordination_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(first))
    with cron_capture_custody(timeout_seconds=2.0) as first_lease:
        monkeypatch.setenv("HERMES_HOME", str(second))
        with cron_capture_custody(timeout_seconds=2.0) as second_lease:
            assert first_lease.lock_path != second_lease.lock_path
            assert first_lease.profile_root == first
            assert second_lease.profile_root == second
        monkeypatch.setenv("HERMES_HOME", str(first))
        first_lease.assert_current()


@pytest.mark.platforms("posix")
def test_profile_symlink_substitution_refuses_and_recovers(
    profile: Path, tmp_path: Path,
) -> None:
    moved = tmp_path / "moved-profile"
    lease = None
    try:
        with pytest.raises(CronCustodyRefused):
            with cron_capture_custody(timeout_seconds=2.0) as lease:
                profile.rename(moved)
                profile.symlink_to(moved, target_is_directory=True)
                lease.assert_current()
    finally:
        if profile.is_symlink():
            profile.unlink()
        if moved.exists():
            moved.rename(profile)
    with cron_writer_custody(timeout_seconds=2.0):
        pass
    assert lease is not None


@pytest.mark.platforms("posix")
def test_coordination_symlink_substitution_refuses_and_recovers(
    profile: Path, tmp_path: Path,
) -> None:
    replacement = tmp_path / "replacement-lock"
    lock_path = profile / COORDINATION_FILENAME
    with pytest.raises(CronCustodyRefused):
        with cron_writer_custody(timeout_seconds=2.0) as lease:
            lock_path.rename(replacement)
            lock_path.symlink_to(replacement)
            lease.assert_current()
    lock_path.unlink()
    replacement.rename(lock_path)
    with cron_capture_custody(timeout_seconds=2.0):
        pass


def test_os_termination_releases_real_lock(profile: Path) -> None:
    child = _child(profile, "capture")
    try:
        _wait_ready(child)
        child.kill()
        child.wait(timeout=10.0)
        with cron_writer_custody(timeout_seconds=2.0):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10.0)


def test_release_checks_identity_and_closed_leases_refuse_use(profile: Path) -> None:
    with cron_writer_custody(timeout_seconds=2.0) as lease:
        lease.assert_current()
        lease.release()
        with pytest.raises(CronCustodyRefused) as error:
            lease.assert_current()
        assert error.value.code == "cron_custody_released"


@pytest.mark.platforms("posix")
def test_parent_directory_alias_preserves_real_profile_custody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "physical"
    home = parent / "profile"
    home.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias / "profile"))
    with cron_writer_custody(timeout_seconds=2.0) as lease:
        lease.assert_current()
        with pytest.raises(CronCustodyRefused, match="cron_custody_busy"):
            with cron_capture_custody(timeout_seconds=2.0):
                pass
    with cron_capture_custody(timeout_seconds=2.0):
        pass
    assert not (home / "cron").exists()


@pytest.mark.platforms("posix")
def test_parent_directory_alias_retarget_refuses_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for parent in (first, second):
        (parent / "profile").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(first, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias / "profile"))
    manager = cron_capture_custody(timeout_seconds=2.0)
    lease = manager.__enter__()
    try:
        alias.unlink()
        alias.symlink_to(second, target_is_directory=True)
        with pytest.raises(CronCustodyRefused):
            lease.assert_current()
        alias.unlink()
        alias.symlink_to(first, target_is_directory=True)
        # Restoring an equivalent target still replaces the retained alias
        # identity, so a fresh generation is required.
        with pytest.raises(CronCustodyRefused):
            lease.assert_current()
    finally:
        # Acquisition above must succeed; only stale-lease cleanup may refuse.
        with pytest.raises(CronCustodyRefused):
            manager.__exit__(None, None, None)
    with cron_writer_custody(timeout_seconds=2.0):
        pass


@pytest.mark.platforms("posix")
def test_parent_alias_target_permission_drift_refuses_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "physical"
    (parent / "profile").mkdir(parents=True)
    parent.chmod(0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias / "profile"))
    manager = cron_capture_custody(timeout_seconds=2.0)
    lease = manager.__enter__()
    try:
        parent.chmod(0o700)
        with pytest.raises(CronCustodyRefused):
            lease.assert_current()
    finally:
        try:
            with pytest.raises(CronCustodyRefused):
                manager.__exit__(None, None, None)
        finally:
            parent.chmod(0o755)
    with cron_writer_custody(timeout_seconds=2.0):
        pass


@pytest.mark.platforms("posix")
def test_parent_alias_intermediate_physical_permission_drift_refuses_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = tmp_path / "physical"
    parent = physical / "parent"
    (parent / "profile").mkdir(parents=True)
    physical.chmod(0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(parent, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias / "profile"))
    manager = cron_capture_custody(timeout_seconds=2.0)
    lease = manager.__enter__()
    try:
        physical.chmod(0o700)
        with pytest.raises(CronCustodyRefused):
            lease.assert_current()
    finally:
        try:
            with pytest.raises(CronCustodyRefused):
                manager.__exit__(None, None, None)
        finally:
            physical.chmod(0o755)
    with cron_writer_custody(timeout_seconds=2.0):
        pass
