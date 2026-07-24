"""Security contract for the cron jobs advisory lock."""

import os
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


@pytest.fixture
def isolated_jobs(monkeypatch):
    """Import cron.jobs only after pinning a disposable Hermes home."""
    with tempfile.TemporaryDirectory(prefix="age3472-") as temp_root:
        hermes_home = (Path(temp_root) / "hermes-home").resolve()
        hermes_home.mkdir()
        default_home = (Path.home() / ".hermes").resolve()
        assert hermes_home != default_home

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
        sys.dont_write_bytecode = True

        from cron import jobs

        assert jobs.get_hermes_home().resolve() == hermes_home
        with jobs.use_cron_store(hermes_home):
            yield jobs, hermes_home


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics required")
def test_new_jobs_lock_is_0600_under_umask_022(isolated_jobs):
    jobs, hermes_home = isolated_jobs
    previous_umask = os.umask(0o022)
    try:
        with jobs._jobs_lock():
            lock_path = jobs._jobs_lock_file()
            lock_stat = os.lstat(lock_path)
    finally:
        os.umask(previous_umask)

    assert lock_path.parent == hermes_home / "cron"
    assert stat.S_ISREG(lock_stat.st_mode)
    assert not lock_path.is_symlink()
    assert lock_stat.st_uid == os.geteuid()  # windows-footgun: ok
    assert lock_stat.st_nlink == 1
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics required")
def test_existing_0644_lock_hardens_in_place(isolated_jobs):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()

    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.write(fd, b"lock-sentinel")
        os.close(fd)
    finally:
        os.umask(previous_umask)

    before = os.lstat(lock_path)
    assert stat.S_IMODE(before.st_mode) == 0o644

    with jobs._jobs_lock():
        during = os.lstat(lock_path)
        assert stat.S_IMODE(during.st_mode) == 0o600
        assert (during.st_dev, during.st_ino) == (before.st_dev, before.st_ino)

    after = os.lstat(lock_path)
    assert lock_path.read_bytes() == b"lock-sentinel"
    assert stat.S_IMODE(after.st_mode) == 0o600
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics required")
def test_jobs_lock_descriptor_is_non_inheritable(isolated_jobs, monkeypatch):
    jobs, _ = isolated_jobs
    if jobs.fcntl is None:
        pytest.skip("fcntl.flock unavailable")

    original_flock = jobs.fcntl.flock
    inheritable_at_lock = []

    def observing_flock(fd, operation):
        if operation & jobs.fcntl.LOCK_EX:
            inheritable_at_lock.append(os.get_inheritable(fd))
        return original_flock(fd, operation)

    monkeypatch.setattr(jobs.fcntl, "flock", observing_flock)
    with jobs._jobs_lock():
        pass

    assert inheritable_at_lock == [False]


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics required")
def test_jobs_lock_rejects_symlink_without_touching_target(isolated_jobs):
    jobs, hermes_home = isolated_jobs
    jobs.ensure_dirs()
    target = hermes_home / "lock-target"
    target.write_bytes(b"unchanged")
    target.chmod(0o640)
    before = os.lstat(target)
    lock_path = jobs._jobs_lock_file()
    lock_path.symlink_to(target)

    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail("unsafe symlink lock yielded its critical section")

    after = os.lstat(target)
    assert target.read_bytes() == b"unchanged"
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert lock_path.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX file types required")
@pytest.mark.parametrize("lock_kind", ["directory", "fifo", "socket"])
def test_jobs_lock_rejects_non_regular_file(isolated_jobs, lock_kind):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    socket_handle = None

    if lock_kind == "directory":
        lock_path.mkdir(mode=0o750)
    elif lock_kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation unavailable")
        os.mkfifo(lock_path, 0o640)
    else:
        import socket

        if not hasattr(socket, "AF_UNIX"):
            pytest.skip("Unix-domain sockets unavailable")
        socket_handle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        socket_handle.bind(str(lock_path))

    before = os.lstat(lock_path)
    try:
        with pytest.raises(jobs.JobsLockSecurityError):
            with jobs._jobs_lock():
                pytest.fail(f"unsafe {lock_kind} lock yielded its critical section")
    finally:
        if socket_handle is not None:
            socket_handle.close()

    after = os.lstat(lock_path)
    assert stat.S_IFMT(after.st_mode) == stat.S_IFMT(before.st_mode)
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="effective UID unavailable")
def test_jobs_lock_rejects_wrong_owner_without_chmod(isolated_jobs, monkeypatch):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)

    before = os.lstat(lock_path)
    monkeypatch.setattr(jobs.os, "geteuid", lambda: before.st_uid + 1)

    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail("wrong-owner lock yielded its critical section")

    after = os.lstat(lock_path)
    assert stat.S_IMODE(after.st_mode) == 0o644
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX hard-link semantics required")
def test_jobs_lock_rejects_unexpected_link_count(isolated_jobs):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    alias_path = lock_path.with_name(".jobs.lock.alias")
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)
    os.link(lock_path, alias_path)

    before = os.lstat(lock_path)
    assert before.st_nlink == 2

    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail("multiply-linked lock yielded its critical section")

    after = os.lstat(lock_path)
    alias = os.lstat(alias_path)
    assert after.st_nlink == alias.st_nlink == 2
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(alias.st_mode) == 0o644
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert (alias.st_dev, alias.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX inode semantics required")
@pytest.mark.parametrize("drift_point", ["before_open", "after_open"])
def test_jobs_lock_rejects_identity_drift(isolated_jobs, monkeypatch, drift_point):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    displaced_path = lock_path.with_name(f".jobs.lock.{drift_point}")
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)

    original_open = jobs.os.open
    drifted = False

    def replacement_file():
        replacement_fd = original_open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644
        )
        os.close(replacement_fd)

    def drifting_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal drifted
        is_existing_lock_open = (
            Path(path) == lock_path and not flags & os.O_CREAT and not drifted
        )
        if not is_existing_lock_open:
            if dir_fd is None:
                return original_open(path, flags, mode)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        drifted = True
        if drift_point == "before_open":
            lock_path.rename(displaced_path)
            replacement_file()
            return original_open(path, flags, mode)

        opened_fd = original_open(path, flags, mode)
        lock_path.rename(displaced_path)
        replacement_file()
        return opened_fd

    monkeypatch.setattr(jobs.os, "open", drifting_open)

    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail(f"{drift_point} identity drift yielded its critical section")

    assert drifted
    assert os.path.samestat(os.lstat(lock_path), os.lstat(displaced_path)) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor chmod required")
@pytest.mark.parametrize("failure_kind", ["error", "no_op", "path_swap"])
def test_jobs_lock_fails_closed_when_hardening_is_unverified(
    isolated_jobs, monkeypatch, failure_kind
):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    displaced_path = lock_path.with_name(".jobs.lock.displaced")
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)

    original_fchmod = jobs.os.fchmod
    original_open = jobs.os.open

    def failing_fchmod(fd, mode):
        if failure_kind == "error":
            raise OSError("simulated descriptor chmod failure")
        if failure_kind == "no_op":
            return None

        original_fchmod(fd, mode)
        lock_path.rename(displaced_path)
        replacement_fd = original_open(
            lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
        )
        os.close(replacement_fd)
        return None

    monkeypatch.setattr(jobs.os, "fchmod", failing_fchmod)

    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail(
                f"unverified {failure_kind} hardening yielded its critical section"
            )

    if failure_kind in {"error", "no_op"}:
        assert stat.S_IMODE(os.lstat(lock_path).st_mode) == 0o644
    else:
        assert displaced_path.exists()
        assert not os.path.samestat(os.lstat(lock_path), os.lstat(displaced_path))


@pytest.mark.parametrize("missing_capability", ["O_NOFOLLOW", "fchmod", "geteuid"])
def test_jobs_lock_fails_closed_without_security_capability(
    isolated_jobs, monkeypatch, missing_capability
):
    jobs, _ = isolated_jobs
    monkeypatch.delattr(jobs.os, missing_capability, raising=False)

    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail(
                f"lock yielded without required {missing_capability} capability"
            )


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics required")
def test_legacy_lock_fails_closed_without_cross_process_backend(
    isolated_jobs, monkeypatch
):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)
    before = os.lstat(lock_path)

    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", None)
    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            pytest.fail("legacy lock yielded without cross-process exclusion")

    after = os.lstat(lock_path)
    assert stat.S_IMODE(after.st_mode) == 0o644
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics required")
def test_legacy_lock_fails_closed_when_cross_process_backend_errors(
    isolated_jobs, monkeypatch
):
    jobs, _ = isolated_jobs
    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)
    before = os.lstat(lock_path)

    class FailingLockBackend:
        LK_LOCK = 1
        LK_UNLCK = 2

        @staticmethod
        def locking(fd, operation, length):
            raise OSError("simulated cross-process lock failure")

    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", FailingLockBackend)
    entered = False
    with pytest.raises(jobs.JobsLockSecurityError):
        with jobs._jobs_lock():
            entered = True

    after = os.lstat(lock_path)
    assert not entered
    assert stat.S_IMODE(after.st_mode) == 0o644
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics required")
def test_private_lock_keeps_process_local_fallback_without_backend(
    isolated_jobs, monkeypatch
):
    jobs, _ = isolated_jobs
    with jobs._jobs_lock():
        pass
    lock_path = jobs._jobs_lock_file()
    before = os.lstat(lock_path)
    assert stat.S_IMODE(before.st_mode) == 0o600

    monkeypatch.setattr(jobs, "fcntl", None)
    monkeypatch.setattr(jobs, "msvcrt", None)
    entered = False
    with jobs._jobs_lock():
        entered = True

    after = os.lstat(lock_path)
    assert entered
    assert stat.S_IMODE(after.st_mode) == 0o600
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock semantics required")
def test_contended_legacy_lock_times_out_without_chmod_then_hardens(
    isolated_jobs, monkeypatch
):
    jobs, hermes_home = isolated_jobs
    if jobs.fcntl is None:
        pytest.skip("fcntl.flock unavailable")

    jobs.ensure_dirs()
    lock_path = jobs._jobs_lock_file()
    previous_umask = os.umask(0)
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(fd)
    finally:
        os.umask(previous_umask)
    before = os.lstat(lock_path)

    ready = hermes_home / "holder-ready"
    release = hermes_home / "holder-release"
    holder_code = """
import fcntl
import os
import pathlib
import sys
import time

hermes_home = pathlib.Path(sys.argv[1]).resolve()
lock_path = pathlib.Path(sys.argv[2])
ready = pathlib.Path(sys.argv[3])
release = pathlib.Path(sys.argv[4])
os.environ["HERMES_HOME"] = str(hermes_home)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
assert hermes_home != (pathlib.Path.home() / ".hermes").resolve()
fd = os.open(lock_path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    ready.write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 15
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
"""
    child_env = os.environ.copy()
    child_env["HERMES_HOME"] = str(hermes_home)
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            holder_code,
            str(hermes_home),
            str(lock_path),
            str(ready),
            str(release),
        ],
        env=child_env,
    )

    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "holder process never acquired the legacy lock"

        monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.2)
        entered = False
        with pytest.raises(jobs.JobsLockSecurityError):
            with jobs._jobs_lock():
                entered = True
        assert not entered

        after_timeout = os.lstat(lock_path)
        assert stat.S_IMODE(after_timeout.st_mode) == 0o644
        assert (after_timeout.st_dev, after_timeout.st_ino) == (
            before.st_dev,
            before.st_ino,
        )
    finally:
        release.write_text("release", encoding="utf-8")
        holder.wait(timeout=15)
        assert holder.returncode == 0

    with jobs._jobs_lock():
        during = os.lstat(lock_path)
        assert stat.S_IMODE(during.st_mode) == 0o600
        assert (during.st_dev, during.st_ino) == (before.st_dev, before.st_ino)

    after = os.lstat(lock_path)
    assert stat.S_IMODE(after.st_mode) == 0o600
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
