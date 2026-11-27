"""Writer-owned OS leases: real spawn processes, never simulated host OSes."""
import errno
import hashlib
import multiprocessing
import os
import signal
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.skill_catalog import locks
from hermes_cli.skill_catalog.validation import CatalogError


_CTX = multiprocessing.get_context("spawn")


def _writer(home, targets, ready, release):
    try:
        with locks.writer_hold(Path(home), targets):
            ready.send("held")
            release.recv()
        ready.send("closed")
    finally:
        ready.close()
        release.close()


@contextmanager
def _held_writer(home, targets):
    ready, child_ready = _CTX.Pipe()
    release, child_release = _CTX.Pipe()
    process = _CTX.Process(target=_writer, args=(str(home), targets, child_ready, child_release))
    process.start()
    child_ready.close()
    child_release.close()
    try:
        assert ready.poll(10), "spawn writer did not report admission"
        assert ready.recv() == "held"
        yield process
    finally:
        if process.is_alive():
            with suppress(BrokenPipeError, EOFError):
                release.send("exit")
        process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        ready.close()
        release.close()


def _busy(home, targets=(), **kwargs):
    with pytest.raises(CatalogError) as caught:
        with locks.writer_hold(home, targets, deadline=time.monotonic(), **kwargs):
            pytest.fail("overlapping writer entered")
    assert caught.value.code == "busy"


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("other", [("a/b",), ("a",), ("a/b/c",)])
def test_spawn_equal_ancestor_descendant_conflict(tmp_path, other):
    with _held_writer(tmp_path, ("a/b",)):
        _busy(tmp_path, other)
        with locks.writer_hold(tmp_path, ("a/c", "z"), deadline=time.monotonic()):
            pass
        _busy(tmp_path, root_exclusive=True)
    with locks.writer_hold(tmp_path, other, root_exclusive=True):
        pass


@pytest.mark.platforms("posix", "windows")
def test_kernel_releases_killed_writer(tmp_path):
    with _held_writer(tmp_path, ("a/b",)) as process:
        process.kill()
        process.join(10)
        assert not process.is_alive()
        with locks.writer_hold(tmp_path, ("a",), root_exclusive=True):
            pass


@pytest.mark.platforms("posix", "windows")
def test_multiple_overlapping_endpoints_coalesce(tmp_path):
    with locks.writer_hold(tmp_path, ("a/b/c", "a", "a/b", "a"), deadline=time.monotonic()):
        _busy(tmp_path, ("a/d",))
    with locks.writer_hold(tmp_path, ("a",)):
        pass


@pytest.mark.platforms("posix", "windows")
def test_shared_and_exclusive_leases_and_idempotent_close(tmp_path):
    path = tmp_path / "lease"
    first = locks.FileLease(path, shared=True).acquire()
    assert first.acquire() is first
    with locks.FileLease(path, shared=True, deadline=time.monotonic()):
        with pytest.raises(CatalogError) as caught:
            locks.FileLease(path, deadline=time.monotonic()).acquire()
        assert caught.value.code == "busy"
    first.close()
    first.close()
    with locks.FileLease(path):
        pass
    assert path.is_file()


@pytest.mark.platforms("posix", "windows")
def test_exception_and_partial_chain_release_without_unlink(tmp_path):
    with _held_writer(tmp_path, ("z",)):
        _busy(tmp_path, ("a/b", "z"))
        with locks.writer_hold(tmp_path, ("a",)):
            pass
    with pytest.raises(RuntimeError, match="effect failed"):
        with locks.writer_hold(tmp_path, ("a/b",)):
            raise RuntimeError("effect failed")
    directory = tmp_path / "cache/skill-catalog/live-v1"
    files = {path: path.stat().st_ino for path in directory.rglob("*") if path.is_file()}
    with locks.writer_hold(tmp_path, ("a",), root_exclusive=True):
        pass
    assert {path: path.stat().st_ino for path in files} == files


@pytest.mark.parametrize("target", ["", "/a", "../a", "a/..", "a//b", "a/", "C:a", "a\\b", ".hub/a", "a/.hidden", "a\x00", "a\ud800", b"a"])
def test_invalid_targets_fail_before_creating_cache(tmp_path, target):
    with pytest.raises(CatalogError) as caught:
        with locks.writer_hold(tmp_path, (target,)):
            pytest.fail("invalid target admitted")
    assert caught.value.code == "ownership_conflict"
    assert not (tmp_path / "cache").exists()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("redirect", ["skills", "skills/a", "cache", "cache/skill-catalog/live-v1/writers"])
def test_symlink_targets_and_lock_directories_refuse(tmp_path, redirect):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / redirect
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(CatalogError):
        with locks.writer_hold(tmp_path, ("a/b",)):
            pytest.fail("redirect admitted")
    assert list(outside.iterdir()) == []


@pytest.mark.platforms("posix", "windows")
def test_exact_utf8_hash_names_and_modes(tmp_path):
    targets = ("é/a", "é/a", "Z", "z")
    with locks.writer_hold(tmp_path, targets):
        directory = tmp_path / "cache/skill-catalog/live-v1/writers"
        names = {path.name for path in directory.iterdir()}
        expected = {hashlib.sha256(path.encode("utf-8")).hexdigest()
                    for path in (*targets, "é", "é")}
        assert names == expected
        _busy(tmp_path, ("é",))
        _busy(tmp_path, ("é",))


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("number", [errno.EIO, errno.EBADF, errno.ENOSPC, errno.EINTR])
def test_noncontention_errors_fail_immediately_and_close(tmp_path, monkeypatch, number):
    attempts = []
    original = locks._try_lock

    def fail(fd, shared):
        attempts.append(fd)
        raise OSError(number, "private diagnostic")

    monkeypatch.setattr(locks, "_try_lock", fail)
    with pytest.raises(CatalogError) as caught:
        locks.FileLease(tmp_path / "lease").acquire()
    assert caught.value.code == "ownership_state_unavailable"
    assert "private diagnostic" not in str(caught.value)
    assert len(attempts) == 1
    with pytest.raises(OSError):
        os.fstat(attempts[0])
    monkeypatch.setattr(locks, "_try_lock", original)
    with locks.FileLease(tmp_path / "lease"):
        pass


@pytest.mark.platforms("posix", "windows")
@pytest.mark.parametrize("number", sorted({errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK, errno.EDEADLK}))
def test_only_contention_errors_retry_with_bounded_deadline(tmp_path, monkeypatch, number):
    attempts = []
    original = locks._try_lock

    def transient(fd, shared):
        attempts.append(fd)
        if len(attempts) == 1:
            raise OSError(number, "contended")
        return original(fd, shared)

    monkeypatch.setattr(locks, "_try_lock", transient)
    with locks.FileLease(tmp_path / "lease", deadline=time.monotonic() + 2):
        assert len(attempts) == 2


@pytest.mark.platforms("posix", "windows")
def test_open_failure_is_unavailable_not_contention(tmp_path, monkeypatch):
    def fail(path):
        raise PermissionError(errno.EACCES, "private open failure")

    monkeypatch.setattr(locks, "_open_file", fail)
    with pytest.raises(CatalogError) as caught:
        locks.FileLease(tmp_path / "lease").acquire()
    assert caught.value.code == "ownership_state_unavailable"


@pytest.mark.platforms("posix", "windows")
def test_descriptor_is_noninheritable_and_close_releases(tmp_path, monkeypatch):
    original = locks._try_lock
    descriptors = []

    def inspect(fd, shared):
        assert not os.get_inheritable(fd)
        descriptors.append(fd)
        return original(fd, shared)

    monkeypatch.setattr(locks, "_try_lock", inspect)
    with locks.FileLease(tmp_path / "lease"):
        assert os.fstat(descriptors[0])
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def _gate_parent(home, ready, release, control, gate_stop):
    with locks.FileLease(Path(home) / "gate.lock"):
        child = _CTX.Process(target=_writer, args=(home, ("a/b",), ready, release))
        child.start()
        control.send(child.pid)
        gate_stop.recv()


@pytest.mark.platforms("posix", "windows")
@pytest.mark.live_system_guard_bypass  # The writer is deliberately orphaned by killing its gate parent.
@pytest.mark.parametrize("kill_writer", [False, True])
def test_writer_hold_survives_gate_parent_death(tmp_path, kill_writer):
    ready, child_ready = _CTX.Pipe()
    release, child_release = _CTX.Pipe()
    control, child_control = _CTX.Pipe()
    gate_stop, child_gate_stop = _CTX.Pipe()
    parent = _CTX.Process(target=_gate_parent, args=(str(tmp_path), child_ready, child_release,
                                                   child_control, child_gate_stop))
    parent.start()
    child_ready.close()
    child_release.close()
    child_control.close()
    child_gate_stop.close()
    writer_pid = None
    try:
        assert control.poll(10)
        writer_pid = control.recv()
        assert ready.poll(10)
        assert ready.recv() == "held"
        with pytest.raises(CatalogError) as caught:
            locks.FileLease(tmp_path / "gate.lock", deadline=time.monotonic()).acquire()
        assert caught.value.code == "busy"
        parent.kill()
        parent.join(10)
        assert not parent.is_alive()
        with locks.FileLease(tmp_path / "gate.lock"):
            for target in ("a/b", "a", "a/b/c"):
                _busy(tmp_path, (target,))
            _busy(tmp_path, root_exclusive=True)
            with locks.writer_hold(tmp_path, ("other",)):
                pass
        if kill_writer:
            os.kill(writer_pid, signal.SIGTERM)
        else:
            release.send("exit")
            assert ready.poll(10)
            assert ready.recv() == "closed"
        with locks.writer_hold(tmp_path, ("a",), root_exclusive=True):
            pass
        writer_pid = None
    finally:
        if parent.is_alive():
            parent.kill()
        parent.join(10)
        if writer_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(writer_pid, signal.SIGTERM)
        for connection in (ready, release, control, gate_stop):
            connection.close()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("where", ["root", "target"])
def test_lock_file_symlink_refuses_without_touching_destination(tmp_path, where):
    destination = tmp_path / "outside"
    destination.write_bytes(b"unchanged")
    directory = tmp_path / "cache/skill-catalog/live-v1"
    filename = (directory / "writer-barrier.lock" if where == "root" else
                directory / "writers" / hashlib.sha256(b"a").hexdigest())
    filename.parent.mkdir(parents=True)
    filename.symlink_to(destination)
    with pytest.raises(CatalogError) as caught:
        with locks.writer_hold(tmp_path, ("a",)):
            pytest.fail("redirected lock admitted")
    assert caught.value.code == "ownership_state_unavailable"
    assert destination.read_bytes() == b"unchanged"


@pytest.mark.platforms("posix", "windows")
def test_default_deadline_is_one_absolute_three_second_budget(tmp_path, monkeypatch):
    clock = [50.0]
    pauses = []

    def monotonic():
        return clock[0]

    def pause(seconds):
        pauses.append(seconds)
        assert 0 < seconds <= 0.01
        clock[0] += seconds

    def contended(fd, shared):
        raise OSError(errno.EAGAIN, "held")

    monkeypatch.setattr(locks, "time", SimpleNamespace(monotonic=monotonic, sleep=pause))
    monkeypatch.setattr(locks, "_try_lock", contended)
    with pytest.raises(CatalogError) as caught:
        locks.FileLease(tmp_path / "lease").acquire()
    assert caught.value.code == "busy"
    assert clock[0] == pytest.approx(53.0)
    assert pauses


@pytest.mark.platforms("posix", "windows")
def test_chain_uses_one_deadline_and_global_depth_utf8_order(tmp_path, monkeypatch):
    acquired = []
    original = locks.FileLease.acquire
    deadline = time.monotonic() + 2

    def record(lease):
        acquired.append((lease.path.name, lease.shared, lease.deadline))
        return original(lease)

    monkeypatch.setattr(locks.FileLease, "acquire", record)
    targets = (target for target in ("z/b", "a/b/c", "a", "z/a", "a/b"))
    with locks.writer_hold(tmp_path, targets, deadline=deadline):
        pass
    expected = [("writer-barrier.lock", True, deadline)]
    for target, shared in (("a", False), ("z", True), ("a/b", False),
                           ("z/a", False), ("z/b", False), ("a/b/c", False)):
        expected.append((hashlib.sha256(target.encode("utf-8")).hexdigest(), shared, deadline))
    assert acquired == expected


@pytest.mark.platforms("windows")
def test_native_windows_shared_lease_rejects_exclusive_open(tmp_path):
    path = tmp_path / "shared-presence"
    with locks.FileLease(path, shared=True):
        with locks.FileLease(path, shared=True):
            with pytest.raises(CatalogError) as caught:
                locks.FileLease(path, deadline=time.monotonic()).acquire()
            assert caught.value.code == "busy"
