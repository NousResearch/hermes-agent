"""Focused tests for machine-global AFK state."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from agent import afk


def _hold_afk_transaction(ready, release):
    with afk.locked_state():
        ready.set()
        release.wait(5)


def _engage_afk_after_signal(attempted, completed, errors):
    attempted.set()
    try:
        afk.engage("from process B")
    except BaseException as exc:
        errors.put(repr(exc))
    else:
        errors.put("<no error>")
        completed.set()


def test_locked_state_yields_valid_off_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with afk.locked_state() as state:
        assert state is None


def test_locked_state_yields_valid_engaged_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    afk.engage("away")

    with afk.locked_state() as state:
        assert state["reason"] == "away"


def test_locked_state_yields_unverifiable_snapshot_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    afk.state_path().write_text("not json", encoding="utf-8")

    with afk.locked_state() as state:
        assert state == {"unverifiable": True}


def test_state_is_shared_by_profiles_under_one_default_root(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    afk.engage("away for lunch")

    assert afk.state_path() == root / "afk.json"
    assert afk.get_state() == {
        "engaged_at": afk.get_state()["engaged_at"],
        "reason": "away for lunch",
    }


def test_path_operations_with_path_only_root_omit_dir_fd_keywords(
    monkeypatch, tmp_path
):
    root = afk._Root(tmp_path, None)
    calls = []

    def record(name):
        def operation(*args, **kwargs):
            calls.append((name, args, kwargs))
            if name == "open":
                return 123
            return None

        return operation

    monkeypatch.setattr(os, "open", record("open"))
    monkeypatch.setattr(os, "stat", record("stat"))
    monkeypatch.setattr(os, "replace", record("replace"))
    monkeypatch.setattr(os, "unlink", record("unlink"))

    afk._path_open(root, afk.STATE_NAME, os.O_RDONLY)
    afk._path_stat(root, afk.STATE_NAME)
    afk._path_replace(root, ".tmp", afk.STATE_NAME)
    afk._path_unlink(root, ".tmp")

    assert [name for name, _args, _kwargs in calls] == [
        "open",
        "stat",
        "replace",
        "unlink",
    ]
    assert all(
        not {"dir_fd", "src_dir_fd", "dst_dir_fd"}.intersection(kwargs)
        for _name, _args, kwargs in calls
    )


def test_root_close_transfers_ownership_before_close_failure(monkeypatch, tmp_path):
    root = object.__new__(afk._Root)
    root.path = tmp_path
    root.fd = 41
    close_calls = []

    def fail_close(fd):
        close_calls.append(fd)
        raise OSError("close failed")

    monkeypatch.setattr(os, "close", fail_close)

    with pytest.raises(OSError, match="close failed"):
        root.close()
    root.close()

    assert root.fd is None
    assert close_calls == [41]


def test_read_state_preserves_body_error_when_close_fails(monkeypatch, tmp_path):
    root = afk._Root(tmp_path, None)
    close_calls = []
    monkeypatch.setattr(afk, "_path_open", lambda *_args, **_kwargs: 31)
    monkeypatch.setattr(os, "fstat", lambda _fd: (_ for _ in ()).throw(ValueError("body failed")))

    def fail_close(fd):
        close_calls.append(fd)
        raise OSError("close failed")

    monkeypatch.setattr(os, "close", fail_close)

    with pytest.raises(ValueError, match="body failed"):
        afk._read_state(root)

    assert close_calls == [31]


def test_read_state_bounds_close_only_failure(monkeypatch, tmp_path):
    root = afk._Root(tmp_path, None)
    close_calls = []
    path = tmp_path / afk.STATE_NAME
    info = path.stat() if path.exists() else tmp_path.stat()
    monkeypatch.setattr(afk, "_path_open", lambda *_args, **_kwargs: 32)
    monkeypatch.setattr(os, "fstat", lambda _fd: os.stat_result((stat.S_IFREG | 0o600, 1, 1, 1, info.st_uid, info.st_gid, 2, 0, 0, 0)))
    monkeypatch.setattr(os, "read", lambda _fd, _size: b"{}")

    def fail_close(fd):
        close_calls.append(fd)
        raise OSError("close failed")

    monkeypatch.setattr(os, "close", fail_close)

    with pytest.raises(afk.AfkStateError, match="close failed") as exc_info:
        afk._read_state(root)

    assert exc_info.value.changed is False
    assert close_calls == [32]


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor ownership")
def test_open_root_closes_child_and_parent_once_when_child_validation_fails(
    monkeypatch,
):
    monkeypatch.setattr(afk, "_root", lambda: Path("/a/b"))
    open_fds = iter((10, 11))
    close_calls = []
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: next(open_fds))
    monkeypatch.setattr(os, "stat", lambda *_args, **_kwargs: os.stat_result((stat.S_IFDIR | 0o700, 1, 1, 1, getattr(os, "geteuid", lambda: 0)(), 0, 0, 0, 0, 0)))
    monkeypatch.setattr(os, "close", close_calls.append)

    def fail_child_check(fd, *, final):
        if fd == 11:
            raise afk.AfkStateError("child validation failed")

    monkeypatch.setattr(afk, "_check_directory", fail_child_check)

    with pytest.raises(afk.AfkStateError, match="child validation failed"):
        afk._open_root()

    assert close_calls == [11, 10]


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor ownership")
def test_open_root_preserves_acquisition_error_when_parent_close_fails(monkeypatch):
    monkeypatch.setattr(afk, "_root", lambda: Path("/a/b"))
    close_calls = []

    def fake_open(*_args, **kwargs):
        if kwargs.get("dir_fd") is None:
            return 10
        raise OSError("child open failed")

    def fail_close(fd):
        close_calls.append(fd)
        raise OSError("parent close failed")

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "stat", lambda *_args, **_kwargs: os.stat_result((stat.S_IFDIR | 0o700, 1, 1, 1, getattr(os, "geteuid", lambda: 0)(), 0, 0, 0, 0, 0)))
    monkeypatch.setattr(afk, "_check_directory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(os, "close", fail_close)

    with pytest.raises(afk.AfkStateError, match="child open failed"):
        afk._open_root()

    assert close_calls == [10]


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory lock")
def test_file_lock_enter_preserves_acquisition_error_when_cleanup_fails(
    monkeypatch, tmp_path
):
    lock = afk._FileLock(tmp_path / "afk.lock")
    current_uid = getattr(os, "geteuid", lambda: 0)()
    regular = os.stat_result((stat.S_IFREG | 0o600, 1, 1, 1, current_uid, 0, 0, 0, 0, 0))
    directory = os.stat_result((stat.S_IFDIR | 0o700, 1, 1, 1, current_uid, 0, 0, 0, 0, 0))
    open_fds = iter((40, 41))
    close_calls = []
    file_close_calls = []

    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: next(open_fds))
    monkeypatch.setattr(os, "fstat", lambda fd: directory if fd == 40 else regular)

    class FailingFile:
        def fileno(self):
            return 41

        def close(self):
            file_close_calls.append(41)
            raise OSError("file close failed")

    monkeypatch.setattr(os, "fdopen", lambda *_args, **_kwargs: FailingFile())
    monkeypatch.setattr(os, "close", lambda fd: (close_calls.append(fd), (_ for _ in ()).throw(OSError("root close failed")))[1])
    import fcntl

    monkeypatch.setattr(fcntl, "flock", lambda *_args: (_ for _ in ()).throw(OSError("lock failed")))

    with pytest.raises(afk.AfkStateError, match="lock failed"):
        lock.__enter__()

    assert lock.file is None
    assert lock.root.fd is None
    assert file_close_calls == [41]
    assert close_calls == [40]


def test_mutation_marker_only_tracks_canonical_state_changes(monkeypatch, tmp_path):
    root = afk._Root(tmp_path, None)
    temporary = ".afk.temporary"
    (tmp_path / temporary).write_text("temporary", encoding="utf-8")

    afk._path_unlink(root, temporary)

    assert root.mutated is False

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    afk.engage("away")
    original_unlink = afk._path_unlink
    observed = []

    def unlink_and_observe(canonical_root, name):
        original_unlink(canonical_root, name)
        observed.append((name, canonical_root))

    monkeypatch.setattr(afk, "_path_unlink", unlink_and_observe)

    assert afk.clear() is True
    assert observed[0][0] == afk.STATE_NAME
    assert observed[0][1].mutated is True


def test_corrupt_or_unreadable_state_is_engaged(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = afk.state_path()
    path.write_text("not json", encoding="utf-8")
    assert afk.is_afk() is True
    path.unlink()
    path.mkdir()
    assert afk.is_afk() is True


def test_unverifiable_state_is_distinct_and_status_is_truthful(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = afk.state_path()
    path.write_text("not json", encoding="utf-8")

    state = afk.get_state()

    assert state["unverifiable"] is True
    assert "None" not in afk.handle_command("status")
    assert "unreadable" in afk.handle_command("status").lower()
    assert "None" not in afk.handle_command("")
    assert "unverifiable" in afk.handle_command("").lower()


def test_status_bounds_non_not_found_state_stat_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fail_state_stat(_root, _name):
        raise OSError("state stat failed")

    monkeypatch.setattr(afk, "_path_stat", fail_state_stat)

    reply = afk.handle_command("status")

    assert "Couldn't safely change" in reply


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_state_with_group_or_other_bits_is_unverifiable(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = afk.state_path()
    path.write_text(
        json.dumps({"engaged_at": "2026-01-01T00:00:00+00:00", "reason": "secret"}),
        encoding="utf-8",
    )
    path.chmod(0o640)

    assert afk.get_state() == {"unverifiable": True}


def test_clear_refuses_symlinked_state_leaf(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "elsewhere"
    target.write_text("keep", encoding="utf-8")
    path = afk.state_path()
    path.symlink_to(target)

    with pytest.raises(afk.AfkStateError, match="symlink"):
        afk.clear()

    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep"


def test_zero_write_result_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = 0

    def no_progress(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0
        raise AssertionError("write loop did not make progress")

    monkeypatch.setattr(os, "write", no_progress)
    with pytest.raises(afk.AfkStateError, match="write"):
        afk.engage("x")


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-relative path safety")
def test_ancestor_symlink_is_refused_without_writing_through_it(monkeypatch, tmp_path):
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(alias / "hermes"))

    with pytest.raises(afk.AfkStateError, match="symlink"):
        afk.engage("blocked")

    assert not (real_parent / "hermes" / afk.STATE_NAME).exists()


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX descriptor-relative publication race"
)
def test_root_replacement_during_publication_never_reports_success(
    monkeypatch, tmp_path
):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    real_replace = os.replace
    moved = tmp_path / "moved-root"
    swapped = False

    def replace_with_root_swap(src, dst, **kwargs):
        nonlocal swapped
        if not swapped:
            swapped = True
            root.rename(moved)
            root.mkdir(mode=0o700)
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(os, "replace", replace_with_root_swap)

    reply = afk.handle_command("on race")

    assert "durability could not be confirmed" in reply
    assert not (root / afk.STATE_NAME).exists()
    assert (moved / afk.STATE_NAME).is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-relative read race")
def test_get_state_rejects_root_replacement_after_read(monkeypatch, tmp_path):
    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    afk.engage("old root")
    moved = tmp_path / "moved-root"
    real_read_state = afk._read_state

    def read_then_swap(bound_root):
        state = real_read_state(bound_root)
        root.rename(moved)
        root.mkdir(mode=0o700)
        return state

    monkeypatch.setattr(afk, "_read_state", read_then_swap)

    with pytest.raises(afk.AfkStateError) as exc_info:
        afk.get_state()

    assert exc_info.value.changed is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory lock test")
def test_file_lock_serializes_across_processes(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    attempted = context.Event()
    completed = context.Event()
    errors = context.SimpleQueue()
    holder = context.Process(target=_hold_afk_transaction, args=(ready, release))
    contender = context.Process(
        target=_engage_afk_after_signal,
        args=(attempted, completed, errors),
    )
    holder.start()
    contender_started = False
    try:
        assert ready.wait(5), "process A did not acquire the AFK transaction"
        contender.start()
        contender_started = True
        assert attempted.wait(5), "process B did not attempt engage"
        assert not completed.wait(0.25), "process B bypassed the AFK lock"
        release.set()
        assert completed.wait(5), "process B did not complete after release"
        holder.join(5)
        contender.join(5)
        assert not holder.is_alive()
        assert not contender.is_alive()
        assert not errors.empty(), "process B did not report its result"
        assert errors.get() == "<no error>"
    finally:
        release.set()
        holder.join(5)
        if contender_started:
            contender.join(5)
        if holder.is_alive():
            holder.terminate()
            holder.join(5)
        if contender_started and contender.is_alive():
            contender.terminate()
            contender.join(5)


def test_reason_is_bounded_and_display_neutralized(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reason = "  [31m" + ("x" * 10_000) + "\nprivate\tdata  "
    afk.engage(reason)
    state = afk.get_state()
    assert afk.is_afk() is True
    assert len(state["reason"] or "") <= afk.MAX_REASON_CHARS
    assert "\x1b" not in (state["reason"] or "")
    assert "[" not in (state["reason"] or "")
    assert "]" not in (state["reason"] or "")


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory lock test")
def test_file_lock_closes_file_when_lock_acquisition_fails(monkeypatch, tmp_path):
    lock = afk._FileLock(tmp_path / "afk.lock")

    import fcntl

    def fail_flock(*_args):
        raise OSError("lock failed")

    monkeypatch.setattr(fcntl, "flock", fail_flock)

    with pytest.raises(afk.AfkStateError, match="lock failed"):
        lock.__enter__()

    assert lock.file is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory lock test")
def test_file_lock_release_closes_file_once(monkeypatch, tmp_path):
    lock = afk._FileLock(tmp_path / "afk.lock")
    real_fdopen = os.fdopen
    close_calls = 0

    class TrackingFile:
        def __init__(self, file):
            self._file = file

        def close(self):
            nonlocal close_calls
            close_calls += 1
            return self._file.close()

        def __getattr__(self, name):
            return getattr(self._file, name)

    def tracked_fdopen(*args, **kwargs):
        return TrackingFile(real_fdopen(*args, **kwargs))

    monkeypatch.setattr(os, "fdopen", tracked_fdopen)
    lock.__enter__()
    lock.__exit__(None, None, None)
    lock.__exit__(None, None, None)

    assert close_calls == 1
    assert lock.file is None


def test_status_handles_symlinked_lock_leaf(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "elsewhere"
    target.write_text("keep", encoding="utf-8")
    afk._lock_path().symlink_to(target)

    assert "Couldn't safely change" in afk.handle_command("status")


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory lock test")
def test_status_bounds_lock_release_error_as_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import fcntl

    real_flock = fcntl.flock

    def fail_unlock(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", fail_unlock)

    reply = afk.handle_command("status")

    assert "Couldn't safely change" in reply


class _FakeTransactionRoot:
    def __init__(self, *, mutated=False):
        self.fd = None
        self.mutated = mutated
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        raise OSError("root close failed")

    def revalidate(self, **_kwargs):
        return None


class _FakeTransactionLock:
    def __init__(self, root, error=None):
        self.root = root
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.error is not None:
            raise self.error
        return False


def test_transaction_bounds_root_close_before_mutation(monkeypatch):
    root = _FakeTransactionRoot()
    monkeypatch.setattr(afk, "_open_root", lambda: root)
    monkeypatch.setattr(afk, "_FileLock", _FakeTransactionLock)

    with pytest.raises(afk.AfkStateError, match="root close failed") as exc_info:
        with afk._transaction():
            pass

    assert exc_info.value.changed is False


def test_handle_command_bounds_root_close_after_canonical_mutation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    def fail_close(_root):
        raise OSError("root close failed")

    monkeypatch.setattr(afk._Root, "close", fail_close)

    reply = afk.handle_command("on lunch")

    assert "AFK state changed" in reply
    assert afk.state_path().is_file()


def test_transaction_preserves_body_error_when_root_close_fails(monkeypatch):
    root = _FakeTransactionRoot()
    monkeypatch.setattr(afk, "_open_root", lambda: root)
    monkeypatch.setattr(afk, "_FileLock", _FakeTransactionLock)

    with pytest.raises(ValueError, match="body failed"):
        with afk._transaction():
            raise ValueError("body failed")


def test_transaction_preserves_lock_cleanup_error_when_root_close_fails(monkeypatch):
    root = _FakeTransactionRoot()
    monkeypatch.setattr(afk, "_open_root", lambda: root)
    monkeypatch.setattr(
        afk,
        "_FileLock",
        lambda lock_root: _FakeTransactionLock(lock_root, OSError("unlock failed")),
    )

    with pytest.raises(OSError, match="unlock failed"):
        with afk._transaction():
            pass


def test_atomic_temp_fd_close_is_bounded_and_owned_once(monkeypatch, tmp_path):
    root = afk._Root(tmp_path, None)
    close_calls = []
    monkeypatch.setattr(afk, "_path_open", lambda *_args, **_kwargs: 101)
    monkeypatch.setattr(os, "fchmod", lambda *_args: None)
    monkeypatch.setattr(os, "write", lambda _fd, data: len(data))
    monkeypatch.setattr(os, "fsync", lambda _fd: None)
    monkeypatch.setattr(
        os, "close", lambda fd: (close_calls.append(fd), (_ for _ in ()).throw(OSError("temp close failed")))[1]
    )
    unlinked = []
    monkeypatch.setattr(afk, "_path_unlink", lambda _root, name: unlinked.append(name))

    with pytest.raises(afk.AfkStateError, match="temp close failed") as exc_info:
        afk._atomic_replace_json(root, {"reason": "x"})

    assert exc_info.value.changed is False
    assert close_calls == [101]
    assert len(unlinked) == 1
    assert root.mutated is False


def test_atomic_temp_unlink_cleanup_failure_is_bounded(monkeypatch, tmp_path):
    root = afk._Root(tmp_path, None)
    monkeypatch.setattr(afk, "_path_open", lambda *_args, **_kwargs: 101)
    monkeypatch.setattr(os, "fchmod", lambda *_args: None)
    monkeypatch.setattr(os, "write", lambda _fd, data: len(data))
    monkeypatch.setattr(os, "fsync", lambda _fd: None)
    monkeypatch.setattr(os, "close", lambda _fd: None)
    monkeypatch.setattr(
        afk, "_path_replace", lambda *_args: (_ for _ in ()).throw(OSError("replace failed"))
    )
    monkeypatch.setattr(
        afk, "_path_unlink", lambda *_args: (_ for _ in ()).throw(OSError("unlink failed"))
    )

    with pytest.raises(afk.AfkStateError, match="replace failed") as exc_info:
        afk._atomic_replace_json(root, {"reason": "x"})

    assert exc_info.value.changed is False
    assert root.mutated is False


def test_atomic_cleanup_failures_preserve_primary_write_error(monkeypatch, tmp_path):
    root = afk._Root(tmp_path, None)
    close_calls = []
    monkeypatch.setattr(afk, "_path_open", lambda *_args, **_kwargs: 101)
    monkeypatch.setattr(os, "fchmod", lambda *_args: None)
    monkeypatch.setattr(
        os, "write", lambda *_args: (_ for _ in ()).throw(OSError("write failed"))
    )
    monkeypatch.setattr(os, "close", lambda fd: close_calls.append(fd))
    monkeypatch.setattr(
        afk, "_path_unlink", lambda *_args: (_ for _ in ()).throw(OSError("unlink failed"))
    )

    with pytest.raises(afk.AfkStateError, match="write failed") as exc_info:
        afk._atomic_replace_json(root, {"reason": "x"})

    assert exc_info.value.changed is False
    assert close_calls == [101]
    assert root.mutated is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory lock test")
def test_engage_reports_lock_release_error_after_state_change(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    import fcntl

    real_flock = fcntl.flock

    def fail_unlock(fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", fail_unlock)

    reply = afk.handle_command("on lunch")

    assert "AFK state changed" in reply
    assert "durability could not be confirmed" in reply
    assert afk.state_path().is_file()


def test_status_handles_unusable_root(monkeypatch, tmp_path):
    root = tmp_path / "not-a-directory"
    root.write_text("keep", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))

    assert "Couldn't safely change" in afk.handle_command("status")


def test_symlink_leaf_is_refused_for_read_and_write(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "elsewhere"
    target.write_text(json.dumps({"engaged": False}), encoding="utf-8")
    path = afk.state_path()
    path.symlink_to(target)

    with pytest.raises(afk.AfkStateError):
        afk.engage("no")
    with pytest.raises(afk.AfkStateError):
        afk.get_state()


@pytest.mark.skipif(os.name == "nt", reason="POSIX durability contract")
def test_write_and_unlink_fsync_file_and_parent_in_order(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []
    real_fsync = os.fsync

    def record_fsync(fd):
        calls.append((
            "fsync",
            "directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file",
        ))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    afk.engage("temporary")
    afk.clear()
    assert calls[-1][0] == "fsync"
    assert calls[-1][1] == "directory"


def test_state_file_is_owner_only(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    afk.engage(None)
    assert afk.state_path().stat().st_mode & 0o077 == 0


@pytest.mark.skipif(os.name == "nt", reason="POSIX durability contract")
def test_file_fsync_failure_is_propagated(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fail_fsync(fd):
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("file barrier failed")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(afk.AfkStateError, match="file barrier") as exc_info:
        afk.engage("x")
    assert exc_info.value.changed is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX durability contract")
def test_replace_failure_is_propagated(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(afk.AfkStateError, match="replace failed") as exc_info:
        afk.engage("x")
    assert exc_info.value.changed is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX durability contract")
def test_command_reports_verification_failure_after_replace_as_changed(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        afk,
        "_verify_owner_only",
        lambda _path: (_ for _ in ()).throw(OSError("verification failed")),
    )

    reply = afk.handle_command("on lunch")

    assert "AFK state changed" in reply
    assert "durability could not be confirmed" in reply
    assert afk.state_path().is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX durability contract")
def test_command_reports_replace_applied_when_parent_sync_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        afk, "_sync_parent", lambda _path: (_ for _ in ()).throw(OSError("sync failed"))
    )

    reply = afk.handle_command("on lunch")

    assert "AFK state changed" in reply
    assert "durability could not be confirmed" in reply
    assert afk.state_path().is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX durability contract")
def test_command_reports_clear_applied_when_parent_sync_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    afk.engage("lunch")
    monkeypatch.setattr(
        afk, "_sync_parent", lambda _path: (_ for _ in ()).throw(OSError("sync failed"))
    )

    reply = afk.handle_command("off")

    assert "AFK state changed" in reply
    assert "durability could not be confirmed" in reply
    assert not afk.state_path().exists()


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("", "AFK recorded since"),
        ("on lunch", "AFK recorded since"),
        ("status", "Not AFK."),
        ("off", "You weren't marked AFK."),
        ("later", "Usage: `/afk [on [reason] | off | status]`"),
    ],
)
def test_command_grammar(monkeypatch, tmp_path, args, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert expected in afk.handle_command(args)
