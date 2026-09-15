"""Real-home guards are exercised against disposable protected roots only."""
from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import sqlite3

import pytest


@pytest.fixture
def protected_home(tmp_path, monkeypatch):
    from tests import conftest

    root = tmp_path / "protected"
    root.mkdir()
    (root / "file.txt").write_text("unchanged", encoding="utf-8")
    (root / "empty").mkdir()
    monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES", [root])
    yield root
    # Restore access before tmp_path/pytest cleanup even when an assertion fails.
    monkeypatch.setattr(conftest, "_REAL_HERMES_ROOT_CANDIDATES", [])


def _open_close(path):
    with open(path, encoding="utf-8"):
        pass


def _io_open_close(path):
    with io.open(path, encoding="utf-8"):
        pass


def _os_open_close(path):
    descriptor = os.open(path, os.O_RDONLY)
    os.close(descriptor)


_OPERATIONS = {
    "builtin-open": _open_close,
    "io-open": _io_open_close,
    "os-open": _os_open_close,
    "read-text": lambda path: path.read_text(encoding="utf-8"),
    "write-text": lambda path: path.write_text("changed", encoding="utf-8"),
    "read-bytes": lambda path: path.read_bytes(),
    "write-bytes": lambda path: path.write_bytes(b"changed"),
    "stat": lambda path: path.stat(),
    "lstat": lambda path: path.lstat(),
    "unlink": lambda path: path.unlink(),
    "remove": lambda path: os.remove(path),
    "rename-out": lambda path: path.rename(path.parent.parent / "moved.txt"),
    "replace-out": lambda path: path.replace(path.parent.parent / "replaced.txt"),
    "mkdir": lambda path: (path.parent / "new").mkdir(),
    "makedirs": lambda path: os.makedirs(path.parent / "deep" / "child"),
    "rmdir": lambda path: (path.parent / "empty").rmdir(),
    "rmtree": lambda path: shutil.rmtree(path.parent),
    "listdir": lambda path: os.listdir(path.parent),
    "scandir": lambda path: list(os.scandir(path.parent)),
    "sqlite": lambda path: sqlite3.connect(path.parent / "state.db").close(),
}


@pytest.mark.parametrize("operation", _OPERATIONS, ids=_OPERATIONS)
def test_io_guard_denies_protected_roots_before_mutation(protected_home, operation):
    with pytest.raises((AssertionError, pytest.fail.Exception), match="REAL hermes home"):
        _OPERATIONS[operation](protected_home / "file.txt")


def test_rename_cannot_overwrite_a_protected_destination(protected_home, tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("external", encoding="utf-8")
    with pytest.raises((AssertionError, pytest.fail.Exception), match="REAL hermes home"):
        source.replace(protected_home / "file.txt")
    assert source.read_text(encoding="utf-8") == "external"


def test_unprotected_paths_and_open_descriptors_still_work(protected_home, tmp_path):
    target = tmp_path / "allowed" / "file.txt"
    target.parent.mkdir()
    target.write_text("ok", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "ok"
    fd = os.open(target, os.O_RDONLY)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        assert stream.read() == "ok"
    moved = target.replace(target.with_name("moved.txt"))
    assert list(moved.parent.iterdir()) == [moved]
    moved.unlink()
    moved.parent.rmdir()
    connection = sqlite3.connect(tmp_path / "allowed.db")
    connection.close()


@pytest.mark.allow_real_home_io
def test_explicit_opt_out_allows_only_the_disposable_canary(protected_home):
    target = protected_home / "file.txt"
    target.write_text("opted out", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "opted out"


def test_close_keeps_a_reused_descriptors_new_owner(tmp_path, monkeypatch):
    from tests.home_io_guard import HomeIOGuard

    first, second = tmp_path / "first", tmp_path / "second"
    first.touch()
    second.touch()
    original_close = os.close
    reopened = []

    def close_and_reopen(fd):
        original_close(fd)
        reopened.append(os.open(second, os.O_RDONLY))

    guard = HomeIOGuard(lambda: [])
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(os, "close", close_and_reopen)
            guard.install(patcher)
            fd = os.open(first, os.O_RDONLY)
            os.close(fd)
            assert reopened == [fd], "the test must exercise descriptor reuse"
            assert guard.directories[fd] == second
    finally:
        for fd in reopened:
            original_close(fd)
