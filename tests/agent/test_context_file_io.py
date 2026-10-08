"""Context reads preserve bounded startup, BOM decoding and handle/identity consistency."""

import os
import threading
from types import SimpleNamespace

import pytest

from agent import context_file_io as io


def test_bom_and_file_identity_are_read_from_one_handle(tmp_path, monkeypatch):
    path = tmp_path / "rules.md"
    path.write_text("\ufefforiginal instructions", encoding="utf-8")
    real_open = os.open
    opened = []

    def tracked_open(*args, **kwargs):
        opened.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(io.os, "open", tracked_open)
    result = io.read_context_file(path, 2)
    assert result.status == "loaded" and result.content == "original instructions"
    assert result.identity == ("inode", path.stat().st_dev, path.stat().st_ino)
    assert opened == [path]


@pytest.mark.platforms("posix")
def test_path_replacement_after_fstat_does_not_change_read_bytes_or_identity(tmp_path, monkeypatch):
    path, replacement = tmp_path / "rules.md", tmp_path / "new.md"
    path.write_text("opened bytes")
    replacement.write_text("replacement bytes")
    expected = path.stat()
    original_fstat = os.fstat

    def replace_after_fstat(fd):
        result = original_fstat(fd)
        replacement.replace(path)
        return result

    monkeypatch.setattr(io.os, "fstat", replace_after_fstat)
    result = io.read_context_file(path, 2)
    assert result.content == "opened bytes"
    assert result.identity == ("inode", expected.st_dev, expected.st_ino)
    assert path.read_text() == "replacement bytes"


def test_zero_inode_files_are_distinguished_by_resolved_path(tmp_path):
    fake_stat = SimpleNamespace(st_ino=0, st_dev=1)
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    assert io._file_identity(a, fake_stat) != io._file_identity(b, fake_stat)
    assert io._file_identity(a, fake_stat) == io._file_identity(tmp_path / "child" / ".." / "a.md", fake_stat)


def test_timed_out_reader_cannot_publish_a_late_loaded_result(tmp_path, monkeypatch, caplog):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def slow(_path):
        started.set()
        try:
            assert release.wait(5)
            return io.ContextFileRead("late", ("inode", 1, 2), "loaded")
        finally:
            finished.set()

    monkeypatch.setattr(io, "_read_once", slow)
    try:
        result = io.read_context_file(tmp_path / "slow.md", .01)
        assert started.wait(2)
        assert result.status == "unreadable" and result.identity is None
        assert "read timed out" in caplog.text
    finally:
        release.set()
        assert finished.wait(2)
    assert result.content == ""


@pytest.mark.platforms("posix")
def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "rules.fifo"
    os.mkfifo(fifo)
    assert io.read_context_file(fifo, 2).status == "unreadable"


def test_read_guard_and_path_resolution_share_the_deadline(tmp_path, monkeypatch, caplog):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def stalled_guard(_path):
        started.set()
        try:
            assert release.wait(5)
            return "blocked"
        finally:
            finished.set()

    monkeypatch.setattr("agent.file_safety.get_read_block_error", stalled_guard)
    try:
        result = io.read_context_file(tmp_path / "network.md", .01, guarded=True)
        assert started.wait(2)
        assert result.status == "unreadable" and result.identity is None
        assert "read timed out" in caplog.text
    finally:
        release.set()
        assert finished.wait(2)


def test_thread_resource_failure_skips_optional_file(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("cannot start new thread")

    monkeypatch.setattr(io, "spawn_context_thread", unavailable)
    assert io.read_context_file(tmp_path / "rules.md", 1).status == "unreadable"


def test_failed_read_guard_is_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "rules.md"
    path.write_text("must not load without validation")

    def failed_guard(_path):
        raise OSError("guard unavailable")

    monkeypatch.setattr("agent.file_safety.get_read_block_error", failed_guard)
    result = io.read_context_file(path, 2, guarded=True)
    assert result.status == "unreadable" and not result.content


def test_nt_namespace_is_rejected_before_filesystem_resolution(monkeypatch):
    from pathlib import Path

    def forbidden_resolve(*_args, **_kwargs):
        raise AssertionError("namespace rejection must precede resolution")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    result = io.read_context_file(Path(r"\??\UNC\example.invalid\share\rules.md"), 2, guarded=True)
    assert result.status == "blocked"


@pytest.mark.platforms("posix")
def test_symlink_to_read_denied_file_is_not_injected(tmp_path):
    target, alias = tmp_path / ".env", tmp_path / "rules.md"
    target.write_text("DO_NOT_INJECT=secret")
    alias.symlink_to(target)
    result = io.read_context_file(alias, 2, guarded=True)
    assert result.status == "blocked" and not result.content
