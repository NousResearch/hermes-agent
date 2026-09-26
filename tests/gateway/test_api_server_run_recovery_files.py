"""Regression coverage for handle-bound local recovery-file snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

from gateway.platforms.api_server_run_recovery_files import inspect_file
from gateway.platforms import api_server_run_recovery_files as recovery_files


def test_inspect_file_hashes_one_open_regular_file_and_returns_json_snapshot(tmp_path):
    target = tmp_path / "receipt.bin"
    data = b"checkpoint payload\n"
    target.write_bytes(data)

    snapshot = inspect_file(str(target), len(data))

    assert snapshot == {
        "sha256": hashlib.sha256(data).hexdigest(),
        "identity": snapshot["identity"],
        "size": len(data),
        "mtime_ns": snapshot["mtime_ns"],
        "ctime_ns": snapshot["ctime_ns"],
    }
    assert set(snapshot["identity"]) == {"device", "file_id"}
    assert all(isinstance(value, int) for value in snapshot["identity"].values())
    assert all(isinstance(snapshot[name], int) for name in ("size", "mtime_ns", "ctime_ns"))
    assert json.loads(json.dumps(snapshot)) == snapshot


def test_inspect_file_rejects_a_size_mismatch(tmp_path):
    target = tmp_path / "receipt.bin"
    target.write_bytes(b"four")

    with pytest.raises(ValueError, match="size"):
        inspect_file(str(target), 3)


def test_inspect_file_rejects_a_directory(tmp_path):
    with pytest.raises(OSError):
        inspect_file(str(tmp_path), 0)


def test_inspect_file_distinguishes_same_byte_os_replacements(tmp_path):
    target = tmp_path / "receipt.bin"
    replacement = tmp_path / "replacement.bin"
    target.write_bytes(b"same")
    first = inspect_file(str(target), 4)
    replacement.write_bytes(b"same")
    os.replace(replacement, target)

    snapshot = inspect_file(str(target), 4)

    assert snapshot["sha256"] == hashlib.sha256(b"same").hexdigest()
    assert snapshot["identity"] != first["identity"]
    assert snapshot["identity"]["file_id"] == os.stat(target).st_ino


@pytest.mark.linux_only
def test_inspect_file_refuses_fifo_without_waiting_for_a_writer(tmp_path):
    target = tmp_path / "receipt.fifo"
    getattr(os, "mkfifo")(target)
    script = (
        "import sys\n"
        "from gateway.platforms.api_server_run_recovery_files import inspect_file\n"
        "try: inspect_file(sys.argv[1], 0)\n"
        "except OSError: print('refused')\n"
        "else: raise AssertionError('FIFO accepted')\n"
    )
    result = subprocess.run([sys.executable, "-c", script, str(target)],
                            check=True, capture_output=True, text=True, timeout=5)
    assert result.stdout.strip() == "refused"


@pytest.mark.linux_only
def test_inspect_file_allows_read_access_time_updates(tmp_path):
    target = tmp_path / "receipt.bin"
    target.write_bytes(b"same")
    os.utime(target, ns=(0, target.stat().st_mtime_ns))
    before = target.stat()

    snapshot = inspect_file(str(target), 4)

    after = target.stat()
    assert snapshot["sha256"] == hashlib.sha256(b"same").hexdigest()
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns == before.st_ctime_ns
    assert inspect_file(str(target), 4) == snapshot


@pytest.mark.linux_only
def test_inspect_file_still_refuses_a_content_change_while_open(tmp_path, monkeypatch):
    target = tmp_path / "receipt.bin"
    target.write_bytes(b"same")
    original = recovery_files._read_hash_fd

    def read_then_change(*args):
        digest = original(*args)
        with target.open("r+b") as stream:
            stream.write(b"B")
            stream.flush()
            os.fsync(stream.fileno())
        return digest

    monkeypatch.setattr(recovery_files, "_read_hash_fd", read_then_change)
    with pytest.raises(OSError, match="changed"):
        inspect_file(str(target), 4)
    assert target.read_bytes() == b"Bame"


@pytest.mark.linux_only
def test_inspect_file_rejects_a_real_symbolic_link(tmp_path):
    target = tmp_path / "receipt.bin"
    target.write_bytes(b"same")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(OSError, match="links"):
        inspect_file(str(link), 4)


@pytest.mark.windows_only
@pytest.mark.parametrize("expected_size", [4, 3])
def test_inspect_file_closes_the_owning_handle_on_success_and_refusal(tmp_path, monkeypatch, expected_size):
    import win32file

    target = tmp_path / "receipt.bin"
    target.write_bytes(b"same")
    opened = []
    create_file = win32file.CreateFile

    def capture_handle(*args, **kwargs):
        handle = create_file(*args, **kwargs)
        opened.append(handle)
        return handle

    monkeypatch.setattr(win32file, "CreateFile", capture_handle)
    try:
        if expected_size == 3:
            with pytest.raises(ValueError, match="size"):
                inspect_file(str(target), expected_size)
        else:
            inspect_file(str(target), expected_size)
        assert len(opened) == 1
        # Closing only int(handle) leaves an armed destructor. A later kernel
        # handle reuse can make that destructor close an unrelated semaphore.
        assert int(opened[0]) == 0
    finally:
        for handle in opened:
            if int(handle):
                # In the RED version the raw handle was already closed. Do not
                # let its stale owning wrapper corrupt this test interpreter.
                handle.Detach()


@pytest.mark.windows_only
def test_inspect_file_refuses_a_same_timestamp_content_change(tmp_path, monkeypatch):
    target = tmp_path / "receipt.bin"
    target.write_bytes(b"same")
    before = target.stat()
    original = recovery_files._read_hash_handle

    def read_then_change(*args):
        digest = original(*args)
        with target.open("r+b") as stream:
            stream.write(b"B")
            stream.flush()
            os.fsync(stream.fileno())
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        return digest

    monkeypatch.setattr(recovery_files, "_read_hash_handle", read_then_change)
    with pytest.raises(OSError, match="changed"):
        inspect_file(str(target), 4)
    assert target.read_bytes() == b"Bame"


@pytest.mark.windows_only
@pytest.mark.parametrize("boundary", ["before_open", "during_read"])
def test_inspect_file_refuses_a_cross_process_path_rebinding_during_inspection(tmp_path, monkeypatch, boundary):
    target = tmp_path / "receipt.bin"
    replacement = tmp_path / "replacement.bin"
    original_object = tmp_path / "original.bin"
    target.write_bytes(b"same")
    replacement.write_bytes(b"same")
    changed = False

    def replace_once():
        nonlocal changed
        if not changed:
            # Windows can refuse overwrite of an open target, while its shared
            # delete handle still permits renaming it and rebinding the path.
            subprocess.run(
                [sys.executable, "-c",
                 "import os,sys;os.rename(sys.argv[2],sys.argv[3]);os.replace(sys.argv[1],sys.argv[2])",
                 str(replacement), str(target), str(original_object)],
                check=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            changed = True

    if boundary == "before_open":
        original = recovery_files._path_chain

        def inspect_then_replace(path):
            result = original(path)
            replace_once()
            return result

        monkeypatch.setattr(recovery_files, "_path_chain", inspect_then_replace)
    else:
        original = recovery_files._read_hash_handle

        def replace_then_read(*args):
            replace_once()
            return original(*args)

        monkeypatch.setattr(recovery_files, "_read_hash_handle", replace_then_read)

    with pytest.raises(OSError, match="changed|escaped"):
        inspect_file(str(target), 4)
    assert changed
    assert not replacement.exists()
    assert original_object.exists()
    assert target.read_bytes() == b"same"


@pytest.mark.windows_only
def test_inspect_file_rejects_a_real_ntfs_junction_ancestor(tmp_path):
    import _winapi

    physical = tmp_path / "physical"
    junction = tmp_path / "junction"
    physical.mkdir()
    payload = physical / "payload.bin"
    payload.write_bytes(b"payload")
    _winapi.CreateJunction(str(physical), str(junction))
    try:
        with pytest.raises(OSError, match="link|reparse|escaped"):
            inspect_file(str(junction / payload.name), len(b"payload"))
    finally:
        junction.rmdir()
