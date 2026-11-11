"""FIFO dotenv reads are bounded, serialized and never persisted."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import errno
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from hermes_cli import env_file


@contextmanager
def producer(path, payload, *, hold=None):
    stop = threading.Event()
    sent = threading.Event()

    def write():
        while not stop.is_set():
            try:
                fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                stop.wait(0.01)
                continue
            try:
                os.set_blocking(fd, True)
                os.write(fd, payload)
                sent.set()
                if hold is not None:
                    hold.wait(3)
            except BrokenPipeError:
                pass  # oversize/timeout tests intentionally close the reader
            finally:
                os.close(fd)
            return

    thread = threading.Thread(target=write)
    thread.start()
    try:
        yield sent
    finally:
        stop.set()
        if hold is not None:
            hold.set()
        thread.join(4)
        assert not thread.is_alive()


@pytest.mark.platforms("posix")
def test_threads_share_one_read_and_refresh_is_explicit(tmp_path):
    pipe = tmp_path / ".env"
    os.mkfifo(pipe, 0o600)
    with producer(pipe, b"KEY=first\n"), ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(env_file.read_env_bytes, [pipe] * 8)) == [b"KEY=first\n"] * 8
    assert env_file.read_env_bytes(pipe) == b"KEY=first\n"
    assert pipe.is_fifo()
    assert (tmp_path / ".env.read.lock").read_bytes() == b""
    assert (tmp_path / ".env.read.lock").stat().st_mode & 0o777 == 0o600
    env_file.invalidate_fifo_cache(pipe)
    with producer(pipe, b"KEY=second\n"):
        assert env_file.read_env_bytes(pipe) == b"KEY=second\n"


@pytest.mark.platforms("posix")
def test_missing_writer_times_out_and_can_retry(tmp_path, monkeypatch):
    pipe = tmp_path / ".env"
    os.mkfifo(pipe, 0o600)
    monkeypatch.setattr(env_file, "_FIFO_TIMEOUT", 0.1)
    started = time.monotonic()
    with pytest.raises(env_file.MountedEnvError, match="unlock/authorize"):
        env_file.read_env_bytes(pipe)
    assert time.monotonic() - started < 2
    with producer(pipe, b"KEY=recovered\n"):
        assert env_file.read_env_bytes(pipe) == b"KEY=recovered\n"


@pytest.mark.platforms("posix")
def test_incomplete_transfer_is_not_cached(tmp_path, monkeypatch):
    pipe = tmp_path / ".env"
    os.mkfifo(pipe, 0o600)
    monkeypatch.setattr(env_file, "_FIFO_TIMEOUT", 0.2)
    hold = threading.Event()
    with producer(pipe, b"KEY=partial", hold=hold):
        with pytest.raises(env_file.MountedEnvError, match="Timed out"):
            env_file.read_env_bytes(pipe)
    with producer(pipe, b"KEY=complete\n"):
        assert env_file.read_env_bytes(pipe) == b"KEY=complete\n"


@pytest.mark.platforms("posix")
def test_oversized_transfer_is_not_cached(tmp_path, monkeypatch):
    pipe = tmp_path / ".env"
    os.mkfifo(pipe, 0o600)
    monkeypatch.setattr(env_file, "_FIFO_MAX_BYTES", 64)
    with producer(pipe, b"KEY=" + b"x" * 128):
        with pytest.raises(env_file.MountedEnvError, match="exceeds"):
            env_file.read_env_bytes(pipe)
    with producer(pipe, b"KEY=small\n"):
        assert env_file.read_env_bytes(pipe) == b"KEY=small\n"


@pytest.mark.platforms("posix")
def test_async_read_never_waits_on_gateway_loop(tmp_path):
    pipe = tmp_path / ".env"
    os.mkfifo(pipe, 0o600)

    async def read():
        with pytest.raises(env_file.MountedEnvError, match="worker thread"):
            env_file.read_env_bytes(pipe)
        with producer(pipe, b"KEY=off-loop\n"):
            assert await asyncio.to_thread(env_file.read_env_bytes, pipe) == b"KEY=off-loop\n"
        assert env_file.read_env_bytes(pipe) == b"KEY=off-loop\n"

    asyncio.run(read())


@pytest.mark.platforms("posix")
def test_profiles_have_distinct_snapshots(tmp_path):
    for name in ("first", "second"):
        home = tmp_path / name
        home.mkdir()
        pipe = home / ".env"
        os.mkfifo(pipe, 0o600)
        payload = f"KEY={name}\n".encode()
        with producer(pipe, payload):
            assert env_file.read_env_bytes(pipe) == payload
    assert env_file.read_env_bytes(tmp_path / "first" / ".env") == b"KEY=first\n"


@pytest.mark.platforms("posix")
def test_cooperating_process_waits_for_read_lock(tmp_path):
    import fcntl

    pipe = tmp_path / ".env"
    os.mkfifo(pipe, 0o600)
    lock = tmp_path / ".env.read.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    code = """
from pathlib import Path
import sys
from hermes_cli import env_file
env_file._FIFO_TIMEOUT = 0.1
try:
    env_file.read_env_bytes(Path(sys.argv[1]))
except env_file.MountedEnvError:
    print("bounded lock timeout")
else:
    raise AssertionError("read bypassed another process's lock")
"""
    try:
        with producer(pipe, b"KEY=not-consumed\n") as sent:
            result = subprocess.run(
                [sys.executable, "-c", code, str(pipe)],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=5, check=False,
            )
            assert result.returncode == 0, result.stderr
            assert result.stdout.strip() == "bounded lock timeout"
            assert not sent.is_set()
    finally:
        os.close(fd)
    with producer(pipe, b"KEY=now-readable\n"):
        assert env_file.read_env_bytes(pipe) == b"KEY=now-readable\n"


def test_regular_files_are_reread_without_fifo_cache(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"KEY=first\n")
    assert env_file.read_env_bytes(path) == b"KEY=first\n"
    path.write_bytes(b"KEY=other\n")
    assert env_file.read_env_bytes(path) == b"KEY=other\n"
    assert not (tmp_path / ".env.read.lock").exists()


@pytest.mark.platforms("posix")
def test_symlink_aliases_share_the_canonical_mount_lock(tmp_path):
    pipe = tmp_path / ".env"
    alias = tmp_path / "alias.env"
    os.mkfifo(pipe, 0o600)
    alias.symlink_to(pipe)
    with producer(pipe, b"KEY=alias\n"):
        assert env_file.read_env_bytes(alias) == b"KEY=alias\n"
    assert env_file.read_env_bytes(pipe) == b"KEY=alias\n"
    assert (tmp_path / ".env.read.lock").exists()
    assert not (tmp_path / "alias.env.read.lock").exists()
