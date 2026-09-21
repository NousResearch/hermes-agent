"""Native Windows control-pipe deadlines, framing and cancelled-I/O cleanup.

Peers bind only fresh tmp_path-derived pipe names. Their six-second safety stop
also makes a regression to blocking client I/O fail rather than hang the suite.
"""
from __future__ import annotations

import contextlib
import gc
import json
import threading
import time

import pytest

from gateway.control_socket import _MAX_RESPONSE_BYTES, query_gateway_control, windows_pipe_name

pytestmark = pytest.mark.windows_only


@contextlib.contextmanager
def _native_peer(home, mode):
    import _winapi

    if mode == "eof-json":
        import ctypes
        from ctypes import wintypes

        flush = ctypes.WinDLL("kernel32", use_last_error=True).FlushFileBuffers
        flush.argtypes = [wintypes.HANDLE]
        flush.restype = wintypes.BOOL

    pipe = windows_pipe_name(home)
    handle = _winapi.CreateNamedPipe(
        pipe, _winapi.PIPE_ACCESS_DUPLEX | _winapi.FILE_FLAG_OVERLAPPED,
        0, 1, 1024, 1024, 1000, _winapi.NULL)  # byte mode, blocking server semantics
    stopped = threading.Event()
    listening = threading.Event()
    errors = []
    received = []
    holder = None

    def serve():
        deadline = time.monotonic() + 6.0

        def complete(operation):
            try:
                while _winapi.WaitForSingleObject(operation.event, 20) == _winapi.WAIT_TIMEOUT:
                    if stopped.is_set() or time.monotonic() >= deadline:
                        raise TimeoutError("synthetic peer safety stop")
                _, error = operation.GetOverlappedResult(False)
                if error not in (0, _winapi.ERROR_MORE_DATA):
                    raise OSError(error, "synthetic peer I/O failed")
            except BaseException:
                with contextlib.suppress(OSError):
                    operation.cancel()
                with contextlib.suppress(OSError):
                    operation.GetOverlappedResult(True)
                raise

        try:
            operation = _winapi.ConnectNamedPipe(handle, overlapped=True)
            listening.set()
            complete(operation)
            if mode in {"write-stall", "busy"}:
                stopped.wait(max(0, deadline - time.monotonic()))
                return
            request = bytearray()
            while b"\n" not in request:
                operation, _ = _winapi.ReadFile(handle, 65536, overlapped=True)
                complete(operation)
                chunk = operation.getbuffer()
                if not chunk:
                    return
                request.extend(chunk)
            payload = json.loads(request.partition(b"\n")[0])
            received.append(payload)
            if mode == "eof":
                return  # A peer may close after reading, without sending a reply.
            result = {"verb": payload["verb"], "params": payload.get("params", {})}
            response = json.dumps({"ok": True, "result": result}).encode() + b"\n"
            if mode == "malformed":
                response = b"not-json\n"
            elif mode in {"partial-stall", "eof-json"}:
                response = response[:-1]
            elif mode == "oversized":
                response = json.dumps({"ok": True, "result": {"text": "x" * _MAX_RESPONSE_BYTES}}).encode() + b"\n"
            if mode != "read-stall":
                parts = [response[:5], response[5:]] if mode == "fragmented" else [response]
                for part in parts:
                    operation, _ = _winapi.WriteFile(handle, part, overlapped=True)
                    complete(operation)
                    if mode == "fragmented":
                        stopped.wait(0.03)
            if mode == "eof-json":
                # Closing a pipe may discard unread bytes. Flush synchronizes
                # with the real client consuming this reply, without a sleep.
                if not flush(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                return
            # Keep the connection open until the client closes it. In stalled
            # cases no reply/newline will arrive, regardless of this peer's life.
            operation, _ = _winapi.ReadFile(handle, 1, overlapped=True)
            complete(operation)
        except TimeoutError:
            pass
        except OSError as error:
            # The client closes early for malformed/oversized/timed-out replies.
            if getattr(error, "winerror", error.errno) not in (109, 232, 233, 995):
                errors.append((type(error).__name__, str(error)))
        finally:
            listening.set()
            _winapi.CloseHandle(handle)

    worker = threading.Thread(target=serve, name="synthetic-control-pipe-peer")
    worker.start()
    try:
        assert listening.wait(10), "synthetic native peer did not start"
        assert not errors, errors
        if mode == "busy":
            holder = _winapi.CreateFile(
                pipe, _winapi.GENERIC_READ | _winapi.GENERIC_WRITE,
                0, _winapi.NULL, _winapi.OPEN_EXISTING,
                _winapi.FILE_FLAG_OVERLAPPED, _winapi.NULL)
        yield received
    finally:
        if holder is not None:
            _winapi.CloseHandle(holder)
        stopped.set()
        worker.join(3)
        assert not worker.is_alive(), "synthetic native peer failed to clean up"
        assert not errors, errors


@pytest.mark.parametrize("mode", [
    "responsive", "fragmented", "malformed", "eof", "eof-json", "partial-stall",
    "read-stall", "write-stall", "busy", "oversized",
])
def test_native_control_pipe_exchange_is_bounded(tmp_path, mode):
    home = tmp_path / mode
    home.mkdir()
    params = {"value": "synthetic Unicode كويتي"}
    if mode == "write-stall":
        params = {"value": "x" * 32768}  # exceeds the unread peer's pipe buffer
    with _native_peer(home, mode) as received:
        started = time.monotonic()
        result = query_gateway_control(home, "status", params=params, timeout=0.2)
        elapsed = time.monotonic() - started
        if mode in {"responsive", "fragmented", "eof-json"}:
            assert result == {"verb": "status", "params": params}
            assert received == [{"verb": "status", "id": 1, "protocol": 1, "params": params}]
        else:
            assert result is None
        # Generous scheduling headroom, but shorter than the peer's safety stop.
        assert elapsed < 4.0, f"{mode} ignored the pipe deadline ({elapsed:.2f}s)"


@pytest.mark.parametrize("mode", ["read-stall", "write-stall"])
def test_repeated_native_timeouts_release_handles_and_threads(tmp_path, mode):
    import psutil

    process = psutil.Process()
    # Warm up native imports before taking the handle/thread baseline.
    home = tmp_path / "warmup"
    home.mkdir()
    with _native_peer(home, "responsive"):
        assert query_gateway_control(home, "status", timeout=2) == {"verb": "status", "params": {}}
    gc.collect()
    handles = process.num_handles()
    threads = {thread.ident for thread in threading.enumerate()}
    for attempt in range(8):
        home = tmp_path / str(attempt)
        home.mkdir()
        params = {"value": "x" * 32768} if mode == "write-stall" else None
        with _native_peer(home, mode):
            started = time.monotonic()
            assert query_gateway_control(home, "status", params=params, timeout=0.05) is None
            assert time.monotonic() - started < 4.0
    gc.collect()
    assert process.num_handles() <= handles + 2
    assert {thread.ident for thread in threading.enumerate()} == threads
    # Cancellation leaves the transport usable for the next request.
    home = tmp_path / "after-timeouts"
    home.mkdir()
    with _native_peer(home, "responsive"):
        assert query_gateway_control(home, "status", timeout=2) == {"verb": "status", "params": {}}
