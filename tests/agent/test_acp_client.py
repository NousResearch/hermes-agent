"""Tests for agent.transports.acp_client — ACP wire client.

Tests drive the three-arm dispatch, timeout handling, initialize handshake,
respond/respond_error, and close behavior. All tests use fake subprocesses
(threads writing to queues) — no real ACP agent binary required.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from io import BytesIO
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from agent.transports.acp_client import ACPClient, ACPClientError


# ---------------------------------------------------------------------------
# Helpers — fake subprocess
# ---------------------------------------------------------------------------


class FakeProc:
    """Minimal Popen-compatible fake for testing ACPClient without a real process.

    Writes pre-programmed responses to an internal pipe; the ACPClient's reader
    thread reads from it just as it would from a real subprocess.
    """

    def __init__(self) -> None:
        self._stdin_r, self._stdin_w = self._make_pipe()
        self._stdout_r, self._stdout_w = self._make_pipe()
        self._stderr_r, self._stderr_w = self._make_pipe()

        # Public attrs ACPClient accesses
        self.stdin = _WritableStream(self._stdin_w)
        self.stdout = _ReadableStream(self._stdout_r)
        self.stderr = _ReadableStream(self._stderr_r)
        self._returncode = None

    def _make_pipe(self):
        import os
        r, w = os.pipe()
        return os.fdopen(r, "rb", 0), os.fdopen(w, "wb", 0)

    def poll(self) -> None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = -15
        self._close_all()

    def kill(self) -> None:
        self._returncode = -9
        self._close_all()

    def wait(self, timeout: float = None) -> int:
        return self._returncode or 0

    def _close_all(self) -> None:
        for f in (self._stdin_w, self._stdout_w, self._stderr_w,
                  self._stdin_r, self._stdout_r, self._stderr_r):
            try:
                f.close()
            except Exception:
                pass

    def push_stdout(self, msg: dict) -> None:
        """Push a JSON-RPC message to the fake stdout (agent → client)."""
        data = (json.dumps(msg) + "\n").encode("utf-8")
        try:
            self._stdout_w.write(data)
            self._stdout_w.flush()
        except Exception:
            pass

    def push_stderr(self, text: str) -> None:
        data = (text + "\n").encode("utf-8")
        try:
            self._stderr_w.write(data)
            self._stderr_w.flush()
        except Exception:
            pass

    def drain_stdin(self) -> list[dict]:
        """Read all bytes written to stdin and parse as newline-delimited JSON."""
        chunks = []
        try:
            # Non-blocking read via direct fd
            import select
            fd = self._stdin_r.fileno()
            buf = b""
            while True:
                ready, _, _ = select.select([fd], [], [], 0.05)
                if not ready:
                    break
                data = self._stdin_r.read(4096)
                if not data:
                    break
                buf += data
        except Exception:
            buf = b""
        for line in buf.splitlines():
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return chunks


class _WritableStream:
    def __init__(self, f) -> None:
        self._f = f
        self.closed = False

    def write(self, data: bytes) -> int:
        return self._f.write(data)

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self.closed = True
        try:
            self._f.close()
        except Exception:
            pass


class _ReadableStream:
    def __init__(self, f) -> None:
        self._f = f

    def readline(self) -> bytes:
        return self._f.readline()

    def fileno(self) -> int:
        return self._f.fileno()


def make_client_with_fake_proc() -> tuple[ACPClient, FakeProc]:
    """Create an ACPClient wired to a FakeProc, bypassing subprocess.Popen."""
    fake = FakeProc()
    client = ACPClient.__new__(ACPClient)
    # Manually initialize the client fields (mirrors __init__ without Popen)
    import queue as q_mod
    client._command = "fake-acp-agent"
    client._proc = fake
    client._next_id = 1
    client._pending = {}
    client._pending_lock = threading.Lock()
    client._send_lock = threading.Lock()
    client._notifications = q_mod.Queue()
    client._server_requests = q_mod.Queue()
    client._stderr_lines = []
    client._stderr_lock = threading.Lock()
    client._closed = False
    client._initialized = False

    # Start reader threads
    client._reader = threading.Thread(target=client._read_stdout, daemon=True)
    client._reader.start()
    client._stderr_reader = threading.Thread(target=client._read_stderr, daemon=True)
    client._stderr_reader.start()

    return client, fake


# ---------------------------------------------------------------------------
# Tests: dispatch arms
# ---------------------------------------------------------------------------


class TestDispatch:
    def test_reply_unblocks_pending(self):
        """Arm 1: a reply (id + result, no method) resolves the pending request."""
        client, fake = make_client_with_fake_proc()
        # Enqueue a reply before the request thread blocks on it
        def push_reply():
            time.sleep(0.05)
            fake.push_stdout({"id": 1, "result": {"ok": True}})

        t = threading.Thread(target=push_reply, daemon=True)
        t.start()
        result = client.request("test/method", {}, timeout=2.0)
        assert result == {"ok": True}
        t.join(timeout=1.0)
        client.close()

    def test_reply_error_raises_acp_client_error(self):
        """Arm 1 error path: error response raises ACPClientError."""
        client, fake = make_client_with_fake_proc()

        def push_error():
            time.sleep(0.05)
            fake.push_stdout({"id": 1, "error": {"code": -32601, "message": "not found"}})

        t = threading.Thread(target=push_error, daemon=True)
        t.start()
        with pytest.raises(ACPClientError) as exc_info:
            client.request("test/method", {}, timeout=2.0)
        assert exc_info.value.code == -32601
        assert "not found" in exc_info.value.message
        t.join(timeout=1.0)
        client.close()

    def test_server_request_goes_to_server_requests_queue(self):
        """Arm 2: a message with id + method lands in _server_requests."""
        client, fake = make_client_with_fake_proc()
        fake.push_stdout({"id": 99, "method": "session/request_permission", "params": {"permissionId": "exec"}})
        time.sleep(0.1)
        req = client.take_server_request(timeout=0.5)
        assert req is not None
        assert req["method"] == "session/request_permission"
        assert req["id"] == 99
        client.close()

    def test_notification_goes_to_notifications_queue(self):
        """Arm 3: a message with method but no id lands in _notifications."""
        client, fake = make_client_with_fake_proc()
        fake.push_stdout({"method": "session/update", "params": {"sessionId": "abc", "update": {"sessionUpdate": "agent_message_chunk"}}})
        time.sleep(0.1)
        note = client.take_notification(timeout=0.5)
        assert note is not None
        assert note["method"] == "session/update"
        client.close()

    def test_non_json_stdout_captured_as_stderr(self):
        """Non-JSON stdout is stored in stderr buffer for diagnostics."""
        client, fake = make_client_with_fake_proc()
        fake.push_stdout.__func__  # just touching to avoid lint
        # Inject non-JSON directly to the stdout pipe
        data = b"not json at all\n"
        fake._stdout_w.write(data)
        fake._stdout_w.flush()
        time.sleep(0.1)
        tail = client.stderr_tail(5)
        assert any("non-json" in line for line in tail)
        client.close()


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    def test_request_timeout_raises_timeout_error(self):
        """Request times out when no reply arrives."""
        client, fake = make_client_with_fake_proc()
        with pytest.raises(TimeoutError) as exc_info:
            client.request("session/prompt", {}, timeout=0.1)
        assert "session/prompt" in str(exc_info.value)
        assert "timed out" in str(exc_info.value)
        client.close()

    def test_timed_out_request_removed_from_pending(self):
        """After timeout, the pending entry is cleaned up so no memory leak."""
        client, fake = make_client_with_fake_proc()
        with pytest.raises(TimeoutError):
            client.request("session/prompt", {}, timeout=0.05)
        with client._pending_lock:
            assert len(client._pending) == 0
        client.close()


# ---------------------------------------------------------------------------
# Tests: initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_initialize_sends_correct_params_and_returns_result(self):
        """initialize() sends the right wire params and returns the result dict."""
        client, fake = make_client_with_fake_proc()

        def push_response():
            time.sleep(0.05)
            fake.push_stdout({
                "id": 1,
                "result": {
                    "protocolVersion": 1,
                    "agentInfo": {"name": "test-agent", "version": "1.0"},
                    "agentCapabilities": {},
                }
            })

        t = threading.Thread(target=push_response, daemon=True)
        t.start()
        result = client.initialize(client_name="hermes", client_version="0.1", timeout=2.0)
        assert result["agentInfo"]["name"] == "test-agent"
        assert client._initialized is True
        t.join(timeout=1.0)

        # Verify what was sent on the wire
        sent = fake.drain_stdin()
        assert len(sent) == 1
        msg = sent[0]
        assert msg["method"] == "initialize"
        assert msg["params"]["protocolVersion"] == 1
        assert msg["params"]["clientInfo"]["name"] == "hermes"
        client.close()

    def test_initialize_twice_raises(self):
        """initialize() called twice raises RuntimeError."""
        client, fake = make_client_with_fake_proc()

        def push_response():
            time.sleep(0.05)
            fake.push_stdout({"id": 1, "result": {"protocolVersion": 1}})

        t = threading.Thread(target=push_response, daemon=True)
        t.start()
        client.initialize(timeout=2.0)
        t.join(timeout=1.0)

        with pytest.raises(RuntimeError, match="already initialized"):
            client.initialize()
        client.close()


# ---------------------------------------------------------------------------
# Tests: respond / respond_error
# ---------------------------------------------------------------------------


class TestRespond:
    def test_respond_sends_json_rpc_result(self):
        """respond() sends a proper JSON-RPC result message."""
        client, fake = make_client_with_fake_proc()
        client.respond(42, {"action": "accept"})
        time.sleep(0.05)
        sent = fake.drain_stdin()
        assert len(sent) == 1
        msg = sent[0]
        assert msg["id"] == 42
        assert msg["result"] == {"action": "accept"}
        assert "error" not in msg
        client.close()

    def test_respond_error_sends_json_rpc_error(self):
        """respond_error() sends a proper JSON-RPC error message."""
        client, fake = make_client_with_fake_proc()
        client.respond_error(7, code=-32601, message="not found", data={"method": "fs/write"})
        time.sleep(0.05)
        sent = fake.drain_stdin()
        assert len(sent) == 1
        msg = sent[0]
        assert msg["id"] == 7
        assert msg["error"]["code"] == -32601
        assert msg["error"]["message"] == "not found"
        assert msg["error"]["data"]["method"] == "fs/write"
        client.close()


# ---------------------------------------------------------------------------
# Tests: notify
# ---------------------------------------------------------------------------


class TestNotify:
    def test_notify_sends_no_id(self):
        """notify() sends a notification with no id field."""
        client, fake = make_client_with_fake_proc()
        client.notify("session/cancel", {"sessionId": "abc"})
        time.sleep(0.05)
        sent = fake.drain_stdin()
        assert len(sent) == 1
        msg = sent[0]
        assert "id" not in msg
        assert msg["method"] == "session/cancel"
        assert msg["params"]["sessionId"] == "abc"
        client.close()


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_idempotent(self):
        """close() can be called multiple times without error."""
        client, fake = make_client_with_fake_proc()
        client.close()
        client.close()  # second call must not raise

    def test_send_after_close_raises(self):
        """Calling request() after close() raises RuntimeError."""
        client, fake = make_client_with_fake_proc()
        client.close()
        with pytest.raises(RuntimeError, match="closed"):
            client.request("session/prompt", {}, timeout=0.1)

    def test_is_alive_false_after_terminate(self):
        """is_alive() returns False after the fake process is terminated."""
        client, fake = make_client_with_fake_proc()
        assert client.is_alive() is True
        fake.terminate()
        assert client.is_alive() is False
        client.close()


# ---------------------------------------------------------------------------
# Tests: stderr capture
# ---------------------------------------------------------------------------


class TestStderrCapture:
    def test_stderr_tail_returns_lines(self):
        """stderr lines pushed by the agent are captured in stderr_tail."""
        client, fake = make_client_with_fake_proc()
        fake.push_stderr("agent diagnostic line 1")
        fake.push_stderr("agent diagnostic line 2")
        time.sleep(0.1)
        tail = client.stderr_tail(5)
        assert any("diagnostic line 1" in l for l in tail)
        assert any("diagnostic line 2" in l for l in tail)
        client.close()

    def test_stderr_bounded_at_500_lines(self):
        """stderr buffer is bounded to 500 lines."""
        client, fake = make_client_with_fake_proc()
        for i in range(600):
            fake.push_stderr(f"line {i}")
        time.sleep(0.3)
        tail = client.stderr_tail(1000)
        assert len(tail) <= 500
        client.close()


# ---------------------------------------------------------------------------
# Tests: concurrent-write regression (_send_lock)
# ---------------------------------------------------------------------------


class TestConcurrentWrite:
    def test_concurrent_sends_produce_valid_json_lines(self):
        """Each concurrent _send call must produce a complete, valid JSON line.

        Regression guard for the race on BufferedWriter.write: without
        _send_lock, two threads writing simultaneously interleave bytes,
        producing a single mangled line instead of two well-formed ones.
        If _send_lock is ever removed, this test must FAIL.
        """
        client, fake = make_client_with_fake_proc()

        n_threads = 10
        errors: list[str] = []
        barrier = threading.Barrier(n_threads)

        def send_one(idx: int) -> None:
            barrier.wait()  # all threads start writing at the same instant
            try:
                client._send({"method": "test/concurrent", "params": {"idx": idx}})
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=send_one, args=(i,), daemon=True) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert not errors, f"_send raised in threads: {errors}"

        # Allow the reader-side a moment to drain the pipe
        time.sleep(0.1)
        sent = fake.drain_stdin()

        # Every message must be well-formed JSON-RPC with the expected method
        assert len(sent) == n_threads, (
            f"Expected {n_threads} complete messages, got {len(sent)}. "
            "Interleaved writes corrupt lines and cause json.loads to fail in drain_stdin."
        )
        for msg in sent:
            assert msg.get("method") == "test/concurrent"
            assert isinstance(msg.get("params", {}).get("idx"), int)

        client.close()
