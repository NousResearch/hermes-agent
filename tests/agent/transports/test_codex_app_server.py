"""Invariants for CodexAppServerClient's transport-loss contract (#87433, #83127).

Bare clients are built via ``object.__new__`` with a mocked, inert subprocess:
pure in-memory queue/pipe logic, no real process.
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import Mock

import pytest

from agent.transports.codex_app_server import (
    CodexAppServerClient, CodexAppServerError, CodexAppServerTransportError,
)


def _bare_client() -> CodexAppServerClient:
    client = object.__new__(CodexAppServerClient)
    client._closed = False
    client._pending = {}
    client._pending_lock = threading.Lock()
    client._next_id = 1
    client._proc = Mock(stdin=None, terminate=Mock(), kill=Mock(), wait=Mock(return_value=0))
    return client


def test_close_fails_in_flight_request_immediately():
    """#87433: close() on another thread must unblock request() with a transport error,
    not leave it riding out its own per-call timeout."""
    client = _bare_client()
    client._send = Mock()
    outcome: dict = {}

    def blocked():
        start = time.monotonic()
        with pytest.raises(CodexAppServerTransportError) as info:
            client.request("turn/start", {}, timeout=10.0)
        outcome["elapsed"] = time.monotonic() - start
        outcome["err"] = info.value

    t = threading.Thread(target=blocked)
    t.start()
    time.sleep(0.05)
    client.close()
    t.join(timeout=5)
    assert not t.is_alive()
    assert outcome["elapsed"] < 2.0
    assert "clos" in str(outcome["err"])
    assert client._pending == {}


def test_write_failure_raises_transport_error_and_drops_pending():
    """#83127: a torn-down stdin surfaces as CodexAppServerTransportError (a CodexAppServerError,
    never a bare RuntimeError) and leaves no orphaned pending slot behind."""
    client = _bare_client()
    client._proc = Mock()
    client._proc.stdin.write.side_effect = BrokenPipeError(32, "Broken pipe")
    with pytest.raises(CodexAppServerTransportError) as info:
        client.request("turn/start", {}, timeout=1.0)
    assert isinstance(info.value, CodexAppServerError)
    assert "stdin closed unexpectedly" in info.value.message
    assert client._pending == {}


def test_stdout_eof_fails_in_flight_request_promptly():
    """Losing the app-server (stdout EOF) must fail a blocked request() with a transport
    error right away rather than letting it ride out its per-call timeout; the later
    close() -> _fail_pending_requests must stay a harmless no-op."""
    client = _bare_client()
    client._send = Mock()
    read_end, write_end = os.pipe()
    client._proc.stdout = os.fdopen(read_end, "rb")
    outcome: dict = {}

    def blocked():
        start = time.monotonic()
        with pytest.raises(CodexAppServerTransportError) as info:
            client.request("turn/start", {}, timeout=10.0)
        outcome["elapsed"] = time.monotonic() - start
        outcome["err"] = info.value

    reader = threading.Thread(target=client._read_stdout)
    reader.start()
    t = threading.Thread(target=blocked)
    t.start()
    time.sleep(0.05)
    os.close(write_end)  # codex died: stdout hits EOF mid-request
    t.join(timeout=5)
    reader.join(timeout=5)
    assert not t.is_alive() and not reader.is_alive()
    assert outcome["elapsed"] < 2.0
    assert "stdout closed" in str(outcome["err"])
    assert client._pending == {}
    client._fail_pending_requests("codex app-server client is closing")  # idempotent
    client._proc.stdout.close()
