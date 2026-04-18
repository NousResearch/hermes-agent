"""Shared fixtures/utilities for ``tests/agent`` — import these from tests
instead of redefining them per file.

The ACP test suites (``test_acp_client_base.py``,
``test_claude_code_acp_client.py``) both need an in-process stand-in for a
JSON-RPC subprocess. Keep the fakes here so they stay in sync as the real
transport evolves.
"""
from __future__ import annotations

import json
import queue
import threading


class _FakeStdin:
    """Captures lines the client writes to the subprocess."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            self.lines.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeProc:
    """Pretends to be a ``subprocess.Popen`` with manually-pushed stdout frames.

    Tests call ``push(obj)`` to enqueue a JSON-RPC message; the client
    reader thread pops them via iteration over ``stdout``. ``close_stdout``
    simulates EOF (the reader raises ``StopIteration``).
    """

    def __init__(self) -> None:
        self.stdin = _FakeStdin()
        self._inbox: "queue.Queue[str | None]" = queue.Queue()
        self.stdout = self  # iterator
        self.stderr = None
        self._return_code: int | None = None
        self._killed = False
        self._terminated = False

    # Popen-compatible stub surface
    def poll(self):
        return self._return_code

    def terminate(self):
        self._terminated = True
        self._return_code = 0

    def kill(self):
        self._killed = True
        self._return_code = -9

    def wait(self, timeout=None):
        return self._return_code if self._return_code is not None else 0

    # Stdout as an iterator over JSON-RPC lines
    def __iter__(self):
        return self

    def __next__(self):
        item = self._inbox.get()
        if item is None:
            raise StopIteration
        return item

    # Test helpers
    def push(self, obj: dict) -> None:
        self._inbox.put(json.dumps(obj) + "\n")

    def close_stdout(self) -> None:
        self._inbox.put(None)
