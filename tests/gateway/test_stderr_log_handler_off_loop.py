"""The gateway's -v/-q stderr log handler must never write on the caller's thread.

Under a service manager (launchd/systemd) stderr is a log file. A StreamHandler attached
directly to root makes every WARNING+ logged on the event loop a synchronous disk write, so a
slow disk blocks the loop (observed: 10 s loop blocks on a platform heartbeat path).
"""
from __future__ import annotations

import ast
import logging
import threading
import time
from pathlib import Path

import hermes_logging

RUN_PY = Path(__file__).resolve().parents[2] / "gateway" / "run.py"


def _stderr_handler_attach_calls(src: str) -> list[str]:
    """Names of the calls that take ``_stderr_handler`` as their only positional arg."""
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and len(node.args) == 1:
            arg = node.args[0]
            if isinstance(arg, ast.Name) and arg.id == "_stderr_handler":
                func = node.func
                out.append(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "?"))
    return out


def test_stderr_handler_is_registered_on_the_queue_not_root():
    calls = _stderr_handler_attach_calls(RUN_PY.read_text())
    assert "_register_queued_handler" in calls
    assert "addHandler" not in calls


class _BlockingStream:
    def __init__(self, block_s):
        self.block_s = block_s
        self.lines = []
        self.writer_threads = set()

    def write(self, s):
        self.writer_threads.add(threading.get_ident())
        time.sleep(self.block_s)
        self.lines.append(s)

    def flush(self):
        pass


def test_queued_stream_handler_does_not_block_the_caller():
    stream = _BlockingStream(1.0)
    h = logging.StreamHandler(stream)
    h.setLevel(logging.WARNING)
    hermes_logging._register_queued_handler(h)
    try:
        log = logging.getLogger("tests.gateway.stderr_offloop")
        t0 = time.monotonic()
        log.warning("loop-thread warning must return immediately")
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"logger.warning blocked the caller for {elapsed:.2f}s"
        deadline = time.monotonic() + 5
        while not stream.lines and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stream.lines, "queued handler never wrote the record"
        assert threading.get_ident() not in stream.writer_threads, "write ran on the caller's thread"
    finally:
        try:
            hermes_logging._queued_file_handlers.remove(h)
        except ValueError:
            pass
