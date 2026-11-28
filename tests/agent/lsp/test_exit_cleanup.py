"""Regression tests for LSP teardown when the background loop stops.

``loop.stop()`` makes ``run_forever`` return immediately, leaving the reader
parked on the server's stdout and any in-flight shutdown coroutine pending.
Closing the loop then strands those coroutines: the GC finalises them with no
running loop, so ``LSPClient._reader_loop``'s ``finally`` used to die in
``asyncio.current_task()`` with ``RuntimeError: no running event loop`` and
CPython printed it as an ignored exception on exit.

``test_hermes_exit_after_lsp_teardown_prints_no_warnings`` is the regression
test: it runs the reproducer from the bug report in a real interpreter and
asserts the exit is silent.  The other three pin the pieces that keep it that
way (the loop's drain, and the reader's no-loop guard).
"""
from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.manager import _BackgroundLoop
from tests.waiting import wait_until

# The reproducer from the bug report, plus a ``sys.path`` pin so the child imports this checkout
# and not an installed ``hermes-agent`` (an install can .pth-shadow PYTHONPATH entirely).
_REPRO = '''\
import asyncio, gc, sys
sys.path.insert(0, sys.argv[1])
from agent.lsp.client import LSPClient
from agent.lsp.manager import _BackgroundLoop

async def spawn():
    c = LSPClient(server_id="repro", workspace_root=sys.argv[2], command=["true"])
    c._proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(120)",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    print("fake pid:", c._proc.pid, flush=True)
    c._state = "running"
    c._reader_task = asyncio.create_task(c._reader_loop())
    c._stderr_task = asyncio.create_task(c._drain_stderr())
    await asyncio.sleep(0.2)  # let the reader block on stdout
    return c

loop = _BackgroundLoop()
loop.start()
client = loop.run(spawn(), timeout=10)
loop.stop()   # what LSPService.shutdown() does once run() returns or raises
del client
gc.collect()
'''


def _process_gone(pid: int) -> bool:
    """True once ``pid`` is gone; POSIX only (``os.kill`` is not a probe on Windows)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def test_background_loop_drains_pending_tasks_before_closing():
    """A task still parked when the loop stops is cancelled and its cleanup runs."""
    loop = _BackgroundLoop()
    loop.start()
    cleaned = threading.Event()
    holder: dict[str, asyncio.Task] = {}

    async def parked() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def spawn() -> asyncio.Task:
        holder["task"] = asyncio.create_task(parked())
        await asyncio.sleep(0)  # let the task reach its first suspension point
        return holder["task"]

    try:
        task = loop.run(spawn(), timeout=5.0)
        assert not task.done()
    finally:
        loop.stop()

    assert task.done(), "loop.close() ran with the task still pending"
    assert cleaned.is_set(), "task cleanup did not run on the loop"


def test_reader_cleanup_is_skipped_without_a_running_loop(monkeypatch):
    """The GC may finalise the reader after ``loop.close()``; cleanup must not await.

    Driving the coroutine with ``send(None)`` outside a loop is what CPython's
    generator close does at that point: ``asyncio.get_running_loop()`` raises and
    the guard must return instead of calling into ``_cleanup_process``.
    """
    client = LSPClient(server_id="test", workspace_root="/tmp", command=["true"])
    called: list[bool] = []

    async def fake_cleanup() -> None:
        called.append(True)

    monkeypatch.setattr(client, "_cleanup_process", fake_cleanup)
    coro = client._cleanup_after_unexpected_close()
    with pytest.raises(StopIteration):
        coro.send(None)

    assert called == []


def test_reader_cleanup_runs_when_a_loop_is_alive(monkeypatch):
    """With a live loop the unexpected-close path still reaps the server."""
    client = LSPClient(server_id="test", workspace_root="/tmp", command=["true"])
    called: list[bool] = []

    async def fake_cleanup() -> None:
        called.append(True)

    monkeypatch.setattr(client, "_cleanup_process", fake_cleanup)
    asyncio.run(client._cleanup_after_unexpected_close())

    assert called == [True]


def test_hermes_exit_after_lsp_teardown_prints_no_warnings(tmp_path):
    """Stopping the loop under a live reader exits silently and still reaps the server.

    This is the bug as the user sees it: an ignored ``RuntimeError: no running event loop``
    from ``_reader_loop``'s cleanup, plus ``Task was destroyed but it is pending!`` for the
    reader and the stderr drain.  Fails on unmodified ``main``.
    """
    root = Path(__file__).resolve().parents[3]
    script = tmp_path / "exit_cleanup_repro.py"
    script.write_text(_REPRO, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(script), str(root), str(tmp_path)],
        capture_output=True, text=True, timeout=180, check=False, cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": str(root)},
    )
    pid_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("fake pid:")]
    assert pid_line, f"reproducer never started its fake server:\n{proc.stdout}\n{proc.stderr}"
    pid = int(pid_line[0].split(":", 1)[1])

    try:
        assert proc.returncode == 0, proc.stderr
        assert "no running event loop" not in proc.stderr, proc.stderr
        assert "Task was destroyed" not in proc.stderr, proc.stderr
        if os.name == "posix":
            wait_until(lambda: _process_gone(pid), timeout=5.0,
                       what=f"the fake language server (pid {pid}) to be reaped")
    finally:
        if os.name == "posix":
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # already reaped by teardown
                pass
