"""Regression tests for LSP teardown when the background loop stops.

``loop.stop()`` makes ``run_forever`` return immediately, leaving the reader
parked on the server's stdout and any in-flight shutdown coroutine pending.
Closing the loop then strands those coroutines: the GC finalises them with no
running loop, so ``LSPClient._reader_loop``'s ``finally`` used to die in
``asyncio.current_task()`` with ``RuntimeError: no running event loop`` and
CPython printed it as an ignored exception on exit.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from agent.lsp.client import LSPClient
from agent.lsp.manager import _BackgroundLoop


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
