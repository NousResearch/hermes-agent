"""Thread-affinity test for the Group Chat hosted-room worker startup import.

Issue #123347: ``_ensure_hosted_room_worker`` ran ``import tui_gateway.server``
inside ``asyncio.to_thread``. That import executes
``methods_connectors.register()`` at import time, which pulls in the
``tools.connectors`` package; racing with concurrent startup imports on the
event-loop thread, the cross-thread import-lock cycle raises
``_frozen_importlib._DeadlockError`` and the worker fails closed.

The contract: the heavyweight import chain must be bound on the calling
(loop) thread before delegating to the worker thread. Same-thread imports
are sequential under the event loop, so no import-lock cycle is possible.
"""

from __future__ import annotations

import asyncio
import importlib.abc
import sys
import threading

import pytest

from gateway.run import GatewayRunner

_WARMED = ("tui_gateway.server",)


class _ImportThreadRecorder(importlib.abc.MetaPathFinder):
    """Records the thread that imports each watched module."""

    def __init__(self, targets) -> None:
        self.targets = set(targets)
        self.threads: dict[str, threading.Thread] = {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.targets and fullname not in self.threads:
            self.threads[fullname] = threading.current_thread()
        return None


@pytest.mark.asyncio
async def test_hosted_room_worker_import_binds_on_caller_thread(monkeypatch):
    """The worker-thread body must never be the importer of tui_gateway.server."""
    caller_thread = threading.current_thread()
    worker_threads: list[threading.Thread] = []

    async def fake_to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _record_worker_thread, func, worker_threads, args, kwargs,
        )

    def _record_worker_thread(func, seen, args, kwargs):
        seen.append(threading.current_thread())
        return func(*args, **kwargs)

    def stub_sync():
        return "started-without-importing"

    monkeypatch.setattr(
        GatewayRunner, "_start_hosted_room_worker_sync", staticmethod(stub_sync),
    )
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    saved = {name: sys.modules.pop(name) for name in _WARMED if name in sys.modules}
    recorder = _ImportThreadRecorder(_WARMED)
    sys.meta_path.insert(0, recorder)
    try:
        runner = GatewayRunner.__new__(GatewayRunner)
        assert await runner._ensure_hosted_room_worker() == "started-without-importing"
    finally:
        sys.meta_path.remove(recorder)
        sys.modules.update(saved)

    assert worker_threads, "expected the worker body to still run off the loop thread"
    assert all(t is not caller_thread for t in worker_threads)
    for name in _WARMED:
        assert name in recorder.threads, (
            f"{name} was imported by the worker thread's stub (or not at all): "
            "the caller thread must bind it before thread delegation (#123347)"
        )
        assert recorder.threads[name] is caller_thread, (
            f"{name} imported on {recorder.threads[name].name}, "
            f"expected {caller_thread.name} (#123347)"
        )
