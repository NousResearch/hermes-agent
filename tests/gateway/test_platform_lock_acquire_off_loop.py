"""The platform-lock acquire must not run on the event loop.

``_acquire_platform_lock`` is file I/O, and on the explicit ``--replace`` takeover path it polls for the
old owner's exit with ``time.sleep`` for up to ~15 s. Every adapter calls it from ``async def connect()``,
so inline it stalls every other adapter and heartbeat in the process.
"""

import ast
import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import BasePlatformAdapter

REPO = Path(__file__).resolve().parents[2]


class _StubAdapter(BasePlatformAdapter):
    platform = MagicMock(value="telegram")

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        pass

    async def send(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {}


@pytest.fixture()
def adapter():
    obj = _StubAdapter.__new__(_StubAdapter)
    obj._platform_lock_scope = None
    obj._platform_lock_identity = None
    return obj


def _slow_acquire(record, result=True, hold=0.3):
    def acquire(self, scope, identity, resource_desc):
        record.append(threading.current_thread().name)
        time.sleep(hold)  # stands in for the takeover poll
        self._platform_lock_scope, self._platform_lock_identity = scope, identity
        return result
    return acquire


def test_acquire_runs_off_the_loop_and_the_loop_keeps_ticking(adapter, monkeypatch):
    threads = []
    monkeypatch.setattr(_StubAdapter, "_acquire_platform_lock", _slow_acquire(threads))

    async def scenario():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        try:
            ok = await adapter._acquire_platform_lock_async("telegram-bot-token", "tok", "Telegram bot token")
        finally:
            t.cancel()
        return ok, ticks, threading.current_thread().name

    ok, ticks, loop_thread = asyncio.run(scenario())
    assert ok is True
    assert threads and threads[0] != loop_thread
    assert ticks >= 10, f"loop only ticked {ticks}x while the acquire blocked"


def test_cancelled_acquire_releases_the_lock_it_took(adapter, monkeypatch):
    monkeypatch.setattr(_StubAdapter, "_acquire_platform_lock", _slow_acquire([]))
    released = []
    monkeypatch.setattr(_StubAdapter, "_release_platform_lock", lambda self: released.append(True))

    async def scenario():
        task = asyncio.create_task(adapter._acquire_platform_lock_async("s", "i", "desc"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.5)  # let the shielded worker finish

    asyncio.run(scenario())
    assert released == [True], "an orphaned acquire kept a lock nobody will use"


def test_cancelled_acquire_leaves_a_newer_acquire_alone(adapter, monkeypatch):
    monkeypatch.setattr(_StubAdapter, "_acquire_platform_lock", _slow_acquire([]))
    released = []
    monkeypatch.setattr(_StubAdapter, "_release_platform_lock", lambda self: released.append(True))

    async def scenario():
        first = asyncio.create_task(adapter._acquire_platform_lock_async("s", "i", "desc"))
        await asyncio.sleep(0.05)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        # A retry connect on the same adapter acquires the same pair before the orphan lands.
        assert await adapter._acquire_platform_lock_async("s", "i", "desc") is True
        await asyncio.sleep(0.2)

    asyncio.run(scenario())
    assert released == [], "the cancelled acquire released the lock a retry connect now holds"


def test_failed_cancelled_acquire_releases_nothing(adapter, monkeypatch):
    monkeypatch.setattr(_StubAdapter, "_acquire_platform_lock", _slow_acquire([], result=False))
    released = []
    monkeypatch.setattr(_StubAdapter, "_release_platform_lock", lambda self: released.append(True))

    async def scenario():
        task = asyncio.create_task(adapter._acquire_platform_lock_async("s", "i", "desc"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.5)

    asyncio.run(scenario())
    assert released == []


def _direct_acquire_calls_in_async_bodies(tree):
    hits = []

    def scan(node, fn):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.AsyncFunctionDef) and node is not fn:
            walk_async(node)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_acquire_platform_lock"):
            hits.append((fn.name, node.lineno))
        for child in ast.iter_child_nodes(node):
            scan(child, fn)

    def walk_async(fn):
        for stmt in fn.body:
            scan(stmt, fn)

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            walk_async(node)
    return sorted(set(hits))


def test_no_coroutine_calls_the_blocking_acquire_directly():
    files = sorted((REPO / "gateway").rglob("*.py")) + sorted((REPO / "plugins" / "platforms").rglob("*.py"))
    offenders = []
    for path in files:
        hits = _direct_acquire_calls_in_async_bodies(ast.parse(path.read_text(encoding="utf-8")))
        offenders += [f"{path.relative_to(REPO)}:{ln} in {fn}()" for fn, ln in hits]
    assert not offenders, "blocking _acquire_platform_lock() on the loop; use _acquire_platform_lock_async: " + ", ".join(offenders)


def test_contract_fires_on_a_direct_call():
    tree = ast.parse(
        "async def connect(self):\n"
        "    if not self._acquire_platform_lock('s', 'i', 'd'):\n"
        "        return False\n"
        "    await asyncio.to_thread(lambda: self._acquire_platform_lock('s', 'i', 'd'))\n")
    assert _direct_acquire_calls_in_async_bodies(tree) == [("connect", 2)]
