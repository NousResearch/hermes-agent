"""Unauthorized-DM pairing replies must not block the gateway event loop on the pairing lock.

``PairingStore`` edits hold a cross-process flock (the dashboard and the ``hermes pairing`` CLI
mutate the same files). Waiting on it inline in an async handler stalls every chat the gateway
serves for up to the lock timeout, then raises a ``TimeoutError`` the handler never handled.
"""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.pairing import PairingStore
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.lock_contention import watch_file_lock_contention


def _runner(store, adapter):
    runner = object.__new__(GatewayRunner)
    runner._pairing_store_for = lambda _source: store
    runner._delivery_adapter_for = lambda _source: adapter
    runner.config = None
    return runner


def _source(user_id="stranger"):
    return SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm", user_id=user_id)


class _OtherProcessHoldsTheLock:
    """Holds the pairing file lock from another thread until ``release()``; a backstop timer frees
    it if the test dies first."""

    def __init__(self):
        self._locked = threading.Event()
        self._release = threading.Event()
        self._thread = threading.Thread(target=self._hold)
        self._backstop = threading.Timer(30.0, self._release.set)

    def _hold(self):
        with PairingStore()._locked():
            self._locked.set()
            self._release.wait()

    def __enter__(self):
        self._thread.start()
        assert self._locked.wait(5)
        self._backstop.start()
        return self

    def release(self):
        self._release.set()

    def __exit__(self, *exc):
        self._backstop.cancel()
        self._release.set()
        self._thread.join(5)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", ["_hm_offer_pairing_code", "_hm_send_unauthorized_decline"])
async def test_waiting_on_the_pairing_lock_does_not_stall_the_event_loop(monkeypatch, handler):
    """While the pairing edit waits on another process's lock, the handler is parked at an await
    and this coroutine keeps running on the same loop. Inline, the handler's first step would
    hold the loop until the lock wait timed out, and be done by the time this coroutine resumes."""
    contended = watch_file_lock_contention(monkeypatch)
    adapter = AsyncMock()
    runner = _runner(PairingStore(), adapter)

    with _OtherProcessHoldsTheLock() as holder:
        task = asyncio.create_task(getattr(runner, handler)(_source()))
        assert await asyncio.to_thread(contended.wait, 15), "the pairing edit never reached the lock"
        assert not task.done(), "the handler ran the lock wait on the event loop"
        holder.release()
        await asyncio.wait_for(task, 15)

    adapter.send.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("handler, method", [
    ("_hm_offer_pairing_code", "generate_code"),
    ("_hm_send_unauthorized_decline", "record_decline"),
])
async def test_a_pairing_lock_timeout_stays_silent(monkeypatch, handler, method):
    store = PairingStore()

    def timed_out(*_args, **_kwargs):
        raise TimeoutError("Timed out waiting for the pairing store lock")

    monkeypatch.setattr(store, method, timed_out)
    adapter = AsyncMock()

    await getattr(_runner(store, adapter), handler)(_source())

    adapter.send.assert_not_awaited()


def test_an_approval_check_does_not_wait_behind_an_edit_blocked_on_the_file_lock(monkeypatch):
    """Message admission calls ``is_approved`` on the event loop. A pairing edit offloaded to a worker
    that is still waiting for another process's file lock must not hold the store's thread lock
    meanwhile, or that synchronous check stalls the loop for the whole wait."""
    contended = watch_file_lock_contention(monkeypatch)
    store = PairingStore()
    checked = threading.Event()

    with _OtherProcessHoldsTheLock():
        worker = threading.Thread(target=store.generate_code, args=("telegram", "stranger"))
        worker.start()
        assert contended.wait(15), "the pairing edit never reached the lock"
        checker = threading.Thread(target=lambda: store.is_approved("telegram", "someone-else") or checked.set())
        checker.start()
        # The holder never lets go before this answer: behind the edit's thread lock it never comes.
        answered = checked.wait(10)
        edit_still_waiting = worker.is_alive()
    worker.join(15)
    checker.join(15)

    assert edit_still_waiting
    assert answered, "is_approved waited behind an edit blocked on the file lock"
