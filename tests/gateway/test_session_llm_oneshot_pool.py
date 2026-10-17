"""The owner's one-shot pool: bounded, scoped like the handler, and owned by connection and owner.

A one-shot's model call is the one long blocking call the owner answers off the connection's serial
dispatch. It runs on the owner's own small pool, never the loop's default executor, and its answer
lives only as long as the socket that asked for it and the owner that serves it."""
import asyncio
import contextvars
import gc
import threading
from types import SimpleNamespace
import weakref

import pytest

from gateway import session_llm_oneshot as oneshot
from hermes_state_runtime import RuntimeStoreError
from tests.waiting import async_wait_until


class _Transport:
    def __init__(self):
        self.frames = []

    async def write_async(self, frame):
        self.frames.append(frame)
        return True


def _connection(authority):
    return SimpleNamespace(authority=authority, transport=_Transport())


class _Calls:
    """Model calls parked on one gate, counting how many ran at once."""

    def __init__(self):
        self.gate, self.lock = threading.Event(), threading.Lock()
        self.ran, self.live, self.peak = [], 0, 0

    def work(self, label):
        def run():
            with self.lock:
                self.ran.append(label)
                self.live += 1
                self.peak = max(self.peak, self.live)
            self.gate.wait(10)
            with self.lock:
                self.live -= 1
            return {'text': label}
        return run


def test_a_full_pool_refuses_by_name_and_never_runs_past_its_workers():
    calls = _Calls()

    async def scenario():
        connection = _connection(SimpleNamespace())
        for n in range(oneshot._ADMITTED):
            oneshot.DeferredReply(calls.work(f'call{n}')).start(connection, n)
        with pytest.raises(RuntimeStoreError) as refused:
            oneshot.DeferredReply(calls.work('extra')).start(connection, 'extra')
        await async_wait_until(lambda: len(calls.ran) == oneshot._WORKERS)
        # Unrelated default-executor work is untouched by the parked calls.
        async with asyncio.timeout(5):
            assert await asyncio.to_thread(lambda: 'free') == 'free'
        calls.gate.set()
        await async_wait_until(lambda: len(connection.transport.frames) == oneshot._ADMITTED)
        return refused.value.reason, connection.transport.frames
    reason, frames = asyncio.run(scenario())
    assert reason == 'oneshot_busy'
    assert calls.peak == oneshot._WORKERS
    assert 'extra' not in calls.ran and len(calls.ran) == oneshot._ADMITTED
    assert sorted(frame['id'] for frame in frames) == list(range(oneshot._ADMITTED))


def test_a_closed_socket_drops_its_queued_calls_and_its_running_reply():
    calls = _Calls()

    async def scenario():
        authority = SimpleNamespace()
        gone, kept = _connection(authority), _connection(authority)
        for n in range(oneshot._WORKERS):  # fill every worker: the next calls queue
            oneshot.DeferredReply(calls.work(f'running{n}')).start(gone, f'running{n}')
        await async_wait_until(lambda: len(calls.ran) == oneshot._WORKERS)
        oneshot.DeferredReply(calls.work('queued')).start(gone, 'queued')
        oneshot.DeferredReply(calls.work('other')).start(kept, 'other')
        pool = authority._oneshot_pool
        oneshot.forget(gone)
        await async_wait_until(lambda: all(owner is not gone for owner in pool.answers.values()))
        # A running call can't be interrupted: it keeps its slot until its thread returns.
        held = len(pool.calls)
        calls.gate.set()
        await async_wait_until(lambda: not pool.answers and not pool.calls)
        return held, gone.transport.frames, kept.transport.frames
    held, gone_frames, kept_frames = asyncio.run(scenario())
    assert held == oneshot._WORKERS + 1  # the running calls and the other socket's queued one
    assert 'queued' not in calls.ran  # dropped before it started
    assert gone_frames == []  # nobody reads a closed socket's replies
    assert kept_frames == [{'jsonrpc': '2.0', 'id': 'other', 'result': {'text': 'other'}}]


class _Payload:
    """Stands in for a queued call's params and copied context."""


def test_a_reconnect_loop_never_retains_more_queued_payloads_than_the_bound():
    """Every worker parked on the model; sockets keep filling the queue and closing. A cancelled call
    must leave the queue with its payload, not wait for a worker to dequeue it."""
    calls, payloads = _Calls(), weakref.WeakSet()

    def queued(label):
        payload = _Payload()
        payloads.add(payload)
        return lambda: {'text': label, 'payload': repr(payload)}

    async def scenario():
        authority = SimpleNamespace()
        busy = _connection(authority)
        for n in range(oneshot._WORKERS):
            oneshot.DeferredReply(calls.work(f'running{n}')).start(busy, f'running{n}')
        await async_wait_until(lambda: len(calls.ran) == oneshot._WORKERS)
        pool, retained = authority._oneshot_pool, []
        for cycle in range(5):
            gone = _connection(authority)
            for n in range(oneshot._ADMITTED - oneshot._WORKERS):
                oneshot.DeferredReply(queued(f'{cycle}.{n}')).start(gone, f'{cycle}.{n}')
            oneshot.forget(gone)
            await async_wait_until(lambda: len(pool.answers) == oneshot._WORKERS)
            gc.collect()
            retained.append(len(payloads))
        calls.gate.set()
        await async_wait_until(lambda: not pool.answers and not pool.calls)
        return retained, busy.transport.frames
    retained, frames = asyncio.run(scenario())
    assert max(retained) <= oneshot._ADMITTED - oneshot._WORKERS, retained
    assert retained == [0] * 5, retained  # nothing queued survives its socket
    assert len(calls.ran) == oneshot._WORKERS and len(frames) == oneshot._WORKERS


def test_owner_shutdown_refuses_new_calls_and_cancels_the_queue():
    calls = _Calls()

    async def scenario():
        authority = SimpleNamespace()
        connection = _connection(authority)
        idle = SimpleNamespace()
        oneshot.stop(idle)  # an owner that never served a one-shot refuses from its stop on too
        with pytest.raises(RuntimeStoreError):
            oneshot.DeferredReply(calls.work('idle')).start(_connection(idle), 'idle')
        for n in range(oneshot._WORKERS + 2):
            oneshot.DeferredReply(calls.work(f'call{n}')).start(connection, n)
        await async_wait_until(lambda: len(calls.ran) == oneshot._WORKERS)
        oneshot.stop(authority)
        with pytest.raises(RuntimeStoreError) as refused:
            oneshot.DeferredReply(calls.work('late')).start(connection, 'late')
        pool = authority._oneshot_pool
        calls.gate.set()
        await async_wait_until(lambda: not pool.calls and not pool.answers)
        return refused.value.reason, connection.transport.frames
    reason, frames = asyncio.run(scenario())
    assert reason == 'runtime_draining'
    assert len(calls.ran) == oneshot._WORKERS  # the queued two never started
    assert frames == []


def test_the_call_runs_in_the_handlers_context_and_its_result_is_checked_before_the_reply():
    scope = contextvars.ContextVar('scope', default='ambient')

    def bad_result(result):
        raise ValueError(f'result violates its contract: {result}')

    async def scenario():
        connection = _connection(SimpleNamespace())
        token = scope.set('secondary-profile')
        try:
            oneshot.DeferredReply(lambda: {'text': scope.get()}).start(connection, 'scoped', lambda result: None)
        finally:
            scope.reset(token)
        oneshot.DeferredReply(lambda: {'text': 'x'}).start(connection, 'checked', bad_result)
        await async_wait_until(lambda: len(connection.transport.frames) == 2)
        return {frame['id']: frame for frame in connection.transport.frames}
    frames = asyncio.run(scenario())
    assert frames['scoped']['result'] == {'text': 'secondary-profile'}
    assert frames['checked']['error']['code'] == 5030
    assert 'violates its contract' in frames['checked']['error']['message']


def test_a_worker_that_cannot_start_gives_its_slot_back(monkeypatch):
    """Thread exhaustion at spawn: the call fails with the start error, and its admission and the
    worker count go back, so the owner does not answer ``oneshot_busy`` for good after 16 of them."""
    from agent import memory_provider
    real = memory_provider.spawn_context_thread

    class _Unstartable:
        def start(self):
            raise RuntimeError("can't start new thread")
    monkeypatch.setattr(memory_provider, 'spawn_context_thread', lambda *args, **kwargs: _Unstartable())
    pool, refusals = oneshot._Pool(), []
    for _ in range(oneshot._ADMITTED + 1):
        try:
            pool.submit(lambda: {'text': 'never'})
        except Exception as exc:  # the start error, or a busy refusal once slots leak
            refusals.append(type(exc).__name__)
    assert refusals == ['RuntimeError'] * (oneshot._ADMITTED + 1)
    assert (pool.calls, pool.queue, pool.threads) == (set(), {}, 0)
    monkeypatch.setattr(memory_provider, 'spawn_context_thread', real)
    assert pool.submit(lambda: {'text': 'ok'}).result(timeout=10) == {'text': 'ok'}
