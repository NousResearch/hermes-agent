"""``ctx.post_card`` / ``ctx.apost_card`` never raise, including across a gateway shutdown.

The thread side (``hermes_cli/plugin_cards.py``) creates the coroutine, schedules it on the gateway loop
and waits as one guarded operation: a loop closed between the lookup and the scheduling, a loop closed
before the scheduled task ran, a cancelled task and a timeout all come back as the documented result
shape, and a coroutine that never ran is closed (no ``RuntimeWarning: coroutine ... was never awaited``,
which this module turns into an error).
"""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_cli import plugin_cards as thread_side

pytestmark = pytest.mark.filterwarnings('error::RuntimeWarning')

_CARD = {'id': 'share-1', 'text': 'Share it?'}


def _ctx(tmp_path):
    return SimpleNamespace(plugin_id='cwplug', _manager=SimpleNamespace(home_path=tmp_path),
                           _granted_in_own_home=lambda capability: capability == thread_side.CAPABILITY)


def _serve(monkeypatch, loop, runner=None):
    runner = runner or SimpleNamespace(_plugin_injection_context=None)
    monkeypatch.setattr('hermes_cli.plugin_messages._gateway_loop', lambda: (runner, loop))
    return runner


def _gateway_post(monkeypatch, calls, release=None):
    """The gateway-side ``post``: records the call and, with *release*, waits for it before answering."""
    async def post(runner, card, **kwargs):
        calls.append(card)
        if release is not None:
            await release.wait()
        return {'ok': True, 'card_id': card['id'], 'status': 'queued'}
    monkeypatch.setattr('gateway.plugin_cards.post', post)


class _LoopThread:
    """A gateway loop running in its own thread, the way a plugin thread sees it."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def call(self, fn, *args):
        return asyncio.run_coroutine_threadsafe(fn(*args), self.loop) if asyncio.iscoroutinefunction(fn) \
            else self.loop.call_soon_threadsafe(fn, *args)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


def test_loop_closed_between_lookup_and_scheduling_is_a_result(tmp_path, monkeypatch):
    loop = asyncio.new_event_loop()
    loop.close()
    _serve(monkeypatch, loop)
    calls = []
    _gateway_post(monkeypatch, calls)

    result = thread_side.post(_ctx(tmp_path), _CARD, session_key='agent:main:local:dm:x')

    assert result == {'ok': False, 'error': 'gateway_unavailable',
                      'detail': 'the gateway loop closed before the card was scheduled'}
    assert calls == [], 'nothing was admitted'


def test_loop_closed_between_scheduling_and_result_is_a_result(tmp_path, monkeypatch):
    """The callback was queued on a loop that then shut down without running it."""
    loop = asyncio.new_event_loop()  # never run: the scheduling lands, the task never starts
    _serve(monkeypatch, loop)
    calls = []
    _gateway_post(monkeypatch, calls)
    monkeypatch.setattr(thread_side, '_TIMEOUT_S', 10.0)
    monkeypatch.setattr(thread_side, '_POLL_S', 0.02, raising=False)
    results = []
    worker = threading.Thread(target=lambda: results.append(
        thread_side.post(_ctx(tmp_path), _CARD, session_key='agent:main:local:dm:x')))
    worker.start()
    deadline = time.monotonic() + 5
    while not loop._ready and time.monotonic() < deadline:  # the scheduling callback is queued
        time.sleep(0.01)
    assert loop._ready, 'the post was scheduled'
    loop.close()
    worker.join(5)

    assert results == [{'ok': False, 'error': 'gateway_unavailable',
                        'detail': 'the gateway loop closed before the card was admitted'}]
    assert calls == []


def test_cancelled_task_is_a_result(tmp_path, monkeypatch):
    gateway = _LoopThread()
    try:
        _serve(monkeypatch, gateway.loop)
        calls, release = [], asyncio.Event()
        _gateway_post(monkeypatch, calls, release)
        results = []
        worker = threading.Thread(target=lambda: results.append(
            thread_side.post(_ctx(tmp_path), _CARD, session_key='agent:main:local:dm:x')))
        worker.start()
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls, 'the gateway side is running and waiting'

        def shutdown():  # what a gateway shutdown does to its tasks
            for task in asyncio.all_tasks(gateway.loop):
                task.cancel()
        gateway.call(shutdown)
        worker.join(5)
        assert results == [{'ok': False, 'error': 'cancelled',
                            'detail': 'the gateway stopped before answering; the card may not have landed'}]
    finally:
        gateway.stop()


def test_timeout_is_a_result_and_the_card_may_still_land(tmp_path, monkeypatch):
    gateway = _LoopThread()
    try:
        _serve(monkeypatch, gateway.loop)
        calls, release = [], asyncio.Event()
        _gateway_post(monkeypatch, calls, release)
        monkeypatch.setattr(thread_side, '_TIMEOUT_S', 0.3)
        monkeypatch.setattr(thread_side, '_POLL_S', 0.05, raising=False)

        result = thread_side.post(_ctx(tmp_path), _CARD, session_key='agent:main:local:dm:x')

        assert result == {'ok': False, 'error': 'timeout',
                          'detail': 'the gateway did not answer within 0s; the card may still land'}
        gateway.call(release.set)  # the admission finishes on its own, nothing is torn down
        deadline = time.monotonic() + 5
        while any(not t.done() for t in asyncio.all_tasks(gateway.loop)) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls == [_CARD]
    finally:
        gateway.stop()


def test_apost_translates_the_gateways_cancellation_and_keeps_the_callers(tmp_path, monkeypatch):
    async def scenario():
        loop = asyncio.get_running_loop()
        _serve(monkeypatch, loop)
        calls, release = [], asyncio.Event()
        _gateway_post(monkeypatch, calls, release)
        ctx = _ctx(tmp_path)

        # The gateway cancels every task at shutdown: the plugin gets a result.
        caller = asyncio.ensure_future(thread_side.apost(ctx, _CARD, session_key='agent:main:local:dm:x'))
        while not calls:
            await asyncio.sleep(0)
        for task in asyncio.all_tasks(loop):
            if task is not caller and task is not asyncio.current_task():
                task.cancel()
        assert await caller == {'ok': False, 'error': 'cancelled',
                                'detail': 'the gateway stopped before answering; the card may not have landed'}

        # The caller's own cancellation is the caller's: it propagates.
        calls.clear()
        caller = asyncio.ensure_future(thread_side.apost(ctx, {**_CARD, 'id': 'share-2'}, session_key='agent:main:local:dm:x'))
        while not calls:
            await asyncio.sleep(0)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller

        # A loop that refuses the task (closing) answers with a result and closes the coroutine.
        refusing = SimpleNamespace(create_task=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('closed')),
                                   is_closed=lambda: False)
        _serve(monkeypatch, refusing)
        assert await thread_side.apost(ctx, _CARD, session_key='agent:main:local:dm:x') == {
            'ok': False, 'error': 'gateway_unavailable',
            'detail': 'the gateway loop closed before the card was scheduled'}
    asyncio.run(scenario())
