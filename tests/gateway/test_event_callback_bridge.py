"""Leak-safety of the gateway hook-event bridge (``TurnRunner._event_callback_sync``).

``_event_callback_sync`` is the sync -> async bridge that carries lifecycle hook
events (notably ``session:compress``, emitted from
``agent/conversation_compression.py`` and ``agent/codex_runtime.py``) from the
agent thread onto the gateway's turn loop.  It is wired unconditionally in
``_run_agent_inner`` -- unlike ``step_callback``, which is only attached when
hooks are loaded -- so every gateway turn goes through it.

Bridging a coroutine onto a loop that is closing or gone is the shutdown race
that ``agent.async_utils.safe_schedule_threadsafe`` exists to absorb: a bare
``asyncio.run_coroutine_threadsafe`` raises before the loop ever takes
ownership, leaving the ``emit(...)`` coroutine created-but-never-awaited, which
both drops the hook silently and leaks the coroutine frame with a
``RuntimeWarning: coroutine ... was never awaited``.

These tests pin the invariant on the two failure branches the helper
distinguishes: the loop is closed, and the loop is missing entirely.  They
assert the coroutine ends up *closed*, not merely that no exception escaped --
swallowing the exception was never the missing half.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

from gateway.turn_context import TurnContext


class _RecordingHooks:
    """Stands in for the gateway's hook registry.

    ``emit`` is a real coroutine function, so calling it constructs a coroutine
    object exactly as the production ``hooks.emit`` does.  The object is kept so
    a test can inspect whether the bridge disposed of it.
    """

    def __init__(self) -> None:
        self.coros: list = []
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, event_type: str, context: dict) -> None:
        self.emitted.append((event_type, context))


def _make_runner(ctx: TurnContext):
    from gateway.run import TurnRunner

    class _StubGatewayRunner:
        def _adapter_for_source(self, source):
            return None

    return TurnRunner(_StubGatewayRunner(), ctx)


def _bridge_one_event(loop) -> tuple[object, _RecordingHooks]:
    """Drive one ``session:compress`` event through the bridge onto ``loop``.

    Returns the ``emit()`` coroutine object the bridge was handed, so the caller
    can assert what happened to it.
    """
    hooks = _RecordingHooks()
    real_emit = hooks.emit

    def _capturing_emit(event_type: str, context: dict):
        coro = real_emit(event_type, context)
        hooks.coros.append(coro)
        return coro

    hooks.emit = _capturing_emit  # type: ignore[method-assign]

    ctx = TurnContext(_hooks_ref=hooks, _loop_for_step=loop)
    runner = _make_runner(ctx)
    runner._event_callback_sync("session:compress", {"session_id": "s-1"})

    assert len(hooks.coros) == 1
    return hooks.coros[0], hooks


class TestEventCallbackBridgeLeakSafety:
    def test_closed_step_loop_closes_the_event_coroutine(self, recwarn):
        """A loop closed by shutdown must not strand the emit coroutine.

        This is the live race: the compression path fires ``session:compress``
        from the agent thread while the gateway is tearing its loop down.
        ``run_coroutine_threadsafe`` raises inside ``call_soon_threadsafe``
        before the loop adopts the coroutine, so unless the bridge closes it,
        the hook is dropped *and* the frame leaks.
        """
        loop = asyncio.new_event_loop()
        loop.close()

        coro, hooks = _bridge_one_event(loop)

        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        # The coroutine never ran, so the hook body did not execute -- the
        # point is that it was disposed of, not that it was delivered.
        assert hooks.emitted == []
        assert not [w for w in recwarn.list if "never awaited" in str(w.message)]

    def test_missing_step_loop_closes_the_event_coroutine(self, recwarn):
        """``_loop_for_step`` can be ``None`` outright, a distinct branch.

        The turn context carries ``_loop_for_step=None`` by default, and the
        helper handles that without going near asyncio at all.  The bare bridge
        instead reaches ``None.call_soon_threadsafe`` and leaks the coroutine on
        the way out.
        """
        coro, hooks = _bridge_one_event(None)

        assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        assert hooks.emitted == []
        assert not [w for w in recwarn.list if "never awaited" in str(w.message)]

    def test_live_step_loop_still_delivers_the_event(self):
        """Invariant guard: the leak-safe path must not change the happy path.

        A running loop still receives and runs the hook coroutine, so routing
        through the helper is a pure hardening of the failure branches.
        """
        loop = asyncio.new_event_loop()
        thread = None
        try:
            ready = threading.Event()

            def _run() -> None:
                asyncio.set_event_loop(loop)
                loop.call_soon(ready.set)
                loop.run_forever()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            assert ready.wait(5.0)

            coro, hooks = _bridge_one_event(loop)

            waited = 0.0
            while not hooks.emitted and waited < 5.0:
                time.sleep(0.02)
                waited += 0.02

            assert hooks.emitted == [("session:compress", {"session_id": "s-1"})]
            assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED
        finally:
            loop.call_soon_threadsafe(loop.stop)
            if thread is not None:
                thread.join(timeout=5.0)
            loop.close()


@pytest.mark.parametrize("loop_state", ["closed", "missing"])
def test_bridge_stays_non_raising_and_returns_none(loop_state):
    """Invariant guard on the bridge's own contract, both failure branches.

    ``_event_callback_sync`` is called from the agent thread and is typed
    ``-> None``; the ``except Exception`` it replaces already guaranteed the
    non-raising half, and the helper must keep it while also returning nothing
    (the future is deliberately not consumed here, matching the sibling
    ``_step_callback_sync`` bridge).
    """
    if loop_state == "closed":
        loop = asyncio.new_event_loop()
        loop.close()
    else:
        loop = None

    hooks = _RecordingHooks()
    ctx = TurnContext(_hooks_ref=hooks, _loop_for_step=loop)
    runner = _make_runner(ctx)

    assert runner._event_callback_sync("session:compress", {"session_id": "s-1"}) is None
