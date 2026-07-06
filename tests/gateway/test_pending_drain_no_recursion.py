"""Regression test for #17758 — chained pending-message drains must not
grow the call stack.

Before the fix, ``_process_message_background`` finished a turn, found a
pending follow-up, and drained it via ``await
self._process_message_background(pending_event, session_key)``.  Each
queued follow-up added a frame to the call stack instead of starting
fresh, so under sustained pending-queue activity the C stack would
exhaust at ~2000 nested frames and the process would crash with
SIGSEGV.

After the fix, the in-band drain spawns a fresh task (mirroring the
late-arrival drain pattern), so the stack stays bounded regardless of
chain length.

We assert the invariant directly: count nested
``_process_message_background`` frames at handler entry across a chain
of N follow-ups.  Recursion makes depth grow linearly (1, 2, 3, …, N);
task spawning keeps it constant (1 every time).
"""

import asyncio
import sys
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
)
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(return_value=None)
    return adapter


def _make_event(text="hi", chat_id="42"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
    )


def _sk(chat_id="42"):
    return build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm")
    )


def _count_pmb_frames() -> int:
    """Walk the current call stack and count nested
    ``_process_message_background`` frames.  Used to detect recursive
    in-band drains."""
    f = sys._getframe()
    n = 0
    while f is not None:
        if f.f_code.co_name == "_process_message_background":
            n += 1
        f = f.f_back
    return n


@pytest.mark.asyncio
async def test_in_band_drain_does_not_grow_stack():
    """Issue #17758: chained pending-message drains must not recurse.

    Queue a fresh pending message inside each handler invocation so the
    in-band drain block fires for every turn in the chain.  After N
    turns, the recorded stack depth at handler entry must stay bounded.
    Pre-fix, depths would be 1, 2, 3, …, N; post-fix, depths are 1
    every time because each drain runs in its own task.
    """
    N = 12
    adapter = _make_adapter()
    sk = _sk()

    depths: list[int] = []
    next_index = [1]

    async def handler(event):
        depths.append(_count_pmb_frames())
        if next_index[0] < N:
            adapter._pending_messages[sk] = _make_event(text=f"M{next_index[0]}")
            next_index[0] += 1
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="M0"))

    # Drain the chain.  Each turn schedules the next via the in-band
    # drain block, so we wait until N handler runs have completed and
    # the session has been released.
    for _ in range(400):
        if len(depths) >= N and sk not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()

    assert len(depths) == N, (
        f"expected {N} handler runs in the chain, got {len(depths)}: depths={depths!r}"
    )
    max_depth = max(depths)
    assert max_depth <= 2, (
        f"in-band drain is recursing instead of spawning a fresh task — "
        f"stack depth grew with chain length: {depths!r}"
    )


@pytest.mark.asyncio
async def test_in_band_drain_preserves_active_session_guard():
    """The original task must NOT release ``_active_sessions[session_key]``
    after handing off to the drain task.

    When the in-band drain spawns ``drain_task`` and transfers ownership
    via ``_session_tasks[session_key] = drain_task``, the original task
    still unwinds through the ``finally`` block.  The drain task picks
    up the same ``interrupt_event`` in its own
    ``_process_message_background`` entry, so a naive
    ``_release_session_guard(session_key, guard=interrupt_event)`` in
    the unwind matches and deletes ``_active_sessions[session_key]``.
    That briefly reopens the Level-1 guard between the original task's
    finally and the drain task's first await — a concurrent inbound
    arriving in that window passes the guard and spawns a second
    handler for the same session.

    Invariant: ``_active_sessions[sk]`` must hold the SAME interrupt
    Event identity at every handler entry across an in-band drain
    chain.  Pre-fix, the original task's finally deletes the entry, so
    the drain task falls through to the ``or asyncio.Event()`` branch
    in ``_process_message_background`` and installs a *new* Event —
    the identity diverges.  Post-fix, the entry is preserved across
    handoff and the drain task reuses the original Event.
    """
    adapter = _make_adapter()
    sk = _sk()

    seen_guards: list = []

    async def handler(event):
        seen_guards.append(adapter._active_sessions.get(sk))
        if len(seen_guards) == 1:
            adapter._pending_messages[sk] = _make_event(text="M1")
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="M0"))

    for _ in range(400):
        if len(seen_guards) >= 2 and sk not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()

    assert len(seen_guards) == 2, f"expected 2 handler runs, got {len(seen_guards)}"
    assert seen_guards[0] is not None, "M0 saw no active-session guard"
    assert seen_guards[1] is not None, "M1 saw no active-session guard"
    assert seen_guards[0] is seen_guards[1], (
        "in-band drain handoff replaced the active-session guard — the "
        "original task's finally deleted _active_sessions[sk] and the "
        "drain task installed a new Event.  Concurrent inbounds during "
        "the handoff window would bypass the Level-1 guard and spawn a "
        "second handler for the same session."
    )


# ---------------------------------------------------------------------------
# Follow-up guardrails (belt-and-suspenders on top of the #17758 fix).
#
# The in-band drain hand-off changed cleanup semantics in three subtle ways
# that the original fix reasoned about but didn't test directly.  These
# tests pin each invariant so future refactors can't silently regress them.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_path_releases_session_guard():
    """The common path — one message, nothing queued — must still
    fully release ``_active_sessions[sk]`` and ``_session_tasks[sk]``
    through the end-of-finally block.

    The #17758 fix moved ``_release_session_guard(...)`` under an
    ``if current_task is self._session_tasks.get(session_key)``
    conditional.  For the 99%-common case (no pending message, no
    handoff) ``current_task`` IS the stored task, so the guard must
    still fire.  This test would fail if the conditional were ever
    tightened in a way that dropped the normal path."""
    adapter = _make_adapter()
    sk = _sk()

    async def handler(event):
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="solo"))

    # Wait for the single-shot handler to fully unwind.
    for _ in range(200):
        if sk not in adapter._active_sessions and sk not in adapter._session_tasks:
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()

    assert sk not in adapter._active_sessions, (
        "normal-path unwind left _active_sessions[sk] populated — future "
        "messages would take the busy-handler path forever"
    )
    assert sk not in adapter._session_tasks, (
        "normal-path unwind left _session_tasks[sk] populated — "
        "stale-lock detection will treat a dead task as alive"
    )


@pytest.mark.asyncio
async def test_drain_task_cancellation_releases_session():
    """If the in-band drain task is cancelled (e.g. user sent ``/stop``
    mid-drain), the session guard and task registry must still get
    cleaned up — the cancelled drain task's own ``finally`` runs and
    fires ``_release_session_guard``.

    The #17758 fix transfers ownership of ``_session_tasks[sk]`` to
    the drain task; the drain task's ``except asyncio.CancelledError``
    branch must then own the cleanup.  Without this test a future
    refactor could move cancellation handling in a way that leaves
    the session permanently pinned as busy after a cancel."""
    adapter = _make_adapter()
    sk = _sk()

    turn_started = asyncio.Event()
    drain_hit_handler = asyncio.Event()

    async def handler(event):
        if event.text == "M0":
            # Queue a pending follow-up so an in-band drain task gets spawned.
            adapter._pending_messages[sk] = _make_event(text="M1")
            turn_started.set()
            return "ok"
        # M1 is the drained follow-up — hang so we can cancel the drain task.
        drain_hit_handler.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="M0"))

    # Wait for the drain task to actually start running M1.
    await asyncio.wait_for(drain_hit_handler.wait(), timeout=2)

    # Cancel the drain task mid-handler.
    drain_task = adapter._session_tasks.get(sk)
    assert drain_task is not None, "in-band drain did not install a drain task"
    assert not drain_task.done(), "drain task finished before we could cancel"
    drain_task.cancel()

    # Drain task's finally must release both registries.
    for _ in range(200):
        if sk not in adapter._active_sessions and sk not in adapter._session_tasks:
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()

    assert sk not in adapter._active_sessions, (
        "cancelled drain task did not release _active_sessions[sk] — "
        "the session stays permanently pinned as busy after a /stop mid-drain"
    )
    assert sk not in adapter._session_tasks, (
        "cancelled drain task did not release _session_tasks[sk] — "
        "stale-lock detection will treat the dead task as alive"
    )


@pytest.mark.asyncio
async def test_late_arrival_drain_still_fires_when_no_in_band_drain():
    """The late-arrival drain in ``finally`` must still spawn a fresh
    task when no in-band drain preceded it.

    Pre-#17758 this path already existed; the #17758 follow-up guard
    only re-queues when ``_session_tasks[sk] is not current_task``.
    For a late-arrival with no in-band drain, ``_session_tasks[sk]``
    IS the current task, so the ``else`` branch must fire and spawn
    a drain task for the queued message.

    Queue a pending message *after* M0's handler returns (so the
    in-band drain block sees nothing) but *before* ``finally`` runs
    the late-arrival check — we do this by hooking ``_stop_typing``,
    which runs in finally before the late-arrival check."""
    adapter = _make_adapter()
    sk = _sk()

    results: list[str] = []
    original_stop_typing = getattr(adapter, "stop_typing", None)

    async def injecting_stop_typing(chat_id):
        # Simulate a message landing during the cleanup awaits.
        adapter._pending_messages[sk] = _make_event(text="late")
        if original_stop_typing:
            await original_stop_typing(chat_id)

    adapter.stop_typing = injecting_stop_typing

    async def handler(event):
        results.append(event.text)
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(_make_event(text="first"))

    # Wait for the late-arrival drain task to finish the second event.
    for _ in range(400):
        if "late" in results and sk not in adapter._active_sessions:
            break
        await asyncio.sleep(0.01)

    await adapter.cancel_background_tasks()

    assert "first" in results, "original message handler did not run"
    assert "late" in results, (
        "late-arrival drain did not spawn a drain task — a message that "
        "landed during cleanup awaits was silently dropped"
    )
