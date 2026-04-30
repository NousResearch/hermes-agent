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
