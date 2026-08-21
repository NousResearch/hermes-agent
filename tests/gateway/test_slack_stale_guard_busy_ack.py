"""Regression: an orphaned session guard must not busy-ack a brand-new turn.

Root cause of "Red always starts with ⚡ Interrupting current task".

The busy branch in ``BasePlatformAdapter.handle_message`` triggers purely on
``session_key in self._active_sessions``. Before that check it calls
``_heal_stale_session_lock()``, which is the only safety net that frees a guard
left behind by a run that already ended (issue #11016 split-brain).

``_cancel_session_processing`` pops ``_session_tasks[key]`` unconditionally and,
when called with ``release_guard=False`` (the reset-like /stop, /new, /reset
path), deliberately leaves ``_active_sessions[key]`` installed so the command
can finish atomically. Ownership is expected to pass to a replacement task. If
that replacement never spawns, the guard survives with **no** owner task and
nothing left to release it.

``_session_task_is_stale`` could not detect that from a missing ``_session_tasks``
entry alone, because tests and non-``handle_message`` paths legitimately install
guards with no owner task — treating those as stale breaks that contract. So the
orphan is now recorded explicitly at the one place that creates it
(``_note_orphaned_session_guard``) and cleared wherever a real owner takes over.

Symptom before the fix: every later message on that session key was routed to
the busy handler with ``running_agent = None``, so under
``busy_input_mode: interrupt`` a brand-new turn got a "⚡ Interrupting current
task" ack — with no status detail, because no agent was running.

Complements the #48300 release-then-conditional-delete ordering, which keeps a
*done* task's entry alive so its guard stays recognisable as stale. This closes
the remaining hole where the entry is already gone.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter


class _StubAdapter(BasePlatformAdapter):
    """Concrete stub: real guard logic, abstract API satisfied but unused."""

    async def connect(self, is_reconnect: bool = False):  # pragma: no cover
        raise NotImplementedError

    async def disconnect(self):  # pragma: no cover
        raise NotImplementedError

    async def get_chat_info(self, chat_id):  # pragma: no cover
        raise NotImplementedError

    async def send(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


def _make_adapter():
    """Stub adapter with only the guard plumbing initialised.

    ``object.__new__`` skips ``__init__`` (needs a full GatewayConfig) while
    keeping the genuine ``_heal_stale_session_lock`` under test.
    """
    adapter = object.__new__(_StubAdapter)
    adapter.platform = Platform.SLACK
    adapter._active_sessions = {}
    adapter._session_tasks = {}
    adapter._pending_messages = {}
    adapter._text_debounce = {}
    adapter._orphaned_session_guards = set()
    return adapter


KEY = "agent:main:slack:group:T025KND0E:C0BF1EYUA9H:1787104575.961829"


def test_guard_with_finished_owner_task_is_healed():
    """Baseline: a guard whose owner task already finished is freed."""
    adapter = _make_adapter()
    adapter._active_sessions[KEY] = asyncio.Event()

    finished = MagicMock()
    finished.done.return_value = True
    adapter._session_tasks[KEY] = finished

    healed = adapter._heal_stale_session_lock(KEY)

    assert healed is True
    assert KEY not in adapter._active_sessions


def test_guard_with_live_owner_task_is_preserved():
    """Baseline: a genuinely running turn keeps its guard.

    Counterpart invariant — real mid-run follow-ups must keep reaching the busy
    handler so interrupt/queue behaviour is intact when it is actually wanted.
    """
    adapter = _make_adapter()
    guard = asyncio.Event()
    adapter._active_sessions[KEY] = guard

    live = MagicMock()
    live.done.return_value = False
    adapter._session_tasks[KEY] = live

    healed = adapter._heal_stale_session_lock(KEY)

    assert healed is False
    assert adapter._active_sessions.get(KEY) is guard


def test_unmarked_ownerless_guard_is_left_alone():
    """A directly-installed guard with no owner task is NOT treated as stale.

    Tests and non-``handle_message`` paths legitimately install guards without
    an owner task; the documented contract is to leave those alone. Only a
    guard explicitly marked as orphaned may be healed.
    """
    adapter = _make_adapter()
    guard = asyncio.Event()
    adapter._active_sessions[KEY] = guard

    healed = adapter._heal_stale_session_lock(KEY)

    assert healed is False
    assert adapter._active_sessions.get(KEY) is guard


def test_marked_orphan_guard_is_healed_not_treated_as_busy():
    """THE BUG: a guard orphaned by cancellation must not keep the session busy.

    ``_cancel_session_processing`` pops the owner task and, with
    ``release_guard=False``, leaves the guard installed for a replacement owner
    that never arrives. The orphan marker lets the healer recognise it; without
    healing, every later brand-new turn on this key is busy-acked with
    "⚡ Interrupting current task" while no agent is running.
    """
    adapter = _make_adapter()
    adapter._active_sessions[KEY] = asyncio.Event()
    adapter._note_orphaned_session_guard(KEY)

    healed = adapter._heal_stale_session_lock(KEY)

    assert healed is True, (
        "a guard marked orphaned has no owner and nothing can ever clear it — "
        "it must be healed on entry"
    )
    assert KEY not in adapter._active_sessions, (
        "leaving the orphan installed makes the next brand-new user message "
        "take the busy path and emit '⚡ Interrupting current task' with no "
        "running agent"
    )
    assert KEY not in adapter._orphaned_session_guards, (
        "the marker must be dropped once healed so it cannot free a future "
        "live guard on the same session key"
    )


def test_orphan_marker_is_cleared_when_a_real_owner_takes_over():
    """A replacement owner invalidates the marker.

    Ordering guard: if the marker survived a legitimate ownership handoff, the
    healer would free a guard whose task is still running and two handlers
    could run on one session.
    """
    adapter = _make_adapter()
    adapter._active_sessions[KEY] = asyncio.Event()
    adapter._note_orphaned_session_guard(KEY)

    # A drain/replacement task takes ownership.
    live = MagicMock()
    live.done.return_value = False
    adapter._session_tasks[KEY] = live
    adapter._clear_orphaned_session_guard(KEY)

    assert adapter._session_task_is_stale(KEY) is False
    assert adapter._heal_stale_session_lock(KEY) is False
    assert KEY in adapter._active_sessions
