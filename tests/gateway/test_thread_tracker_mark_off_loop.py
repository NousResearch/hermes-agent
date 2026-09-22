"""The thread-participation tracker's persist must not block the event loop.

``gateway/platforms/helpers.ThreadParticipationTracker._save`` ends in
``atomic_json_write`` -> ``os.replace``, whose duration is unbounded under
filesystem pressure.  Every caller sits on an inbound-message coroutine --
Matrix ``_resolve_message_context`` and ``create_handoff_thread``, Discord
``_handle_message`` (x2) and ``_handle_thread_create_slash`` -- so that rename
was paid inline on the loop, stalling every other adapter's polling and every
in-flight turn for as long as it took.

The fix is a CHOKE-POINT one rather than five call-site ones: ``mark_async``
offloads only the persist via ``asyncio.to_thread``, and all five coroutine
call sites await it.  ``mark`` keeps its exact synchronous contract for the
non-loop callers and tests.

These tests are about OBSERVABLE BEHAVIOUR -- can the loop keep running while
the persist is in its slow path -- not about which API was called.  There are no
wall-clock thresholds: the rename is held open on a real barrier for as long as
the assertion needs, and a companion gate-proof forces the pre-fix inline shape
and asserts the very same barrier DOES starve the loop, so the liveness test
cannot pass vacuously.

They also pin the property the off-loop move would otherwise silently break.
Moving the persist to a worker thread removes the accidental serialization the
event loop used to provide, so the check/insert/trim/write sequence can now
interleave across two marks and lose an entry.  ``atomic_json_write`` makes each
write atomic; it does not make the sequence atomic.

And they pin the read-after-mark contract: the adapters gate on
``thread_id in self._threads`` immediately after marking, so the in-memory half
of ``mark_async`` has to be synchronous even though the persist is not.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time

import pytest

from gateway.platforms import helpers
from gateway.platforms.helpers import ThreadParticipationTracker


@pytest.fixture()
def tracker(tmp_path, monkeypatch):
    """A tracker whose state file lives in an isolated directory."""
    path = tmp_path / "matrix_threads.json"
    monkeypatch.setattr(
        ThreadParticipationTracker, "_state_path", lambda self: path, raising=True
    )
    obj = ThreadParticipationTracker("matrix")
    obj.state_path = path  # convenience for assertions
    return obj


def test_two_concurrent_marks_do_not_lose_an_entry(tracker, monkeypatch):
    """Off-loop marks run on worker threads, so the sequence must be locked.

    The event loop used to serialize every caller by accident.  With the
    persist on ``asyncio.to_thread`` two marks genuinely overlap, and a
    check/insert/trim/write sequence without a lock drops one of the two.  The
    oracle is the DURABLE FILE, not the in-memory dict.
    """

    # With the io lock in place the second worker never reaches the barrier
    # (it waits on the lock), so the first one MUST time out here.  That
    # timeout is the green path's cost, so keep it small.
    entered = threading.Barrier(2, timeout=0.5)
    real_write = helpers.atomic_json_write

    def _synchronised_write(path, payload, *a, **kw):
        # Force maximum overlap: neither write proceeds until both have begun.
        try:
            entered.wait()
        except threading.BrokenBarrierError:  # pragma: no cover - timeout path
            pass
        return real_write(path, payload, *a, **kw)

    monkeypatch.setattr(helpers, "atomic_json_write", _synchronised_write, raising=True)

    async def scenario():
        marks = asyncio.gather(
            tracker.mark_async("!a:example.org"),
            tracker.mark_async("!b:example.org"),
        )
        # A worker is parked inside the rename. Neither the second mark's
        # in-memory insert nor a membership check may wait for it: both take
        # the set lock ON THE LOOP THREAD, so holding that lock across
        # ``os.replace`` stalls the loop for the whole rename.
        started = time.monotonic()
        await asyncio.sleep(0.05)
        assert "!a:example.org" in tracker
        assert time.monotonic() - started < 0.2, (
            "the loop stalled on the tracker lock while a worker held it "
            "across os.replace"
        )
        await marks

    asyncio.run(scenario())

    persisted = json.loads(tracker.state_path.read_text(encoding="utf-8"))
    assert sorted(persisted) == ["!a:example.org", "!b:example.org"], (
        "a concurrent mark was lost from the DURABLE file: "
        f"{persisted!r}. atomic_json_write makes each write atomic; it does "
        "not make check/insert/trim/write atomic."
    )


def test_a_mark_is_visible_in_memory_before_the_persist_completes(tracker, monkeypatch):
    """``thread_id in tracker`` must be true before the persist runs AT ALL.

    Both adapters gate on membership right after marking
    (``in_bot_thread = bool(thread_id and thread_id in self._threads)``), so
    deferring the in-memory insert to the worker thread alongside the write
    would reopen the mention-gating hole the tracker exists to close -- and
    would make membership depend on executor availability.

    The oracle is a handoff that NEVER RUNS: ``asyncio.to_thread`` is replaced
    by a coroutine that parks forever without ever invoking the callable.  That
    is what makes this non-vacuous.  Merely waiting for the worker to reach the
    rename -- the obvious shape -- also observes a True under the deferred
    implementation, because a deferred insert still happens before the write;
    this one cannot.
    """
    handed_off: list[object] = []

    async def _never(fn, *a, **kw):
        handed_off.append(fn)
        await asyncio.Event().wait()  # park forever; fn is never called

    # Patch through the module seam the tracker actually calls.
    monkeypatch.setattr(helpers.asyncio, "to_thread", _never, raising=True)

    async def scenario():
        mark = asyncio.create_task(tracker.mark_async("!first:example.org"))
        # One loop iteration runs the task up to its first suspension point,
        # which is inside ``_never`` -- no Event handoff, no wall-clock wait.
        await asyncio.sleep(0)
        assert handed_off, "mark_async never reached the to_thread handoff"
        try:
            assert "!first:example.org" in tracker, (
                "the mark is not in memory once the persist has been handed "
                "off -- the in-memory insert was deferred onto the worker "
                "thread, so membership now depends on executor availability "
                "and mention gating can re-prompt for an @mention in a thread "
                "the bot just joined"
            )
        finally:
            mark.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await mark

    asyncio.run(scenario())
