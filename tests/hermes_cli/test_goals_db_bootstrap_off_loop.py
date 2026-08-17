"""SessionDB bootstrap must never run schema init on an event-loop thread.

Coatue field report (2026-08-14): after an unclean shutdown plus a
double-instance startup race, the v25 schema migration inside
``SessionDB.__init__`` blocked the gateway's event-loop thread (reached via
``_post_turn_goal_continuation`` → ``GoalManager()`` → ``load_goal`` →
``_get_session_db``). The loop-liveness watchdog missed 3 probes and killed
the process with exit 75; the supervisor restarted into the same state —
an unbounded crash loop.

Contract fixed here, at the shared boundary (goals.py ``_get_session_db``,
which the heartbeat module reuses):

- Called on a thread with a RUNNING event loop and no cached DB, it must
  NOT construct SessionDB inline. It returns None immediately (callers
  already degrade gracefully on None) and populates the cache from a
  background thread.
- Called on a plain worker thread, it constructs inline as before.
- Once cached, every thread gets the cached instance.
"""

import asyncio
import threading
import time

import pytest

import hermes_cli.goals as goals


class _RecordingDB:
    """Stands in for SessionDB; records which thread constructed it."""

    constructed_on: list = []

    def __init__(self):
        _RecordingDB.constructed_on.append(threading.get_ident())

    def get_meta(self, key):
        return None


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch):
    _RecordingDB.constructed_on = []
    monkeypatch.setattr(goals, "_DB_CACHE", {})
    yield


def _patch_sessiondb(monkeypatch):
    import hermes_state

    monkeypatch.setattr(hermes_state, "SessionDB", _RecordingDB)


def test_loop_thread_cache_miss_returns_none_without_constructing(monkeypatch):
    """On the event-loop thread with a cold cache: no inline construction,
    immediate None, background population."""
    _patch_sessiondb(monkeypatch)
    loop_thread_id = None
    inline_result = "UNSET"

    async def main():
        nonlocal loop_thread_id, inline_result
        loop_thread_id = threading.get_ident()
        inline_result = goals._get_session_db()

    asyncio.run(main())

    # The loop-thread call must not have blocked on construction.
    assert inline_result is None
    # Background thread eventually populates the cache, OFF the loop thread.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not goals._DB_CACHE:
        time.sleep(0.02)
    assert goals._DB_CACHE, "background bootstrap never populated the cache"
    assert _RecordingDB.constructed_on, "SessionDB never constructed"
    assert all(t != loop_thread_id for t in _RecordingDB.constructed_on), (
        "SessionDB was constructed on the event-loop thread"
    )


def test_loop_thread_cache_hit_returns_cached_instance(monkeypatch):
    """A warm cache is returned directly even on the loop thread."""
    _patch_sessiondb(monkeypatch)
    sentinel = _RecordingDB()
    from hermes_constants import get_hermes_home

    goals._DB_CACHE[str(get_hermes_home())] = sentinel
    result = "UNSET"

    async def main():
        nonlocal result
        result = goals._get_session_db()

    asyncio.run(main())
    assert result is sentinel


def test_worker_thread_constructs_inline(monkeypatch):
    """No running loop on the calling thread → construct inline, return it."""
    _patch_sessiondb(monkeypatch)
    db = goals._get_session_db()
    assert db is not None
    assert isinstance(db, _RecordingDB)
    assert _RecordingDB.constructed_on == [threading.get_ident()]


def test_slow_construction_does_not_block_the_loop(monkeypatch):
    """A SessionDB whose init blocks (locked-DB migration) must not stall
    the event loop past the watchdog probe window."""
    import hermes_state

    class _BlockingDB:
        def __init__(self):
            time.sleep(2.0)  # simulated contended migration

        def get_meta(self, key):
            return None

    monkeypatch.setattr(hermes_state, "SessionDB", _BlockingDB)
    elapsed = None

    async def main():
        nonlocal elapsed
        t0 = time.monotonic()
        goals._get_session_db()
        elapsed = time.monotonic() - t0

    asyncio.run(main())
    assert elapsed is not None and elapsed < 0.5, (
        f"loop-thread call blocked for {elapsed:.2f}s — watchdog territory"
    )
