"""Regression test for 'NoneType has no attribute execute' under concurrent close.

Bug: a shared SessionDB instance (the gateway's ``self._session_db._db``,
fan-out to N thread-pool workers) had ``self._conn = None`` after
``close()`` while another worker was mid-``_execute_write``. The retry loop
acquired ``self._lock``, did NOT re-check ``_conn``, and crashed with
``AttributeError: 'NoneType' object has no attribute 'execute'`` — fired
6119+ times across all profiles since 2026-06-30.

The fix gates ``_execute_write`` on a ``_closed`` flag read UNDER the lock,
plus makes ``close()`` idempotent so simultaneous shutdown + worker writes
cleanly no-op instead of raising.

Sibling gap: the same race blew up on the READ path too — every read method
called ``self._conn.execute()`` directly without a ``_closed`` check. The
``_execute_read`` helper now guards every read site with the same
lock-and-flag pattern as ``_execute_write``. This file also covers that
path with one hammer test (8 readers + 1 closer, expect zero
``'NoneType' object has no attribute 'execute'`` during the window).

Ponytail: one regression test per race axis; no mock suites.
"""

import threading
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    sdb = SessionDB(db_path=tmp_path / "state.db")
    sdb.create_session(
        session_id="sess-race",
        source="test",
        model="m",
        system_prompt="x",
    )
    yield sdb
    sdb.close()


def test_close_during_concurrent_writers_does_not_crash(db):
    """Hammer append_message from 8 threads while one calls close()."""
    errors = []
    counts = {"ok": 0, "fail": 0}
    stop = threading.Event()

    def writer(idx):
        while not stop.is_set():
            try:
                db.append_message(
                    session_id="sess-race",
                    role="user",
                    content=f"msg-{idx}-{time.time()}",
                )
                counts["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["fail"] += 1
                errors.append(repr(exc))
                if counts["fail"] >= 5:
                    return

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()

    # Let writers warm up, then close mid-flight.
    time.sleep(0.1)
    db.close()

    # Keep writers running a bit so they hit the closed flag — must no-op.
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert counts["fail"] == 0, (
        f"close-during-write raised {counts['fail']} times: {errors[:3]}"
    )
    assert counts["ok"] > 0, "writers never landed a successful append"


def test_close_is_idempotent(db):
    """Calling close() repeatedly must not raise — used by gateway shutdown."""
    db.close()
    db.close()
    db.close()
    assert db._closed is True


def test_close_during_concurrent_readers_does_not_crash(db):
    """Hammer read methods from 8 threads while one calls close().

    Mirror of the writer race above for the read path. Pre-fix, a reader
    holding ``self._lock`` could still see ``self._conn`` nulled between
    the lock acquisition and the ``self._conn.execute()`` call — they hold
    the lock against writers but not against ``close()`` itself, since
    ``close()`` also acquires the same lock and nulls ``_conn`` after the
    flag flips. ``AttributeError: 'NoneType' object has no attribute
    'execute'`` was the silent bleed.

    The fix routes every read through ``_execute_read`` which checks
    ``_closed`` under the lock; the expected outcome is the same as the
    writer path — clean no-op (None / empty list), zero exceptions.
    """
    errors = []
    counts = {"ok": 0, "fail": 0}
    stop = threading.Event()

    # Cover the three highest-traffic read shapes: single-row lookup,
    # integer count (count_empty_sessions), and parameterized count
    # (message_count). All previously went through `with self._lock:
    # self._conn.execute(...)` directly.
    def reader(idx):
        while not stop.is_set():
            try:
                if idx % 3 == 0:
                    db.get_session("sess-race")
                elif idx % 3 == 1:
                    db.count_empty_sessions()
                else:
                    db.message_count("sess-race")
                counts["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["fail"] += 1
                errors.append(repr(exc))
                if counts["fail"] >= 5:
                    return

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()

    # Let readers warm up, then close mid-flight.
    time.sleep(0.1)
    db.close()

    # Keep readers running a bit so they hit the closed flag — must no-op.
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert counts["fail"] == 0, (
        f"close-during-read raised {counts['fail']} times: {errors[:3]}"
    )
    assert counts["ok"] > 0, "readers never landed a successful read"


def test_close_during_search_messages_does_not_crash(db):
    """Hammer search_messages (cursor-returning crash site) during close().

    Pre-fix, ``search_messages`` captured a cursor from ``_execute_read`` and
    called ``.fetchall()`` on it outside the helper. When ``_execute_read``
    short-circuited to ``None`` (closed DB), ``None.fetchall()`` raised
    ``AttributeError``. The fix moves ``.fetchall()`` into the lambda so the
    helper returns ``[]`` (not a cursor) on close — this test pins that.
    """
    # Seed searchable content so the FTS path is exercised.
    for i in range(5):
        db.append_message(
            session_id="sess-race",
            role="user",
            content=f"searchable token-{i} needle",
        )
        db.append_message(
            session_id="sess-race",
            role="assistant",
            content=f"reply token-{i} haystack",
        )

    errors = []
    counts = {"ok": 0, "fail": 0}
    stop = threading.Event()

    def searcher():
        while not stop.is_set():
            try:
                results = db.search_messages("token")
                # Must be a list, never None.
                assert isinstance(results, list)
                counts["ok"] += 1
            except Exception as exc:  # noqa: BLE001
                counts["fail"] += 1
                errors.append(repr(exc))
                if counts["fail"] >= 5:
                    return

    threads = [threading.Thread(target=searcher) for _ in range(8)]
    for t in threads:
        t.start()

    time.sleep(0.1)
    db.close()

    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=2)

    assert counts["fail"] == 0, (
        f"close-during-search raised {counts['fail']} times: {errors[:3]}"
    )
    assert counts["ok"] > 0, "searchers never landed a successful search"
