"""The search index administration verbs on SQLite.

The first section drives the three administration verbs and observes through ``search_messages``; race cases run two contenders in two threads released together, so they contend for the store's cross-process rebuild authority. The second section writes rows through the backend into one session: a rebuild denied while another holder has the rebuild authority, recovery deferred by a damaged index, and read-only access.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pytest

import hermes_state_common
from hermes_state_common import fts_rebuild_admission
from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.search import SearchIndexRebuildStatus
from persistence.conversation.sessions import NewSession
from persistence.sqlite import SQLiteBackend
from persistence.sqlite._holder import StoreHolder
from tests.persistence_sqlite.conftest import store_for

EPOCH = 1767225600.0  # 2026-01-01T00:00:00Z


def _seed(backend, scope, session_id: str = "s-idx-1") -> set:
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id, "cli", model="idx-model"))
    append = store.messages.append
    return {
        append(session_id, NewMessage(role="user", content="quick brown fox", timestamp=EPOCH + 1)),
        append(session_id, NewMessage(role="assistant", content="quick reply", timestamp=EPOCH + 2)),
        append(session_id, NewMessage(role="tool", content="quick tool body", tool_name="web_search", timestamp=EPOCH + 3)),
    }


def _match_set(backend, scope, text: str = "quick") -> set:
    return {hit.message_id for hit in backend.open(scope).search.search_messages(text, limit=50)}


def _seed_deferred_rebuild(scope, *, total: int, indexed: int) -> None:
    """Plant the deferred-rebuild progress marker the store's chunked backfill keeps in ``state_meta``."""
    with sqlite3.connect(store_for(scope)) as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)",
            [("fts_rebuild_high_water", str(total)), ("fts_rebuild_progress", str(indexed))],
        )


def _race(*calls: Callable[[], Any]) -> list:
    """Run each call in its own thread, released together by a barrier; results in call order."""
    gate = threading.Barrier(len(calls))

    def run(call: Callable[[], Any]) -> Any:
        gate.wait()
        return call()

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return [future.result() for future in [pool.submit(run, call) for call in calls]]


# ── rebuild ─────────────────────────────────────────────────────────────────


def test_rebuild_reports_a_count_and_keeps_the_match_set(backend, scope_a):
    ids = _seed(backend, scope_a)
    before = _match_set(backend, scope_a)
    assert before == ids
    rebuilt = backend.open(scope_a).search.rebuild_search_index()
    assert isinstance(rebuilt, int) and rebuilt >= 0
    assert _match_set(backend, scope_a) == before


def test_rebuild_is_repeatable(backend, scope_a):
    ids = _seed(backend, scope_a)
    search = backend.open(scope_a).search
    first = search.rebuild_search_index()
    second = search.rebuild_search_index()
    assert second == first
    assert _match_set(backend, scope_a) == ids


def test_rebuild_two_contenders_each_report_and_the_index_stays_consistent(backend, scope_a):
    ids = _seed(backend, scope_a)
    outcomes = _race(
        lambda: backend.open(scope_a).search.rebuild_search_index(),
        lambda: backend.open(scope_a).search.rebuild_search_index(),
    )
    uncontended = backend.open(scope_a).search.rebuild_search_index()
    # Two contenders never rebuild at once: each rebuilt after the other released (the uncontended count) or was
    # denied within the bounded wait (0).
    assert all(outcome in (0, uncontended) for outcome in outcomes), (outcomes, uncontended)
    assert _match_set(backend, scope_a) == ids
    assert backend.open(scope_a).search.search_index_rebuild_status() is None


def test_rebuild_leaves_other_scopes_untouched(backend, scope_a, scope_b):
    ids_a = _seed(backend, scope_a)
    ids_b = _seed(backend, scope_b, "s-idx-b")
    backend.open(scope_a).search.rebuild_search_index()
    assert _match_set(backend, scope_b) == ids_b
    assert _match_set(backend, scope_a) == ids_a
    assert backend.open(scope_b).search.search_index_rebuild_status() is None


# ── status ──────────────────────────────────────────────────────────────────


def test_status_is_none_while_nothing_is_deferred(backend, scope_a):
    search = backend.open(scope_a).search
    assert search.search_index_rebuild_status() is None
    _seed(backend, scope_a)
    search.rebuild_search_index()
    assert search.search_index_rebuild_status() is None


def test_status_reports_a_pending_deferred_rebuild_and_survives_a_sync_rebuild(backend, scope_a, scope_b):
    ids = _seed(backend, scope_a)
    search = backend.open(scope_a).search
    _seed_deferred_rebuild(scope_a, total=10, indexed=4)
    status = search.search_index_rebuild_status()
    assert status == SearchIndexRebuildStatus(total=10, indexed=4)
    assert status.percent == 40
    assert backend.open(scope_b).search.search_index_rebuild_status() is None
    search.rebuild_search_index()
    assert search.search_index_rebuild_status() == status
    assert _match_set(backend, scope_a) == ids
    _seed_deferred_rebuild(scope_a, total=0, indexed=0)
    assert search.search_index_rebuild_status() is None


def test_status_reads_a_negative_progress_marker_as_stored(backend, scope_a):
    _seed(backend, scope_a)
    _seed_deferred_rebuild(scope_a, total=10, indexed=-3)
    assert backend.open(scope_a).search.search_index_rebuild_status() == SearchIndexRebuildStatus(total=10, indexed=-3)


def test_status_record_carries_the_legacy_projection():
    status = SearchIndexRebuildStatus(total=10, indexed=4)
    assert (status.total, status.indexed, status.percent) == (10, 4, 40)
    assert SearchIndexRebuildStatus(total=3, indexed=7).percent == 100
    assert SearchIndexRebuildStatus(total=3, indexed=0).percent == 0
    with pytest.raises(ValueError):
        SearchIndexRebuildStatus(total=0, indexed=0)


# ── deferred recovery retry ─────────────────────────────────────────────────


def test_retry_is_false_while_nothing_is_deferred(backend, scope_a):
    search = backend.open(scope_a).search
    assert search.retry_deferred_search_recovery() is False
    ids = _seed(backend, scope_a)
    search.rebuild_search_index()
    assert search.retry_deferred_search_recovery() is False
    assert _match_set(backend, scope_a) == ids


def test_retry_two_contenders_never_raise(backend, scope_a):
    _seed(backend, scope_a)
    outcomes = _race(
        lambda: backend.open(scope_a).search.retry_deferred_search_recovery(),
        lambda: backend.open(scope_a).search.retry_deferred_search_recovery(),
    )
    assert outcomes == [False, False]


# ── rebuild authority, recovery and read-only access ────────────────────────

BRIDGE = SessionId("br-1")


@pytest.fixture
def no_foreign_holders(monkeypatch):
    """No other process holds the store: the premise of an admitted rebuild (a hermetic run's process scan fails
    closed)."""
    monkeypatch.setattr(StoreHolder, "_foreign_state_db_holders", lambda self: [])


def _write(store, session_id: str, message: NewMessage) -> int:
    """Append one row through the backend, creating its session first."""
    if store.sessions.get(session_id) is None:
        store.sessions.create(NewSession(session_id, "cli"))
    return int(store.messages.append(session_id, message))


def _bridge_seed(store, rows: int = 3) -> int:
    last = 0
    for i in range(rows):
        last = _write(store, BRIDGE, NewMessage(role="user", content=f"quick brown fox {i}", timestamp=EPOCH + i))
    return last


def test_bridge_rebuild_denied_by_a_live_holder_returns_zero_and_touches_nothing(backend, scope_a, monkeypatch,
                                                                                  no_foreign_holders):
    monkeypatch.setattr(hermes_state_common, "_FTS_REBUILD_LOCK_TIMEOUT_SECONDS", 0.3)
    store = backend.open(scope_a)
    _bridge_seed(store)
    with fts_rebuild_admission(store_for(scope_a)) as holder:
        assert holder is True
        assert store.search.rebuild_search_index() == 0
    assert store.search.rebuild_search_index() >= 1


def test_bridge_retry_completes_a_deferred_recovery(backend, scope_a, no_foreign_holders):
    store = backend.open(scope_a)
    _bridge_seed(store)
    with sqlite3.connect(store_for(scope_a)) as raw:
        raw.execute("UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'")
    # The backend's own write fails open: the row lands and the store marks its index stale.
    needle = store.messages.append(BRIDGE, NewMessage(role="user", content="healed needle", timestamp=EPOCH + 60))
    assert store.search.retry_deferred_search_recovery() is True
    assert {hit.message_id for hit in store.search.search_messages("needle", limit=10)} == {needle}
    assert store.search.retry_deferred_search_recovery() is False


def test_bridge_read_only_store_rebuilds_nothing_serves_status_and_answers_false(backend, scope_a):
    _bridge_seed(backend.open(scope_a))
    _seed_deferred_rebuild(scope_a, total=8, indexed=2)
    reader = SQLiteBackend({scope_a.hermes_home_key: store_for(scope_a)}, read_only=True)
    try:
        search = reader.open(scope_a).search
        assert search.rebuild_search_index() == 0
        assert search.search_index_rebuild_status() == SearchIndexRebuildStatus(total=8, indexed=2)
        assert search.retry_deferred_search_recovery() is False
    finally:
        reader.close()
