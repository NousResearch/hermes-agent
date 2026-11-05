"""The accounting queue behind ``UsageRepository.queue_usage`` and ``flush_usage`` on SQLite: reads that flush first, a flush that times out while the writer is busy, coalescing of adjacent same-route deltas, and the stopped-writer fallback. A sibling connection holding the write lock keeps the writer busy; the holder's internals only stop the writer and count the applier's calls.
"""

import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from pathlib import Path

import pytest

from persistence import PermanentStoreError
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from persistence.sqlite import SQLiteBackend, _holder
from persistence.sqlite.conversation import usage as sqlite_usage
from tests.persistence_sqlite.conftest import store_for

SESSION = "acct-1"
PROVIDER, BASE_URL = "prov", "https://api.example.test"
# Queued while the writer is busy on the first delta; the next batch coalesces only adjacent equal routes.
FOLLOWING = ((2, "model-a"), (3, "model-a"), (4, "model-b"), (5, "model-a"))
APPLIED = [(SESSION, 1, "model-a"), (SESSION, 5, "model-a"), (SESSION, 4, "model-b"), (SESSION, 5, "model-a")]


class _Backend:
    """The backend's repositories over one profile store."""

    write_refusal = PermanentStoreError

    def __init__(self, store, path: Path, owned=None):
        self.store, self.path, self.owned = store, path, owned

    def create(self, session_id):
        self.store.sessions.create(NewSession(session_id=session_id, source="cli", model="model-a"))

    def queue(self, session_id, input_tokens, model="model-a"):
        self.store.usage.queue_usage(session_id, UsageDelta(
            tokens=TokenCounts(input_tokens=input_tokens, output_tokens=1), api_call_count=1,
            route=BillingRoute(model=model, billing_provider=PROVIDER, billing_base_url=BASE_URL), source="cli"))

    def flush(self, timeout=5.0):
        return self.store.usage.flush_usage(timeout)

    def totals(self, session_id):
        record = self.store.sessions.get(session_id)
        return record.input_tokens, record.output_tokens, record.api_call_count

    def route(self, session_id):
        usage = self.store.usage.recent_route(session_id)
        route = usage.route
        return route.model, route.billing_provider, route.billing_base_url, route.billing_mode, usage.api_call_count

    def stop_writer(self):
        self.store.holder()._stop_token_writer()

    def count_applies(self, monkeypatch, calls, entered):
        apply = _holder._token_delta_applier

        def counting(holder, session_id, kwargs):
            calls.append((session_id, kwargs["input_tokens"], kwargs["model"]))
            entered.set()
            return apply(holder, session_id, kwargs)

        monkeypatch.setattr(_holder, "_token_delta_applier", counting)

    def reader(self):
        backend = SQLiteBackend({self.store.scope.hermes_home_key: self.path}, read_only=True)
        return _Backend(backend.open(self.store.scope), self.path, owned=backend)

    def close(self):
        if self.owned is not None:
            self.owned.close()


@pytest.fixture(params=("backend",))
def side(backend, scope_a):
    subject = _Backend(backend.open(scope_a), store_for(scope_a))
    subject.create(SESSION)
    yield subject
    subject.close()


def _raw_totals(path: Path, session_id: str):
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as conn:
        return conn.execute(
            "SELECT input_tokens, output_tokens, api_call_count FROM sessions WHERE id = ?", (session_id,)).fetchone()


def _hold_write_lock(path, hold_s, started):
    conn = sqlite3.connect(str(path), timeout=1.0, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        started.set()
        time.sleep(hold_s)  # health: allow HX013 -- the sibling's held write lock is the busy writer under test
        conn.execute("COMMIT")
    finally:
        conn.close()


@contextmanager
def _busy_writer(side, monkeypatch, calls):
    """The writer applying the first delta, stuck behind a sibling's write lock until the block exits."""
    entered, started = threading.Event(), threading.Event()
    side.count_applies(monkeypatch, calls, entered)
    sibling = threading.Thread(target=_hold_write_lock, args=(side.path, 1.5, started))
    sibling.start()
    try:
        assert started.wait(5.0)
        side.queue(SESSION, 1)
        assert entered.wait(5.0)
        yield
    finally:
        sibling.join(10.0)


def test_the_holder_applies_deltas_through_the_usage_repository():
    assert _holder._token_delta_applier is sqlite_usage._apply_usage_delta


def test_flush_times_out_while_the_writer_is_busy_and_never_raises(side, monkeypatch):
    calls = []
    with _busy_writer(side, monkeypatch, calls):
        assert side.flush(timeout=0.2) is False
    assert side.flush() is True
    assert calls == [(SESSION, 1, "model-a")]
    assert _raw_totals(side.path, SESSION) == (1, 1, 1)


def test_get_flushes_queued_deltas_first(side, monkeypatch):
    calls, result = [], {}
    with _busy_writer(side, monkeypatch, calls):
        reader = threading.Thread(target=lambda: result.update(totals=side.totals(SESSION)))
        reader.start()
        reader.join(0.3)
        assert reader.is_alive(), "the read returned while a queued delta was unapplied"
    reader.join(10.0)
    assert result["totals"] == (1, 1, 1)


def test_recent_route_flushes_queued_deltas_first(side, monkeypatch):
    calls, result = [], {}
    with _busy_writer(side, monkeypatch, calls):
        side.queue(SESSION, 2, "model-b")
        reader = threading.Thread(target=lambda: result.update(route=side.route(SESSION)))
        reader.start()
        reader.join(0.3)
        assert reader.is_alive(), "the read returned while a queued delta was unapplied"
    reader.join(10.0)
    assert result["route"] == ("model-b", PROVIDER, BASE_URL, None, 1)


def test_adjacent_same_route_deltas_coalesce(side, monkeypatch):
    calls = []
    with _busy_writer(side, monkeypatch, calls):
        for tokens, model in FOLLOWING:
            side.queue(SESSION, tokens, model)
    assert side.flush() is True
    assert calls == APPLIED
    assert side.totals(SESSION) == (15, 5, 5)
    assert side.route(SESSION) == ("model-a", PROVIDER, BASE_URL, None, 4)


def test_a_stopped_writer_applies_the_delta_synchronously(side):
    side.queue(SESSION, 7)
    assert side.flush() is True
    side.stop_writer()
    side.queue(SESSION, 100)
    assert _raw_totals(side.path, SESSION) == (107, 2, 2)


def test_the_stopped_writer_fallback_raises_where_the_writer_only_logs(side):
    reader = side.reader()
    try:
        reader.queue(SESSION, 7)
        assert reader.flush() is True
        assert _raw_totals(side.path, SESSION) == (0, 0, 0)
        reader.stop_writer()
        with pytest.raises(side.write_refusal, match="readonly database"):
            reader.queue(SESSION, 7)
    finally:
        reader.close()
    assert _raw_totals(side.path, SESSION) == (0, 0, 0)
