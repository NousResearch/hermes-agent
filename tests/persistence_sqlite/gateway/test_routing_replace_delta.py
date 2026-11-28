"""``replace_routing_entries`` writes only the delta: a no-change call writes zero rows (the old code deleted
the scope's whole index and re-inserted it on every call, holding the state.db writer mutex 10-12s on a
saturated disk). Changed rows land identically to the old full rewrite; removed or newly-retired rows are
deleted; other scopes are untouched; retired keys are still returned.
"""

from __future__ import annotations

import json
import time

import pytest

NOW = 1_800_000_000.0
TS = 1699999980.0
DROP = '{"session_id":"drop"}'
KEEP = '{"session_id":"keep"}'


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> float:
        return self.now


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> Clock:
    fixed = Clock()
    monkeypatch.setattr(time, "time", fixed)
    return fixed


def _seed(store, rows) -> None:
    store.holder()._write_sql("INSERT INTO gateway_routing VALUES (?, ?, ?, ?)", rows, many=True)


def _rows(store) -> list[tuple]:
    return [tuple(row) for row in store.holder()._read_all("SELECT * FROM gateway_routing ORDER BY scope, session_key")]


def test_replace_no_change_writes_zero_rows(backend, scope_a):
    # A rewrite with identical content must leave every row byte for byte (payload AND timestamp):
    # the old full rewrite re-stamped updated_at on all rows, so unchanged timestamps prove zero writes.
    store = backend.open(scope_a)
    _seed(store, [("one", "agent:one:a:x", '{"session_id":"x"}', TS),
                  ("one", "agent:one:a:y", '{"session_id":"y"}', TS)])
    before = _rows(store)
    assert store.routing.replace_routing_entries(
        {"agent:one:a:x": '{"session_id":"x"}', "agent:one:a:y": '{"session_id":"y"}'},
        routing_scope="one") == set()
    assert _rows(store) == before


def test_replace_mixed_change_lands_identically(backend, scope_a):
    store = backend.open(scope_a)
    _seed(store, [("one", "agent:one:a:a", '{"session_id":"a","v":1}', TS),
                  ("one", "agent:one:a:b", '{"session_id":"b"}', TS),
                  ("one", "agent:one:a:c", '{"session_id":"c"}', TS)])
    assert store.routing.replace_routing_entries(
        {"agent:one:a:a": '{"session_id":"a","v":2}', "agent:one:a:b": '{"session_id":"b"}',
         "agent:one:a:d": '{"session_id":"d"}'},
        routing_scope="one") == set()
    assert _rows(store) == [("one", "agent:one:a:a", '{"session_id":"a","v":2}', NOW),
                            ("one", "agent:one:a:b", '{"session_id":"b"}', TS),
                            ("one", "agent:one:a:d", '{"session_id":"d"}', NOW)]


def test_replace_returns_retired_keys_and_drops_them(backend, scope_a):
    store = backend.open(scope_a)
    _seed(store, [("one", "agent:one:a:drop", DROP, TS), ("one", "agent:one:a:keep", KEEP, TS)])
    store.routing.fence_retired_routes({"drop"}, namespace="one")
    assert store.routing.replace_routing_entries(
        {"agent:one:a:drop": DROP, "agent:one:a:keep": KEEP}, routing_scope="one") == {"agent:one:a:drop"}
    assert _rows(store) == [("one", "agent:one:a:keep", KEEP, TS)]


def test_replace_retired_row_already_in_table_is_dropped(backend, scope_a):
    # A row inserted before its route was retired must go: the old code deleted it with the whole
    # scope and skipped re-inserting it, so the delta must delete it too.
    store = backend.open(scope_a)
    store.routing.upsert_routing_entry("agent:one:a:drop", DROP, routing_scope="one")
    store.routing.fence_retired_routes({"drop"}, namespace="one")
    # The fence already deleted the row; re-seed it to simulate a row that survived (raced the fence).
    _seed(store, [("one", "agent:one:a:drop", DROP, TS)])
    assert store.routing.replace_routing_entries(
        {"agent:one:a:drop": DROP}, routing_scope="one") == {"agent:one:a:drop"}
    assert _rows(store) == []


def test_replace_empty_entries_clears_only_its_scope(backend, scope_a):
    store = backend.open(scope_a)
    _seed(store, [("one", "agent:one:a:x", '{"session_id":"x"}', TS),
                  ("two", "agent:one:b:y", '{"session_id":"y"}', TS)])
    assert store.routing.replace_routing_entries({}, routing_scope="one") == set()
    assert _rows(store) == [("two", "agent:one:b:y", '{"session_id":"y"}', TS)]


def test_replace_drops_empty_keys_and_entries(backend, scope_a):
    store = backend.open(scope_a)
    assert store.routing.replace_routing_entries(
        {"": '{"session_id":"no"}', "agent:one:a:e": "", "agent:one:a:ok": '{"session_id":"ok"}'},
        routing_scope="one") == set()
    assert store.routing.load_routing_entries(routing_scope="one") == {"agent:one:a:ok": '{"session_id":"ok"}'}


def test_replace_first_write_from_empty(backend, scope_a):
    store = backend.open(scope_a)
    assert store.routing.replace_routing_entries(
        {"agent:one:a:x": '{"session_id":"x"}'}, routing_scope="one") == set()
    assert _rows(store) == [("one", "agent:one:a:x", '{"session_id":"x"}', NOW)]
    assert json.loads(store.routing.load_routing_entries(routing_scope="one")["agent:one:a:x"]) == {"session_id": "x"}
