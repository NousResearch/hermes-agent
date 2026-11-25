"""Dropping the routes of retired sessions (``fence_retired_routes``) removes only the rows whose key's namespace is
fenced for the id their entry names, in every routing scope, and leaves every other row byte for byte: its payload
(well-formed or not, even empty) and its timestamp. Routes published after the fence write survive; a later save
of a fenced route is skipped."""

from __future__ import annotations

import json

import pytest

TS = 1699999980.0
DROP = '{"session_id":"drop"}'


def _seed(store, rows) -> None:
    store.holder()._write_sql("INSERT INTO gateway_routing VALUES (?, ?, ?, ?)", rows, many=True)


def _rows(store) -> list[tuple]:
    return [tuple(row) for row in store.holder()._read_all("SELECT * FROM gateway_routing ORDER BY scope, session_key")]


@pytest.mark.parametrize("payload", ['{"session_id":"keep"}', "[]", "0", "null", '"text"', "{"])
def test_delete_keeps_survivor_payload_and_timestamp(backend, scope_a, payload):
    store = backend.open(scope_a)
    _seed(store, [("one", "agent:one:a:drop", DROP, TS), ("one", "agent:one:a:keep", payload, TS),
                  ("one", "agent:two:a:drop", DROP, TS), ("two", "agent:one:b:drop", DROP, TS)])
    store.routing.fence_retired_routes({"drop"}, namespace="one")
    assert _rows(store) == [("one", "agent:one:a:keep", payload, TS), ("one", "agent:two:a:drop", DROP, TS)]


def test_delete_keeps_entries_published_after_the_fence(backend, scope_a):
    store = backend.open(scope_a)
    store.routing.upsert_routing_entry("agent:one:a:drop", DROP, routing_scope="one")
    store.routing.fence_retired_routes(["drop"], namespace="one")
    store.routing.upsert_routing_entry("agent:one:a:new", json.dumps({"session_id": "new"}), routing_scope="one")
    store.routing.upsert_routing_entry("agent:one:a:drop", DROP, routing_scope="one")
    assert store.routing.load_routing_entries(routing_scope="one") == {"agent:one:a:new": '{"session_id": "new"}'}


def test_delete_keeps_an_empty_survivor(backend, scope_a):
    # An empty entry names no session (the legacy loader reads it as no target), so the fence keeps it as stored.
    store = backend.open(scope_a)
    _seed(store, [("one", "agent:one:a:drop", DROP, TS), ("one", "agent:one:a:keep", "", TS)])
    store.routing.fence_retired_routes(["drop"], namespace="one")
    assert _rows(store) == [("one", "agent:one:a:keep", "", TS)]
