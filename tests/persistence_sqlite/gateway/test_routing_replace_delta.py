"""A whole-index routing save writes only what changed.

``gateway_routing`` is re-sent in full on every structural session change, and the save runs inside the
store's single in-process writer mutex. Rewriting every route (thousands of rows, megabytes of JSON) held
that mutex for tens of seconds once another thread was CPU-bound — CPython's sqlite3 releases the GIL per
``sqlite3_step`` — and every other writer past its patience died with ``WriterMutexTimeout`` (#156978).
So a save now deletes only the keys that left the index, inserts/updates only the keys whose payload
changed, and judges the retirement predicate for the whole index in one read. These cases pin that
contract, including the case a delta must not break: a fenced route is still dropped even when its
payload is unchanged.
"""

from __future__ import annotations

import json

TS = 1699999980.0


def _entry(session_id: str) -> str:
    return json.dumps({"session_id": session_id})


def _seed(store, scope: str, count: int) -> dict[str, tuple[str, float]]:
    """Seed *count* routes through the public save, then read back payload + timestamp per key."""
    store.routing.replace_routing_entries(
        {f"agent:main:cron:job:{i}": _entry(f"s{i}") for i in range(count)}, routing_scope=scope)
    return _stored(store, scope)


def _stored(store, scope: str) -> dict[str, tuple[str, float]]:
    return {row["session_key"]: (row["entry_json"], row["updated_at"]) for row in store.holder()._read_all(
        "SELECT session_key, entry_json, updated_at FROM gateway_routing WHERE scope = ?", (scope,))}


def test_an_unchanged_save_rewrites_no_row(backend, scope_a):
    store = backend.open(scope_a)
    before = _seed(store, "work", 200)
    store.routing.replace_routing_entries(
        {key: payload for key, (payload, _) in before.items()}, routing_scope="work")
    assert _stored(store, "work") == before


def test_a_changed_entry_rewrites_only_its_own_row(backend, scope_a):
    store = backend.open(scope_a)
    before = _seed(store, "work", 200)
    changed = "agent:main:cron:job:7"
    store.routing.replace_routing_entries(
        {key: (_entry("s7-moved") if key == changed else payload) for key, (payload, _) in before.items()},
        routing_scope="work")
    after = _stored(store, "work")
    assert set(after) == set(before)
    assert after[changed][0] == _entry("s7-moved")
    assert after[changed][1] >= before[changed][1]
    # Every other row keeps payload and timestamp: they were never written.
    assert {key: row for key, row in after.items() if key != changed} == {
        key: row for key, row in before.items() if key != changed}


def test_a_key_leaving_the_index_is_deleted_and_reported(backend, scope_a):
    store = backend.open(scope_a)
    before = _seed(store, "work", 3)
    gone = "agent:main:cron:job:1"
    kept = {key: payload for key, (payload, _) in before.items() if key != gone}
    assert store.routing.replace_routing_entries(kept, routing_scope="work") == set()
    assert set(_stored(store, "work")) == set(kept)


def test_a_fence_drops_a_route_whose_payload_never_changed(backend, scope_a):
    # The delta must not turn "unchanged" into "unchecked": a fence landing between two saves still
    # retires the route on the next save, even though the payload the caller re-sends is identical.
    store = backend.open(scope_a)
    before = _seed(store, "work", 3)
    store.routing.fence_retired_routes(["s2"], namespace="main")
    skipped = store.routing.replace_routing_entries(
        {key: payload for key, (payload, _) in before.items()}, routing_scope="work")
    assert skipped == {"agent:main:cron:job:2"}
    assert "agent:main:cron:job:2" not in _stored(store, "work")
    assert store.routing.route_retired("agent:main:cron:job:2", _entry("s2")) is True
