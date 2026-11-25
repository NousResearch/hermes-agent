"""Contract cases for ``StateMetaRepository``.

Every case runs against every registered backend. The read-only refusal arm lives in ``test_r4_translation_boundary.py`` (the sqlite-only typed-refusal suite), because only the SQLite backend has a read-only assembly to refuse from.
"""

from __future__ import annotations

import pytest

from persistence import PersistenceError
from persistence.housekeeping.state_meta import StateMetaRepository


def test_repository_satisfies_protocol(backend, scope_a):
    # CONTRACT: the assembled attribute presents both state_meta get and set operations.
    assert isinstance(backend.open(scope_a).state_meta, StateMetaRepository)


def test_unknown_key_reads_none(backend, scope_a):
    # CONTRACT: absence is the typed None return, never NotFound.
    store_a = backend.open(scope_a)
    assert store_a.state_meta.get("never-set") is None


def test_set_then_get_and_upsert_overwrites(backend, scope_a):
    store_a = backend.open(scope_a)
    result = store_a.state_meta.set("gate:one", "1")
    # CONTRACT: set is a None-returning command.
    assert result is None
    assert store_a.state_meta.get("gate:one") == "1"
    store_a.state_meta.set("gate:one", "2")
    # CONTRACT: a second write of the same key overwrites; the read sees the latest value.
    assert store_a.state_meta.get("gate:one") == "2"


def test_values_are_opaque_strings(backend, scope_a):
    store_a = backend.open(scope_a)
    payload = '{"state": "running", "n": 3, "path": "/tmp/%_x"}'
    store_a.state_meta.set("loop:sess-1", payload)
    # CONTRACT: the value round-trips byte-for-byte; the repository never parses or normalizes it.
    assert store_a.state_meta.get("loop:sess-1") == payload
    store_a.state_meta.set("empty", "")
    # CONTRACT: the empty string is a stored value, distinct from an unset key.
    assert store_a.state_meta.get("empty") == ""


def test_keys_are_literal_not_patterns(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.state_meta.set("k_1", "underscore")
    store_a.state_meta.set("k%1", "percent")
    # CONTRACT: keys match literally; wildcard characters in a key address only that key.
    assert store_a.state_meta.get("k_1") == "underscore"
    assert store_a.state_meta.get("k%1") == "percent"
    assert store_a.state_meta.get("k11") is None


def test_empty_key_is_a_caller_error(backend, scope_a):
    # CONTRACT: an empty key is an input-shape violation (ValueError), never a persistence error and never a stored row.
    store_a = backend.open(scope_a)
    with pytest.raises(ValueError) as get_exc:
        store_a.state_meta.get("")
    assert not isinstance(get_exc.value, PersistenceError)
    with pytest.raises(ValueError) as set_exc:
        store_a.state_meta.set("", "x")
    assert not isinstance(set_exc.value, PersistenceError)


def test_scope_isolation(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.state_meta.set("shared-key", "a-value")
    # CONTRACT: a key set in one scope is unset in another.
    assert store_b.state_meta.get("shared-key") is None
    store_b.state_meta.set("shared-key", "b-value")
    # CONTRACT: the same key holds independent values per scope; a write in one scope never touches the other.
    assert store_a.state_meta.get("shared-key") == "a-value"
    assert store_b.state_meta.get("shared-key") == "b-value"


# ── list_prefix ────────────────────────────


def test_list_prefix_returns_the_prefix_matches_in_stored_order(backend, scope_a):
    repo = backend.open(scope_a).state_meta
    for key, value in (("loop:s2", "b"), ("loop:s1", "a"), ("loops:x", "not-under-loop-colon"), ("goal:s1", "g"), ("loop:s10", "c")):
        repo.set(key, value)
    # CONTRACT: exactly the keys under the prefix, as records, in the order the keys were first stored.
    from persistence.housekeeping.state_meta import StateMetaEntry

    assert repo.list_prefix("loop:") == (
        StateMetaEntry("loop:s2", "b"),
        StateMetaEntry("loop:s1", "a"),
        StateMetaEntry("loop:s10", "c"),
    )
    # CONTRACT: the listing reflects the latest upsert, and the upserted key keeps its place.
    repo.set("loop:s1", "a2")
    assert [e.value for e in repo.list_prefix("loop:")] == ["b", "a2", "c"]
    # CONTRACT: an unmatched prefix reads empty, never an error.
    assert repo.list_prefix("nothing:") == ()


def test_list_prefix_wildcards_are_literal_ascii_case_folds_and_empty_prefix_is_a_caller_error(backend, scope_a):
    repo = backend.open(scope_a).state_meta
    repo.set("a_b", "underscore")
    repo.set("axb", "x")
    repo.set("50%", "percent")
    repo.set("500", "digits")
    repo.set("Loop:s1", "upper")
    repo.set("loop:s1", "lower")
    repo.set("\u00c9t\u00e9:1", "upper accent")
    repo.set("\u00e9t\u00e9:1", "lower accent")
    # CONTRACT: ``_`` and ``%`` in the prefix match only themselves.
    assert [e.key for e in repo.list_prefix("a_")] == ["a_b"]
    assert [e.key for e in repo.list_prefix("A_")] == ["a_b"]
    assert [e.key for e in repo.list_prefix("50%")] == ["50%"]
    # CONTRACT: ASCII letters match either case, in stored order; any other character matches only itself.
    assert [e.key for e in repo.list_prefix("loop:")] == ["Loop:s1", "loop:s1"]
    assert [e.key for e in repo.list_prefix("LOOP:")] == ["Loop:s1", "loop:s1"]
    assert [e.key for e in repo.list_prefix("\u00e9T\u00e9:")] == ["\u00e9t\u00e9:1"]
    assert [e.key for e in repo.list_prefix("\u00c9t\u00e9:")] == ["\u00c9t\u00e9:1"]
    assert repo.list_prefix("\u00c9T\u00c9:") == ()
    # CONTRACT: no enumeration; the prefix is required.
    with pytest.raises(ValueError):
        repo.list_prefix("")


def test_list_prefix_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    store_a.state_meta.set("loop:only-a", "1")
    store_b.state_meta.set("loop:only-b", "2")
    assert [e.key for e in store_a.state_meta.list_prefix("loop:")] == ["loop:only-a"]
    assert [e.key for e in store_b.state_meta.list_prefix("loop:")] == ["loop:only-b"]
