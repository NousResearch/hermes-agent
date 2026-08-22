"""MemoryStore.search_facts must not crash on FTS5 query syntax in user text.

The memory ``search`` action is exposed to the LLM, which routinely emits
queries containing FTS5 metasyntax — a stray double-quote, a ``key:value``
colon, a bare ``AND``/``OR``/``NEAR``, or a parenthesis. Passing those straight
to ``MATCH`` raised an unhandled ``sqlite3.OperationalError`` and crashed the
memory tool. search_facts now retries with each token quoted as a literal
phrase so the search degrades gracefully.
"""

import sqlite3

import pytest

from plugins.memory.holographic.store import MemoryStore, _fts5_safe_query


@pytest.fixture(autouse=True)
def _clean_shared_registry():
    """Each test starts and ends with an empty shared-connection registry."""
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()
    yield
    for entry in list(MemoryStore._shared.values()):
        try:
            entry["conn"].close()
        except sqlite3.Error:
            pass
    MemoryStore._shared.clear()


@pytest.fixture
def store(tmp_path):
    """A MemoryStore backed by a per-test database file.

    ``MemoryStore`` resolves ``db_path`` before ``sqlite3.connect`` and keys its
    process-wide shared-connection registry on the resolved path, so passing
    ``":memory:"`` yields a shared *on-disk* file rather than SQLite's in-memory
    sentinel — leaking rows between tests. Use ``tmp_path`` and close the store,
    matching ``test_holographic_store.py``.
    """
    store = MemoryStore(tmp_path / "memory_store.db")
    store.add_fact("Python is a programming language", category="tech")
    store.add_fact("Rust is memory safe and fast", category="tech")
    try:
        yield store
    finally:
        store.close()


@pytest.mark.parametrize(
    "query",
    [
        '"unterminated',
        "foo:bar",
        "AND",
        "a OR",
        "(python",
        "memory:safe",
        "C++ AND rust",
        'say "hi',
        "NEAR(",
    ],
)
def test_fts5_metasyntax_does_not_crash(store, query):
    # Must return a list, never raise sqlite3.OperationalError.
    result = store.search_facts(query)
    assert isinstance(result, list)


def test_bare_boolean_operators_no_longer_crash(store):
    # `_sanitize_fts_query` neutralises most metasyntax before MATCH (colons,
    # stray quotes, parens, NEAR), but a bare boolean operator survives it and
    # still reaches FTS5 as a syntax error — on an unguarded store,
    # `search_facts("AND")` raises `fts5: syntax error near "AND"` and
    # `search_facts("a OR")` raises `fts5: syntax error near ""`. These are the
    # cases the quoted-phrase retry still has to absorb.
    for query in ("AND", "a OR"):
        assert isinstance(store.search_facts(query), list), query


def test_normal_query_unaffected(store):
    results = store.search_facts("python")
    assert any("Python" in r["content"] for r in results)


def test_prefix_query_still_works(store):
    # A valid FTS5 prefix query must keep working (try-raw before fallback).
    results = store.search_facts("rust*")
    assert any("Rust" in r["content"] for r in results)


def test_empty_after_sanitization_returns_empty(store):
    # A query that's only quotes -> no usable tokens -> [] (no crash).
    assert store.search_facts('"') == []


def test_fts5_safe_query_helper():
    assert _fts5_safe_query("foo:bar baz") == '"foo:bar" "baz"'
    # An embedded double-quote is doubled (FTS5 phrase escaping).
    assert _fts5_safe_query('a"b') == '"a""b"'
    assert _fts5_safe_query("   ") == ""
