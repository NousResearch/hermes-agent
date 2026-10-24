"""Gateway session search stays fuzzy-only unless an embedding model is configured."""

from hermes_state_search_hybrid import (
    EmbeddingIndex,
    content_search_matches,
    cosine,
    rrf_fuse,
    semantic_enabled,
)


def test_rrf_promotes_an_id_that_ranks_well_in_both_lists():
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]])
    assert fused[0] == "b"
    assert "d" in fused
    assert fused.index("b") < fused.index("a")


def test_cosine_is_one_for_the_same_direction():
    assert cosine([1.0, 0.0], [2.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0


def test_default_config_is_fuzzy_only():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_state_search_hybrid import embedding_settings

    assert embedding_settings(DEFAULT_CONFIG)["model"] == ""
    assert semantic_enabled(embedding_settings(DEFAULT_CONFIG)) is False
    assert semantic_enabled({"model": "", "base_url": "http://embed", "api_key": "k"}) is False
    assert semantic_enabled({"model": "m", "base_url": "", "api_key": "k"}) is False
    assert semantic_enabled({"model": "m", "base_url": "http://embed", "api_key": ""}) is False
    assert semantic_enabled({"model": "m", "base_url": "http://embed", "api_key": "k"}) is True


class _FuzzyDB:
    def __init__(self, rows, db_path=None):
        self.rows = rows
        self.calls = []
        self.db_path = db_path

    def search_messages(self, query, source_filter=None, exclude_sources=None, limit=20, fields=None):
        self.calls.append(query)
        return list(self.rows)[:limit]


def test_no_embedding_model_returns_fuzzy_hits_unchanged():
    db = _FuzzyDB([
        {"session_id": "s1", "snippet": "alpha"},
        {"session_id": "s2", "snippet": "beta"},
    ])
    rows = content_search_matches(
        db, "alpha*", limit=10, settings={"model": "", "base_url": "", "api_key": ""},
    )
    assert [row["session_id"] for row in rows] == ["s1", "s2"]
    assert db.calls == ["alpha*"]


def test_embedding_model_fuses_fuzzy_and_semantic_ranks(tmp_path):
    import sqlite3

    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, source TEXT, model TEXT, started_at REAL)")
    conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('semantic-only', 'cli', 'm', 1)")
    conn.execute("INSERT INTO messages VALUES (2, 'semantic-only', 'user', 'nearby meaning')")
    conn.commit()
    conn.close()

    db = _FuzzyDB([
        {"session_id": "fuzzy-only", "snippet": "keyboard"},
        {"session_id": "both", "snippet": "shared"},
    ], db_path=db_path)
    store = EmbeddingIndex(db.db_path)
    store.upsert("m", [
        (1, "both", [1.0, 0.0]),
        (2, "semantic-only", [0.9, 0.1]),
    ])

    def embed(texts):
        return [[1.0, 0.0] for _ in texts]

    rows = content_search_matches(
        db,
        "shared*",
        limit=10,
        settings={"model": "m", "base_url": "http://embed", "api_key": "k"},
        embed=embed,
        index=store,
    )
    store.close()
    order = [row["session_id"] for row in rows]
    assert order[0] == "both"
    assert "semantic-only" in order
    assert "fuzzy-only" in order
