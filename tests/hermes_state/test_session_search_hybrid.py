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
        self.filter_calls = []
        self.db_path = db_path

    def search_messages(self, query, source_filter=None, exclude_sources=None, limit=20, fields=None):
        self.calls.append(query)
        self.filter_calls.append((source_filter, exclude_sources))
        rows = list(self.rows)
        if source_filter is not None:
            allowed = set(source_filter)
            rows = [row for row in rows if row.get("source") in allowed]
        if exclude_sources is not None:
            blocked = set(exclude_sources)
            rows = [row for row in rows if row.get("source") not in blocked]
        return rows[:limit]


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


def _seed_state(db_path, sessions, messages):
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE sessions (id TEXT, source TEXT, model TEXT, started_at REAL)")
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT)"
    )
    conn.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?)", sessions)
    conn.executemany("INSERT INTO messages VALUES (?, ?, ?, ?)", messages)
    conn.commit()
    conn.close()


def test_semantic_lane_respects_source_and_exclude_filters(tmp_path):
    db_path = tmp_path / "state.db"
    _seed_state(
        db_path,
        sessions=[
            ("telegram-hit", "telegram", "m", 1),
            ("cli-semantic", "cli", "m", 2),
        ],
        messages=[
            (1, "telegram-hit", "user", "shared phrase"),
            (2, "cli-semantic", "user", "nearby meaning"),
        ],
    )
    db = _FuzzyDB(
        [{"session_id": "telegram-hit", "snippet": "shared phrase", "source": "telegram"}],
        db_path=db_path,
    )
    store = EmbeddingIndex(db.db_path)
    store.upsert(
        "m",
        [
            (1, "telegram-hit", [0.5, 0.5]),
            (2, "cli-semantic", [1.0, 0.0]),
        ],
    )

    def embed(texts):
        return [[1.0, 0.0] for _ in texts]

    settings = {"model": "m", "base_url": "http://embed", "api_key": "k"}
    included = content_search_matches(
        db,
        "shared*",
        source_filter=["telegram"],
        limit=10,
        settings=settings,
        embed=embed,
        index=store,
    )
    excluded = content_search_matches(
        db,
        "shared*",
        exclude_sources=["cli"],
        limit=10,
        settings=settings,
        embed=embed,
        index=store,
    )
    store.close()
    assert [row["session_id"] for row in included] == ["telegram-hit"]
    assert [row["session_id"] for row in excluded] == ["telegram-hit"]
    assert db.filter_calls[0] == (["telegram"], None)
    assert db.filter_calls[1] == (None, ["cli"])


def test_index_batch_reaches_messages_older_than_the_recent_window(tmp_path, monkeypatch):
    import hermes_state_search_hybrid as hybrid

    db_path = tmp_path / "state.db"
    # One old unindexed row, then a full recent window of already-indexed rows.
    recent_count = hybrid._INDEX_BATCH * 4
    sessions = [("old", "cli", "m", 1), ("new", "cli", "m", 2)]
    messages = [(1, "old", "user", "ancient meaning")]
    messages.extend(
        (mid, "new", "user", f"recent {mid}")
        for mid in range(2, recent_count + 2)
    )
    _seed_state(db_path, sessions=sessions, messages=messages)

    store = EmbeddingIndex(db_path)
    store.upsert(
        "m",
        [(mid, "new", [0.0, 1.0]) for mid in range(2, recent_count + 2)],
    )
    embedded = []

    def embed(texts):
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    hybrid._index_batch(store, _FuzzyDB([], db_path=db_path), "m", embed)
    assert embedded == ["ancient meaning"]
    assert 1 in store.stored_ids("m")
    store.close()


def test_sidecar_open_or_read_failure_degrades_to_fuzzy(tmp_path, monkeypatch):
    import hermes_state_search_hybrid as hybrid

    fuzzy_rows = [
        {"session_id": "s1", "snippet": "alpha", "source": "cli"},
        {"session_id": "s2", "snippet": "beta", "source": "cli"},
    ]
    db = _FuzzyDB(fuzzy_rows, db_path=tmp_path / "state.db")

    class _BoomIndex:
        def __init__(self, *_args, **_kwargs):
            raise OSError("sidecar unreadable")

        def close(self):
            return None

    monkeypatch.setattr(hybrid, "EmbeddingIndex", _BoomIndex)
    rows = content_search_matches(
        db,
        "alpha*",
        limit=10,
        settings={"model": "m", "base_url": "http://embed", "api_key": "k"},
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    assert [row["session_id"] for row in rows] == ["s1", "s2"]

    class _BoomRead:
        def close(self):
            return None

        def stored_ids(self, model):
            return set()

        def ranked_sessions(self, model, query_vector, limit):
            raise sqlite3.OperationalError("database is locked")

        def upsert(self, *args, **kwargs):
            return None

    import sqlite3

    rows = content_search_matches(
        db,
        "alpha*",
        limit=10,
        settings={"model": "m", "base_url": "http://embed", "api_key": "k"},
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
        index=_BoomRead(),
    )
    assert [row["session_id"] for row in rows] == ["s1", "s2"]
