"""PostgreSQL search projection/visibility through constructed SessionDB handles.

The SQL boundary executes real queries against a temporary SQLite store. Only
PostgreSQL dialect primitives are adapted: FTS predicates use a real FTS5
index containing every message, ILIKE uses SQLite's ASCII-insensitive LIKE,
and rank is constant because these tests do not assert relevance ordering.
No search results, filter builders, or context enrichers are mocked. These
contracts do not replace validation against a PostgreSQL server.
"""

from __future__ import annotations

import re
import sqlite3

import pytest

import hermes_state
import hermes_state_postgres


class _SearchSqlBoundary:
    def __init__(self, connection):
        self.connection = connection
        self._conn = self  # raw query-builder/catalog probes use the same SQL boundary
        self.raw = self
        self._dsn = "postgresql://test.invalid/search-contract"
        self.context_queries = 0
        self.match_routes = []

    def execute(self, sql, params=()):
        normalized = " ".join(sql.split())
        if "pg_catalog.pg_attribute" in normalized:
            return self.connection.execute(
                "SELECT 1 FROM pragma_table_info('messages') WHERE name = 'fts_content'",
            )
        if re.search(r"\bIN\s*\(\s*\)", sql, re.IGNORECASE):
            raise RuntimeError("PostgreSQL rejects empty IN lists")
        if normalized.upper().startswith("WITH TARGET AS"):
            self.context_queries += 1
        if "m.fts_content @@" in sql:
            self.match_routes.append("fts")
            sql = sql.replace(
                "m.fts_content @@ %s::tsquery",
                "m.id IN (SELECT rowid FROM test_pg_fts WHERE test_pg_fts MATCH ?)",
            )
        if " ILIKE " in sql:
            self.match_routes.append("ilike")
            sql = sql.replace(" ILIKE ", " LIKE ")
        sql = re.sub(r"::(?:tsquery|text)\b", "", sql).replace("%s", "?")
        return self.connection.execute(sql, params)

    def close(self):
        self.connection.close()


@pytest.fixture(params=["fts", "ilike"])
def search_store(request, tmp_path, monkeypatch):
    home = tmp_path / "pg_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    remote_path = tmp_path / "search.sqlite"
    ids = {}
    with hermes_state.SessionDB(db_path=remote_path) as seed:
        seed.create_session("projection", source="cli", model="search-model")
        for stamp, (role, content) in enumerate((
            ("user", "before the match"),
            ("assistant", "projectionneedle answer"),
            ("user", "after the match"),
        ), start=1):
            mid = seed.append_message("projection", role, content, timestamp=stamp)
            if role == "assistant":
                ids["projection"] = mid
        for stamp, (name, source) in enumerate((
            ("visible", "cli"), ("desktop", "desktop"), ("cron", "cron"),
            ("hidden", "cli"), ("rewound", "cli"), ("compacted", "cli"),
        ), start=10):
            seed.create_session(name, source=source)
            ids[name] = seed.append_message(
                name, "user", f"visibilityneedle {name}", timestamp=stamp,
                display_kind="hidden" if name == "hidden" else None,
            )
        for name, compacted in (("rewound", 0), ("compacted", 1)):
            seed._write_sql("UPDATE messages SET active = 0, compacted = ? WHERE id = ?",
                            (compacted, ids[name]))

    raw = sqlite3.connect(remote_path, isolation_level=None)
    raw.row_factory = sqlite3.Row
    raw.create_function("websearch_to_tsquery", 2, lambda dictionary, query: query)
    raw.create_function("plainto_tsquery", 2, lambda dictionary, query: query)
    raw.create_function("ts_rank", 2, lambda content, query: 1.0)
    if request.param == "fts":
        raw.execute("ALTER TABLE messages ADD COLUMN fts_content TEXT")
        raw.execute("UPDATE messages SET fts_content = content")
        # This index deliberately includes hidden and rewound rows. The actual
        # PostgreSQL WHERE predicates must enforce visibility themselves.
        raw.execute("CREATE VIRTUAL TABLE test_pg_fts USING fts5("
                    "content, content='messages', content_rowid='id')")
        raw.execute("INSERT INTO test_pg_fts(test_pg_fts) VALUES ('rebuild')")
    boundary = _SearchSqlBoundary(raw)
    monkeypatch.setattr(
        hermes_state_postgres, "maybe_open_postgres",
        lambda read_only, schema_version, dsn_override=None: boundary,
    )
    db = hermes_state.SessionDB(db_path=home / "state.db", postgres_dsn=boundary._dsn)
    try:
        yield db, boundary, ids
    finally:
        db.close()
        # A silently failing FTS query must not pass these tests through the
        # ILIKE fallback, which has independent SQL and enrichment dispatch.
        assert boundary.match_routes
        assert set(boundary.match_routes) == {request.param}


def test_include_context_false_avoids_enrichment(search_store):
    db, boundary, ids = search_store
    matches = hermes_state_postgres.search_messages_postgres(
        db._conn, db._decode_content, "projectionneedle", include_context=False,
    )
    assert [row["id"] for row in matches] == [ids["projection"]]
    assert "context" not in matches[0]
    assert boundary.context_queries == 0

    enriched = hermes_state_postgres.search_messages_postgres(
        db._conn, db._decode_content, "projectionneedle", include_context=True,
    )
    assert enriched[0]["id"] == matches[0]["id"]
    assert [row["content"] for row in enriched[0]["context"]] == [
        "before the match", "projectionneedle answer", "after the match",
    ]
    assert boundary.context_queries > 0


@pytest.mark.parametrize("fields", [("id", "snippet"), ("session_id", "context"), ()])
def test_requested_fields_control_result_and_enrichment(search_store, fields):
    db, boundary, ids = search_store
    matches = db.search_messages("projectionneedle", fields=fields)
    assert len(matches) == 1
    assert set(matches[0]) == set(fields)
    if "id" in fields:
        assert matches[0]["id"] == ids["projection"]
        assert matches[0]["snippet"] == "projectionneedle answer"
    if "context" in fields:
        assert matches[0]["session_id"] == "projection"
        assert matches[0]["context"]
        assert boundary.context_queries > 0
    else:
        assert boundary.context_queries == 0


@pytest.mark.parametrize("include_inactive", [False, True])
def test_hidden_hits_stay_excluded_when_inactive_history_is_requested(search_store, include_inactive):
    db, _, ids = search_store
    matches = db.search_messages(
        "visibilityneedle", include_inactive=include_inactive, fields=("id",),
    )
    expected = {ids[name] for name in ("visible", "desktop", "cron", "compacted")}
    if include_inactive:
        expected.add(ids["rewound"])
    assert {row["id"] for row in matches} == expected
    assert ids["hidden"] not in expected


def test_empty_source_filter_selects_no_sources(search_store):
    db, _, _ = search_store
    assert db.search_messages("visibilityneedle", source_filter=None, fields=("id",))
    assert db.search_messages("visibilityneedle", source_filter=[], fields=("id",)) == []


def test_empty_exclusions_leave_all_sources_available(search_store):
    db, _, _ = search_store
    unrestricted = db.search_messages("visibilityneedle", fields=("id", "source"))
    assert {row["source"] for row in unrestricted} == {"cli", "desktop", "cron"}
    assert db.search_messages(
        "visibilityneedle", exclude_sources=[], fields=("id", "source"),
    ) == unrestricted
    excluded = db.search_messages("visibilityneedle", exclude_sources=["cli"], fields=("source",))
    assert {row["source"] for row in excluded} == {"desktop", "cron"}
