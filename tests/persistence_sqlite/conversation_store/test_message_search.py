"""The conversation store's search functions on a real StoreHolder: FTS5 message search, the CJK LIKE fallback, the FTS5 query sanitizer, session search and listing filters, and session id search."""

import time

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import (
    _contains_cjk,
    _sanitize_fts5_query,
    search_messages,
    search_sessions_by_id,
)
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    list_sessions_rich,
    search_sessions,
    set_session_archived,
)


@pytest.fixture()
def db(tmp_path):
    """Create a StoreHolder with a temp database file."""
    db_path = tmp_path / "test_state.db"
    holder = StoreHolder(db_path=db_path)
    yield holder
    holder.close()


# =========================================================================
# FTS5 search
# =========================================================================

class TestFTS5Search:
    def test_search_finds_content(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="How do I deploy with Docker?")
        append_message(db, "s1", role="assistant", content="Use docker compose up.")

        results = search_messages(db, "docker")
        assert len(results) == 2
        # At least one result should mention docker
        snippets = [r.get("snippet", "") for r in results]
        assert any("docker" in s.lower() or "Docker" in s for s in snippets)
        # Results never carry full content; snippet + metadata only.
        assert all("content" not in r for r in results)

    def test_search_returns_context(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="Tell me about Kubernetes")
        append_message(db, "s1", role="assistant", content="Kubernetes is an orchestrator.")

        results = search_messages(db, "Kubernetes")
        assert len(results) == 2
        assert "context" in results[0]
        assert isinstance(results[0]["context"], list)
        assert len(results[0]["context"]) > 0

    def test_search_fields_project_results_without_changing_default(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="Tell me about Kubernetes")
        append_message(db, "s1", role="assistant", content="Kubernetes is an orchestrator.")

        projected = search_messages(db,
            "Kubernetes", fields=("session_id", "role", "snippet")
        )
        default = search_messages(db, "Kubernetes")

        assert len(projected) == len(default) == 2
        assert all(set(row) == {"session_id", "role", "snippet"} for row in projected)
        assert [
            (row["session_id"], row["role"], row["snippet"])
            for row in projected
        ] == [
            (row["session_id"], row["role"], row["snippet"])
            for row in default
        ]
        assert all("context" in row and row["context"] for row in default)

    def test_sanitize_fts5_query_strips_dangerous_chars(self):
        """Unit test for the _sanitize_fts5_query function."""
        s = _sanitize_fts5_query
        assert s('hello world') == 'hello world'
        assert '+' not in s('C++')
        assert '"' not in s('"unterminated')
        assert '(' not in s('(problem')
        assert '{' not in s('{test}')
        # Dangling operators removed
        assert s('hello AND') == 'hello'
        assert s('OR world') == 'world'
        # Leading bare * removed
        assert s('***') == ''
        # Valid prefix kept
        assert s('deploy*') == 'deploy*'
        # Colon (FTS5 column-filter operator) stripped, both terms preserved
        assert ':' not in s('TODO: fix')
        assert s('TODO: fix').split() == ['TODO', 'fix']
        assert ':' not in s('error:timeout')

    def test_long_search_query_is_capped_and_does_not_crash(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="bounded sanitizer target")

        query = ('"' * 50_000) + (" bounded" * 10_000)
        start = time.perf_counter()
        results = search_messages(db, query)
        elapsed = time.perf_counter() - start

        assert isinstance(results, list)
        assert elapsed < 1.0


# =========================================================================
# CJK (Chinese/Japanese/Korean) LIKE fallback
# =========================================================================

class TestCJKSearchFallback:
    """Regression tests for CJK search (see #11511).

    SQLite FTS5's default tokenizer treats contiguous CJK runs as a single
    token ("和其他agent的聊天记录" → one token), so substring queries like
    "记忆断裂" return 0 rows despite the data being present. search_messages falls
    back to LIKE substring matching whenever FTS5 returns no results and
    the query contains CJK characters.
    """

    def test_cjk_detection_covers_all_ranges(self):
        f = _contains_cjk
        # Chinese (CJK Unified Ideographs)
        assert f("记忆断裂") is True
        # Japanese Hiragana + Katakana
        assert f("こんにちは") is True
        assert f("カタカナ") is True
        # Korean Hangul syllables (both early and late — guards against
        # the \ud7a0-\ud7af typo seen in one of the duplicate PRs)
        assert f("안녕하세요") is True
        assert f("기억") is True
        # Non-CJK
        assert f("hello world") is False
        assert f("日本語mixedwithenglish") is True
        assert f("") is False

        # No CJK in query → LIKE fallback must not run. We don't assert this
        # directly (no instrumentation), but the FTS5 path produces an
        # FTS5-style snippet with highlight markers when the term is short.
        # At minimum: english queries must still match.

    def test_mixed_cjk_english_query(self, db):
        """Mixed queries should still fall back to LIKE when FTS5 misses."""
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="讨论Agent通信协议")
        # "Agent通信" is CJK+English — FTS5 default tokenizer indexes the
        # whole CJK run with embedded "agent" as separate tokens; the LIKE
        # fallback handles the substring correctly.
        results = search_messages(db, "Agent通信")
        assert len(results) == 1

    def test_cjk_like_escapes_wildcards(self, db):
        """Special characters (%, _) in CJK queries are treated as literals."""
        create_session(db, session_id="s1", source="cli")
        create_session(db, session_id="s2", source="cli")
        append_message(db, "s1", role="user", content="达成100%完成率")
        append_message(db, "s2", role="user", content="达成100完成率是目标")
        # The % in the query must be literal — should only match s1
        results = search_messages(db, "100%完成")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"


# =========================================================================
# Session search and listing
# =========================================================================

class TestSearchSessions:
    def test_list_all_sessions(self, db):
        create_session(db, session_id="s1", source="cli")
        create_session(db, session_id="s2", source="telegram")

        sessions = search_sessions(db)
        assert len(sessions) == 2

    def test_pagination(self, db):
        for i in range(5):
            create_session(db, session_id=f"s{i}", source="cli")

        page1 = search_sessions(db, limit=2)
        page2 = search_sessions(db, limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]["id"] != page2[0]["id"]


# =========================================================================
# Session source exclusion (--source flag for third-party isolation)
# =========================================================================

class TestExcludeSources:
    """Tests for exclude_sources on list_sessions_rich and search_messages."""

    def test_list_sessions_rich_excludes_tool_source(self, db):
        create_session(db, "s1", "cli")
        create_session(db, "s2", "tool")
        create_session(db, "s3", "telegram")
        sessions = list_sessions_rich(db, exclude_sources=["tool"])
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s3" in ids
        assert "s2" not in ids

    def test_search_messages_excludes_tool_source(self, db):
        create_session(db, "s1", "cli")
        append_message(db, "s1", "user", "Python deployment question")
        create_session(db, "s2", "tool")
        append_message(db, "s2", "user", "Python automated question")
        results = search_messages(db, "Python", exclude_sources=["tool"])
        sources = [r["source"] for r in results]
        assert "cli" in sources
        assert "tool" not in sources


# =========================================================================
# FTS5 indexing of tool_calls / tool_name (#16751)
# =========================================================================

class TestFTS5ToolCallIndexing:
    """Regression tests: search_messages must see tool_name and tool_calls.

    Before #16751's fix, `messages_fts` only indexed `messages.content`, so
    tokens that only appeared in `tool_name` or the serialized `tool_calls`
    JSON were invisible to session_search even though the row was in the DB.
    """

    def test_tool_name_is_searchable(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db,
            "s1", role="assistant", content="",
            tool_name="UNIQUETOOLNAME",
        )
        results = search_messages(db, "UNIQUETOOLNAME")
        assert len(results) == 1

    def test_tool_calls_args_are_searchable(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db,
            "s1", role="assistant", content="",
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "UNIQUESEARCHTOKEN"}',
                },
            }],
        )
        results = search_messages(db, "UNIQUESEARCHTOKEN")
        assert len(results) == 1


class TestSessionIdSearch:
    """Session id search backs Desktop's Search Sessions UX."""

    def _seed(self, db, sid, *, content="ordinary message", archived=False, source="cli"):
        create_session(db, session_id=sid, source=source, model="test-model")
        append_message(db, session_id=sid, role="user", content=content)
        if archived:
            set_session_archived(db, sid, True)

    def test_search_sessions_by_id_matches_exact_prefix_and_substring(self, db):
        self._seed(db, "20260603_090200_abcd12", content="content without id")
        self._seed(db, "20260602_111111_other99", content="other content")

        assert [s["id"] for s in search_sessions_by_id(db, "20260603_090200_abcd12")] == [
            "20260603_090200_abcd12"
        ]
        assert [s["id"] for s in search_sessions_by_id(db, "20260603")] == ["20260603_090200_abcd12"]
        assert [s["id"] for s in search_sessions_by_id(db, "ABCD12")] == ["20260603_090200_abcd12"]


class TestFts5SanitizerCharacterClass:
    """Every character FTS5 rejects outside a quoted phrase must be stripped.

    A survivor reaches MATCH raw and raises, which the execute site swallows
    into zero results — so the search silently finds nothing rather than
    erroring. Assertions run the sanitized text against a real FTS5 table.
    """

    @staticmethod
    def _fts_table():
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        conn.execute(
            "INSERT INTO t (content) VALUES "
            "('meet me at user host about gateway run py it s 50 a b')"
        )
        return conn

    @staticmethod
    def _sanitize(query):
        return _sanitize_fts5_query(query)

    @pytest.mark.parametrize(
        "query",
        [
            "it's",                 # apostrophe — ordinary prose
            "gateway/run.py",       # path separator
            "user@host",            # email / handle
            "a,b",                  # comma
            "why?",                 # question mark
            "e=mc2",                # equals
            "a;b", "a!b", "a&b", "a|b", "x~y",
            "#tag", "$dollar", "[bracket]", "<tag>",
            r"C:\path\file",        # backslash
        ],
    )
    def test_query_stays_parsable(self, query):
        conn = self._fts_table()
        sanitized = self._sanitize(query)
        if not sanitized.strip():
            return
        # Raises sqlite3.OperationalError if a special character survived.
        conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (sanitized,)).fetchone()

    def test_plain_terms_are_untouched(self):
        assert self._sanitize("hello world").split() == ["hello", "world"]

    def test_quoted_phrase_survives(self):
        assert '"exact phrase"' in self._sanitize('"exact phrase"')

    def test_hyphen_dotted_term_still_quoted(self):
        # Step 5's behaviour must not regress: my-app.config.ts stays one term.
        assert '"my-app.config.ts"' in self._sanitize("my-app.config.ts")

    def test_prefix_star_still_works(self):
        conn = self._fts_table()
        sanitized = self._sanitize("gate*")
        rows = conn.execute(
            "SELECT count(*) FROM t WHERE t MATCH ?", (sanitized,)
        ).fetchone()
        assert rows[0] == 1

    def test_percent_stripped_for_non_cjk_query(self):
        # % is kept only for the CJK LIKE fallback; a non-CJK query never
        # reaches that fallback, so % must be stripped before MATCH.
        conn = self._fts_table()
        sanitized = self._sanitize("50%")
        assert "%" not in sanitized
        conn.execute(
            "SELECT count(*) FROM t WHERE t MATCH ?", (sanitized,)
        ).fetchone()

    def test_percent_preserved_for_cjk_query(self):
        # The CJK LIKE fallback builds its own pattern from the sanitized
        # text; keep % intact there (pre-existing contract).
        sanitized = self._sanitize("完成50%")
        assert "%" in sanitized
