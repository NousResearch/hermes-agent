"""Tests for term_index — inverted index for session search fast path.

Covers: stop word filtering, term extraction, term insertion at write time,
term-based search with session-level results, multi-term intersection.
"""

import time
import pytest
from pathlib import Path

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# =========================================================================
# Stop word filtering
# =========================================================================

class TestStopWords:
    def test_common_english_words_are_stopped(self):
        from stop_words import is_stop_word
        for w in ["the", "and", "is", "in", "it", "of", "to", "a", "was", "for"]:
            assert is_stop_word(w), f"'{w}' should be a stop word"

    def test_case_insensitive_stop_words(self):
        from stop_words import is_stop_word
        assert is_stop_word("The")
        assert is_stop_word("AND")
        assert is_stop_word("Is")

    def test_non_stop_words_pass(self):
        from stop_words import is_stop_word
        for w in ["docker", "kubernetes", "python", "hermes", "session"]:
            assert not is_stop_word(w), f"'{w}' should NOT be a stop word"

    def test_short_words_not_auto_stopped(self):
        """Single letters and 2-letter words that aren't in the list should pass."""
        from stop_words import is_stop_word
        # 'go' is a real tech term, 'I' is a stop word
        assert not is_stop_word("go")
        assert is_stop_word("I")


# =========================================================================
# Term extraction
# =========================================================================

class TestTermExtraction:
    def test_extracts_words_from_content(self):
        from term_index import extract_terms
        terms = extract_terms("docker compose up -d")
        assert "docker" in terms
        assert "compose" in terms

    def test_strips_punctuation(self):
        from term_index import extract_terms
        terms = extract_terms("It's working! Check the file.py, okay?")
        assert "working" in terms
        assert "file.py" in terms  # dots in filenames preserved
        assert "okay" in terms

    def test_filters_stop_words(self):
        from term_index import extract_terms
        terms = extract_terms("the docker container is running in the background")
        assert "the" not in terms
        assert "is" not in terms
        assert "in" not in terms
        assert "docker" in terms
        assert "container" in terms
        assert "running" in terms

    def test_case_folded(self):
        from term_index import extract_terms
        terms = extract_terms("Docker DOCKER docker")
        # Should be case-folded to single term
        assert len(terms) == len(set(terms)), "Terms should be deduplicated after case folding"

    def test_empty_content(self):
        from term_index import extract_terms
        terms = extract_terms("")
        assert terms == []

    def test_none_content(self):
        from term_index import extract_terms
        terms = extract_terms(None)
        assert terms == []

    def test_preserves_paths_and_commands(self):
        from term_index import extract_terms
        terms = extract_terms("edited /etc/hosts and ran git push origin main")
        assert "/etc/hosts" in terms or "etc/hosts" in terms  # path fragment
        assert "git" in terms
        assert "push" in terms


# =========================================================================
# Term index insertion
# =========================================================================

class TestTermIndexInsertion:
    def test_terms_inserted_on_append_message(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            session_id="s1",
            role="user",
            content="I need to deploy the docker container",
        )

        # Should be able to find the message by term
        results = db.search_by_terms(["docker"])
        assert len(results) >= 1
        assert any(r["session_id"] == "s1" for r in results)

    def test_stop_words_not_indexed(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            session_id="s1",
            role="user",
            content="the and is in of to a",
        )

        # All stop words — should find nothing
        results = db.search_by_terms(["the", "and", "is"])
        assert len(results) == 0

    def test_same_term_multiple_messages_same_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="docker is great")
        db.append_message(session_id="s1", role="assistant", content="docker compose ready")

        results = db.search_by_terms(["docker"])
        # Should return session once, not twice
        sids = [r["session_id"] for r in results]
        assert sids.count("s1") == 1

    def test_term_indexed_across_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message(session_id="s1", role="user", content="fix the docker bug")
        db.append_message(session_id="s2", role="user", content="docker pull failed")

        results = db.search_by_terms(["docker"])
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" in sids


# =========================================================================
# Term-based search
# =========================================================================

class TestTermSearch:
    def test_single_term_search(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            session_id="s1",
            role="user",
            content="I need to configure kubernetes",
        )

        results = db.search_by_terms(["kubernetes"])
        assert len(results) >= 1
        assert results[0]["session_id"] == "s1"
        # Should include session metadata
        assert "source" in results[0]
        assert "started_at" in results[0] or "session_started" in results[0]

    def test_multi_term_intersection(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.create_session(session_id="s3", source="cli")

        db.append_message(session_id="s1", role="user", content="docker networking issue")
        db.append_message(session_id="s2", role="user", content="docker container running")
        db.append_message(session_id="s3", role="user", content="kubernetes networking problem")

        # Both "docker" AND "networking" should only match s1
        results = db.search_by_terms(["docker", "networking"])
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" not in sids
        assert "s3" not in sids

    def test_search_returns_empty_for_stop_words_only(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="the and is")

        results = db.search_by_terms(["the", "and"])
        assert results == []

    def test_search_excludes_hidden_sources(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="tool")
        db.append_message(session_id="s1", role="user", content="docker deployment")
        db.append_message(session_id="s2", role="user", content="docker deployment tool")

        results = db.search_by_terms(["docker"], exclude_sources=["tool"])
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" not in sids

    def test_search_with_limit(self, db):
        for i in range(5):
            sid = f"s{i}"
            db.create_session(session_id=sid, source="cli")
            db.append_message(session_id=sid, role="user", content="python script")

        results = db.search_by_terms(["python"], limit=3)
        assert len(results) <= 3

    def test_nonexistent_term_returns_empty(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hello world")

        results = db.search_by_terms(["nonexistent_xyzzy"])
        assert results == []

    def test_term_result_includes_match_count(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="docker docker docker")
        db.append_message(session_id="s1", role="assistant", content="docker ready")

        results = db.search_by_terms(["docker"])
        assert len(results) >= 1
        # Should tell us how many messages matched in the session
        assert "match_count" in results[0]


# =========================================================================
# Schema and migration
# =========================================================================

class TestTermIndexSchema:
    def test_term_index_table_exists(self, db):
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='term_index'"
        )
        assert cursor.fetchone() is not None

    def test_term_index_is_without_rowid(self, db):
        cursor = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='term_index'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert "WITHOUT ROWID" in row[0]

    def test_schema_version_bumped(self, db):
        cursor = db._conn.execute("SELECT version FROM schema_version LIMIT 1")
        version = cursor.fetchone()[0]
        assert version >= 9

    def test_existing_data_survives_migration(self, tmp_path):
        """Create a v6 DB, then open it with current code -- data should survive."""
        # Build a v6 DB manually
        db_path = tmp_path / "migrate.db"
        db = SessionDB(db_path=db_path)
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hello world")
        db.close()

        # Re-open -- migration should run, data intact
        db2 = SessionDB(db_path=db_path)
        session = db2.get_session("s1")
        assert session is not None
        assert session["source"] == "cli"
        # term_index should now exist
        cursor = db2._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='term_index'"
        )
        assert cursor.fetchone() is not None
        db2.close()

    def test_v9_migration_auto_reindexes(self, tmp_path):
        """When a v6 DB with existing messages is opened, the v9 migration
        should create the term_index and backfill it automatically."""
        db_path = tmp_path / "migrate_v9.db"

        # Step 1: Create a fresh DB, add messages, then manually downgrade
        # to v6 so the next open triggers the migration path.
        db = SessionDB(db_path=db_path)
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message(session_id="s1", role="user", content="deploy the kubernetes cluster")
        db.append_message(session_id="s2", role="user", content="debug docker networking issue")
        db.close()

        # Step 2: Re-open raw, manually set version to 6 and wipe term_index
        # to simulate a pre-v7 DB.
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("UPDATE schema_version SET version = 6")
        conn.execute("DROP TABLE IF EXISTS term_index")
        conn.commit()
        conn.close()

        # Step 3: Open with SessionDB — should migrate to v9 and auto-reindex.
        db2 = SessionDB(db_path=db_path)
        # Verify version is now 9
        cursor = db2._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == 9

        # Verify term_index is populated — search should find the terms
        results = db2.search_by_terms(["kubernetes"])
        assert len(results) >= 1
        assert results[0]["session_id"] == "s1"

        results2 = db2.search_by_terms(["docker"])
        assert len(results2) >= 1
        assert results2[0]["session_id"] == "s2"

        db2.close()


# =========================================================================
# Regression tests for red-team QA bugs
# =========================================================================

class TestClearMessagesCleansTermIndex:
    """BUG 3: clear_messages() left stale term_index entries.

    After clearing messages, search_by_terms should return zero results
    for that session, not ghost matches pointing to deleted message IDs.
    """

    def test_clear_messages_removes_term_entries(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="docker networking issue")

        # Confirm indexed
        results = db.search_by_terms(["docker"])
        assert len(results) >= 1

        # Clear messages
        db.clear_messages(session_id="s1")

        # Term entries should be gone
        results = db.search_by_terms(["docker"])
        assert results == []

    def test_clear_messages_does_not_affect_other_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message(session_id="s1", role="user", content="docker test")
        db.append_message(session_id="s2", role="user", content="docker prod")

        db.clear_messages(session_id="s1")

        # s2 should still be searchable
        results = db.search_by_terms(["docker"])
        sids = [r["session_id"] for r in results]
        assert "s2" in sids
        assert "s1" not in sids

    def test_clear_messages_no_stray_term_rows(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="kubernetes deployment")

        db.clear_messages(session_id="s1")

        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM term_index WHERE session_id = 's1'"
        )
        assert cursor.fetchone()[0] == 0


class TestSearchByTermsParamBinding:
    """BUG 1: search_by_terms() had dead code with wrong param binding.

    The multi-term GROUP BY + HAVING path is the one that actually runs.
    These tests verify parameter binding is correct for both single and
    multi-term queries, including with exclude_sources.
    """

    def test_single_term_with_exclude_sources(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="tool")
        db.append_message(session_id="s1", role="user", content="docker deploy")
        db.append_message(session_id="s2", role="user", content="docker deploy")

        results = db.search_by_terms(["docker"], exclude_sources=["tool"])
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" not in sids

    def test_multi_term_and_semantics(self, db):
        """Multi-term search should use AND: only sessions with ALL terms match."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.create_session(session_id="s3", source="cli")
        db.append_message(session_id="s1", role="user", content="docker networking issue")
        db.append_message(session_id="s2", role="user", content="docker container only")
        db.append_message(session_id="s3", role="user", content="networking problem only")

        results = db.search_by_terms(["docker", "networking"])
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" not in sids
        assert "s3" not in sids

    def test_multi_term_with_exclude_sources(self, db):
        """Multi-term + exclude_sources: param binding must be correct."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="tool")
        db.append_message(session_id="s1", role="user", content="docker networking setup")
        db.append_message(session_id="s2", role="user", content="docker networking deploy")

        results = db.search_by_terms(
            ["docker", "networking"], exclude_sources=["tool"]
        )
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" not in sids

    def test_three_term_intersection(self, db):
        """Three-term AND: all three must be present in the session."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message(session_id="s1", role="user", content="docker kubernetes aws deployment")
        db.append_message(session_id="s2", role="user", content="docker kubernetes only two terms")

        results = db.search_by_terms(["docker", "kubernetes", "aws"])
        sids = [r["session_id"] for r in results]
        assert "s1" in sids
        assert "s2" not in sids


class TestDeleteSessionCleansTermIndex:
    """Verify delete_session() and prune_sessions() clean term_index."""

    def test_delete_session_removes_term_entries(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="docker deploy")
        db.append_message(session_id="s1", role="assistant", content="docker is running")

        db.delete_session(session_id="s1")

        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM term_index WHERE session_id = 's1'"
        )
        assert cursor.fetchone()[0] == 0

    def test_delete_session_does_not_affect_other_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message(session_id="s1", role="user", content="docker one")
        db.append_message(session_id="s2", role="user", content="docker two")

        db.delete_session(session_id="s1")

        results = db.search_by_terms(["docker"])
        sids = [r["session_id"] for r in results]
        assert "s2" in sids
        assert "s1" not in sids


class TestFastSearchSessionResolution:
    """BUG 2: _fast_search didn't resolve child sessions to parent.

    A delegation child and its parent both containing "docker" would appear
    as two separate results. They should be resolved to the parent session.
    Also, current session lineage exclusion must cover the entire chain.
    """

    def test_child_resolved_to_parent(self, db):
        """Parent + child matching same term should return 1 result (parent)."""
        import json
        from tools.session_search_tool import _fast_search

        db.create_session(session_id="parent-1", source="cli")
        db.create_session(session_id="child-1", source="cli", parent_session_id="parent-1")
        db.append_message(session_id="parent-1", role="user", content="docker setup question")
        db.append_message(session_id="child-1", role="assistant", content="docker setup done")

        result = json.loads(_fast_search(query="docker", db=db, limit=5, current_session_id=None))
        assert result["success"]
        sids = [e["session_id"] for e in result["results"]]
        # Should collapse to parent, not show both
        assert "child-1" not in sids, "Child should be resolved to parent"
        assert "parent-1" in sids
        assert len(result["results"]) == 1

    def test_match_count_accumulates_from_children(self, db):
        """Match_count should sum parent + child matches."""
        import json
        from tools.session_search_tool import _fast_search

        db.create_session(session_id="p", source="cli")
        db.create_session(session_id="c", source="cli", parent_session_id="p")
        db.append_message(session_id="p", role="user", content="docker question")
        db.append_message(session_id="c", role="assistant", content="docker answer")

        result = json.loads(_fast_search(query="docker", db=db, limit=5, current_session_id=None))
        entry = result["results"][0]
        assert entry["session_id"] == "p"
        assert entry["match_count"] >= 2, f"Expected accumulated count >= 2, got {entry['match_count']}"

    def test_current_session_lineage_excludes_children(self, db):
        """When current session is a child, parent should also be excluded."""
        import json
        from tools.session_search_tool import _fast_search

        db.create_session(session_id="parent-2", source="cli")
        db.create_session(session_id="child-2", source="cli", parent_session_id="parent-2")
        db.create_session(session_id="unrelated", source="cli")
        db.append_message(session_id="parent-2", role="user", content="docker deploy")
        db.append_message(session_id="child-2", role="assistant", content="docker deployed")
        db.append_message(session_id="unrelated", role="user", content="docker build")

        # Current session = child -> should exclude parent-2 AND child-2, keep unrelated
        result = json.loads(_fast_search(query="docker", db=db, limit=5, current_session_id="child-2"))
        sids = [e["session_id"] for e in result["results"]]
        assert "parent-2" not in sids, "Parent of current should be excluded"
        assert "child-2" not in sids, "Current child should be excluded"
        assert "unrelated" in sids, "Unrelated session should appear"


class TestGetChildSessionIds:
    """Tests for SessionDB.get_child_session_ids -- public API replacing
    direct db._lock/db._conn access in _fast_search."""

    def test_returns_child_ids(self, db):
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child-1", source="delegation", parent_session_id="parent")
        db.create_session(session_id="child-2", source="compression", parent_session_id="parent")
        db.create_session(session_id="orphan", source="cli")

        children = db.get_child_session_ids("parent")
        assert set(children) == {"child-1", "child-2"}

    def test_returns_empty_for_leaf_session(self, db):
        db.create_session(session_id="leaf", source="cli")
        assert db.get_child_session_ids("leaf") == []

    def test_returns_empty_for_no_args(self, db):
        assert db.get_child_session_ids() == []

    def test_multiple_parent_ids(self, db):
        db.create_session(session_id="p1", source="cli")
        db.create_session(session_id="p2", source="cli")
        db.create_session(session_id="c1", source="delegation", parent_session_id="p1")
        db.create_session(session_id="c2", source="delegation", parent_session_id="p2")

        children = db.get_child_session_ids("p1", "p2")
        assert set(children) == {"c1", "c2"}

    def test_does_not_recurse(self, db):
        """Only direct children, not grandchildren."""
        db.create_session(session_id="root", source="cli")
        db.create_session(session_id="child", source="delegation", parent_session_id="root")
        db.create_session(session_id="grandchild", source="delegation", parent_session_id="child")

        children = db.get_child_session_ids("root")
        assert children == ["child"]


class TestCJKFallbackInFastSearch:
    """CJK queries should fall through to the slow path even when fast=True.

    The term index can't handle CJK because extract_terms() splits on
    whitespace, and CJK languages don't use spaces between words.
    session_search should detect this and use the FTS5+LIKE fallback.
    """

    def test_cjk_query_bypasses_fast_path(self, db):
        """A CJK query with fast=True should be downgraded to fast=False."""
        import json
        from tools.session_search_tool import session_search

        db.create_session(session_id="cjk-1", source="cli")
        db.append_message(session_id="cjk-1", role="user", content="测试中文搜索")

        # fast=True, but CJK query should fall through to full search
        result = json.loads(session_search(
            query="中文", db=db, limit=3, fast=True, current_session_id=None
        ))
        # The result should come from the slow path (mode="full")
        # not the fast path (mode="fast") since CJK triggers fallback
        assert result["success"]
        # mode should be "full" (not "fast") because CJK forced the fallback
        assert result.get("mode") != "fast"

    def test_english_query_stays_fast(self, db):
        """Non-CJK queries should still use the fast path."""
        import json
        from tools.session_search_tool import session_search

        db.create_session(session_id="eng-1", source="cli")
        db.append_message(session_id="eng-1", role="user", content="deploy the server")

        result = json.loads(session_search(
            query="deploy", db=db, limit=3, fast=True, current_session_id=None
        ))
        assert result["success"]
        assert result.get("mode") == "fast"