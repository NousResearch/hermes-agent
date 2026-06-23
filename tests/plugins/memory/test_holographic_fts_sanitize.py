"""Tests for the holographic memory provider's FTS5 query sanitizer.

These tests cover the root-cause fix for the silent-empty-results bug where
SQLite FTS5's default ``unicode61`` tokenizer treats ``-``, ``:``, ``(``,
``)``, and ``"`` as token separators / column-filter operators. The
original code passed the raw user query into a ``MATCH ?`` clause, which
raised ``OperationalError`` on hyphenated terms (e.g. ``hermes-agent``);
the retrieval layer's ``except Exception: return []`` then silently
swallowed the error and the agent lost all relevant memory.

The fix mirrors the reference implementation in
``hermes_state.SessionDB._sanitize_fts5_query`` (see upstream commit
``d1771114e`` which fixed the analogous colon-in-search bug):

  1. Preserve properly paired quoted phrases (``"exact phrase"``) verbatim.
  2. Replace remaining FTS5 special characters with a space (NOT delete)
     so token boundaries are kept.
  3. Collapse runs of whitespace.
  4. Strip a leading ``*`` (FTS5 requires at least one char before ``*``).
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Make the plugin importable without installing it as a package.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "memory" / "holographic"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plugins.memory.holographic.store import (  # noqa: E402
    MemoryStore,
    _sanitize_fts_query,
)


# ---------------------------------------------------------------------------
# Pure-function tests for _sanitize_fts_query
# ---------------------------------------------------------------------------


class TestSanitizeFtsQuery:
    """Direct unit tests of the sanitizer — no database required."""

    @pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
    def test_empty_or_whitespace_returns_empty_string(self, raw: str) -> None:
        """Empty input must short-circuit to '' so callers return [] cleanly."""
        assert _sanitize_fts_query(raw) == ""

    def test_none_input_returns_empty(self) -> None:
        """Defensive: ``None`` (or any falsy value) must not crash."""
        assert _sanitize_fts_query(None) == ""  # type: ignore[arg-type]

    def test_plain_hyphenated_word_becomes_safe_token(self) -> None:
        """The canonical bug trigger: ``hermes-agent`` must be rewritten so
        unicode61 does not split it on the hyphen."""
        # Hyphen is in the FTS5 special set and replaced with space, then
        # collapsed — output is the two safe tokens "hermes agent".
        assert _sanitize_fts_query("hermes-agent") == "hermes agent"

    def test_multi_word_with_spaces_kept(self) -> None:
        """Words separated by spaces stay separated."""
        assert _sanitize_fts_query("hello world") == "hello world"

    def test_already_quoted_phrase_preserved(self) -> None:
        """Caller-supplied phrase must survive unchanged (no re-wrapping)."""
        assert _sanitize_fts_query('"exact phrase"') == '"exact phrase"'

    def test_phrases_around_other_content(self) -> None:
        """A query mixing a phrase and surrounding bare tokens must keep the
        phrase intact while sanitizing the rest."""
        result = _sanitize_fts_query('"hermes-agent" OR plugin')
        assert '"hermes-agent"' in result
        assert "plugin" in result
        # Parens are FTS5-special — they become spaces.
        assert "(" not in result and ")" not in result

    def test_colon_replaced_with_space(self) -> None:
        """``foo:bar`` would parse as ``column:term`` and raise. The
        sanitizer must rewrite it to ``foo bar`` so FTS5 sees two safe
        tokens. (Same shape as upstream commit d1771114e for the search
        module's colon bug.)"""
        assert _sanitize_fts_query("foo:bar") == "foo bar"

    def test_colon_inside_phrase_preserved(self) -> None:
        """A colon inside a quoted phrase stays literal — the phrase is
        stashed before the strip pass and restored verbatim."""
        assert _sanitize_fts_query('"namespace:token"') == '"namespace:token"'

    def test_parens_become_spaces(self) -> None:
        """FTS5 expression parens are user-input hazards; replace with space."""
        result = _sanitize_fts_query("(foo OR bar)")
        assert "(" not in result and ")" not in result
        assert "foo" in result and "bar" in result

    def test_wildcard_inside_phrase_preserved(self) -> None:
        """A bare ``*`` outside a phrase is stripped (FTS5 requires a leading
        char), but inside a quoted phrase it stays as a literal ``*``."""
        assert _sanitize_fts_query('"agent*"') == '"agent*"'

    def test_leading_wildcard_stripped(self) -> None:
        """Leading ``*`` with nothing before it would error — strip it."""
        assert _sanitize_fts_query("*") == ""
        assert _sanitize_fts_query("**foo") == "foo"

    def test_unicode_kept_intact(self) -> None:
        """Non-ASCII content survives sanitization."""
        result = _sanitize_fts_query("中文-测试")
        assert result == "中文 测试"

    def test_underscore_word_kept(self) -> None:
        """Underscores are token chars in unicode61; bare words pass through."""
        assert _sanitize_fts_query("snake_case") == "snake_case"

    def test_path_like_input_becomes_tokens(self) -> None:
        """Real-world hazard: file paths. ``/path/to/foo.txt`` should
        become ``path to foo txt`` so FTS5 tokenizes it sensibly rather
        than choking on ``/`` (which is in the FTS5 special set)."""
        # Forward slash is in the FTS5-special set; result is space-joined.
        result = _sanitize_fts_query("/path/to/foo.txt")
        assert "/" not in result
        assert "path" in result and "to" in result and "foo" in result


# ---------------------------------------------------------------------------
# Bug reproduction: confirm the original OperationalError is gone
# ---------------------------------------------------------------------------


class TestFts5MatchRegression:
    """End-to-end test reproducing the bug against an in-memory SQLite FTS5
    table, then proving the sanitizer makes ``search_facts`` robust."""

    @pytest.fixture()
    def fts5_demo_db(self) -> sqlite3.Connection:
        """A bare-bones FTS5 table that mirrors the unicode61 tokenizer config
        used by ``holographic.MemoryStore``. We use this to demonstrate the
        raw error mode and the sanitized success mode side-by-side."""
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE VIRTUAL TABLE demo_fts USING fts5("
            "content, tokenize='unicode61 remove_diacritics 2'"
            ")"
        )
        con.execute(
            "INSERT INTO demo_fts(content) VALUES (?), (?), (?)",
            (
                "hermes-agent plugin handles FTS5",
                "we use hugging-face embeddings",
                "plain word without separators",
            ),
        )
        con.commit()
        return con

    def test_raw_hyphenated_query_raises(self, fts5_demo_db: sqlite3.Connection) -> None:
        """The original bug: raw ``hermes-agent`` against unicode61 raises."""
        with pytest.raises(sqlite3.OperationalError):
            fts5_demo_db.execute(
                "SELECT rowid FROM demo_fts WHERE demo_fts MATCH ?",
                ("hermes-agent",),
            ).fetchall()

    def test_sanitized_hyphenated_query_returns_match(
        self, fts5_demo_db: sqlite3.Connection
    ) -> None:
        """After sanitization the same query becomes ``hermes agent`` and
        matches the row that contains ``hermes-agent``."""
        safe = _sanitize_fts_query("hermes-agent")
        assert safe == "hermes agent"
        rows = fts5_demo_db.execute(
            "SELECT rowid FROM demo_fts WHERE demo_fts MATCH ?",
            (safe,),
        ).fetchall()
        assert len(rows) == 1

    def test_raw_colon_query_raises(self, fts5_demo_db: sqlite3.Connection) -> None:
        """Bare colon queries are parsed as ``column:term`` and raise."""
        fts5_demo_db.execute(
            "INSERT INTO demo_fts(content) VALUES (?)", ("namespace token",)
        )
        fts5_demo_db.commit()
        with pytest.raises(sqlite3.OperationalError):
            fts5_demo_db.execute(
                "SELECT rowid FROM demo_fts WHERE demo_fts MATCH ?",
                ("namespace:token",),
            ).fetchall()

    def test_sanitized_colon_query_returns_match(
        self, fts5_demo_db: sqlite3.Connection
    ) -> None:
        """Sanitization also fixes the colon-as-column-filter failure mode."""
        fts5_demo_db.execute(
            "INSERT INTO demo_fts(content) VALUES (?)", ("namespace:token",)
        )
        fts5_demo_db.commit()
        safe = _sanitize_fts_query("namespace:token")
        assert safe == "namespace token"
        rows = fts5_demo_db.execute(
            "SELECT rowid FROM demo_fts WHERE demo_fts MATCH ?",
            (safe,),
        ).fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Integration with MemoryStore.search_facts
# ---------------------------------------------------------------------------


class TestMemoryStoreSearchFactsIntegration:
    """Verify ``MemoryStore.search_facts`` uses the sanitizer and never raises
    on user queries that previously triggered the silent-failure mode."""

    @pytest.fixture()
    def store(self, tmp_path: Path) -> MemoryStore:
        """An on-disk MemoryStore backed by a tmp file so we get a real
        SQLite connection (MemoryStore opens its own connection)."""
        db = tmp_path / "holo.db"
        return MemoryStore(db_path=str(db))

    def test_search_facts_with_hyphenated_query_returns_results(
        self, store: MemoryStore
    ) -> None:
        """The original failure scenario: storing a fact whose content has a
        hyphen and then searching for the hyphenated term must succeed and
        return that fact (not [] silently)."""
        fact_id = store.add_fact(
            content="hermes-agent plugin handles FTS5 sanitize",
            category="technical",
        )
        assert fact_id > 0

        results = store.search_facts(query="hermes-agent", limit=5)
        assert len(results) >= 1
        assert any(r["fact_id"] == fact_id for r in results)

    def test_search_facts_empty_query_returns_empty_list(
        self, store: MemoryStore
    ) -> None:
        """Sanitizer returns ``""`` for whitespace-only input; store must
        short-circuit and return ``[]`` rather than hitting SQLite."""
        assert store.search_facts(query="   ", limit=5) == []
        assert store.search_facts(query="", limit=5) == []

    def test_search_facts_with_quoted_phrase(self, store: MemoryStore) -> None:
        """Phrase queries survive sanitization round-trip unchanged."""
        store.add_fact(content="we use phrase-aware queries sometimes", category="t")
        results = store.search_facts(query='"phrase-aware queries"', limit=5)
        assert len(results) >= 1

    def test_search_facts_with_colon_query(self, store: MemoryStore) -> None:
        """Colon queries — the upstream ``d1771114e`` analog — now find
        matches instead of silently returning []."""
        store.add_fact(content="TODO fix the deployment script", category="t")
        results = store.search_facts(query="TODO: fix", limit=5)
        assert len(results) >= 1