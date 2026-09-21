"""Tests for idea_box.dedup — DedupChecker against kanban, session_search, mnemosyne.

All backends are mocked to keep tests hermetic.  Verifies:
  - Each source is queried independently
  - Similarity threshold filters correctly
  - Match aggregation produces correct DedupResult
  - Missing/unavailable backends are handled gracefully
  - The Jaccard similarity helper behaves correctly
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from idea_box.dedup import (
    DedupChecker,
    DedupMatch,
    DedupSource,
    _jaccard_similarity,
    _tokenize,
)
from idea_box.models import DedupResult


def _recent_iso_date(days_ago: int = 5) -> str:
    """Return an ISO date a fixed number of days in the past.

    Keeps mock session ``when`` fields inside the dedup 30-day window so
    the tests don't rot as real time advances (the ``when`` field must be
    recent for ``_check_session_search`` to include it).
    """
    from datetime import datetime, timezone, timedelta

    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).date().isoformat()


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert _jaccard_similarity("build a feature", "build a feature") == 1.0

    def test_completely_different(self):
        assert _jaccard_similarity("alpha", "omega") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity("build feature for dashboard", "build feature for ui")
        assert 0.0 < sim < 1.0
        # Common words: "build", "feature", "for" (all > 2 chars, all pass filter)
        # "ui" is 2 chars, filtered by _tokenize
        # tokens a = {build, feature, for, dashboard}, b = {build, feature, for}
        # intersection = {build, feature, for} = 3, union = {build, feature, for, dashboard} = 4
        # sim = 3/4 = 0.75
        assert abs(sim - 0.75) < 0.01

    def test_empty_string(self):
        assert _jaccard_similarity("", "test") == 0.0

    def test_both_empty(self):
        assert _jaccard_similarity("", "") == 0.0

    def test_case_insensitive(self):
        assert _jaccard_similarity("Build Feature", "build feature") == 1.0

    def test_short_words_filtered(self):
        # Words of length <= 2 are filtered by _tokenize
        assert _tokenize("a bc def") == {"def"}
        # Words of length <= 2: "a" (1 char), "bc" (2 chars) filtered


# ---------------------------------------------------------------------------
# DedupChecker with mocked backends
# ---------------------------------------------------------------------------

class TestDedupCheckerKanban:
    def test_kanban_match_found(self):
        """When kanban has a similar task title, it should be a match."""
        mock_task = MagicMock()
        mock_task.id = "t_001"
        mock_task.title = "Build dashboard feature for analytics"
        mock_task.body = "Create a dashboard with charts"
        mock_task.status = "todo"

        mock_conn = MagicMock()
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[mock_task]):
            checker = DedupChecker(kanban_conn=mock_conn)
            result = checker.check("Build dashboard feature")
        assert result.is_duplicate is True
        assert any(m["source"] == "kanban" for m in result.matches)
        assert "kanban" in result.checked_sources

    def test_kanban_no_match(self):
        """When kanban has no similar tasks, no match."""
        mock_task = MagicMock()
        mock_task.id = "t_002"
        mock_task.title = "Completely different task"
        mock_task.body = "Unrelated work"
        mock_task.status = "todo"

        mock_conn = MagicMock()
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[mock_task]):
            checker = DedupChecker(kanban_conn=mock_conn)
            result = checker.check("Build dashboard feature")
        # "Build dashboard feature" vs "Completely different task" — very low overlap
        # Kanban is checked but no match; other sources will also be empty (no mocks)
        # The result depends on whether session_search/mnemosyne produce matches.
        # With no mocks for those, they'll return empty lists.
        kanban_matches = [m for m in result.matches if m["source"] == "kanban"]
        assert len(kanban_matches) == 0

    def test_kanban_skips_done_archived(self):
        """Done and archived tasks should be skipped during dedup."""
        done_task = MagicMock()
        done_task.id = "t_done"
        done_task.title = "Build dashboard feature for analytics"
        done_task.body = ""
        done_task.status = "done"

        archived_task = MagicMock()
        archived_task.id = "t_archived"
        archived_task.title = "Build dashboard features more"
        archived_task.body = ""
        archived_task.status = "archived"

        mock_conn = MagicMock()
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[done_task, archived_task]):
            checker = DedupChecker(kanban_conn=mock_conn)
            result = checker.check("Build dashboard features for analytics")
        kanban_matches = [m for m in result.matches if m["source"] == "kanban"]
        assert len(kanban_matches) == 0

    def test_kanban_body_similarity(self):
        """When the title is different but body is similar, should match."""
        mock_task = MagicMock()
        mock_task.id = "t_003"
        mock_task.title = "Random title"
        mock_task.body = "Build dashboard feature for analytics platform"
        mock_task.status = "running"

        mock_conn = MagicMock()
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[mock_task]):
            checker = DedupChecker(kanban_conn=mock_conn)
            result = checker.check("Build dashboard feature for analytics")
        kanban_matches = [m for m in result.matches if m["source"] == "kanban"]
        assert len(kanban_matches) >= 1
        assert kanban_matches[0]["ref_id"] == "t_003"


class TestDedupCheckerSessionSearch:
    def test_session_search_match(self):
        """When session_search returns a similar session, it's a match."""
        mock_search_fn = MagicMock(return_value=json.dumps({
            "success": True,
            "mode": "discover",
            "query": "test",
            "results": [
                {
                    "session_id": "s_001",
                    "title": "Build dashboard feature",
                    "snippet": "Discussed building a dashboard feature for analytics",
                    "when": _recent_iso_date(),
                }
            ],
            "count": 1,
            "sessions_searched": 10
        }))
        checker = DedupChecker(
            kanban_conn=MagicMock(),  # empty kanban
            session_search_fn=mock_search_fn,
        )
        # Mock kanban to return no tasks
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature for analytics")
        session_matches = [m for m in result.matches if m["source"] == "session_search"]
        assert len(session_matches) >= 1
        assert session_matches[0]["ref_id"] == "s_001"

    def test_session_search_no_match(self):
        """When session_search returns unrelated sessions, no match."""
        mock_search_fn = MagicMock(return_value=json.dumps({
            "success": True,
            "results": [
                {
                    "session_id": "s_002",
                    "title": "Unrelated topic",
                    "snippet": "Discussed something completely different",
                    "when": _recent_iso_date(),
                }
            ],
            "count": 1,
        }))
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=mock_search_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature for analytics")
        session_matches = [m for m in result.matches if m["source"] == "session_search"]
        assert len(session_matches) == 0

    def test_session_search_unavailable(self):
        """When session_search_fn raises, dedup still runs for other sources."""
        mock_search_fn = MagicMock(side_effect=Exception("DB unavailable"))
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=mock_search_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature")
        # session_search failed but the result should still have checked_sources
        assert "session_search" in result.checked_sources
        # No crash, no session matches
        session_matches = [m for m in result.matches if m["source"] == "session_search"]
        assert len(session_matches) == 0

    def test_session_search_30_day_window_filters_old_sessions(self):
        """Sessions older than 30 days should be excluded from dedup."""
        from datetime import datetime, timezone, timedelta
        old_date = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        mock_search_fn = MagicMock(return_value=json.dumps({
            "success": True,
            "results": [
                {
                    "session_id": "s_old",
                    "title": "Build dashboard feature for analytics",
                    "snippet": "Build dashboard feature for analytics discussion",
                    "when": old_date,
                }
            ],
            "count": 1,
        }))
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=mock_search_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature for analytics")
        session_matches = [m for m in result.matches if m["source"] == "session_search"]
        assert len(session_matches) == 0, "Old session should be filtered by 30-day window"

    def test_session_search_recent_session_included(self):
        """Sessions within the last 30 days should be included in dedup."""
        from datetime import datetime, timezone, timedelta
        recent_date = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        mock_search_fn = MagicMock(return_value=json.dumps({
            "success": True,
            "results": [
                {
                    "session_id": "s_recent",
                    "title": "Build dashboard feature for analytics",
                    "snippet": "Build dashboard feature for analytics discussion",
                    "when": recent_date,
                }
            ],
            "count": 1,
        }))
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=mock_search_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature for analytics")
        session_matches = [m for m in result.matches if m["source"] == "session_search"]
        assert len(session_matches) >= 1, "Recent session should not be filtered"


class TestDedupCheckerMnemosyne:
    def test_mnemosyne_match(self):
        """When mnemosyne returns a similar memory, it's a match."""
        mock_recall_fn = MagicMock(return_value={
            "memories": [
                {
                    "id": "mem_001",
                    "content": "Build dashboard feature for analytics platform",
                    "importance": 0.8,
                }
            ]
        })
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=MagicMock(return_value='{"results": []}'),
            mnemosyne_recall_fn=mock_recall_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature for analytics")
        mnemosyne_matches = [m for m in result.matches if m["source"] == "mnemosyne"]
        assert len(mnemosyne_matches) >= 1
        assert mnemosyne_matches[0]["ref_id"] == "mem_001"

    def test_mnemosyne_no_match(self):
        """When mnemosyne returns unrelated memories, no match."""
        mock_recall_fn = MagicMock(return_value={
            "memories": [
                {
                    "id": "mem_002",
                    "content": "Completely unrelated topic about cooking",
                    "importance": 0.5,
                }
            ]
        })
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=MagicMock(return_value='{"results": []}'),
            mnemosyne_recall_fn=mock_recall_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature for analytics")
        mnemosyne_matches = [m for m in result.matches if m["source"] == "mnemosyne"]
        assert len(mnemosyne_matches) == 0

    def test_mnemosyne_unavailable(self):
        """When mnemosyne_recall_fn raises, dedup still runs."""
        mock_recall_fn = MagicMock(side_effect=Exception("Mnemosyne offline"))
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=MagicMock(return_value='{"results": []}'),
            mnemosyne_recall_fn=mock_recall_fn,
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("Build dashboard feature")
        assert "mnemosyne" in result.checked_sources


class TestDedupCheckerAggregation:
    def test_all_sources_checked(self):
        """All three sources should appear in checked_sources even if no matches."""
        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=MagicMock(return_value='{"results": []}'),
            mnemosyne_recall_fn=MagicMock(return_value='{"memories": []}'),
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[]):
            result = checker.check("some unique idea")
        assert "kanban" in result.checked_sources
        assert "session_search" in result.checked_sources
        assert "mnemosyne" in result.checked_sources
        assert result.is_duplicate is False

    def test_multiple_source_matches(self):
        """When multiple sources find matches, all should be in result."""
        mock_task = MagicMock()
        mock_task.id = "t_001"
        mock_task.title = "Build dashboard feature"
        mock_task.body = ""
        mock_task.status = "todo"

        checker = DedupChecker(
            kanban_conn=MagicMock(),
            session_search_fn=MagicMock(return_value=json.dumps({
                "success": True,
                "results": [
                    {
                        "session_id": "s_001",
                        "title": "Build dashboard feature",
                        "snippet": "Build dashboard feature discussion",
                        "when": _recent_iso_date(),
                    }
                ],
                "count": 1,
            })),
            mnemosyne_recall_fn=MagicMock(return_value={
                "memories": [
                    {
                        "id": "mem_001",
                        "content": "Build dashboard feature for analytics",
                        "importance": 0.8,
                    }
                ]
            }),
        )
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[mock_task]):
            result = checker.check("Build dashboard feature for analytics")
        assert result.is_duplicate is True
        sources_in_matches = {m["source"] for m in result.matches}
        assert "kanban" in sources_in_matches
        assert "session_search" in sources_in_matches
        assert "mnemosyne" in sources_in_matches

    def test_custom_threshold(self):
        """A higher threshold should reject more matches."""
        mock_task = MagicMock()
        mock_task.id = "t_001"
        mock_task.title = "Build dashboard feature for analytics platform visualization"
        mock_task.body = ""
        mock_task.status = "todo"

        mock_conn = MagicMock()

        # With default threshold (0.35), this should match
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[mock_task]):
            checker_low = DedupChecker(kanban_conn=mock_conn, similarity_threshold=0.35)
            result_low = checker_low.check("Build dashboard feature for analytics")

        # With very high threshold (0.99), this should not match
        with patch("hermes_cli.kanban_db.list_tasks", return_value=[mock_task]):
            checker_high = DedupChecker(kanban_conn=mock_conn, similarity_threshold=0.99)
            result_high = checker_high.check("Build dashboard feature for analytics")

        kanban_low = [m for m in result_low.matches if m["source"] == "kanban"]
        kanban_high = [m for m in result_high.matches if m["source"] == "kanban"]
        assert len(kanban_low) >= 1
        assert len(kanban_high) == 0


class TestDedupMatchToDict:
    def test_to_dict(self):
        m = DedupMatch(
            source=DedupSource.KANBAN,
            title="Test task",
            summary="Test summary",
            ref_id="t_001",
            score=0.75,
            extra={"status": "todo"},
        )
        d = m.to_dict()
        assert d["source"] == "kanban"
        assert d["title"] == "Test task"
        assert d["summary"] == "Test summary"
        assert d["ref_id"] == "t_001"
        assert d["score"] == 0.75
        assert d["extra"] == {"status": "todo"}