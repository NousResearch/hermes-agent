"""Tests for hybrid search with Reciprocal Rank Fusion."""

from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.tiered.search import (
    FTS_WEIGHT,
    RRF_K,
    VECTOR_WEIGHT,
    hybrid_search,
    reciprocal_rank_fusion,
)


# ---------------------------------------------------------------------------
# RRF unit tests (no DB needed)
# ---------------------------------------------------------------------------

class TestReciprocalRankFusion:
    def test_rrf_both_lists(self):
        """Entry in both FTS and vec ranks higher than entry in only one."""
        fts = [
            {"id": "a", "title": "A", "content": "A content", "category": "project"},
            {"id": "b", "title": "B", "content": "B content", "category": "project"},
        ]
        vec = [
            {"id": "a", "title": "A", "content": "A content", "category": "project"},
            {"id": "c", "title": "C", "content": "C content", "category": "project"},
        ]
        results = reciprocal_rank_fusion(fts, vec, limit=10)

        # "a" appears in both lists, so it should rank first
        assert results[0]["id"] == "a"
        # "a" should have higher score than "b" or "c"
        scores = {r["id"]: r["score"] for r in results}
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_rrf_fts_only(self):
        """Only FTS results, no vec."""
        fts = [
            {"id": "x", "title": "X", "content": "X content", "category": "project"},
            {"id": "y", "title": "Y", "content": "Y content", "category": "project"},
        ]
        results = reciprocal_rank_fusion(fts, [], limit=10)
        assert len(results) == 2
        assert results[0]["id"] == "x"  # rank 1 scores higher
        assert results[1]["id"] == "y"

    def test_rrf_vec_only(self):
        """Only vec results, no FTS."""
        vec = [
            {"id": "p", "title": "P", "content": "P content", "category": "project"},
            {"id": "q", "title": "Q", "content": "Q content", "category": "project"},
        ]
        results = reciprocal_rank_fusion([], vec, limit=10)
        assert len(results) == 2
        assert results[0]["id"] == "p"
        assert results[1]["id"] == "q"

    def test_rrf_limit(self):
        """Returns at most `limit` results."""
        fts = [{"id": str(i), "title": f"T{i}", "content": f"C{i}", "category": "project"} for i in range(10)]
        results = reciprocal_rank_fusion(fts, [], limit=3)
        assert len(results) == 3

    def test_rrf_empty(self):
        """Both empty lists returns empty."""
        results = reciprocal_rank_fusion([], [], limit=5)
        assert results == []

    def test_rrf_scoring(self):
        """Verify exact scores for known ranks."""
        fts = [{"id": "z", "title": "Z", "content": "Z", "category": "project"}]
        vec = [{"id": "z", "title": "Z", "content": "Z", "category": "project"}]

        results = reciprocal_rank_fusion(fts, vec, limit=5)
        assert len(results) == 1

        # FTS rank 1 + vec rank 1 = 0.4/(60+1) + 0.6/(60+1) = 1.0/61
        expected_score = FTS_WEIGHT / (RRF_K + 1) + VECTOR_WEIGHT / (RRF_K + 1)
        assert abs(results[0]["score"] - expected_score) < 1e-10
        assert abs(expected_score - 1.0 / 61) < 1e-10

    def test_rrf_preserves_entry_data(self):
        """RRF results preserve original entry fields."""
        fts = [{"id": "m", "title": "My Title", "content": "My Content", "category": "briefing", "extra": "fts_data"}]
        vec = []
        results = reciprocal_rank_fusion(fts, vec, limit=5)
        assert results[0]["title"] == "My Title"
        assert results[0]["content"] == "My Content"
        assert results[0]["category"] == "briefing"

    def test_rrf_fts_entry_preferred_on_conflict(self):
        """When entry exists in both, FTS entry data is used (entries.setdefault)."""
        fts = [{"id": "x", "title": "FTS Title", "content": "FTS Content", "category": "project"}]
        vec = [{"id": "x", "title": "Vec Title", "content": "Vec Content", "category": "project"}]
        results = reciprocal_rank_fusion(fts, vec, limit=5)
        # FTS is added first, so entries["x"] = fts entry; vec uses setdefault (no overwrite)
        assert results[0]["title"] == "FTS Title"


# ---------------------------------------------------------------------------
# hybrid_search tests (mocked DB)
# ---------------------------------------------------------------------------

class TestHybridSearch:
    def test_hybrid_search_with_embedding(self):
        """Mock generate_embedding to return a vector, verify both search paths called."""
        mock_db = MagicMock()
        fts_results = [
            {"id": "a", "title": "A", "content": "A content", "category": "project", "fts_rank": -1.0},
        ]
        vec_results = [
            {"id": "b", "title": "B", "content": "B content", "category": "project", "distance": 0.1},
        ]
        mock_db.search_fts.return_value = fts_results
        mock_db.search_vec.return_value = vec_results

        fake_embedding = [0.1] * 384
        with patch("plugins.memory.tiered.search.generate_embedding", return_value=fake_embedding):
            results = hybrid_search(mock_db, "test query", limit=5)

        mock_db.search_fts.assert_called_once_with("test query", limit=10, category=None)
        mock_db.search_vec.assert_called_once_with(fake_embedding, limit=10, category=None)
        assert len(results) == 2
        # Both entries should have a score
        for r in results:
            assert "score" in r

    def test_hybrid_search_fts_fallback(self):
        """Mock generate_embedding to return None, verify only FTS used."""
        mock_db = MagicMock()
        fts_results = [
            {"id": "a", "title": "A", "content": "A content", "category": "project", "fts_rank": -1.0},
            {"id": "b", "title": "B", "content": "B content", "category": "project", "fts_rank": -0.5},
        ]
        mock_db.search_fts.return_value = fts_results

        with patch("plugins.memory.tiered.search.generate_embedding", return_value=None):
            results = hybrid_search(mock_db, "test query", limit=5)

        mock_db.search_fts.assert_called_once()
        mock_db.search_vec.assert_not_called()
        assert len(results) == 2
        # FTS-only fallback uses 1/(i+1) scoring
        assert results[0]["score"] == 1.0
        assert results[1]["score"] == 0.5

    def test_hybrid_search_with_category(self):
        """Category filter is passed through to both search methods."""
        mock_db = MagicMock()
        mock_db.search_fts.return_value = []
        mock_db.search_vec.return_value = []

        fake_embedding = [0.1] * 384
        with patch("plugins.memory.tiered.search.generate_embedding", return_value=fake_embedding):
            hybrid_search(mock_db, "test", limit=3, category="briefing")

        mock_db.search_fts.assert_called_once_with("test", limit=6, category="briefing")
        mock_db.search_vec.assert_called_once_with(fake_embedding, limit=6, category="briefing")

    def test_hybrid_search_respects_limit(self):
        """Output is capped to requested limit even if inputs are larger."""
        mock_db = MagicMock()
        fts_results = [
            {"id": str(i), "title": f"T{i}", "content": f"C{i}", "category": "project"}
            for i in range(10)
        ]
        mock_db.search_fts.return_value = fts_results

        with patch("plugins.memory.tiered.search.generate_embedding", return_value=None):
            results = hybrid_search(mock_db, "query", limit=3)

        assert len(results) == 3
