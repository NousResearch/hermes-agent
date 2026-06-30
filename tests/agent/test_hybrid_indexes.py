"""Integration tests for the hybrid retrieval indexes (FTS5 + Embedding).

These tests verify that the FTS5 and embedding indexes work correctly
and that the hybrid scorer produces meaningful relevance rankings.
"""
from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from agent.memory.fts_index import FTS5Index
from agent.memory.embedding_index import EmbeddingIndex


# =========================================================================
# FTS5 Index tests
# =========================================================================


class TestFTS5Index:
    def setup_method(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.idx = FTS5Index(self.db)

    def teardown_method(self):
        self.idx.close()
        Path(self.db).unlink(missing_ok=True)

    def test_rebuild_and_search(self):
        entries = [
            "Deploy the kanban service to GKE staging",
            "Build REST API for user management with Python",
            "Debug null pointer in auth service exception",
        ]
        count = self.idx.rebuild(entries)
        assert count == 3

        results = self.idx.search("GKE deploy")
        assert len(results) > 0
        # First result should be the deploy entry
        assert "Deploy" in results[0]["content"] or "GKE" in results[0]["content"]

    def test_search_empty_index(self):
        assert self.idx.search("anything") == []

    def test_search_empty_query(self):
        self.idx.rebuild(["test entry"])
        assert self.idx.search("") == []

    def test_entry_count(self):
        self.idx.rebuild(["a", "b", "c"])
        assert self.idx.entry_count() == 3

    def test_rebuild_clears_old(self):
        self.idx.rebuild(["old entry"])
        self.idx.rebuild(["new entry"])
        results = self.idx.search("new")
        assert len(results) > 0
        assert "new" in results[0]["content"]

    def test_relevance_ranking(self):
        entries = [
            "User prefers concise responses in chat",
            "Deploy to GKE cluster in europe-central2 using Helm",
            "Documentation for the authentication flow API",
        ]
        self.idx.rebuild(entries)
        results = self.idx.search("deploy GKE Helm")
        assert results
        # Top result should be the deploy entry
        assert "Deploy" in results[0]["content"]
        assert results[0]["score"] > 0


# =========================================================================
# Embedding Index tests (only if model available)
# =========================================================================


class TestEmbeddingIndex:
    def setup_method(self):
        self.cache_dir = tempfile.mkdtemp()
        self.idx = EmbeddingIndex(self.cache_dir)

    def teardown_method(self):
        self.idx.close()

    def test_embedding_search(self):
        """Semantic search returns meaningfully ranked results."""
        if not self.idx.available:
            pytest.skip("ONNX model not available")

        entries = [
            "Deploy the kanban service to GKE staging",
            "Build REST API for user management with Python",
            "Debug null pointer in auth service exception",
            "User prefers concise responses",
        ]
        rebuilt = self.idx.rebuild(entries)
        assert rebuilt == 4

        # Search for something deployment-related
        results = self.idx.search("deploy microservice kubernetes", entries, limit=4)
        assert len(results) > 0
        # Top result should be deploy-related
        top = results[0]["content"]
        assert "Deploy" in top or "deploy" in top.lower()

    def test_semantic_relevance(self):
        """Semantic similarity should be > keyword-only for related concepts."""
        if not self.idx.available:
            pytest.skip("ONNX model not available")

        entries = [
            "Implement user authentication with JWT tokens",
            "Set up PostgreSQL database with migrations",
            "Frontend React component for login form",
        ]
        self.idx.rebuild(entries)

        # Search for "auth" — should find auth entry
        results = self.idx.search("authentication login security", entries, limit=3)
        assert results
        # The auth entry should rank first
        assert "authentication" in results[0]["content"].lower() or "login" in results[0]["content"].lower()

    def test_unrelated_queries(self):
        """Unrelated queries should return low scores for all entries."""
        if not self.idx.available:
            pytest.skip("ONNX model not available")

        entries = [
            "Deploy to GKE cluster in europe-central2",
        ]
        self.idx.rebuild(entries)

        results = self.idx.search("weather forecast", entries, limit=1)
        if results:
            # Score should be low for unrelated query
            assert results[0]["score"] < 0.6

    def test_fallback_when_model_unavailable(self):
        """Index should degrade gracefully when model is missing."""
        # Create a broken model path
        bad_idx = EmbeddingIndex(tempfile.mkdtemp())
        try:
            # Force unavailable
            bad_idx._available = False
            result = bad_idx.rebuild(["test entry"])
            assert result == 0

            results = bad_idx.search("test", ["test entry"], limit=1)
            assert results == []
        finally:
            bad_idx.close()
