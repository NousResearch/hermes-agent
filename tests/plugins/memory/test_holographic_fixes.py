"""Tests for holographic memory FTS5 query sanitization (#14024, #21102).

Hyphenated tokens and special characters in FTS5 MATCH queries crash with
"no such column" errors. This verifies the _sanitize_fts_query() fix.
"""

import pytest

from plugins.memory.holographic.store import MemoryStore
from plugins.memory.holographic.retrieval import FactRetriever


@pytest.fixture
def store(tmp_path):
    """Create a fresh MemoryStore with a temp DB."""
    db = str(tmp_path / "test_memory.db")
    return MemoryStore(db_path=db, default_trust=0.5, hrr_dim=128)


@pytest.fixture
def retriever(store):
    """Create a FactRetriever for the temp store."""
    return FactRetriever(store=store, hrr_dim=128, hrr_weight=0.0)


# =========================================================================
# FTS5 query sanitization (#14024)
# =========================================================================

class TestFTS5Sanitization:
    """Hyphenated and special-char queries must not crash FTS5."""

    def test_hyphenated_query(self, store, retriever):
        """pve-01 should work without 'no such column' error."""
        store.add_fact("PVE-01 hardware: i5-13500T, IP 10.20.90.00", category="hardware", tags="pve-01")
        results = retriever.search("pve-01", min_trust=0.0)
        assert len(results) >= 1
        assert "PVE-01" in results[0]["content"]

    def test_hyphenated_hostnames(self, store, retriever):
        """Multiple hyphenated hostnames should all be findable."""
        store.add_fact("pihole-02 blocks ads on DNS", tags="pihole-02")
        store.add_fact("lxc-103 runs Ubuntu 24.04", tags="lxc-103")
        store.add_fact("gw-router handles VLAN trunking", tags="gw-router")

        assert len(retriever.search("pihole-02", min_trust=0.0)) >= 1
        assert len(retriever.search("lxc-103", min_trust=0.0)) >= 1
        assert len(retriever.search("gw-router", min_trust=0.0)) >= 1

    def test_special_fts5_operators(self, store, retriever):
        """Characters that are FTS5 operators: AND, OR, NOT, *, ^, :"""
        store.add_fact("C++ programming language is widely used")
        store.add_fact("search:elasticsearch for logging")
        store.add_fact("NOT a drill, this is real")

        # These should not crash
        results = retriever.search("C++", min_trust=0.0)
        assert isinstance(results, list)

        results = retriever.search("search:elasticsearch", min_trust=0.0)
        assert isinstance(results, list)

        results = retriever.search("NOT drill", min_trust=0.0)
        assert isinstance(results, list)

    def test_sanitize_fts_query_modes(self):
        """Test the _sanitize_fts_query static method directly."""
        from plugins.memory.holographic.retrieval import FactRetriever

        # default mode: tokens quoted
        assert FactRetriever._sanitize_fts_query("pve-01") == '"pve" "01"'

        # and mode: same as default
        assert FactRetriever._sanitize_fts_query("pve-01", mode="and") == '"pve" "01"'

        # or mode: OR-joined
        result = FactRetriever._sanitize_fts_query("pve-01", mode="or")
        assert "OR" in result
        assert '"pve"' in result
        assert '"01"' in result

    def test_empty_query(self, retriever):
        """Empty query should return empty results, not crash."""
        assert retriever.search("", min_trust=0.0) == []

    def test_query_only_hyphens(self, store, retriever):
        """Query that is only hyphens/punctuation should not crash."""
        store.add_fact("some regular fact content here")
        results = retriever.search("---", min_trust=0.0)
        assert isinstance(results, list)  # may be empty, just don't crash


# =========================================================================
# FTS5 query fuzzing
# =========================================================================

class TestFTS5Fuzzing:
    """Fuzz the FTS5 query path with adversarial inputs."""

    @pytest.mark.parametrize("query", [
        "'; DROP TABLE facts; --",
        "null\x00byte",
        "UPPER CASE QUERY",
        "   lots   of   spaces   ",
        "emoji 🎉 unicode café résumé",
        "a",
        "x" * 500,
        "1+1=2",
        "field:value",
        '"already quoted"',
        "'single quoted'",
        "fact AND fiction OR reality NOT dream",
        "term*",
        "^boosted",
        "NEAR/3 term",
        "pve-01 lxc-103",
        "10.20.90.00",
        "path/to/file.py",
        "user@domain.com",
        "a-b-c-d-e",
        "!@#$%^&*()",
    ])
    def test_fuzz_queries(self, store, retriever, query):
        """None of these queries should crash the search."""
        store.add_fact("Regular fact about Python and Rust programming languages")
        store.add_fact("PVE-01 server at 10.20.90.00 running Proxmox")
        results = retriever.search(query, min_trust=0.0)
        assert isinstance(results, list)
