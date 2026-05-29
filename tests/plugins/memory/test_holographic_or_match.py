"""Regression tests for holographic FactRetriever FTS5 OR-match behavior.

Bug: ``_fts_candidates`` passed the raw query straight to FTS5 MATCH, which
ANDs bare terms by default. Natural-language recall queries (the kind
auto-recall feeds every turn) therefore required *every* token to appear in a
single fact and almost always returned nothing — silently disabling recall.

Fix: ``_build_or_match`` lowercases, strips punctuation, drops FTS5 operator
keywords, double-quotes each token, and OR-joins them so FTS5 casts a wide net;
the downstream Jaccard + HRR + trust scoring then ranks the candidates.
"""

import os
import sys

import pytest

PLUGIN_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "plugins", "memory", "holographic"
)
PLUGIN_DIR = os.path.abspath(PLUGIN_DIR)
if PLUGIN_DIR not in sys.path:
    sys.path.insert(0, PLUGIN_DIR)

store_mod = pytest.importorskip("store")
ret_mod = pytest.importorskip("retrieval")


def _make_retriever(tmp_path):
    db = str(tmp_path / "mem.db")
    st = store_mod.MemoryStore(db)
    st.add_fact(
        "Matvii prefers Verdict-Evidence-Next structure and compact Telegram replies.",
        category="user_pref",
        tags="communication,format",
    )
    st.add_fact(
        "Matvii's tools: Xcode, VS Code, Claude Code, Hermes, Slack, Jira.",
        category="tool",
        tags="tools",
    )
    st.add_fact(
        "Update Hermes via the hermes-fork-update skill, not blind hermes update.",
        category="tool",
        tags="hermes,config",
    )
    return ret_mod.FactRetriever(st)


class TestBuildOrMatch:
    def test_multiword_joined_with_or(self):
        out = ret_mod.FactRetriever._build_or_match("communication preferences telegram format")
        assert " OR " in out
        assert out.count('"') == 8  # four quoted tokens

    def test_operator_keywords_dropped(self):
        out = ret_mod.FactRetriever._build_or_match("tools and hermes or vault")
        assert "and" not in out.lower().replace('"', "").split()
        assert '"tools"' in out and '"hermes"' in out and '"vault"' in out

    def test_punctuation_stripped_no_injection(self):
        # A raw query with FTS5 syntax chars must not raise / inject operators.
        out = ret_mod.FactRetriever._build_or_match('vault* (path) "quote"')
        assert "*" not in out
        assert "(" not in out and ")" not in out

    def test_empty_query_returns_empty(self):
        assert ret_mod.FactRetriever._build_or_match("") == ""
        assert ret_mod.FactRetriever._build_or_match("   ") == ""


class TestRecallSearch:
    def test_multiword_query_returns_hits(self, tmp_path):
        r = _make_retriever(tmp_path)
        res = r.search("communication preferences telegram format", min_trust=0.3, limit=3)
        assert res, "multi-word recall query must return candidates"
        assert "Telegram" in res[0]["content"]

    def test_natural_language_question(self, tmp_path):
        r = _make_retriever(tmp_path)
        res = r.search("how should I update hermes", min_trust=0.3, limit=3)
        assert res
        assert "hermes-fork-update" in res[0]["content"]

    def test_irrelevant_query_returns_nothing(self, tmp_path):
        r = _make_retriever(tmp_path)
        res = r.search("totally unrelated quantum banana spaceship", min_trust=0.3, limit=3)
        assert res == [], "irrelevant query must inject zero context (no bloat)"
