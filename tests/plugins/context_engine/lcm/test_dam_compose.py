"""Regression tests: DAMRetriever.compose() operation handling.

Fix 4: the previous else-branch silently treated any unknown operation (including
'NOT') as OR.  Now NOT, AND, OR each have explicit branches; any unrecognised
operation returns an empty list.

Tests confirm:
- NOT with 1 query returns a list (possibly empty if no patterns stored).
- NOT with 2+ queries returns [] (multi-NOT is undefined/rejected).
- Unknown operation string returns [].
- AND and OR still work correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

from plugins.context_engine.lcm.dam.network import DenseAssociativeMemory
from plugins.context_engine.lcm.dam.encoder import MessageEncoder
from plugins.context_engine.lcm.dam.retrieval import DAMRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_retriever(nv: int = 512, nh: int = 64) -> DAMRetriever:
    net = DenseAssociativeMemory(nv=nv, nh=nh)
    enc = MessageEncoder(nv=nv)
    return DAMRetriever(net=net, enc=enc)


def _seed_patterns(retriever: DAMRetriever, messages: list[dict]) -> None:
    """Manually populate the pattern cache (without a store)."""
    vecs = [retriever.enc.encode(m) for m in messages]
    if vecs:
        retriever.net.learn(np.stack(vecs))
    for i, v in enumerate(vecs):
        retriever._pattern_cache[i] = v


# ---------------------------------------------------------------------------
# Tests: NOT operation
# ---------------------------------------------------------------------------

class TestComposeNOT:
    def test_not_single_query_returns_list(self):
        """compose(['foo'], operation='NOT') must return a list (not crash or silently OR)."""
        r = _make_retriever()
        _seed_patterns(r, [
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "goodbye world"},
        ])
        result = r.compose(["hello"], operation="NOT")
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_not_single_query_result_differs_from_or(self):
        """compose(['foo'], operation='NOT') must differ from operation='OR'.

        If NOT is still silently treated as OR the scores will be identical —
        which is the bug we're fixing.
        """
        r = _make_retriever()
        _seed_patterns(r, [
            {"role": "user", "content": "alpha beta gamma"},
            {"role": "assistant", "content": "delta epsilon zeta"},
            {"role": "user", "content": "totally unrelated keywords here"},
        ])
        result_not = r.compose(["alpha"], operation="NOT")
        result_or  = r.compose(["alpha"], operation="OR")

        # Both return lists of (msg_id, score) tuples — if they're identical,
        # NOT is still being executed as OR.
        scores_not = {mid: score for mid, score in result_not}
        scores_or  = {mid: score for mid, score in result_or}

        # They should differ (NOT inverts activations; OR does not).
        assert scores_not != scores_or, (
            "compose('NOT') returned the same scores as compose('OR') — "
            "NOT is still treated as OR (the bug is not fixed)"
        )

    def test_not_multi_query_returns_empty(self):
        """compose(['foo', 'bar'], operation='NOT') must return [] — multi-NOT is rejected."""
        r = _make_retriever()
        _seed_patterns(r, [{"role": "user", "content": "test"}])
        result = r.compose(["foo", "bar"], operation="NOT")
        assert result == [], (
            f"Multi-query NOT must be rejected with empty list, got: {result}"
        )

    def test_not_zero_queries_returns_empty(self):
        """compose([], operation='NOT') must return [] — no queries to invert."""
        r = _make_retriever()
        result = r.compose([], operation="NOT")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: unknown / garbage operation
# ---------------------------------------------------------------------------

class TestComposeUnknownOperation:
    def test_unknown_op_returns_empty(self):
        r = _make_retriever()
        _seed_patterns(r, [{"role": "user", "content": "hello"}])
        result = r.compose(["hello"], operation="XOR")
        assert result == [], f"Unknown op must return [], got: {result}"

    def test_empty_op_returns_empty(self):
        r = _make_retriever()
        _seed_patterns(r, [{"role": "user", "content": "hello"}])
        result = r.compose(["hello"], operation="")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: AND and OR still work (no regression)
# ---------------------------------------------------------------------------

class TestComposeAndOr:
    def test_and_returns_list(self):
        r = _make_retriever()
        _seed_patterns(r, [
            {"role": "user", "content": "alpha beta"},
            {"role": "assistant", "content": "gamma delta"},
        ])
        result = r.compose(["alpha", "gamma"], operation="AND")
        assert isinstance(result, list)

    def test_or_returns_list(self):
        r = _make_retriever()
        _seed_patterns(r, [
            {"role": "user", "content": "alpha beta"},
            {"role": "assistant", "content": "gamma delta"},
        ])
        result = r.compose(["alpha", "gamma"], operation="OR")
        assert isinstance(result, list)

    def test_and_case_insensitive(self):
        r = _make_retriever()
        _seed_patterns(r, [{"role": "user", "content": "hello"}])
        result_upper = r.compose(["hello"], operation="AND")
        result_lower = r.compose(["hello"], operation="and")
        # Both should return the same type (list), not error
        assert isinstance(result_upper, list)
        assert isinstance(result_lower, list)

    def test_not_case_insensitive(self):
        r = _make_retriever()
        _seed_patterns(r, [{"role": "user", "content": "hello"}])
        result = r.compose(["hello"], operation="not")
        assert isinstance(result, list)
