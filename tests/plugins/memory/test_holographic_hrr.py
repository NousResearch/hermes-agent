"""Tests for holographic HRR core functions — dimension-mismatch safety.

Covers the fix for issue #68682: cross-session hrr_dim mismatches used to
raise an opaque numpy broadcast ValueError deep inside similarity(), crashing
the whole retrieval pipeline. similarity() now raises a clear, actionable
ValueError instead — every pipeline call site filters mismatches at decode
time (retrieval._safe_phases) before reaching similarity(), so the primitive
keeps an exceptional contract rather than silently reporting "unrelated"
(0.0 would read as meaningful divergence in the contradiction-scoring math).
"""
from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import holographic as hrr


class TestSimilarityDimensionGuard:
    def test_matching_dims_unaffected(self):
        a = hrr.encode_atom("hello", dim=256)
        b = hrr.encode_atom("hello", dim=256)
        assert hrr.similarity(a, b) == pytest.approx(1.0)

    def test_mismatched_dims_raises_value_error(self):
        a = hrr.encode_atom("hello", dim=256)
        b = hrr.encode_atom("hello", dim=1024)
        with pytest.raises(ValueError, match="dimension mismatch"):
            hrr.similarity(a, b)

    def test_mismatched_dims_error_mentions_rebuild(self):
        a = hrr.encode_atom("hello", dim=64)
        b = hrr.encode_atom("hello", dim=2048)
        with pytest.raises(ValueError, match="rebuild_all_vectors"):
            hrr.similarity(a, b)
