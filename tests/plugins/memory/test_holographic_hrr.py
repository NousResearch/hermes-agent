"""Tests for holographic HRR core functions — dimension-mismatch safety.

Covers the fix for issue #68682: cross-session hrr_dim mismatches used to
raise a numpy broadcast ValueError deep inside similarity(), crashing the
whole retrieval pipeline. similarity() must now degrade gracefully.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("numpy")

from plugins.memory.holographic import holographic as hrr


class TestSimilarityDimensionGuard:
    def test_matching_dims_unaffected(self):
        a = hrr.encode_atom("hello", dim=256)
        b = hrr.encode_atom("hello", dim=256)
        assert hrr.similarity(a, b) == pytest.approx(1.0)

    def test_mismatched_dims_returns_neutral_zero(self):
        a = hrr.encode_atom("hello", dim=256)
        b = hrr.encode_atom("hello", dim=1024)
        assert hrr.similarity(a, b) == 0.0

    def test_mismatched_dims_does_not_raise(self):
        a = hrr.encode_atom("hello", dim=64)
        b = hrr.encode_atom("hello", dim=2048)
        # Must not raise ValueError from numpy broadcasting.
        result = hrr.similarity(a, b)
        assert isinstance(result, float)

    def test_mismatched_dims_logs_warning(self, caplog):
        a = hrr.encode_atom("hello", dim=256)
        b = hrr.encode_atom("hello", dim=1024)
        with caplog.at_level(logging.DEBUG, logger=hrr.logger.name):
            hrr.similarity(a, b)
        assert any("dim" in rec.message.lower() for rec in caplog.records)
