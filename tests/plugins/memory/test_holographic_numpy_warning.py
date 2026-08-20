"""Tests for holographic memory numpy-degradation warning and capability reporting."""

from __future__ import annotations

import logging
from unittest.mock import patch, MagicMock, PropertyMock

import pytest


def _make_plugin_with_facts(count: int):
    """Create a minimal HolographicMemoryProvider with a mocked fact store."""
    from plugins.memory.holographic import HolographicMemoryProvider

    plugin = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
    plugin._db_path = ":memory:"
    plugin._auto_extract = False
    plugin._default_trust = 0.5
    plugin._hrr_dim = 1024

    mock_store = MagicMock()
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchone.return_value = (count,)
    mock_store._conn = mock_conn
    plugin._store = mock_store
    return plugin


class TestHrrNumpyWarning:
    """Verify that missing numpy triggers a warning and accurate capability description."""

    def test_retrieval_init_warns_when_numpy_missing(self, caplog):
        """FactRetriever.__init__ should log a warning when numpy is unavailable."""
        from plugins.memory.holographic.retrieval import FactRetriever
        from plugins.memory.holographic import holographic as hrr_mod

        mock_store = MagicMock()

        with patch.object(hrr_mod, "_HAS_NUMPY", False):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.holographic.retrieval"):
                retriever = FactRetriever(mock_store, hrr_weight=0.3)

        assert any("numpy not installed" in r.message for r in caplog.records)
        assert retriever.hrr_weight == 0.0
        assert retriever.fts_weight == 0.6
        assert retriever.jaccard_weight == 0.4

    def test_retrieval_init_no_warning_when_numpy_present(self, caplog):
        """FactRetriever.__init__ should NOT warn when numpy is available."""
        from plugins.memory.holographic.retrieval import FactRetriever
        from plugins.memory.holographic import holographic as hrr_mod

        mock_store = MagicMock()

        with patch.object(hrr_mod, "_HAS_NUMPY", True):
            with caplog.at_level(logging.WARNING, logger="plugins.memory.holographic.retrieval"):
                retriever = FactRetriever(mock_store, hrr_weight=0.3)

        assert not any("numpy not installed" in r.message for r in caplog.records)
        assert retriever.hrr_weight == 0.3

    def test_status_prompt_degraded_without_numpy(self):
        """Status prompt should mention keyword fallback when numpy is missing."""
        from plugins.memory.holographic import holographic as hrr_mod

        plugin = _make_plugin_with_facts(5)

        with patch.object(hrr_mod, "_HAS_NUMPY", False):
            status = plugin.system_prompt_block()

        assert "keyword fallback" in status
        assert "install numpy" in status.lower()
        assert "5 facts" in status

    def test_status_prompt_full_with_numpy(self):
        """Status prompt should describe algebraic capabilities when numpy is available."""
        from plugins.memory.holographic import holographic as hrr_mod

        plugin = _make_plugin_with_facts(10)

        with patch.object(hrr_mod, "_HAS_NUMPY", True):
            status = plugin.system_prompt_block()

        assert "algebraic" in status
        assert "keyword fallback" not in status
        assert "10 facts" in status

    def test_status_prompt_empty_store(self):
        """Status prompt for empty store should not mention numpy at all."""
        plugin = _make_plugin_with_facts(0)
        status = plugin.system_prompt_block()
        assert "Empty fact store" in status
        assert "numpy" not in status.lower()
