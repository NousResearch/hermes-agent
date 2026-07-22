"""Tests for holographic memory numpy degradation warning (#17350).

When numpy is missing, HRR operations silently fell back to FTS5-only.
No warning was logged and hermes doctor couldn't detect it.
"""

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.holographic import HolographicMemoryProvider


class TestNumpyWarning:
    """Holographic memory should warn when numpy is unavailable."""

    def test_system_prompt_with_numpy(self):
        """With numpy available, system prompt should not show warning."""
        provider = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
        provider._store = MagicMock()
        provider._hrr_available = True
        provider._store._conn.execute.return_value.fetchone.return_value = (5,)

        prompt = provider.system_prompt_block()
        assert "WARNING" not in prompt
        assert "5 facts" in prompt

    def test_system_prompt_without_numpy(self):
        """Without numpy, system prompt should show degradation warning."""
        provider = HolographicMemoryProvider.__new__(HolographicMemoryProvider)
        provider._store = MagicMock()
        provider._hrr_available = False
        provider._store._conn.execute.return_value.fetchone.return_value = (3,)

        prompt = provider.system_prompt_block()
        assert "WARNING" in prompt
        assert "numpy" in prompt
        assert "3 facts" in prompt

    def test_init_warns_without_numpy(self, caplog):
        """initialize() should log a warning when numpy is missing."""
        with tempfile.TemporaryDirectory() as tmp:
            provider = HolographicMemoryProvider(config={
                "db_path": os.path.join(tmp, "test.db"),
            })
            with patch("plugins.memory.holographic.holographic._HAS_NUMPY", False):
                with caplog.at_level(logging.WARNING, logger="plugins.memory.holographic"):
                    provider.initialize("test-session")
                assert hasattr(provider, "_hrr_available")
                assert not provider._hrr_available
                assert any("numpy not found" in r.message for r in caplog.records)
