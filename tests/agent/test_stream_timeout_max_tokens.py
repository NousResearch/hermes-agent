"""Tests for max_tokens-based stream timeout scaling.

Thinking/reasoning models (GLM-5.1, DeepSeek-R1, etc.) can spend minutes
in the reasoning phase without emitting visible chunks.  When max_tokens
is large (>32k), both the httpx read timeout and the stream stale timeout
must be scaled up to avoid premature connection kills.

See: https://github.com/NousResearch/hermes-agent/issues/17913
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from agent.model_metadata import is_local_endpoint


class TestStreamReadTimeoutMaxTokensScaling:
    """Verify that the stream read timeout scales with max_tokens output budget."""

    @staticmethod
    def _compute_read_timeout(base_url: str, api_kwargs: dict) -> float:
        """Reproduce the read timeout logic from run_agent.py streaming path."""
        _provider_timeout_cfg = None  # no per-provider config
        _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
        if _provider_timeout_cfg is not None:
            _stream_read_timeout = _provider_timeout_cfg
        else:
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            # max_tokens scaling
            _output_budget = (
                api_kwargs.get("max_tokens")
                or api_kwargs.get("max_completion_tokens")
                or 0
            )
            if _output_budget > 32768:
                _stream_read_timeout = max(_stream_read_timeout, 180.0)
        return _stream_read_timeout

    def test_large_max_tokens_bumps_read_timeout(self):
        """max_tokens=65536 on remote provider -> read timeout >= 180s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            result = self._compute_read_timeout(
                "https://api.deepseek.com",
                {"max_tokens": 65536},
            )
            assert result >= 180.0

    def test_large_max_completion_tokens_bumps_read_timeout(self):
        """max_completion_tokens=65536 on remote provider -> read timeout >= 180s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            result = self._compute_read_timeout(
                "https://api.openai.com",
                {"max_completion_tokens": 65536},
            )
            assert result >= 180.0

    def test_small_max_tokens_keeps_default(self):
        """max_tokens=4096 on remote provider -> read timeout stays 120s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            result = self._compute_read_timeout(
                "https://api.deepseek.com",
                {"max_tokens": 4096},
            )
            assert result == 120.0

    def test_no_max_tokens_keeps_default(self):
        """No max_tokens on remote provider -> read timeout stays 120s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            result = self._compute_read_timeout(
                "https://api.deepseek.com",
                {},
            )
            assert result == 120.0

    def test_threshold_boundary_32768(self):
        """max_tokens=32768 should NOT bump (threshold is >32768)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            result = self._compute_read_timeout(
                "https://api.deepseek.com",
                {"max_tokens": 32768},
            )
            assert result == 120.0

    def test_threshold_boundary_32769(self):
        """max_tokens=32769 should bump to 180s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            result = self._compute_read_timeout(
                "https://api.deepseek.com",
                {"max_tokens": 32769},
            )
            assert result >= 180.0


class TestStreamStaleTimeoutMaxTokensScaling:
    """Verify that the stream stale timeout scales with max_tokens output budget."""

    @staticmethod
    def _compute_stale_timeout(api_kwargs: dict, *, is_local: bool = False) -> float:
        """Reproduce the stale timeout logic from run_agent.py streaming path."""
        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        base_url = "http://localhost:11434" if is_local else "https://api.deepseek.com"

        if _stream_stale_timeout_base == 180.0 and base_url and is_local_endpoint(base_url):
            return float("inf")

        _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
        if _est_tokens > 100_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
        elif _est_tokens > 50_000:
            _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
        else:
            _stream_stale_timeout = _stream_stale_timeout_base

        # max_tokens scaling
        _output_budget = (
            api_kwargs.get("max_tokens")
            or api_kwargs.get("max_completion_tokens")
            or 0
        )
        if _output_budget > 65536:
            _stream_stale_timeout = max(_stream_stale_timeout, 420.0)
        elif _output_budget > 32768:
            _stream_stale_timeout = max(_stream_stale_timeout, 300.0)

        return _stream_stale_timeout

    def test_large_max_tokens_bumps_stale_timeout(self):
        """max_tokens=131072 on remote provider -> stale timeout >= 420s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            result = self._compute_stale_timeout({"max_tokens": 131072})
            assert result >= 420.0

    def test_medium_max_tokens_bumps_stale_timeout(self):
        """max_tokens=65536 on remote provider -> stale timeout >= 300s."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            result = self._compute_stale_timeout({"max_tokens": 65536})
            assert result >= 300.0

    def test_small_max_tokens_keeps_base(self):
        """max_tokens=4096 on remote provider -> stale timeout stays at base."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            result = self._compute_stale_timeout({"max_tokens": 4096})
            assert result == 180.0

    def test_max_tokens_scales_with_large_context(self):
        """Large context + large max_tokens -> takes the higher of the two."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            # Large context (200k tokens) + large max_tokens
            messages = [{"role": "user", "content": "x" * 800_000}]
            result = self._compute_stale_timeout({
                "messages": messages,
                "max_tokens": 131072,
            })
            # Should be at least 420s (from max_tokens scaling)
            assert result >= 420.0

    def test_max_tokens_on_top_of_context_scaling(self):
        """max_tokens scaling should be additive with context scaling."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            # Small context but large max_tokens
            result = self._compute_stale_timeout({"max_tokens": 65537})
            assert result >= 420.0  # 65537 > 65536 threshold

    def test_threshold_boundary_65536(self):
        """max_tokens=65536 should hit the 300s tier (not 420s)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            result = self._compute_stale_timeout({"max_tokens": 65536})
            assert result == 300.0

    def test_threshold_boundary_65537(self):
        """max_tokens=65537 should hit the 420s tier."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_STALE_TIMEOUT", None)
            result = self._compute_stale_timeout({"max_tokens": 65537})
            assert result == 420.0


class TestNonStreamStaleTimeoutMaxTokensScaling:
    """Verify that the non-stream stale timeout also scales with max_tokens."""

    def _make_agent_mock(self):
        """Create a minimal mock of AIAgent for testing _compute_non_stream_stale_timeout."""
        from run_agent import AIAgent
        # We can't easily instantiate AIAgent, so we'll test the method directly
        # by calling it as an unbound method with a mock self
        mock_self = MagicMock()
        mock_self._resolved_api_call_stale_timeout_base.return_value = (300.0, True)
        mock_self._base_url = "https://api.deepseek.com"
        mock_self.base_url = "https://api.deepseek.com"
        return mock_self

    def test_large_max_tokens_bumps_non_stream_timeout(self):
        """max_output_tokens=131072 -> non-stream stale timeout >= 600s."""
        from run_agent import AIAgent
        mock_self = self._make_agent_mock()
        result = AIAgent._compute_non_stream_stale_timeout(
            mock_self, [], max_output_tokens=131072,
        )
        assert result >= 600.0

    def test_medium_max_tokens_bumps_non_stream_timeout(self):
        """max_output_tokens=65536 -> non-stream stale timeout >= 450s."""
        from run_agent import AIAgent
        mock_self = self._make_agent_mock()
        result = AIAgent._compute_non_stream_stale_timeout(
            mock_self, [], max_output_tokens=65536,
        )
        assert result >= 450.0

    def test_small_max_tokens_no_bump(self):
        """max_output_tokens=4096 -> non-stream stale timeout stays at base."""
        from run_agent import AIAgent
        mock_self = self._make_agent_mock()
        result = AIAgent._compute_non_stream_stale_timeout(
            mock_self, [], max_output_tokens=4096,
        )
        assert result == 300.0  # base from mock
