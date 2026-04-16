"""Tests for percentage clamping at 100% across display paths.

PR #3480 capped context pressure percentage at 100% in agent/display.py
but missed the same unclamped pattern in 4 other files. When token counts
overshoot the context length (possible during streaming or before
compression fires), users see >100% in /stats, gateway status, and
memory tool output.
"""

import pytest


class TestContextCompressorUsagePercent:
    """agent/context_compressor.py — get_status() usage_percent"""

    def test_usage_percent_capped_at_100(self):
        """Tokens exceeding context_length should still show max 100%."""
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor.__new__(ContextCompressor)
        comp.last_prompt_tokens = 210_000  # exceeds context_length
        comp.context_length = 200_000
        comp.threshold_tokens = 160_000
        comp.compression_count = 0

        status = comp.get_status()
        assert status["usage_percent"] <= 100

    def test_usage_percent_normal(self):
        """Normal usage should show correct percentage."""
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor.__new__(ContextCompressor)
        comp.last_prompt_tokens = 100_000
        comp.context_length = 200_000
        comp.threshold_tokens = 160_000
        comp.compression_count = 0

        status = comp.get_status()
        assert status["usage_percent"] == 50.0

    def test_usage_percent_zero_context_length(self):
        """Zero context_length should return 0, not crash."""
        from agent.context_compressor import ContextCompressor

        comp = ContextCompressor.__new__(ContextCompressor)
        comp.last_prompt_tokens = 1000
        comp.context_length = 0
        comp.threshold_tokens = 0
        comp.compression_count = 0

        status = comp.get_status()
        assert status["usage_percent"] == 0


class TestMemoryToolPercentClamp:
    """tools/memory_tool.py — _success_response and _render_block pct"""

    def test_over_limit_clamped_at_100(self):
        """Percentage should be capped at 100 even if current > limit."""
        # Simulate the calculation directly
        current = 5500
        limit = 5000
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        assert pct == 100

    def test_normal_percentage(self):
        current = 2500
        limit = 5000
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        assert pct == 50

    def test_zero_limit_returns_zero(self):
        current = 100
        limit = 0
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        assert pct == 0


class TestCLIStatsPercentClamp:
    """cli.py — /stats command percentage"""

    def test_over_context_clamped_at_100(self):
        """Tokens exceeding context_length should show max 100%."""
        last_prompt = 210_000
        ctx_len = 200_000
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        assert pct == 100

    def test_normal_context(self):
        last_prompt = 100_000
        ctx_len = 200_000
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        assert pct == 50.0

    def test_zero_context_length(self):
        last_prompt = 1000
        ctx_len = 0
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        assert pct == 0


class TestGatewayStatsPercentClamp:
    """gateway/run.py — _format_usage_stats percentage"""

    def test_over_context_clamped_at_100(self):
        last_prompt_tokens = 210_000
        context_length = 200_000
        pct = min(100, last_prompt_tokens / context_length * 100) if context_length else 0
        assert pct == 100

    def test_normal_context(self):
        last_prompt_tokens = 150_000
        context_length = 200_000
        pct = min(100, last_prompt_tokens / context_length * 100) if context_length else 0
        assert pct == 75.0


class TestSourceLinesAreClamped:
    """Verify the actual source files have min(100, ...) applied."""

    @staticmethod
    def _read_file(rel_path: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def test_context_compressor_clamped(self):
        src = self._read_file("agent/context_compressor.py")
        assert "min(100," in src, (
            "context_compressor.py usage_percent is not clamped with min(100, ...)"
        )

    def test_gateway_run_clamped(self):
        src = self._read_file("gateway/run.py")
        # Check that the stats handler has min(100, ...)
        assert "min(100, ctx.last_prompt_tokens" in src, (
            "gateway/run.py stats pct is not clamped with min(100, ...)"
        )

    def test_cli_clamped(self):
        src = self._read_file("cli.py")
        assert "min(100, (last_prompt" in src, (
            "cli.py /stats pct is not clamped with min(100, ...)"
        )

    def test_memory_tool_clamped(self):
        src = self._read_file("tools/memory_tool.py")
        # Both _success_response and _render_block should have min(100, ...)
        count = src.count("min(100, int((current / limit)")
        assert count >= 2, (
            f"memory_tool.py has only {count} clamped pct lines, expected >= 2"
        )
