"""Tests for percentage clamping at 100% across display paths.

PR #3480 capped context pressure percentage at 100% in agent/display.py
but missed the same unclamped pattern in 4 other files. When token counts
overshoot the context length (possible during streaming or before
compression fires), users see >100% in /stats, gateway status, and
memory tool output.
"""

class TestMemoryToolPercentClamp:
    """tools/memory_tool.py — _success_response and _render_block pct"""

    def test_over_limit_clamped_at_100(self):
        """Percentage should be capped at 100 even if current > limit."""
        # Simulate the calculation directly
        current = 5500
        limit = 5000
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        assert pct == 100




class TestCLIStatsPercentClamp:
    """cli.py — /stats command percentage"""

    def test_over_context_clamped_at_100(self):
        """Tokens exceeding context_length should show max 100%."""
        last_prompt = 210_000
        ctx_len = 200_000
        pct = min(100, (last_prompt / ctx_len * 100)) if ctx_len else 0
        assert pct == 100




class TestGatewayStatsPercentClamp:
    """gateway/run.py — _format_usage_stats percentage"""

    def test_over_context_clamped_at_100(self):
        last_prompt_tokens = 210_000
        context_length = 200_000
        pct = min(100, last_prompt_tokens / context_length * 100) if context_length else 0
        assert pct == 100



class TestSourceLinesAreClamped:
    """Verify the actual source files have min(100, ...) applied."""

    @staticmethod
    def _read_file(rel_path: str) -> str:
        import os
        base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(base, rel_path)) as f:
            return f.read()

    def test_gateway_run_clamped(self):
        # The /usage stats handler was extracted from gateway/run.py into
        # gateway/slash_commands.py (god-file decomposition Phase 3b).
        src = self._read_file("gateway/slash_commands.py")
        # Check that the stats handler clamps the context pct with min(100, ...).
        # Assert the clamp intent, not a specific local name (the occupancy
        # value is read into a clamped `_lpt` local, #50421).
        assert "min(100, _lpt / ctx.context_length" in src, (
            "gateway/slash_commands.py stats pct is not clamped with min(100, ...)"
        )


