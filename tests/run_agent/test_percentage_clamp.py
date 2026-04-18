"""Tests for percentage clamping at 100% across display paths.

PR #3480 capped context pressure percentage at 100% in agent/display.py
but missed the same unclamped pattern in 4 other files. When token counts
overshoot the context length (possible during streaming or before
compression fires), users see >100% in /stats, gateway status, and
memory tool output.
"""

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

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
        from tools.memory_tool import MemoryStore

        store = MemoryStore(memory_char_limit=10, user_char_limit=10)
        store.memory_entries = ["x" * 12]

        response = store._success_response("memory")

        assert response["usage"].startswith("100%")
        rendered = store._render_block("memory", store.memory_entries)
        assert "[100% — 12/10 chars]" in rendered

    def test_normal_percentage(self):
        from tools.memory_tool import MemoryStore

        store = MemoryStore(memory_char_limit=20, user_char_limit=20)
        store.memory_entries = ["x" * 10]

        response = store._success_response("memory")

        assert response["usage"].startswith("50%")

    def test_zero_limit_returns_zero(self):
        from tools.memory_tool import MemoryStore

        store = MemoryStore(memory_char_limit=0, user_char_limit=0)
        store.memory_entries = ["x" * 10]

        response = store._success_response("memory")

        assert response["usage"].startswith("0%")


class TestCLIStatsPercentClamp:
    """cli.py — /stats command percentage"""

    @staticmethod
    def _make_cli_with_agent(*, context_tokens: int, context_length: int):
        from cli import HermesCLI

        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.model = "local/test-model"
        cli_obj.session_start = datetime.now() - timedelta(minutes=5)
        cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
        cli_obj.verbose = False
        cli_obj.agent = SimpleNamespace(
            model="local/test-model",
            provider=None,
            base_url="",
            session_input_tokens=10,
            session_output_tokens=5,
            session_cache_read_tokens=0,
            session_cache_write_tokens=0,
            session_prompt_tokens=10,
            session_completion_tokens=5,
            session_total_tokens=15,
            session_api_calls=1,
            context_compressor=SimpleNamespace(
                last_prompt_tokens=context_tokens,
                context_length=context_length,
                compression_count=0,
            ),
        )
        return cli_obj

    def test_over_context_clamped_at_100(self, capsys):
        """Tokens exceeding context_length should show max 100%."""
        cli_obj = self._make_cli_with_agent(
            context_tokens=210_000,
            context_length=200_000,
        )

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Current context:  210,000 / 200,000 (100%)" in output

    def test_normal_context(self, capsys):
        cli_obj = self._make_cli_with_agent(
            context_tokens=100_000,
            context_length=200_000,
        )

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Current context:  100,000 / 200,000 (50%)" in output

    def test_zero_context_length(self, capsys):
        cli_obj = self._make_cli_with_agent(
            context_tokens=1_000,
            context_length=0,
        )

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Current context:  1,000 / 0 (0%)" in output


class TestGatewayStatsPercentClamp:
    """gateway/run.py — _format_usage_stats percentage"""

    @staticmethod
    def _make_runner_with_agent(*, context_tokens: int, context_length: int):
        from gateway.run import GatewayRunner

        runner = GatewayRunner.__new__(GatewayRunner)
        runner._session_key_for_source = lambda _source: "session-key"
        runner._running_agents = {
            "session-key": SimpleNamespace(
                session_total_tokens=15,
                session_api_calls=1,
                session_prompt_tokens=10,
                session_completion_tokens=5,
                context_compressor=SimpleNamespace(
                    last_prompt_tokens=context_tokens,
                    context_length=context_length,
                    compression_count=0,
                ),
            )
        }
        return runner

    def test_over_context_clamped_at_100(self):
        runner = self._make_runner_with_agent(
            context_tokens=210_000,
            context_length=200_000,
        )
        event = SimpleNamespace(source=SimpleNamespace())

        output = asyncio.run(runner._handle_usage_command(event))

        assert "Context: 210,000 / 200,000 (100%)" in output

    def test_normal_context(self):
        runner = self._make_runner_with_agent(
            context_tokens=150_000,
            context_length=200_000,
        )
        event = SimpleNamespace(source=SimpleNamespace())

        output = asyncio.run(runner._handle_usage_command(event))

        assert "Context: 150,000 / 200,000 (75%)" in output
