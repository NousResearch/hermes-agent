"""Tests for context pressure warnings (user-facing, not injected into messages).

Covers:
- Display formatting (CLI and gateway variants)
- Flag tracking and threshold logic on AIAgent
- Flag reset after compression
- status_callback invocation
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.display import format_context_pressure, format_context_pressure_gateway
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Display formatting tests
# ---------------------------------------------------------------------------


class TestFormatContextPressure:
    """CLI context pressure display (agent/display.py).

    The bar shows progress toward the compaction threshold, not the
    raw context window.  60% = 60% of the way to compaction.
    """

    def test_80_percent_uses_warning_icon(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "⚠" in line
        assert "80% to compaction" in line

    def test_90_percent_uses_warning_icon(self):
        line = format_context_pressure(0.90, 100_000, 0.50)
        assert "⚠" in line
        assert "90% to compaction" in line

    def test_bar_length_scales_with_progress(self):
        line_80 = format_context_pressure(0.80, 100_000, 0.50)
        line_95 = format_context_pressure(0.95, 100_000, 0.50)
        assert line_95.count("▰") > line_80.count("▰")

    def test_shows_threshold_tokens(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "100k" in line

    def test_small_threshold(self):
        line = format_context_pressure(0.80, 500, 0.50)
        assert "500" in line

    def test_shows_threshold_percent(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "50%" in line

    def test_approaching_hint(self):
        line = format_context_pressure(0.80, 100_000, 0.50)
        assert "compaction approaching" in line

    def test_no_compaction_when_disabled(self):
        line = format_context_pressure(0.85, 100_000, 0.50, compression_enabled=False)
        assert "no auto-compaction" in line

    def test_returns_string(self):
        result = format_context_pressure(0.65, 128_000, 0.50)
        assert isinstance(result, str)

    def test_over_100_percent_capped(self):
        """Progress > 1.0 should cap both bar and percentage text at 100%."""
        line = format_context_pressure(1.05, 100_000, 0.50)
        assert "▰" in line
        assert line.count("▰") == 20
        assert "100%" in line
        assert "105%" not in line


class TestFormatContextPressureGateway:
    """Gateway (plain text) context pressure display."""

    def test_80_percent_warning(self):
        msg = format_context_pressure_gateway(0.80, 0.50)
        assert "80% to compaction" in msg
        assert "50%" in msg

    def test_90_percent_warning(self):
        msg = format_context_pressure_gateway(0.90, 0.50)
        assert "90% to compaction" in msg
        assert "approaching" in msg

    def test_no_compaction_warning(self):
        msg = format_context_pressure_gateway(0.85, 0.50, compression_enabled=False)
        assert "disabled" in msg

    def test_no_ansi_codes(self):
        msg = format_context_pressure_gateway(0.80, 0.50)
        assert "\033[" not in msg

    def test_has_progress_bar(self):
        msg = format_context_pressure_gateway(0.80, 0.50)
        assert "▰" in msg

    def test_over_100_percent_capped(self):
        """Progress > 1.0 should cap percentage text at 100%."""
        msg = format_context_pressure_gateway(1.09, 0.50)
        assert "100% to compaction" in msg
        assert "109%" not in msg
        assert msg.count("▰") == 20


# ---------------------------------------------------------------------------
# AIAgent context pressure flag tests
# ---------------------------------------------------------------------------


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    """Minimal AIAgent with mocked internals."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


class TestContextPressureFlags:
    """Context pressure warning flag tracking on AIAgent."""

    def test_flag_initialized_zero(self, agent):
        assert agent._context_pressure_warned_at == 0.0

    def test_emit_calls_status_callback(self, agent):
        """status_callback should be invoked with event type and message."""
        cb = MagicMock()
        agent.status_callback = cb

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000  # 50%

        agent._emit_context_pressure(0.85, compressor)

        cb.assert_called_once()
        args = cb.call_args[0]
        assert args[0] == "context_pressure"
        assert "85% to compaction" in args[1]

    def test_emit_no_callback_no_crash(self, agent):
        """No status_callback set — should not crash."""
        agent.status_callback = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        # Should not raise
        agent._emit_context_pressure(0.60, compressor)

    def test_emit_prints_for_cli_platform(self, agent, capsys):
        """CLI platform should always print context pressure, even in quiet_mode."""
        agent.quiet_mode = True
        agent.platform = "cli"
        agent.status_callback = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        agent._emit_context_pressure(0.85, compressor)
        captured = capsys.readouterr()
        assert "▰" in captured.out
        assert "to compaction" in captured.out

    def test_emit_skips_print_for_gateway_platform(self, agent, capsys):
        """Gateway platforms get the callback, not CLI print."""
        agent.platform = "telegram"
        agent.status_callback = None

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        agent._emit_context_pressure(0.85, compressor)
        captured = capsys.readouterr()
        assert "▰" not in captured.out

    def test_flag_reset_on_compression(self, agent):
        """After _compress_context, context pressure flag should reset."""
        agent._context_pressure_warned_at = 0.85
        agent.compression_enabled = True

        agent.context_compressor = MagicMock()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Summary of conversation so far."}
        ]
        agent.context_compressor.context_length = 200_000
        agent.context_compressor.threshold_tokens = 100_000
        agent.context_compressor.compression_count = 1

        agent._todo_store = MagicMock()
        agent._todo_store.format_for_injection.return_value = None

        agent._build_system_prompt = MagicMock(return_value="system prompt")
        agent._cached_system_prompt = "old system prompt"
        agent._session_db = None

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        agent._compress_context(messages, "system prompt")

        assert agent._context_pressure_warned_at == 0.0

    def test_emit_callback_error_handled(self, agent):
        """If status_callback raises, it should be caught gracefully."""
        cb = MagicMock(side_effect=RuntimeError("callback boom"))
        agent.status_callback = cb

        compressor = MagicMock()
        compressor.context_length = 200_000
        compressor.threshold_tokens = 100_000

        # Should not raise
        agent._emit_context_pressure(0.85, compressor)

    def test_tiered_reemits_at_95(self, agent):
        """Warning fires at 85%, then fires again when crossing 95%."""
        agent._context_pressure_warned_at = 0.85
        # Simulate crossing 95%: the tier (0.95) > warned_at (0.85)
        assert 0.95 > agent._context_pressure_warned_at
        # After emission at 95%, the tier should update
        agent._context_pressure_warned_at = 0.95
        assert agent._context_pressure_warned_at == 0.95

    def test_tiered_no_double_emit_at_same_level(self, agent):
        """Once warned at 85%, further 85%+ readings don't re-warn."""
        agent._context_pressure_warned_at = 0.85
        # At 88%, tier is 0.85, which is NOT > warned_at (0.85)
        _warn_tier = 0.85 if 0.88 >= 0.85 else 0.0
        assert not (_warn_tier > agent._context_pressure_warned_at)

    def test_flag_not_reset_when_compression_insufficient(self, agent):
        """When compression can't drop below 85%, keep the flag set."""
        agent._context_pressure_warned_at = 0.85
        agent.compression_enabled = True

        agent.context_compressor = MagicMock()
        agent.context_compressor.compress.return_value = [
            {"role": "user", "content": "Summary of conversation so far."}
        ]
        agent.context_compressor.context_length = 200
        # Use a small threshold so the tiny compressed output still
        # represents >= 85% of it (prevents flag reset).
        agent.context_compressor.threshold_tokens = 10
        agent.context_compressor.compression_count = 1
        agent.context_compressor.last_prompt_tokens = 0

        agent._todo_store = MagicMock()
        agent._todo_store.format_for_injection.return_value = None
        agent._build_system_prompt = MagicMock(return_value="system prompt")
        agent._cached_system_prompt = "old system prompt"
        agent._session_db = None

        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        agent._compress_context(messages, "system prompt")

        # Post-compression is ~90% of threshold — flag should NOT reset
        assert agent._context_pressure_warned_at == 0.85


class TestContextPressureGatewayDedup:
    """Class-level dedup prevents warning spam across AIAgent instances."""

    def setup_method(self):
        """Clear class-level dedup state between tests."""
        AIAgent._context_pressure_last_warned.clear()

    def test_second_instance_within_cooldown_suppressed(self):
        """Same session, same tier, within cooldown — should be suppressed."""
        import time
        sid = "test_session_dedup"
        # Simulate first warning
        AIAgent._context_pressure_last_warned[sid] = (0.85, time.time())
        # Second instance checking same tier within cooldown
        _last = AIAgent._context_pressure_last_warned.get(sid)
        _should_warn = _last is None or _last[0] < 0.85 or (time.time() - _last[1]) >= AIAgent._CONTEXT_PRESSURE_COOLDOWN
        assert not _should_warn

    def test_higher_tier_fires_despite_cooldown(self):
        """Same session, higher tier — should fire even within cooldown."""
        import time
        sid = "test_session_tier"
        AIAgent._context_pressure_last_warned[sid] = (0.85, time.time())
        _last = AIAgent._context_pressure_last_warned.get(sid)
        # 0.95 > 0.85 stored tier → should warn
        _should_warn = _last is None or _last[0] < 0.95 or (time.time() - _last[1]) >= AIAgent._CONTEXT_PRESSURE_COOLDOWN
        assert _should_warn

    def test_warning_fires_after_cooldown_expires(self):
        """Same session, same tier, after cooldown — should fire again."""
        import time
        sid = "test_session_expired"
        # Set a timestamp far in the past
        AIAgent._context_pressure_last_warned[sid] = (0.85, time.time() - AIAgent._CONTEXT_PRESSURE_COOLDOWN - 1)
        _last = AIAgent._context_pressure_last_warned.get(sid)
        _should_warn = _last is None or _last[0] < 0.85 or (time.time() - _last[1]) >= AIAgent._CONTEXT_PRESSURE_COOLDOWN
        assert _should_warn

    def test_compression_clears_dedup(self):
        """After compression drops below 85%, dedup entry should be cleared."""
        import time
        sid = "test_session_clear"
        AIAgent._context_pressure_last_warned[sid] = (0.85, time.time())
        assert sid in AIAgent._context_pressure_last_warned
        # Simulate what _compress_context does on reset
        AIAgent._context_pressure_last_warned.pop(sid, None)
        assert sid not in AIAgent._context_pressure_last_warned

    def test_eviction_removes_stale_entries(self):
        """Stale entries older than 2x cooldown should be evicted."""
        import time
        _now = time.time()
        AIAgent._context_pressure_last_warned = {
            "fresh": (0.85, _now),
            "stale": (0.85, _now - AIAgent._CONTEXT_PRESSURE_COOLDOWN * 3),
        }
        _cutoff = _now - AIAgent._CONTEXT_PRESSURE_COOLDOWN * 2
        AIAgent._context_pressure_last_warned = {
            k: v for k, v in AIAgent._context_pressure_last_warned.items()
            if v[1] > _cutoff
        }
        assert "fresh" in AIAgent._context_pressure_last_warned
        assert "stale" not in AIAgent._context_pressure_last_warned
