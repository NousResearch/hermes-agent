"""Tests for logging.suppress_fallback_messages config option.

When enabled, fallback/rate-limit status messages should be printed to CLI
(via _vprint) but NOT forwarded to status_callback (gateway channel).
"""
import pytest
from unittest.mock import MagicMock, patch


def _make_tool_defs(*names):
    """Build minimal tool definition list accepted by AIAgent.__init__."""
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


class TestSuppressFallbackStatus:
    """Verify _emit_status respects logging.suppress_fallback_messages."""

    @pytest.fixture
    def agent(self):
        from run_agent import AIAgent
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            agent.client = MagicMock()
        return agent

    def test_emit_status_normal_forwards_to_callback(self, agent):
        """Non-fallback messages are forwarded to status_callback normally."""
        agent._suppress_fallback_status = False
        callback_msgs = []
        agent.status_callback = lambda etype, msg: callback_msgs.append((etype, msg))

        agent._emit_status("Something happened")

        assert len(callback_msgs) == 1
        assert callback_msgs[0] == ("lifecycle", "Something happened")

    def test_emit_status_fallback_suppressed_from_callback(self, agent):
        """Fallback messages are NOT forwarded when suppress flag is enabled."""
        agent._suppress_fallback_status = True
        callback_msgs = []
        agent.status_callback = lambda etype, msg: callback_msgs.append((etype, msg))

        agent._emit_status("⚠️ Rate limited — switching to fallback provider...")
        agent._emit_status("⚠️ Empty/malformed response — switching to fallback...")
        agent._emit_status("⚠️ Max retries (3) for invalid responses — trying fallback...")
        agent._emit_status("⚠️ Non-retryable error (HTTP 403) — trying fallback...")
        agent._emit_status("⚠️ Max retries (3) exhausted — trying fallback...")
        agent._emit_status("⏱️ Rate limited. Waiting 5.0s (attempt 2/3)...")

        assert len(callback_msgs) == 0, f"Expected 0 callback messages, got: {callback_msgs}"

    def test_emit_status_non_fallback_still_forwarded_when_suppressed(self, agent):
        """Non-fallback messages are still forwarded even when suppress is enabled."""
        agent._suppress_fallback_status = True
        callback_msgs = []
        agent.status_callback = lambda etype, msg: callback_msgs.append((etype, msg))

        agent._emit_status("❌ API failed after 3 retries — connection timeout")
        agent._emit_status("⚠️ Something unrelated")

        assert len(callback_msgs) == 2

    def test_emit_status_fallback_forwarded_when_not_suppressed(self, agent):
        """Fallback messages ARE forwarded when suppress flag is disabled (default)."""
        agent._suppress_fallback_status = False
        callback_msgs = []
        agent.status_callback = lambda etype, msg: callback_msgs.append((etype, msg))

        agent._emit_status("⚠️ Rate limited — switching to fallback provider...")

        assert len(callback_msgs) == 1

    def test_emit_status_rate_limited_final_error_suppressed(self, agent):
        """Final failure message (❌ Rate limited after...) is suppressed too."""
        agent._suppress_fallback_status = True
        callback_msgs = []
        agent.status_callback = lambda etype, msg: callback_msgs.append((etype, msg))

        agent._emit_status("❌ Rate limited after 3 retries — provider exhausted")

        assert len(callback_msgs) == 0

    def test_emit_status_no_callback_no_crash(self, agent):
        """No crash when status_callback is None (CLI mode)."""
        agent._suppress_fallback_status = True
        agent.status_callback = None

        # Should not raise
        agent._emit_status("⚠️ Rate limited — switching to fallback provider...")
