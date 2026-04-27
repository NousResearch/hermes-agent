"""Tests for _running_agents_ts cleanup — ensures timestamp entries are
removed whenever their corresponding _running_agents entry is deleted.

When an agent entry is removed from _running_agents (stop, new, resume,
shutdown), the matching _running_agents_ts entry must also be cleaned up.
Orphaned timestamps cause memory leaks and can corrupt stale-timeout checks
if session keys are reused.
"""

import re
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _make_runner(tmp_path) -> GatewayRunner:
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")},
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    # Prevent real adapter creation
    runner._create_adapter = MagicMock(return_value=MagicMock())
    return runner


class TestRunningAgentsTsCleanup:
    def test_stop_command_cleans_ts(self, tmp_path):
        """_release_running_agent_state (the /stop path) must pop _running_agents_ts."""
        runner = _make_runner(tmp_path)
        key = "chat_123"
        runner._running_agents[key] = MagicMock()
        runner._running_agents_ts[key] = 1000.0

        runner._release_running_agent_state(key)

        assert key not in runner._running_agents
        assert key not in runner._running_agents_ts

    def test_new_command_cleans_ts(self, tmp_path):
        """_release_running_agent_state (the /new path) must pop _running_agents_ts."""
        runner = _make_runner(tmp_path)
        key = "chat_456"
        runner._running_agents[key] = MagicMock()
        runner._running_agents_ts[key] = 2000.0

        runner._release_running_agent_state(key)

        assert key not in runner._running_agents
        assert key not in runner._running_agents_ts

    def test_resume_command_cleans_ts(self, tmp_path):
        """_release_running_agent_state (the /resume path) must pop _running_agents_ts."""
        runner = _make_runner(tmp_path)
        key = "chat_789"
        runner._running_agents[key] = MagicMock()
        runner._running_agents_ts[key] = 3000.0

        runner._release_running_agent_state(key)

        assert key not in runner._running_agents
        assert key not in runner._running_agents_ts

    def test_release_also_cleans_busy_ack_ts(self, tmp_path):
        """_release_running_agent_state must pop all three tracking dicts atomically."""
        runner = _make_runner(tmp_path)
        key = "chat_999"
        runner._running_agents[key] = MagicMock()
        runner._running_agents_ts[key] = 4000.0
        runner._busy_ack_ts[key] = 4000.0

        runner._release_running_agent_state(key)

        assert key not in runner._running_agents
        assert key not in runner._running_agents_ts
        assert key not in runner._busy_ack_ts

    def test_shutdown_clears_ts(self, tmp_path):
        """Source-level check: the shutdown path must clear _running_agents_ts
        immediately after clearing _running_agents.

        The test verifies that the production code in gateway/run.py contains
        the paired clear() calls in the expected order, rather than testing the
        trivially-correct operation of dict.clear() itself.
        """
        import gateway.run as mod

        source = open(mod.__file__).read()
        # Locate the shutdown clear block — both clears must appear together
        idx_agents = source.find("self._running_agents.clear()")
        idx_ts = source.find("self._running_agents_ts.clear()")

        assert idx_agents != -1, "_running_agents.clear() not found in gateway/run.py"
        assert idx_ts != -1, "_running_agents_ts.clear() not found in gateway/run.py"
        # _running_agents_ts.clear() must follow _running_agents.clear()
        # within a small window (same block)
        assert idx_ts > idx_agents, (
            "_running_agents_ts.clear() must appear after _running_agents.clear() "
            "in the shutdown path"
        )

    def test_clear_site_also_clears_ts(self):
        """Source-level check: every `self._running_agents.clear()` must be
        followed by `self._running_agents_ts.clear()`."""
        import gateway.run as mod

        source = open(mod.__file__).read()

        clear_pattern = re.compile(r'self\._running_agents\.clear\(\)')
        ts_clear_pattern = re.compile(r'self\._running_agents_ts\.clear\(\)')

        clear_lines = []
        ts_clear_lines = []
        for i, line in enumerate(source.splitlines(), 1):
            if clear_pattern.search(line):
                clear_lines.append(i)
            if ts_clear_pattern.search(line):
                ts_clear_lines.append(i)

        missing = []
        for clear_line in clear_lines:
            found = any(
                ts_line > clear_line and ts_line <= clear_line + 5
                for ts_line in ts_clear_lines
            )
            if not found:
                missing.append(clear_line)

        assert missing == [], (
            f"Lines with `_running_agents.clear()` but no "
            f"`_running_agents_ts.clear()` within 5 lines: {missing}"
        )
