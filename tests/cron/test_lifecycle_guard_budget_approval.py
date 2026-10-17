"""Tests for lifecycle scan-budget exhaustion → approval prompt (#119322).

When the lifecycle guard's scan budget is exhausted during a terminal command
check in an interactive session, the guard stores the refusal instead of
hard-blocking, and the approval layer surfaces a human-readable approval prompt
("approve to run anyway / run from outside the gateway").

Unattended sessions (cron, -q, batch) retain the hard deny.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import cron.lifecycle_guard as lifecycle_guard
from cron.lifecycle_guard import is_budget_exhaustion_refusal


# --- is_budget_exhaustion_refusal() --------------------------------------------------

class TestIsBudgetExhaustionRefusal:
    """Distinguish budget exhaustion refusals from other refusal kinds."""

    def test_exact_budget_exhaustion_text(self):
        assert is_budget_exhaustion_refusal(
            "the scan budget was exhausted (remote reads at depth 1)"
        ) is True

    def test_text_budget_variants(self):
        assert is_budget_exhaustion_refusal("the scan budget was exhausted (text at depth 0)")
        assert is_budget_exhaustion_refusal("the scan budget was exhausted (paths at depth 2)")

    def test_non_budget_refusal_oversized(self):
        assert is_budget_exhaustion_refusal(
            "`/opt/data/bin/mail` is larger than the scan cap (1048576 bytes)"
        ) is False

    def test_non_budget_refusal_cloud(self):
        assert is_budget_exhaustion_refusal(
            "`/iCloud/file.sh` lives on a cloud-synced path"
        ) is False

    def test_non_budget_refusal_sqlite(self):
        assert is_budget_exhaustion_refusal(
            "`/tmp/data.db` is a SQLite database open in this gateway process"
        ) is False

    def test_none_input(self):
        assert is_budget_exhaustion_refusal(None) is False

    def test_empty_string(self):
        assert is_budget_exhaustion_refusal("") is False


# --- _LIFECYCLE_BUDGET_GATE spec ----------------------------------------------------

class TestLifecycleBudgetGateSpec:
    """The _LIFECYCLE_BUDGET_GATE has the required message templates."""

    def test_importable(self):
        from tools.approval import _LIFECYCLE_BUDGET_GATE
        assert _LIFECYCLE_BUDGET_GATE.noun == "command"
        assert _LIFECYCLE_BUDGET_GATE.user_approved is True
        assert _LIFECYCLE_BUDGET_GATE.pending_keys is False

    def test_messages_contain_budget_word(self):
        from tools.approval import _LIFECYCLE_BUDGET_GATE
        assert "budget" in _LIFECYCLE_BUDGET_GATE.gateway_refused.lower()
        assert "budget" in _LIFECYCLE_BUDGET_GATE.cli_timeout.lower()
        assert "budget" in _LIFECYCLE_BUDGET_GATE.cli_denied.lower()


# --- _lifecycle_budget_refusals dict -------------------------------------------------

class TestLifecycleBudgetRefusalsStorage:
    """Store and pop lifecycle budget refusals keyed by session."""

    def test_store_and_pop(self):
        from tools.approval import (
            _store_lifecycle_budget_refusal,
            _pop_stored_lifecycle_refusal,
            _lifecycle_budget_refusals,
        )
        _lifecycle_budget_refusals.clear()
        _store_lifecycle_budget_refusal("session-abc", "the scan budget was exhausted (remote reads at depth 1)")
        assert _pop_stored_lifecycle_refusal("session-abc") == "the scan budget was exhausted (remote reads at depth 1)"
        # popped — second call returns None
        assert _pop_stored_lifecycle_refusal("session-abc") is None

    def test_unknown_session_returns_none(self):
        from tools.approval import _pop_stored_lifecycle_refusal, _lifecycle_budget_refusals
        _lifecycle_budget_refusals.clear()
        assert _pop_stored_lifecycle_refusal("nonexistent") is None

    def test_different_sessions_independent(self):
        from tools.approval import (
            _store_lifecycle_budget_refusal,
            _pop_stored_lifecycle_refusal,
            _lifecycle_budget_refusals,
        )
        _lifecycle_budget_refusals.clear()
        _store_lifecycle_budget_refusals("s1", "refusal-1")
        _store_lifecycle_budget_refusals("s2", "refusal-2")
        assert _pop_stored_lifecycle_refusal("s1") == "refusal-1"
        assert _pop_stored_lifecycle_refusal("s2") == "refusal-2"


def _store_lifecycle_budget_refusals(key, val):
    """Helper for multi-session test above."""
    from tools.approval import _store_lifecycle_budget_refusal
    _store_lifecycle_budget_refusal(key, val)


# --- gateway_lifecycle_block routing ------------------------------------------------

class TestGatewayLifecycleBlockBudgetExhaustion:
    """gateway_lifecycle_block routes budget exhaustion to approval in interactive sessions."""

    def _make_mock_env(self):
        env = MagicMock()
        env.execute.return_value = {"returncode": 1, "output": ""}
        return env

    @patch("tools.process_registry._is_supervised_gateway_process", return_value=True)
    @patch("tools.approval._is_interactive_session", return_value=True)
    @patch("tools.approval._store_lifecycle_budget_refusal")
    def test_budget_exhaustion_interactive_stores_and_returns_none(
        self, mock_store, mock_interactive, mock_supervised, monkeypatch
    ):
        """Interactive session: budget exhaustion stores refusal, returns None (no hard block)."""
        from tools.terminal_tool_guards import gateway_lifecycle_block

        # Force scan_gateway_lifecycle to return budget exhaustion
        monkeypatch.setattr(
            lifecycle_guard, "scan_gateway_lifecycle",
            lambda *a, **kw: (True, "the scan budget was exhausted (remote reads at depth 1)")
        )
        # Prevent actual path resolution / remote reads
        monkeypatch.setattr(
            lifecycle_guard, "lifecycle_scan_root_within_budget", lambda *a: False
        )

        result = gateway_lifecycle_block(
            command="/opt/data/bin/mail check",
            env=self._make_mock_env(), env_type="local",
            cwd="/tmp", workdir=None, session_key="test-session",
        )
        # No hard block — None means "command may proceed"
        assert result is None
        mock_store.assert_called_once_with(
            "test-session", "the scan budget was exhausted (remote reads at depth 1)"
        )

    @patch("tools.process_registry._is_supervised_gateway_process", return_value=True)
    @patch("tools.approval._is_interactive_session", return_value=False)
    def test_budget_exhaustion_unattended_hard_blocks(
        self, mock_interactive, mock_supervised, monkeypatch
    ):
        """Unattended session: budget exhaustion stays a hard block."""
        from tools.terminal_tool_guards import gateway_lifecycle_block

        monkeypatch.setattr(
            lifecycle_guard, "scan_gateway_lifecycle",
            lambda *a, **kw: (True, "the scan budget was exhausted (remote reads at depth 1)")
        )
        monkeypatch.setattr(
            lifecycle_guard, "lifecycle_scan_root_within_budget", lambda *a: False
        )

        result = gateway_lifecycle_block(
            command="/opt/data/bin/mail check",
            env=self._make_mock_env(), env_type="local",
            cwd="/tmp", workdir=None, session_key="test-session",
        )
        # Hard block JSON
        assert result is not None
        parsed = json.loads(result)
        assert parsed["exit_code"] == 1
        assert "scan budget" in parsed["error"].lower()

    @patch("tools.process_registry._is_supervised_gateway_process", return_value=True)
    @patch("tools.approval._is_interactive_session", return_value=True)
    def test_non_budget_refusal_still_hard_blocks_interactive(
        self, mock_interactive, mock_supervised, monkeypatch
    ):
        """Even in interactive sessions, non-budget refusals (oversized, cloud, SQLite) hard block."""
        from tools.terminal_tool_guards import gateway_lifecycle_block

        monkeypatch.setattr(
            lifecycle_guard, "scan_gateway_lifecycle",
            lambda *a, **kw: (True, "`/opt/data/bin/mail` is larger than the scan cap (1048576 bytes)")
        )
        monkeypatch.setattr(
            lifecycle_guard, "lifecycle_scan_root_within_budget", lambda *a: False
        )

        result = gateway_lifecycle_block(
            command="/opt/data/bin/mail check",
            env=self._make_mock_env(), env_type="local",
            cwd="/tmp", workdir=None, session_key="test-session",
        )
        assert result is not None
        parsed = json.loads(result)
        assert "scan cap" in parsed["error"]

    @patch("tools.process_registry._is_supervised_gateway_process", return_value=True)
    def test_actual_lifecycle_command_still_hard_blocks(self, mock_supervised, monkeypatch):
        """Actual lifecycle commands (unsafe=True, refusal=None) always hard block."""
        from tools.terminal_tool_guards import gateway_lifecycle_block

        monkeypatch.setattr(
            lifecycle_guard, "scan_gateway_lifecycle",
            lambda *a, **kw: (True, None)  # refusal=None → actual lifecycle command
        )
        monkeypatch.setattr(
            lifecycle_guard, "lifecycle_scan_root_within_budget", lambda *a: False
        )

        result = gateway_lifecycle_block(
            command="hermes gateway restart",
            env=self._make_mock_env(), env_type="local",
            cwd="/tmp", workdir=None, session_key="test-session",
        )
        assert result is not None
        parsed = json.loads(result)
        assert "restart" in parsed["error"].lower()
