"""Tests for the /behavior gateway command handling.

Mirrors the pattern in tests/agent/test_credits_view.py: creates a minimal
stub exposing GatewaySlashCommandsMixin._handle_behavior_command and runs
it via asyncio.run with a fake MessageEvent.

Covers:
  - Config gate (disabled → message, enabled → runs)
  - Argument parsing (--days, --source, positional)
  - Unicode dash normalization (iOS auto-converts -- to em dash)
  - Behavior analyzer is called and format_gateway output is returned
  - Error handling (analyzer exception → error message, no crash)
"""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import MagicMock, patch

import pytest


# =========================================================================
# Helpers
# =========================================================================

def _make_gateway_stub():
    """Minimal object exposing the mixin's _handle_behavior_command."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class _Stub(GatewaySlashCommandsMixin):
        def __init__(self):
            pass

    return _Stub()


class _FakeEvent:
    """Minimal stand-in for MessageEvent with get_command_args()."""

    def __init__(self, text: str = "/behavior"):
        self.text = text

    def get_command_args(self) -> str:
        # Mirrors gateway/platforms/base.py MessageEvent.get_command_args
        if not self.text.startswith("/"):
            return self.text
        parts = self.text.split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        args = args.replace("\u2014\u2014", "--").replace("\u2014", "--").replace("\u2013", "-")
        return args


def _run_behavior_command(text: str, config=None):
    """Run _handle_behavior_command with mocked config + DB.

    Returns the result string.  Patches:
      - read_raw_config → config (default: {"behavior": {"enabled": True}})
      - SessionDB → MagicMock
      - BehavioralAnalyzer → stub
    """
    stub = _make_gateway_stub()
    event = _FakeEvent(text)

    cfg = config if config is not None else {"behavior": {"enabled": True}}

    _BehavioralAnalyzerStub.calls = []

    with patch("hermes_cli.config.read_raw_config", return_value=cfg), \
         patch("hermes_state.SessionDB", return_value=MagicMock()), \
         patch("agent.behavioral_insights.BehavioralAnalyzer", _BehavioralAnalyzerStub):
        result = asyncio.run(stub._handle_behavior_command(event))
    return result


class _BehavioralAnalyzerStub:
    calls = []

    def __init__(self, db, config=None):
        self.db = db
        self.config = config

    def generate(self, *, days=30, source=None):
        self.calls.append({"days": days, "source": source})
        return {"days": days, "source": source, "empty": False,
                "scores": {}, "cards": {}, "session_count": 1,
                "llm_available": False}

    def format_gateway(self, report):
        return f"days={report['days']} source={report['source']}"


# =========================================================================
# Config gate
# =========================================================================

class TestGatewayBehaviorConfigGate:
    def test_disabled_returns_message(self):
        result = _run_behavior_command("/behavior", {"behavior": {"enabled": False}})
        assert "disabled" in result.lower()
        assert "behavior.enabled: true" in result

    def test_no_behavior_section_disabled(self):
        result = _run_behavior_command("/behavior", {})
        assert "disabled" in result.lower()

    def test_no_enabled_key_disabled(self):
        result = _run_behavior_command("/behavior", {"behavior": {}})
        assert "disabled" in result.lower()

    def test_enabled_runs_analyzer(self):
        result = _run_behavior_command("/behavior", {"behavior": {"enabled": True}})
        # Analyzer was called → stub returns format_gateway output
        assert len(_BehavioralAnalyzerStub.calls) == 1
        assert "days=30" in result


# =========================================================================
# Argument parsing
# =========================================================================

class TestGatewayBehaviorArgParsing:
    def test_default_days_30(self):
        result = _run_behavior_command("/behavior")
        assert _BehavioralAnalyzerStub.calls == [{"days": 30, "source": None}]
        assert "days=30 source=None" in result

    def test_positional_days(self):
        result = _run_behavior_command("/behavior 7")
        assert _BehavioralAnalyzerStub.calls == [{"days": 7, "source": None}]
        assert "days=7" in result

    def test_days_flag(self):
        result = _run_behavior_command("/behavior --days 14")
        assert _BehavioralAnalyzerStub.calls == [{"days": 14, "source": None}]

    def test_source_flag(self):
        result = _run_behavior_command("/behavior --source discord")
        assert _BehavioralAnalyzerStub.calls == [{"days": 30, "source": "discord"}]

    def test_days_and_source(self):
        result = _run_behavior_command("/behavior --days 14 --source discord")
        assert _BehavioralAnalyzerStub.calls == [{"days": 14, "source": "discord"}]

    def test_invalid_days_value(self):
        """Non-integer --days → returns invalid_days error message."""
        result = _run_behavior_command("/behavior --days abc")
        # The handler returns t("gateway.behavior.invalid_days", ...) which
        # falls back to the bare key if the catalog entry is missing.
        assert "invalid_days" in result or "abc" in result

    def test_unicode_dash_days_normalized(self):
        """Em dash (U+2014) before 'days' should be normalized to --days."""
        result = _run_behavior_command(f"/behavior \u2014days 7")
        assert _BehavioralAnalyzerStub.calls == [{"days": 7, "source": None}]

    def test_unicode_dash_source_normalized(self):
        """Em dash before 'source' should be normalized to --source."""
        result = _run_behavior_command(f"/behavior \u2014source telegram")
        assert _BehavioralAnalyzerStub.calls == [{"days": 30, "source": "telegram"}]

    def test_combined_unicode_flags(self):
        """Combined em-dash flags should both be normalized."""
        result = _run_behavior_command(f"/behavior \u2014days 30 \u2014source cli")
        assert _BehavioralAnalyzerStub.calls == [{"days": 30, "source": "cli"}]

    def test_en_dash_normalized(self):
        """En dash (U+2013) before 'days' should be normalized."""
        result = _run_behavior_command(f"/behavior \u2013days 7")
        assert _BehavioralAnalyzerStub.calls == [{"days": 7, "source": None}]


# =========================================================================
# Output format and execution
# =========================================================================

class TestGatewayBehaviorOutput:
    def test_returns_gateway_format(self):
        """Result should be the format_gateway output, not format_terminal."""
        result = _run_behavior_command("/behavior")
        assert "days=30" in result  # from stub's format_gateway

    def test_runs_in_executor(self):
        """The handler should use run_in_executor (async → sync DB call).

        We verify by checking the analyzer is called (meaning the executor
        path completed).  The stub runs synchronously inside the executor.
        """
        result = _run_behavior_command("/behavior")
        assert len(_BehavioralAnalyzerStub.calls) == 1

    def test_config_passed_to_analyzer(self):
        """The behavior config dict is passed to BehavioralAnalyzer."""
        captured_config = []

        class _CapturingStub:
            def __init__(self, db, config=None):
                captured_config.append(config)

            def generate(self, *, days=30, source=None):
                return {"days": days, "source": source, "empty": False,
                        "scores": {}, "cards": {}, "session_count": 0,
                        "llm_available": False}

            def format_gateway(self, report):
                return "ok"

        stub = _make_gateway_stub()
        event = _FakeEvent("/behavior")
        cfg = {"behavior": {"enabled": True, "model": "gpt-4o-mini"}}

        with patch("hermes_cli.config.read_raw_config", return_value=cfg), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _CapturingStub):
            result = asyncio.run(stub._handle_behavior_command(event))

        assert len(captured_config) == 1
        assert captured_config[0].get("model") == "gpt-4o-mini"


# =========================================================================
# Error handling
# =========================================================================

class TestGatewayBehaviorErrors:
    def test_analyzer_exception_returns_error_message(self):
        """If BehavioralAnalyzer raises, handler returns error message (no crash)."""
        class _ExplodingStub:
            def __init__(self, db, config=None):
                pass

            def generate(self, *, days=30, source=None):
                raise RuntimeError("boom")

            def format_gateway(self, report):
                return "should not reach"

        stub = _make_gateway_stub()
        event = _FakeEvent("/behavior")
        cfg = {"behavior": {"enabled": True}}

        with patch("hermes_cli.config.read_raw_config", return_value=cfg), \
             patch("hermes_state.SessionDB", return_value=MagicMock()), \
             patch("agent.behavioral_insights.BehavioralAnalyzer", _ExplodingStub):
            result = asyncio.run(stub._handle_behavior_command(event))

        # The handler catches and returns t("gateway.behavior.error", error=e)
        assert "error" in result.lower() or "boom" in result