"""Regression tests for ghost-session classification and burst-detection.

Kanban t_4e906121: 72 telegram session rows from 2026-05-10 04:52-05:02 with
zero messages, zero tokens, zero billing — created by a gateway shutdown/
restart churn loop. The fix bundles three layers:

1. ``InsightsEngine._is_ghost_session`` classifier — display-side filter so
   future husks don't inflate per-platform session counts.
2. ``InsightsEngine._compute_platform_breakdown`` — buckets ghosts into a
   ``ghost_sessions`` field instead of the headline ``sessions`` counter.
3. ``SessionDB.create_session`` — burst-detector warns when one
   ``(source, user_id)`` pair triggers ≥6 create calls within 10s.

These tests guard (1) and (2). (3) is exercised below by monkey-patching
``_insert_session_row`` to keep the test hermetic (no real SQLite write).
"""

from __future__ import annotations

import logging
import time

import pytest

from agent.insights import InsightsEngine


class TestGhostSessionClassifier:
    def test_no_msgs_no_tokens_is_ghost(self):
        s = {"message_count": 0, "input_tokens": 0, "output_tokens": 0,
             "tool_call_count": 0, "billing_provider": None}
        assert InsightsEngine._is_ghost_session(s) is True

    def test_all_nulls_is_ghost(self):
        # NULL columns from SQLite show up as Python None
        s = {"message_count": None, "input_tokens": None, "output_tokens": None,
             "tool_call_count": None, "billing_provider": None}
        assert InsightsEngine._is_ghost_session(s) is True

    def test_has_messages_not_ghost(self):
        s = {"message_count": 3, "input_tokens": 0, "output_tokens": 0,
             "tool_call_count": 0, "billing_provider": None}
        assert InsightsEngine._is_ghost_session(s) is False

    def test_has_input_tokens_not_ghost(self):
        s = {"message_count": 0, "input_tokens": 100, "output_tokens": 0,
             "tool_call_count": 0, "billing_provider": None}
        assert InsightsEngine._is_ghost_session(s) is False

    def test_has_tool_calls_not_ghost(self):
        s = {"message_count": 0, "input_tokens": 0, "output_tokens": 0,
             "tool_call_count": 2, "billing_provider": None}
        assert InsightsEngine._is_ghost_session(s) is False

    def test_has_billing_provider_not_ghost(self):
        # An API call was attempted even if it never produced output
        s = {"message_count": 0, "input_tokens": 0, "output_tokens": 0,
             "tool_call_count": 0, "billing_provider": "anthropic"}
        assert InsightsEngine._is_ghost_session(s) is False


class TestPlatformBreakdownBucketsGhosts:
    def test_ghosts_counted_separately(self):
        # 2 real telegram + 5 ghost telegram + 1 real discord
        sessions = (
            [{"source": "telegram", "message_count": 4, "input_tokens": 100,
              "output_tokens": 50, "tool_call_count": 1, "billing_provider": "anthropic"}] * 2
            + [{"source": "telegram", "message_count": 0, "input_tokens": 0,
                "output_tokens": 0, "tool_call_count": 0, "billing_provider": None}] * 5
            + [{"source": "discord", "message_count": 10, "input_tokens": 500,
                "output_tokens": 200, "tool_call_count": 3, "billing_provider": "anthropic"}]
        )
        # InsightsEngine.__init__ needs a db connection — bypass it.
        result = InsightsEngine._compute_platform_breakdown(InsightsEngine.__new__(InsightsEngine), sessions)
        by_platform = {r["platform"]: r for r in result}

        assert by_platform["telegram"]["sessions"] == 2
        assert by_platform["telegram"]["ghost_sessions"] == 5
        assert by_platform["telegram"]["messages"] == 8  # 2 sessions * 4 msgs
        assert by_platform["telegram"]["input_tokens"] == 200

        assert by_platform["discord"]["sessions"] == 1
        assert by_platform["discord"]["ghost_sessions"] == 0


class TestCreateSessionBurstDetector:
    def test_burst_warning_fires_at_sixth_call(self, caplog):
        from hermes_state import SessionDB
        # Build a stub SessionDB without touching SQLite.
        db = SessionDB.__new__(SessionDB)
        db._insert_session_row = lambda *a, **kw: None  # type: ignore

        caplog.set_level(logging.WARNING, logger="hermes_state")
        for i in range(6):
            db.create_session(f"s{i}", "telegram", user_id="u1")

        burst_warnings = [r for r in caplog.records if "burst detected" in r.message]
        assert len(burst_warnings) == 1, f"expected 1 warning, got {len(burst_warnings)}: {burst_warnings}"

    def test_no_warning_under_threshold(self, caplog):
        from hermes_state import SessionDB
        db = SessionDB.__new__(SessionDB)
        db._insert_session_row = lambda *a, **kw: None  # type: ignore

        caplog.set_level(logging.WARNING, logger="hermes_state")
        for i in range(5):
            db.create_session(f"s{i}", "telegram", user_id="u1")

        burst_warnings = [r for r in caplog.records if "burst detected" in r.message]
        assert burst_warnings == []

    def test_different_users_dont_collide(self, caplog):
        from hermes_state import SessionDB
        db = SessionDB.__new__(SessionDB)
        db._insert_session_row = lambda *a, **kw: None  # type: ignore

        caplog.set_level(logging.WARNING, logger="hermes_state")
        # 5 each for two different users — total 10 calls, neither user hits the threshold
        for i in range(5):
            db.create_session(f"a{i}", "telegram", user_id="userA")
            db.create_session(f"b{i}", "telegram", user_id="userB")

        burst_warnings = [r for r in caplog.records if "burst detected" in r.message]
        assert burst_warnings == []
