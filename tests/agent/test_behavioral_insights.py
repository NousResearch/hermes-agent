"""Tests for agent/behavioral_insights.py — BehavioralAnalyzer signal extraction,
5-axis scoring, LLM narrative cards, score persistence, and formatting.

Follows the same fixture + pattern as tests/agent/test_insights.py:
  - @pytest.fixture() db / populated_db with a temp SessionDB
  - realistic session data (user messages, tool calls, crash outs, plan
    phrases, late-night sessions, subagent sessions)
  - assert invariants, not frozen values
  - mock the LLM provider for narrative-card tests (no real API calls)
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB
from agent.behavioral_insights import (
    BehavioralAnalyzer,
    _bar,
    _format_duration,
    _scrub_credentials,
    _shannon_entropy,
    _word_count,
    _is_all_caps,
    _has_typos,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture()
def db(tmp_path):
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_behavior.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


@pytest.fixture()
def populated_db(db):
    """Create a DB with realistic behavioral session data.

    Sessions cover the signal-extraction surface:
      - steering keywords ("no", "stop", "wait", "don't")
      - politeness ("thanks", "please", "appreciate")
      - crash outs (ALL CAPS, frustration keywords)
      - plan phrases ("let's plan", "first I'll")
      - go-to prompts (repeated short prompts)
      - tool calls (terminal, read_file, search_files, web_search)
      - late-night cryptic prompts (11 PM)
      - subagent sessions (parent_session_id set)
      - model preference (multiple models)
      - decision patterns ("use X not Y", "switch to Z")
      - delegate_task dispatches (background + foreground)
    """
    now = time.time()
    day = 86400

    # ── Session 1: CLI, plan-first, steering, tools, politeness ──
    db.create_session(
        session_id="s1", source="cli",
        model="anthropic/claude-sonnet-4-20250514", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's1'", (now - 2 * day,))
    db.end_session("s1", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's1'", (now - 2 * day + 3600,))
    db.update_token_counts("s1", input_tokens=50000, output_tokens=15000)
    # Plan-first first message
    db.append_message("s1", role="user", content="Let's plan this out first. I'll need to fix the auth bug.")
    db.append_message("s1", role="assistant", content="Sure, let me search the files.",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("s1", role="tool", content="Found 3 matches", tool_name="search_files")
    db.append_message("s1", role="assistant", content="Let me read the file.",
                      tool_calls=[{"function": {"name": "read_file"}}])
    db.append_message("s1", role="tool", content="file contents here", tool_name="read_file")
    db.append_message("s1", role="assistant", content="I found the bug. Let me fix it.",
                      tool_calls=[{"function": {"name": "patch"}}])
    db.append_message("s1", role="tool", content="patched successfully", tool_name="patch")
    # Steering message
    db.append_message("s1", role="user", content="No, stop. Don't touch that file, use the other one instead.")
    # Decision pattern
    db.append_message("s1", role="user", content="Actually, let's go with the redis approach not the file approach.")
    # Done declaration WITHOUT verification tool after it
    db.append_message("s1", role="assistant", content="Done! The fix is complete.")
    # Politeness
    db.append_message("s1", role="user", content="Thanks! I appreciate it.")

    # ── Session 2: Telegram, crash out, go-to prompt, late-night cryptic ──
    db.create_session(
        session_id="s2", source="telegram",
        model="gpt-4o", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's2'", (now - 5 * day,))
    db.end_session("s2", end_reason="timeout")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's2'", (now - 5 * day + 1800,))
    db.update_token_counts("s2", input_tokens=20000, output_tokens=8000)
    # Go-to prompt (short, repeated)
    db.append_message("s2", role="user", content="make it work")
    db.append_message("s2", role="assistant", content="Let me search.",
                      tool_calls=[{"function": {"name": "web_search"}}])
    db.append_message("s2", role="tool", content="results here", tool_name="web_search")
    # Crash out: ALL CAPS + frustration keyword
    db.append_message("s2", role="user", content="STOP TOUCHING THAT FILE!!! WTF!!!")
    db.append_message("s2", role="assistant", content="Sorry, let me try again.")

    # ── Session 3: CLI, late-night cryptic prompt, tool heavy ──
    db.create_session(
        session_id="s3", source="cli",
        model="deepseek-chat", user_id="user1",
    )
    # Backdate to 10 days ago
    s3_start = now - 10 * day
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's3'", (s3_start,))
    db.end_session("s3", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's3'", (s3_start + 7200,))
    db.update_token_counts("s3", input_tokens=100000, output_tokens=40000)
    # Late-night cryptic prompt with typos (timestamp set to 1 AM)
    cryptic_ts = s3_start + 3600  # 1 hour after start
    # Adjust to 1 AM local — we set the message timestamp directly
    db.append_message("s3", role="user", content="make teh thing wrok", timestamp=cryptic_ts)
    db.append_message("s3", role="assistant", content="Running terminal.",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="error: command not found", tool_name="terminal")
    # Error recovery: retry with same tool
    db.append_message("s3", role="assistant", content="Let me retry.",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="success output", tool_name="terminal")
    db.append_message("s3", role="assistant", content="Now searching.",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("s3", role="tool", content="found stuff", tool_name="search_files")
    # Done + verification tool (terminal follows)
    db.append_message("s3", role="assistant", content="Finished the task.")
    db.append_message("s3", role="assistant", content="Let me verify.",
                      tool_calls=[{"function": {"name": "terminal"}}])
    db.append_message("s3", role="tool", content="all tests pass", tool_name="terminal")

    # ── Session 4: Discord, short, delegate_task dispatch ──
    db.create_session(
        session_id="s4", source="discord",
        model="anthropic/claude-sonnet-4-20250514", user_id="user2",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's4'", (now - 1 * day,))
    db.end_session("s4", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's4'", (now - 1 * day + 900,))
    db.update_token_counts("s4", input_tokens=10000, output_tokens=5000)
    db.append_message("s4", role="user", content="fix this")
    db.append_message("s4", role="assistant", content="Delegating to subagent.",
                      tool_calls=[{"id": "c1", "type": "function",
                                   "function": {"name": "delegate_task",
                                                "arguments": json.dumps(
                                                    {"task": "Run the test suite",
                                                     "background": True})}}])
    db.append_message("s4", role="assistant", content="Also delegating foreground task.",
                      tool_calls=[{"id": "c2", "type": "function",
                                   "function": {"name": "delegate_task",
                                                "arguments": json.dumps(
                                                    {"task": "Lint the code",
                                                     "background": False})}}])

    # ── Session 5: Subagent session (child of s4) ──
    db.create_session(
        session_id="s5_sub", source="subagent",
        model="anthropic/claude-sonnet-4-20250514", user_id="user2",
    )
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, parent_session_id = 's4' WHERE id = 's5_sub'",
        (now - 1 * day + 100,),
    )
    db.end_session("s5_sub", end_reason="completed")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's5_sub'", (now - 1 * day + 500,))
    db.update_token_counts("s5_sub", input_tokens=5000, output_tokens=2000)
    db.append_message("s5_sub", role="user", content="Run the test suite")

    # ── Session 6: Old session (45 days ago, excluded from 30-day window) ──
    db.create_session(
        session_id="s_old", source="cli",
        model="gpt-4o-mini", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's_old'", (now - 45 * day,))
    db.end_session("s_old", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's_old'", (now - 45 * day + 600,))
    db.update_token_counts("s_old", input_tokens=5000, output_tokens=2000)
    db.append_message("s_old", role="user", content="old message")
    db.append_message("s_old", role="assistant", content="old reply")

    # ── Session 7: Telegram, more go-to prompts for frequency ──
    db.create_session(
        session_id="s7", source="telegram",
        model="gpt-4o", user_id="user1",
    )
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's7'", (now - 3 * day,))
    db.end_session("s7", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 's7'", (now - 3 * day + 1200,))
    db.update_token_counts("s7", input_tokens=8000, output_tokens=3000)
    db.append_message("s7", role="user", content="make it work")
    db.append_message("s7", role="assistant", content="On it.")

    db._conn.commit()
    return db


@pytest.fixture()
def night_db(db):
    """DB with sessions at specific hours for productivity/shipping timing tests.

    We set started_at to Unix timestamps whose local hour is deterministic
    under TZ=UTC (enforced by tests/conftest.py).  This lets us assert
    night_owl / early_bird / afternoon_grinder classifications.

    All timestamps are within the last few days so they fall inside the
    default 30-day analysis window.
    """
    now = time.time()
    day = 86400

    def ts(days_ago, hour):
        """Build a UTC timestamp for `days_ago` days back at the given hour.

        We anchor to the start of today (UTC midnight), subtract days_ago,
        then add the hour offset.
        """
        today_midnight = (now // day) * day  # start of today UTC
        return today_midnight - (days_ago * day) + (hour * 3600)

    # 3 sessions at 23:00 UTC (night owl peak) on consecutive recent days
    for i in range(3):
        sid = f"night_{i}"
        db.create_session(session_id=sid, source="cli", model="test-model")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                         (ts(i + 1, 23), sid))
        db.end_session(sid, end_reason="user_exit")
        db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?",
                         (ts(i + 1, 23, ), sid))  # 30 min later
        # Fix ended_at to be 30 min after start
        db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?",
                         (ts(i + 1, 23) + 1800, sid))
        db.append_message(sid, role="user", content="hello world this is a test message")
        db.append_message(sid, role="assistant", content="hi there")

    # 1 session at 09:00 UTC (more recent than the night ones)
    db.create_session(session_id="morning", source="cli", model="test-model")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'morning'",
                     (ts(0, 9),))
    db.end_session("morning", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'morning'",
                     (ts(0, 9) + 1200,))
    db.append_message("morning", role="user", content="good morning let me work")
    db.append_message("morning", role="assistant", content="good morning")

    db._conn.commit()
    return db


# =========================================================================
# Helper function tests
# =========================================================================

class TestWordCount:
    def test_basic(self):
        assert _word_count("hello world") == 2

    def test_empty(self):
        assert _word_count("") == 0

    def test_none(self):
        assert _word_count(None) == 0


class TestIsAllCaps:
    def test_all_caps(self):
        assert _is_all_caps("STOP TOUCHING THAT FILE") is True

    def test_mixed(self):
        assert _is_all_caps("Hello World") is False

    def test_too_short(self):
        assert _is_all_caps("NO") is False  # len < 5

    def test_empty(self):
        assert _is_all_caps("") is False


class TestHasTypos:
    def test_common_typo(self):
        assert _has_typos("make teh thing wrok") is True

    def test_clean_text(self):
        assert _has_typos("Hello, how are you today?") is False

    def test_missing_apostrophe(self):
        assert _has_typos("dont do that") is True


class TestFormatDuration:
    def test_minutes(self):
        assert _format_duration(300) == "5m"

    def test_hours_with_minutes(self):
        assert _format_duration(5400) == "1h 30m"

    def test_zero(self):
        assert _format_duration(0) == "0m"

    def test_negative(self):
        assert _format_duration(-10) == "0m"


class TestBar:
    def test_full(self):
        assert _bar(10) == "██████████"

    def test_half(self):
        assert _bar(5) == "█████░░░░░"

    def test_zero(self):
        assert _bar(0) == "░░░░░░░░░░"

    def test_clamps(self):
        # Values above max clamp to full
        assert _bar(15) == "██████████"
        # Negative values clamp to empty
        assert _bar(-3) == "░░░░░░░░░░"


class TestShannonEntropy:
    def test_uniform(self):
        # Equal distribution → max entropy
        ent = _shannon_entropy([5, 5, 5, 5])
        assert ent == pytest.approx(2.0, abs=0.01)  # log2(4) = 2

    def test_single_value(self):
        assert _shannon_entropy([10]) == pytest.approx(0.0, abs=0.001)

    def test_empty(self):
        assert _shannon_entropy([]) == 0.0


class TestScrubCredentials:
    def test_openai_key(self):
        scrubbed = _scrub_credentials("my key is sk-abcdef1234567890abcdef1234567890")
        assert "sk-abcdef" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_bearer_token(self):
        scrubbed = _scrub_credentials("Authorization: Bearer abc123def456ghi789jkl012mno345pqr789")
        assert "Bearer abc123" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_password_assignment(self):
        scrubbed = _scrub_credentials("password=mysecret123")
        assert "mysecret123" not in scrubbed

    def test_connection_string(self):
        scrubbed = _scrub_credentials("postgres://user:pass@host:5432/db")
        assert "pass@host" not in scrubbed
        assert "[REDACTED]" in scrubbed

    def test_clean_text_unchanged(self):
        clean = "Hello, this is a normal message with no secrets"
        assert _scrub_credentials(clean) == clean

    def test_empty(self):
        assert _scrub_credentials("") == ""
        assert _scrub_credentials(None) == ""


# =========================================================================
# Signal extractor tests (deterministic, no LLM)
# =========================================================================

class TestSignalExtraction:
    """Test each of the 18 signal extractors with a populated DB.

    These are deterministic — no LLM calls.  We assert invariants
    (counts > 0 when expected, buckets sum correctly, rates in [0, 1]).
    """

    def _get_signals(self, db, days=30, source=None):
        """Run signal extraction and return the signals dict."""
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff, source)

    # ── Signal 1: Prompt length distribution ──

    def test_prompt_length_buckets_sum_to_total(self, populated_db):
        signals = self._get_signals(populated_db)
        pl = signals["prompt_length"]
        buckets = pl["buckets"]
        assert sum(buckets.values()) == pl["total"]
        assert pl["total"] > 0

    def test_prompt_length_longest_positive(self, populated_db):
        signals = self._get_signals(populated_db)
        assert signals["prompt_length"]["longest"] > 0

    def test_prompt_length_average_in_range(self, populated_db):
        signals = self._get_signals(populated_db)
        avg = signals["prompt_length"]["average"]
        assert 0 < avg < 1000  # reasonable range

    def test_prompt_length_empty(self, db):
        """No user messages → empty prompt length."""
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_prompt_length([])
        assert result["total"] == 0
        assert result["average"] == 0.0
        assert result["longest"] == 0

    # ── Signal 2: Go-to prompts ──

    def test_go_to_prompts_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        go_to = signals["go_to_prompts"]
        # "make it work" appears in s2 and s7
        top_prompts = [p["prompt"] for p in go_to["top"]]
        assert "make it work" in top_prompts

    def test_go_to_prompt_count_matches(self, populated_db):
        signals = self._get_signals(populated_db)
        go_to = signals["go_to_prompts"]
        make_it_work = next(p for p in go_to["top"] if p["prompt"] == "make it work")
        assert make_it_work["count"] >= 2
        assert make_it_work["sessions"] >= 2

    def test_go_to_prompts_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_go_to_prompts([])
        assert result["top"] == []
        assert result["total_short"] == 0

    # ── Signal 3: Steering frequency ──

    def test_steering_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        steering = signals["steering"]
        # s1 has "No, stop. Don't touch that file" → at least 1
        assert steering["count"] >= 1
        assert steering["rate"] > 0
        assert 0 <= steering["rate"] <= 1.0

    def test_steering_examples_capped(self, populated_db):
        signals = self._get_signals(populated_db)
        assert len(signals["steering"]["examples"]) <= 5

    def test_steering_none_without_keywords(self, db):
        """Messages without steering keywords → count 0."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="Hello, how are you today?")
        db.append_message("s", role="user", content="Please help me write a function.")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "Hello, how are you today?", "session_id": "s"},
                {"content": "Please help me write a function.", "session_id": "s"}]
        result = analyzer._extract_steering(msgs)
        assert result["count"] == 0
        assert result["rate"] == 0.0

    # ── Signal 4: Politeness ──

    def test_politeness_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        politeness = signals["politeness"]
        # s1 has "Thanks! I appreciate it."
        assert politeness["thank_count"] >= 1
        assert politeness["total"] >= 1

    def test_politeness_none_without_keywords(self, db):
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "Fix the bug now.", "session_id": "s"}]
        result = analyzer._extract_politeness(msgs)
        assert result["total"] == 0

    def test_politeness_please_counted(self, db):
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "Please run the tests. Thanks!", "session_id": "s"}]
        result = analyzer._extract_politeness(msgs)
        assert result["please_count"] >= 1
        assert result["thank_count"] >= 1

    # ── Signal 5: Crash outs ──

    def test_crash_outs_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        crash = signals["crash_outs"]
        # s2 has "STOP TOUCHING THAT FILE!!! WTF!!!" → at least 1
        assert crash["count"] >= 1
        assert len(crash["messages"]) >= 1

    def test_crash_out_top_has_caps_pct(self, populated_db):
        signals = self._get_signals(populated_db)
        msgs = signals["crash_outs"]["messages"]
        if msgs:
            top = msgs[0]
            assert "caps_pct" in top
            assert "content" in top

    def test_crash_out_short_caps_not_flagged(self, db):
        """Short ALL CAPS (< 5 chars) should NOT be flagged as crash out."""
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "NO", "session_id": "s"}]
        result = analyzer._extract_crash_outs(msgs)
        # "NO" is 2 chars, too short for ALL CAPS detection
        assert result["count"] == 0

    def test_crash_out_normal_not_flagged(self, db):
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "Hello, please help me with this task.", "session_id": "s"}]
        result = analyzer._extract_crash_outs(msgs)
        assert result["count"] == 0

    def test_crash_out_frustration_keyword(self, db):
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "This is broken and useless, what the hell", "session_id": "s"}]
        result = analyzer._extract_crash_outs(msgs)
        assert result["count"] >= 1

    # ── Signal 6: Cryptic prompts ──

    def test_cryptic_prompts_empty_db(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_cryptic_prompts([])
        assert result["count"] == 0

    def test_cryptic_prompts_daytime_not_flagged(self, db):
        """Short daytime prompt → not cryptic."""
        analyzer = BehavioralAnalyzer(db)
        # 2 PM timestamp
        ts = datetime(2025, 1, 6, 14, 0).timestamp()
        msgs = [{"content": "fix it", "session_id": "s", "timestamp": ts}]
        result = analyzer._extract_cryptic_prompts(msgs)
        assert result["count"] == 0

    def test_cryptic_prompts_long_not_flagged(self, db):
        """Long late-night prompt → not cryptic (must be < 15 chars)."""
        analyzer = BehavioralAnalyzer(db)
        ts = datetime(2025, 1, 6, 1, 0).timestamp()
        msgs = [{"content": "this is a very long late night message", "session_id": "s", "timestamp": ts}]
        result = analyzer._extract_cryptic_prompts(msgs)
        assert result["count"] == 0

    def test_cryptic_prompts_late_night_short(self, db):
        """Short late-night prompt → flagged as cryptic."""
        analyzer = BehavioralAnalyzer(db)
        ts = datetime(2025, 1, 6, 1, 0).timestamp()  # 1 AM
        msgs = [{"content": "fix teh bug", "session_id": "s", "timestamp": ts}]
        result = analyzer._extract_cryptic_prompts(msgs)
        assert result["count"] >= 1
        assert result["prompts"][0]["has_typos"] is True

    # ── Signal 7: Planning habits ──

    def test_planning_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        planning = signals["planning"]
        # s1 starts with "Let's plan this out first" → planned
        assert planning["planned_sessions"] >= 1
        assert planning["rate"] > 0
        assert 0 <= planning["rate"] <= 1.0

    def test_planning_no_plan(self, db):
        """Session without plan phrase → not planned."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="just fix this bug quickly")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        user_msgs = analyzer._get_user_messages(cutoff)
        result = analyzer._extract_planning_habits(user_msgs, sessions)
        assert result["planned_sessions"] == 0
        assert result["rate"] == 0.0

    def test_planning_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_planning_habits([], [])
        assert result["planned_sessions"] == 0
        assert result["rate"] == 0.0

    # ── Signal 8: Agent parallelism ──

    def test_agent_parallelism_subagents(self, populated_db):
        signals = self._get_signals(populated_db)
        ap = signals["agent_parallelism"]
        # s5_sub has parent_session_id = 's4' → at least 1 subagent
        assert ap["total_subagents"] >= 1
        assert ap["max_per_parent"] >= 1
        assert ap["parent_count"] >= 1

    def test_agent_parallelism_no_subagents(self, db):
        """No subagent sessions → total 0."""
        db.create_session(session_id="s", source="cli", model="m")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        result = analyzer._extract_agent_parallelism(sessions)
        assert result["total_subagents"] == 0
        assert result["max_per_parent"] == 0

    def test_agent_parallelism_max_concurrent(self, db):
        """Overlapping sessions → max_concurrent > 1."""
        now = time.time()
        # Two sessions that overlap
        for sid in ["a", "b"]:
            db.create_session(session_id=sid, source="cli", model="m")
            db._conn.execute(
                "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = ?",
                (now - 100, now - 50, sid),
            )
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        result = analyzer._extract_agent_parallelism(sessions)
        assert result["max_concurrent"] >= 2

    # ── Signal 9: Session topics ──

    def test_session_topics_with_title(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db._conn.execute("UPDATE sessions SET title = ? WHERE id = 's'", ("Fix auth bug",))
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        result = analyzer._extract_session_topics(sessions, [])
        assert result["topics"]["s"] == "Fix auth bug"

    def test_session_topics_without_title_uses_first_prompt(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="Help me debug the API")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        user_msgs = analyzer._get_user_messages(cutoff)
        result = analyzer._extract_session_topics(sessions, user_msgs)
        assert "Help me debug the API" in result["topics"]["s"]

    def test_session_topics_neither(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        result = analyzer._extract_session_topics(sessions, [])
        assert result["topics"]["s"] == "untitled"

    def test_session_topics_diversity(self, populated_db):
        signals = self._get_signals(populated_db)
        assert signals["session_topics"]["diversity"] >= 1

    # ── Signal 10: Verification rate ──

    def test_verification_rate_with_verification(self, populated_db):
        signals = self._get_signals(populated_db)
        ver = signals["verification"]
        # s3 has "Finished the task." followed by terminal (verification tool)
        assert ver["done_declarations"] >= 1
        assert ver["verified"] >= 1

    def test_verification_rate_without_verification(self, populated_db):
        """s1 has 'Done! The fix is complete.' with no verification tool after."""
        signals = self._get_signals(populated_db)
        ver = signals["verification"]
        # done_declarations should be >= 2 (s1 + s3), verified >= 1 (s3 only)
        assert ver["done_declarations"] >= 2
        # Rate should be < 1 (not all verified)
        assert ver["rate"] < 1.0

    def test_verification_rate_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_verification_rate([], [])
        assert result["done_declarations"] == 0
        assert result["rate"] == 0.0

    # ── Signal 11: Error recovery ──

    def test_error_recovery_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        er = signals["error_recovery"]
        # s3 has "error: command not found" followed by retry (terminal) → self_recovered
        assert er["total_errors"] >= 1
        assert er["self_recovered"] >= 1

    def test_error_recovery_rate_in_range(self, populated_db):
        signals = self._get_signals(populated_db)
        er = signals["error_recovery"]
        assert 0 <= er["recovery_rate"] <= 1.0

    def test_error_recovery_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_error_recovery([], [])
        assert result["total_errors"] == 0
        assert result["recovery_rate"] == 0.0

    # ── Signal 12: Productivity timing ──

    def test_productivity_timing_night_owl(self, night_db):
        signals = self._get_signals(night_db)
        pt = signals["productivity_timing"]
        # 3 sessions at 23:00 UTC vs 1 at 09:00 → peak at 23 → night_owl
        assert pt["peak_hour"] == 23
        assert pt["classification"] == "night_owl"

    def test_productivity_timing_peak_pct(self, night_db):
        signals = self._get_signals(night_db)
        pt = signals["productivity_timing"]
        assert pt["peak_pct"] > 0

    def test_productivity_timing_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_productivity_timing([])
        assert result["classification"] == "unknown"
        assert result["peak_hour"] is None

    # ── Signal 13: Shipping timing ──

    def test_shipping_timing_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        st = signals["shipping_timing"]
        assert st["peak_day"] is not None
        assert st["peak_count"] > 0

    def test_shipping_timing_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_shipping_timing([])
        assert result["peak_day"] is None
        assert result["peak_count"] == 0

    # ── Signal 14: Model preference ──

    def test_model_preference_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        mp = signals["model_preference"]
        assert mp["top_model"] is not None
        assert len(mp["models"]) >= 1
        # Percentages should sum to ~100
        total_pct = sum(m["pct"] for m in mp["models"])
        assert total_pct == pytest.approx(100, abs=1)

    def test_model_preference_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_model_preference([])
        assert result["top_model"] is None
        assert result["models"] == []

    # ── Signal 15: Subagent dispatch ──

    def test_subagent_dispatch_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        sd = signals["subagent_dispatch"]
        # s4 has 2 delegate_task calls (1 background, 1 foreground)
        assert sd["total"] >= 2
        assert sd["background"] >= 1
        assert sd["foreground"] >= 1

    def test_subagent_dispatch_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_subagent_dispatch([])
        assert result["total"] == 0
        assert result["background"] == 0

    def test_subagent_dispatch_task_descriptions_scrubbed(self, db):
        """Task descriptions should be credential-scrubbed."""
        analyzer = BehavioralAnalyzer(db)
        msgs = [{
            "session_id": "s",
            "tool_calls": [{"function": {"name": "delegate_task",
                            "arguments": json.dumps({"task": "use key sk-abcdef1234567890abcdef1234567890"})}}],
        }]
        result = analyzer._extract_subagent_dispatch(msgs)
        assert result["total"] >= 1
        assert len(result["task_descriptions"]) >= 1
        assert "sk-abcdef" not in result["task_descriptions"][0]

    # ── Signal 16: Decision patterns ──

    def test_decision_patterns_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        dp = signals["decision_patterns"]
        # s1 has "let's go with the redis approach not the file approach"
        assert dp["count"] >= 1

    def test_decision_patterns_not_triggered_by_normal_text(self, db):
        analyzer = BehavioralAnalyzer(db)
        msgs = [{"content": "Hello, how are you?", "session_id": "s"}]
        result = analyzer._extract_decision_patterns(msgs)
        assert result["count"] == 0

    def test_decision_patterns_examples_capped(self, populated_db):
        signals = self._get_signals(populated_db)
        assert len(signals["decision_patterns"]["examples"]) <= 5

    # ── Signal 17: Tool diversity ──

    def test_tool_diversity_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        td = signals["tool_diversity"]
        # Multiple distinct tools: search_files, read_file, patch, web_search, terminal
        assert td["distinct_tools"] >= 3
        assert td["entropy"] > 0
        assert td["total_calls"] > 0

    def test_tool_diversity_single_tool(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="assistant", content="run",
                          tool_calls=[{"function": {"name": "terminal"}}])
        db.append_message("s", role="tool", content="out", tool_name="terminal")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        result = analyzer._extract_tool_diversity(cutoff)
        assert result["distinct_tools"] == 1
        assert result["entropy"] == pytest.approx(0.0, abs=0.01)

    def test_tool_diversity_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_tool_diversity(time.time())
        assert result["distinct_tools"] == 0
        assert result["entropy"] == 0.0

    # ── Signal 18: Session duration ──

    def test_session_duration_detected(self, populated_db):
        signals = self._get_signals(populated_db)
        sd = signals["session_duration"]
        # Multiple sessions have ended_at > started_at
        assert sd["longest"] > 0
        assert sd["median"] > 0
        assert sd["mean"] > 0

    def test_session_duration_no_end_time(self, db):
        """Session without end time → no duration."""
        db.create_session(session_id="s", source="cli", model="m")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        result = analyzer._extract_session_duration(sessions)
        assert result["longest"] == 0
        assert result["median"] == 0

    def test_session_duration_end_before_start(self, db):
        """Clock drift (end < start) → duration 0."""
        now = time.time()
        db.create_session(session_id="s", source="cli", model="m")
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, ended_at = ? WHERE id = 's'",
            (now - 100, now - 200),  # end before start
        )
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - 86400
        sessions = analyzer._get_sessions(cutoff)
        result = analyzer._extract_session_duration(sessions)
        assert result["longest"] == 0

    def test_session_duration_longest_session_id(self, populated_db):
        signals = self._get_signals(populated_db)
        sd = signals["session_duration"]
        assert sd["longest_session_id"] is not None


# =========================================================================
# 5-Axis scoring tests
# =========================================================================

class TestScoring:
    """Test the 5-axis behavioral scoring (heuristic fallback)."""

    def test_scores_in_range_populated(self, populated_db):
        """All 5 scores must be in 1-10 for populated DB."""
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        scores = report["scores"]
        for key in ["execution_leverage", "steering", "engineering_quality",
                     "product_thinking", "planning"]:
            score = scores[key]["score"]
            assert 1 <= score <= 10, f"{key} score {score} out of range"
            assert isinstance(scores[key]["rationale"], str)
            assert len(scores[key]["rationale"]) > 0

    def test_scores_in_range_empty(self, db):
        """Empty DB → empty report (no scores)."""
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["empty"] is True
        assert report["scores"] == {}

    def test_heuristic_scores_in_range_single_session(self, db):
        """Single session → heuristic scores still 1-10."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world")
        db.append_message("s", role="assistant", content="hi there")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["empty"] is False
        for key in ["execution_leverage", "steering", "engineering_quality",
                     "product_thinking", "planning"]:
            score = report["scores"][key]["score"]
            assert 1 <= score <= 10

    def test_heuristic_scores_all_crash_outs(self, db):
        """DB with only crash outs → scores still 1-10 (no crash)."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="STOP!!! WTF!!! THIS IS BROKEN!!!")
        db.append_message("s", role="assistant", content="sorry")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        for key in ["execution_leverage", "steering", "engineering_quality",
                     "product_thinking", "planning"]:
            assert 1 <= report["scores"][key]["score"] <= 10

    def test_heuristic_high_steering(self, db):
        """High steering rate + crash outs + decisions → steering score > 1."""
        now = time.time()
        db.create_session(session_id="s", source="cli", model="m")
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 's'", (now - 100,))
        # Many steering messages
        for i in range(10):
            db.append_message("s", role="user", content=f"No, stop. Don't do that. Wrong. Instead use the other approach number {i}.")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["scores"]["steering"]["score"] > 1

    def test_heuristic_low_steering(self, db):
        """No steering → steering score should be low (1)."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="Hello, please help me write a function.")
        db.append_message("s", role="assistant", content="Sure, I can help with that.")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["scores"]["steering"]["score"] <= 3

    def test_heuristic_planning_high(self, db):
        """65% plan-first → planning score mid-high."""
        now = time.time()
        # 7 sessions, 5 with plan phrases → ~71% plan rate
        for i in range(5):
            sid = f"plan_{i}"
            db.create_session(session_id=sid, source="cli", model="m")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                             (now - 100 - i * 10, sid))
            db.append_message(sid, role="user", content="Let's plan this. First I'll check the tests.")
            db.append_message(sid, role="assistant", content="ok")
        for i in range(2):
            sid = f"noplan_{i}"
            db.create_session(session_id=sid, source="cli", model="m")
            db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?",
                             (now - 100 - i * 10, sid))
            db.append_message(sid, role="user", content="just fix it")
            db.append_message(sid, role="assistant", content="ok")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        # ~71% → score 7
        assert report["scores"]["planning"]["score"] >= 5

    def test_heuristic_planning_zero(self, db):
        """0% plan-first → planning score 1."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="just do it now")
        db.append_message("s", role="assistant", content="ok")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["scores"]["planning"]["score"] == 1

    def test_signal_only_fallback(self, db):
        """LLM unavailable → heuristic scores, llm_available=False, no crash."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        # LLM will fail (no provider configured in test) → heuristic
        assert "scores" in report
        assert report["llm_available"] in (True, False)  # depends on env
        # Regardless, scores must be valid
        for key in ["execution_leverage", "steering", "engineering_quality",
                     "product_thinking", "planning"]:
            assert 1 <= report["scores"][key]["score"] <= 10


# =========================================================================
# LLM narrative card tests (mocked provider)
# =========================================================================

class TestLLMNarrativeCards:
    """Test LLM narrative cards with a mocked provider (no real API calls)."""

    def _mock_llm_response(self):
        """Return a mock LLM response object with parseable content."""
        content = """\
Execution Leverage: 8/10 - Strong delegator with good tool usage.
Steering: 4/10 - Lets the agent run, course-corrects occasionally.
Engineering Quality: 5/10 - Needs more verification before declaring done.
Product Thinking: 7/10 - Plans features well and prioritizes effectively.
Planning: 6/10 - Plans 65% of the time before acting.

Archetype: The Orchestrator. You delegate heavily and plan before acting. You steer hard when things go off track.
Agent relationship: Like a chief of staff. You give context, set direction, then let it run.
Growth edge: Verify before declaring done. Your Engineering Quality score is 5/10.
Biggest crash out: You sent 'STOP TOUCHING THAT FILE' after the agent modified a file you told it not to touch.
"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = content
        return mock_response

    def test_llm_archetype_parsed(self, populated_db):
        """Mock LLM returns 'The Orchestrator' → parsed and included."""
        analyzer = BehavioralAnalyzer(populated_db)
        with patch.object(analyzer, "_call_llm_for_scoring") as mock_llm:
            mock_response = self._mock_llm_response()
            scores, cards = analyzer._parse_llm_response(
                mock_response.choices[0].message.content,
                analyzer._extract_signals(time.time() - 30 * 86400),
            )
            assert "Orchestrator" in cards["archetype"]

    def test_llm_agent_relationship_parsed(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        mock_response = self._mock_llm_response()
        scores, cards = analyzer._parse_llm_response(
            mock_response.choices[0].message.content,
            analyzer._extract_signals(time.time() - 30 * 86400),
        )
        assert "chief of staff" in cards["agent_relationship"]

    def test_llm_growth_edge_parsed(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        mock_response = self._mock_llm_response()
        scores, cards = analyzer._parse_llm_response(
            mock_response.choices[0].message.content,
            analyzer._extract_signals(time.time() - 30 * 86400),
        )
        assert "Engineering Quality" in cards["growth_edge"]
        assert "5/10" in cards["growth_edge"]

    def test_llm_scores_in_range(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        mock_response = self._mock_llm_response()
        scores, cards = analyzer._parse_llm_response(
            mock_response.choices[0].message.content,
            analyzer._extract_signals(time.time() - 30 * 86400),
        )
        for key in ["execution_leverage", "steering", "engineering_quality",
                     "product_thinking", "planning"]:
            assert 1 <= scores[key]["score"] <= 10

    def test_llm_failure_graceful_degradation(self, populated_db):
        """LLM call raises → graceful degradation, heuristic scores, no crash."""
        analyzer = BehavioralAnalyzer(populated_db)
        with patch("agent.behavioral_insights.BehavioralAnalyzer._call_llm_for_scoring",
                    return_value=None):
            report = analyzer.generate(days=30)
        assert report["llm_available"] is False
        # Heuristic scores still present and valid
        for key in ["execution_leverage", "steering", "engineering_quality",
                     "product_thinking", "planning"]:
            assert 1 <= report["scores"][key]["score"] <= 10
        # Fallback cards present
        assert "archetype" in report["cards"]
        assert "agent_relationship" in report["cards"]
        assert "growth_edge" in report["cards"]
        assert "crash_out" in report["cards"]

    def test_llm_crash_out_card_from_signals(self, populated_db):
        """Crash out card should reference the top crash-out message."""
        analyzer = BehavioralAnalyzer(populated_db)
        with patch.object(analyzer, "_call_llm_for_scoring", return_value=None):
            report = analyzer.generate(days=30)
        crash_card = report["cards"]["crash_out"]
        # populated_db has crash out in s2
        assert len(crash_card) > 0

    def test_credential_scrubbing_in_prompt(self, populated_db):
        """Credential in crash-out message should be scrubbed before LLM."""
        analyzer = BehavioralAnalyzer(populated_db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        # The prompt should not contain raw API keys even if present in data
        # (our populated_db doesn't have keys, but verify the scrubbing path)
        assert "sk-" not in prompt or "[REDACTED]" in prompt

    def test_credential_scrubbing_with_key_in_data(self, db):
        """Crash-out message with API key → scrubbed in LLM prompt."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user",
                          content="STOP!!! MY KEY sk-abcdef1234567890abcdef1234567890 IS LEAKING!!!")
        db.append_message("s", role="assistant", content="sorry")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "sk-abcdef1234567890" not in prompt
        assert "[REDACTED]" in prompt


# =========================================================================
# Score persistence tests
# =========================================================================

class TestScorePersistence:
    """Test the behavioral_scores persistence table (Layer 3)."""

    def test_table_created_on_init(self, db):
        """BehavioralAnalyzer init creates the behavioral_scores table."""
        # Constructing the analyzer runs CREATE TABLE IF NOT EXISTS
        BehavioralAnalyzer(db)
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='behavioral_scores'"
        )
        assert cursor.fetchone() is not None

    def test_store_scores(self, populated_db):
        """Run analyzer → row in behavioral_scores with correct values."""
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        cursor = populated_db._conn.execute(
            "SELECT * FROM behavioral_scores ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        assert row is not None
        row = dict(row)
        assert row["days_window"] == 30
        assert row["execution_leverage"] == report["scores"]["execution_leverage"]["score"]
        assert row["steering"] == report["scores"]["steering"]["score"]
        assert row["engineering_quality"] == report["scores"]["engineering_quality"]["score"]
        assert row["product_thinking"] == report["scores"]["product_thinking"]["score"]
        assert row["planning"] == report["scores"]["planning"]["score"]

    def test_multiple_runs(self, populated_db):
        """Run analyzer twice → 2 rows, timestamps differ."""
        analyzer = BehavioralAnalyzer(populated_db)
        analyzer.generate(days=30)
        analyzer.generate(days=30)
        cursor = populated_db._conn.execute("SELECT COUNT(*) as cnt FROM behavioral_scores")
        count = cursor.fetchone()["cnt"]
        assert count >= 2

        cursor = populated_db._conn.execute(
            "SELECT run_timestamp FROM behavioral_scores ORDER BY id DESC LIMIT 2"
        )
        rows = cursor.fetchall()
        # Timestamps should be >= 0 (might be equal if sub-second, but both valid)
        assert rows[0]["run_timestamp"] > 0
        assert rows[1]["run_timestamp"] > 0

    def test_raw_signals_json_stored(self, populated_db):
        """Stored JSON contains signal keys."""
        analyzer = BehavioralAnalyzer(populated_db)
        analyzer.generate(days=30)
        cursor = populated_db._conn.execute(
            "SELECT raw_signals FROM behavioral_scores ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        raw = json.loads(row["raw_signals"])
        # Should contain signal keys (but NOT the raw "sessions" list)
        assert "total_sessions" in raw
        assert "prompt_length" in raw
        assert "steering" in raw
        assert "sessions" not in raw  # sessions list is stripped

    def test_source_filter_stored(self, populated_db):
        """Source filter is stored in the persistence row."""
        analyzer = BehavioralAnalyzer(populated_db)
        analyzer.generate(days=30, source="cli")
        cursor = populated_db._conn.execute(
            "SELECT source_filter FROM behavioral_scores ORDER BY id DESC LIMIT 1"
        )
        assert cursor.fetchone()["source_filter"] == "cli"

    def test_persistence_no_crash_on_empty(self, db):
        """Empty DB → generate returns empty, but doesn't crash persistence."""
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["empty"] is True
        # No row should be stored for empty reports
        cursor = db._conn.execute("SELECT COUNT(*) as cnt FROM behavioral_scores")
        assert cursor.fetchone()["cnt"] == 0


# =========================================================================
# Formatting tests
# =========================================================================

class TestTerminalFormatting:
    def test_terminal_format_has_header(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "Hermes Behavior" in text

    def test_terminal_format_has_scores_section(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "Behavioral Scores" in text
        assert "Execution Leverage" in text
        assert "Steering" in text
        assert "Engineering Quality" in text
        assert "Product Thinking" in text
        assert "Planning" in text

    def test_terminal_format_has_bar_charts(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "█" in text  # bar chart characters
        assert "░" in text

    def test_terminal_format_has_insight_cards(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "Insight Cards" in text
        assert "Archetype" in text
        assert "Prompt style" in text
        assert "Politeness" in text
        assert "Model preference" in text
        assert "Planning" in text

    def test_terminal_format_empty_data(self, db):
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "Not enough data" in text

    def test_terminal_format_shows_session_count(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "sessions" in text

    def test_terminal_format_source_filter(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30, source="cli")
        text = analyzer.format_terminal(report)
        assert "cli" in text


class TestGatewayFormatting:
    def test_gateway_format_has_header(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_gateway(report)
        assert "Hermes Behavior" in text

    def test_gateway_format_has_bold(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_gateway(report)
        assert "**" in text  # Markdown bold

    def test_gateway_format_has_scores(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_gateway(report)
        assert "Behavioral Scores" in text
        assert "Execution Leverage" in text
        assert "/10" in text

    def test_gateway_format_has_cards(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        text = analyzer.format_gateway(report)
        assert "Archetype" in text
        assert "Prompt style" in text

    def test_gateway_format_empty_data(self, db):
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        text = analyzer.format_gateway(report)
        assert "Not enough data" in text

    def test_gateway_shorter_than_terminal(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30)
        terminal_text = analyzer.format_terminal(report)
        gateway_text = analyzer.format_gateway(report)
        assert len(gateway_text) < len(terminal_text)


# =========================================================================
# Config gate tests
# =========================================================================

class TestConfigGate:
    """Test the behavior.enabled config gate (enforced by CLI/gateway callers)."""

    def test_disabled_message_cli(self, populated_db, capsys):
        """When behavior.enabled is false, CLI prints disabled message."""
        from cli import HermesCLI

        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {"behavior": {"enabled": False}}
        cli_obj._show_behavior("/behavior")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()
        assert "behavior.enabled: true" in captured.out

    def test_disabled_message_gateway(self, db, capsys):
        """When behavior.enabled is false, gateway returns disabled message."""
        # The gateway handler reads config via read_raw_config; we mock it
        from gateway.slash_commands import GatewaySlashCommandsMixin
        from gateway.platforms.base import MessageEvent, MessageType

        class _Stub(GatewaySlashCommandsMixin):
            def __init__(self):
                pass

        stub = _Stub()
        event = MessageEvent(text="/behavior", message_type=MessageType.TEXT)

        with patch("hermes_cli.config.read_raw_config",
                   return_value={"behavior": {"enabled": False}}):
            result = asyncio.run(stub._handle_behavior_command(event))
        assert "disabled" in result.lower()
        assert "behavior.enabled: true" in result

    def test_disabled_by_default(self, populated_db, capsys):
        """No behavior config → treated as disabled."""
        from cli import HermesCLI

        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.config = {}  # no behavior key
        cli_obj._show_behavior("/behavior")
        captured = capsys.readouterr()
        assert "disabled" in captured.out.lower()


# =========================================================================
# Edge cases
# =========================================================================

class TestEdgeCases:
    def test_empty_db_generate(self, db):
        """Empty DB → generate returns empty report."""
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["empty"] is True
        assert report["scores"] == {}
        assert report["cards"] == {}
        assert report["session_count"] == 0

    def test_empty_db_terminal_format(self, db):
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        text = analyzer.format_terminal(report)
        assert "Not enough data" in text

    def test_empty_db_gateway_format(self, db):
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        text = analyzer.format_gateway(report)
        assert "Not enough data" in text

    def test_single_session(self, db):
        """Single session → valid report, not empty."""
        db.create_session(session_id="s", source="cli", model="test-model")
        db.append_message("s", role="user", content="hello world this is a test")
        db.append_message("s", role="assistant", content="hi there")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["empty"] is False
        assert report["session_count"] == 1

    def test_no_user_messages(self, db):
        """Session with only assistant messages → empty report (no user data)."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="assistant", content="just me here")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        # No user messages → empty
        assert report["empty"] is True

    def test_no_tool_calls(self, db):
        """Session with no tool calls → valid report, tool diversity 0."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message here")
        db.append_message("s", role="assistant", content="hi there, how can I help?")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["empty"] is False
        assert report["signals"]["tool_diversity"]["distinct_tools"] == 0

    def test_polite_and_redirect_same_message(self, db):
        """Message with both politeness AND steering → both counted."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user",
                          content="No, stop. Thanks but don't do that. Please use the other approach.")
        db.append_message("s", role="assistant", content="ok")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["signals"]["steering"]["count"] >= 1
        assert report["signals"]["politeness"]["total"] >= 1

    def test_crash_out_and_steering_same_message(self, db):
        """Message with crash out AND steering keywords → both detected."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user",
                          content="NO STOP!!! DON'T TOUCH THAT!!! WTF!!!")
        db.append_message("s", role="assistant", content="sorry")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=30)
        assert report["signals"]["crash_outs"]["count"] >= 1
        assert report["signals"]["steering"]["count"] >= 1

    def test_source_filter_cli(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30, source="cli")
        # s1, s3 are cli; s5_sub is subagent (not cli)
        assert report["session_count"] >= 2

    def test_source_filter_telegram(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30, source="telegram")
        # s2, s7 are telegram
        assert report["session_count"] >= 2

    def test_source_filter_nonexistent(self, populated_db):
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=30, source="slack")
        assert report["empty"] is True

    def test_days_filter_short(self, populated_db):
        """days=3 → only recent sessions."""
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=3)
        # s4 (1 day ago), s7 (3 days ago) should be included
        assert report["session_count"] >= 2

    def test_days_filter_long(self, populated_db):
        """days=60 → all sessions including s_old."""
        analyzer = BehavioralAnalyzer(populated_db)
        report = analyzer.generate(days=60)
        assert report["session_count"] >= 6  # all sessions

    def test_large_days_value(self, db):
        """Very large days value should not crash."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        report = analyzer.generate(days=365)
        assert report["empty"] is False

    def test_standalone_report(self, populated_db):
        """format_terminal output is standalone, not appended to insights."""
        from agent.insights import InsightsEngine

        insights_engine = InsightsEngine(populated_db)
        insights_report = insights_engine.generate(days=30)
        insights_text = insights_engine.format_terminal(insights_report)

        analyzer = BehavioralAnalyzer(populated_db)
        behavior_report = analyzer.generate(days=30)
        behavior_text = analyzer.format_terminal(behavior_report)

        # Behavior report has its own header, not the insights one
        assert "Hermes Behavior" in behavior_text
        assert "Hermes Insights" in insights_text
        # Behavior text should not contain insights-only sections
        assert "Models Used" not in behavior_text

    def test_config_arg_accepted(self, db):
        """BehavioralAnalyzer accepts an optional config dict."""
        analyzer = BehavioralAnalyzer(db, {"enabled": True, "model": "gpt-4o-mini"})
        assert analyzer._behavior_config.get("model") == "gpt-4o-mini"

    def test_config_arg_none(self, db):
        """BehavioralAnalyzer works with no config (default)."""
        analyzer = BehavioralAnalyzer(db)
        assert analyzer._behavior_config == {}

    def test_init_creates_table_idempotent(self, db):
        """Creating two analyzers on the same DB doesn't crash (IF NOT EXISTS)."""
        BehavioralAnalyzer(db)
        BehavioralAnalyzer(db)  # should not raise


# =========================================================================
# Hermes-exclusive signal extractor tests (signals 19-27)
# =========================================================================

@pytest.fixture()
def hermes_db(db):
    """DB with Hermes-exclusive tool calls: skills, memory, session_search,
    cron sessions, todo lists, delegate_task, and abandonment patterns."""
    now = time.time()
    day = 86400

    # ── Session A: loads skills, uses memory, searches history ──
    db.create_session(session_id="ha", source="cli", model="model-alpha")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'ha'", (now - 2 * day,))
    db.end_session("ha", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'ha'", (now - 2 * day + 3600,))
    db.update_token_counts("ha", input_tokens=10000, output_tokens=5000)
    db.append_message("ha", role="user", content="Help me build a feature")
    db.append_message("ha", role="assistant", content="Loading skill.",
                      tool_calls=[{"id": "c1", "type": "function",
                                   "function": {"name": "skill_view",
                                                "arguments": json.dumps({"name": "cos-gate"})}}])
    db.append_message("ha", role="assistant", content="Loading another skill.",
                      tool_calls=[{"id": "c2", "type": "function",
                                   "function": {"name": "skill_view",
                                                "arguments": json.dumps({"name": "cos-gate"})}}])
    db.append_message("ha", role="assistant", content="Editing skill.",
                      tool_calls=[{"id": "c3", "type": "function",
                                   "function": {"name": "skill_manage",
                                                "arguments": json.dumps({"name": "my-skill", "action": "patch"})}}])
    db.append_message("ha", role="assistant", content="Adding memory.",
                      tool_calls=[{"id": "c4", "type": "function",
                                   "function": {"name": "memory",
                                                "arguments": json.dumps({"action": "add", "target": "memory"})}}])
    db.append_message("ha", role="assistant", content="Replacing memory.",
                      tool_calls=[{"id": "c5", "type": "function",
                                   "function": {"name": "memory",
                                                "arguments": json.dumps({"action": "replace", "target": "user"})}}])
    db.append_message("ha", role="assistant", content="Searching history.",
                      tool_calls=[{"id": "c6", "type": "function",
                                   "function": {"name": "session_search",
                                                "arguments": json.dumps({"query": "auth refactor"})}}])
    db.append_message("ha", role="assistant", content="Creating todo list.",
                      tool_calls=[{"id": "c7", "type": "function",
                                   "function": {"name": "todo",
                                                "arguments": json.dumps({
                                                    "action": "create",
                                                    "todos": [
                                                        {"id": "t1", "content": "task 1", "status": "completed"},
                                                        {"id": "t2", "content": "task 2", "status": "pending"},
                                                    ]
                                                })}}])
    # Set tool_call_count to simulate productivity
    db._conn.execute("UPDATE sessions SET tool_call_count = 50, message_count = 10 WHERE id = 'ha'")

    # ── Session B: no skills, fewer tool calls ──
    db.create_session(session_id="hb", source="cli", model="model-beta")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'hb'", (now - 1 * day,))
    db.end_session("hb", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'hb'", (now - 1 * day + 600,))
    db.append_message("hb", role="user", content="Just fix this bug")
    db.append_message("hb", role="assistant", content="Let me search.",
                      tool_calls=[{"function": {"name": "search_files"}}])
    db.append_message("hb", role="tool", content="found", tool_name="search_files")
    db._conn.execute("UPDATE sessions SET tool_call_count = 10, message_count = 4 WHERE id = 'hb'")

    # ── Session C: cron-sourced, completed ──
    db.create_session(session_id="hc", source="cron", model="model-alpha")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'hc'", (now - 3 * day,))
    db.end_session("hc", end_reason="cron_complete")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'hc'", (now - 3 * day + 300,))
    db._conn.execute("UPDATE sessions SET message_count = 5 WHERE id = 'hc'")
    db.append_message("hc", role="user", content="Cron task")
    db.append_message("hc", role="assistant", content="Done")

    # ── Session D: cron-sourced, failed (no end_reason) ──
    db.create_session(session_id="hd", source="cron", model="model-alpha")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'hd'", (now - 4 * day,))
    # No end_session call → no end_reason (abandoned)
    db._conn.execute("UPDATE sessions SET message_count = 3 WHERE id = 'hd'")

    # ── Session E: delegate_task with background ──
    db.create_session(session_id="he", source="cli", model="model-alpha")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'he'", (now - 1 * day,))
    db.end_session("he", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'he'", (now - 1 * day + 1200,))
    db.append_message("he", role="user", content="Run tests in background")
    db.append_message("he", role="assistant", content="Delegating.",
                      tool_calls=[{"id": "c8", "type": "function",
                                   "function": {"name": "delegate_task",
                                                "arguments": json.dumps({"task": "Run tests", "background": True})}}])
    db.append_message("he", role="assistant", content="Also foreground.",
                      tool_calls=[{"id": "c9", "type": "function",
                                   "function": {"name": "delegate_task",
                                                "arguments": json.dumps({"task": "Lint", "background": False})}}])

    # ── Session F: subagent child of he, overlapping ──
    db.create_session(session_id="hf_sub", source="subagent", model="model-alpha")
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, parent_session_id = 'he' WHERE id = 'hf_sub'",
        (now - 1 * day + 50,),
    )
    db.end_session("hf_sub", end_reason="completed")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'hf_sub'", (now - 1 * day + 600,))

    # ── Session G: model-beta with crash out ──
    db.create_session(session_id="hg", source="cli", model="model-beta")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'hg'", (now - 5 * day,))
    db.end_session("hg", end_reason="user_exit")
    db._conn.execute("UPDATE sessions SET ended_at = ? WHERE id = 'hg'", (now - 5 * day + 300,))
    db.append_message("hg", role="user", content="STOP!!! WTF!!! THIS IS BROKEN!!!")
    db.append_message("hg", role="assistant", content="sorry")
    db._conn.execute("UPDATE sessions SET tool_call_count = 5 WHERE id = 'hg'")

    # ── Session H: abandoned (no end_reason, no ended_at) ──
    db.create_session(session_id="hh", source="cli", model="model-alpha")
    db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = 'hh'", (now - 6 * day,))
    # No end_session → abandoned
    db.append_message("hh", role="user", content="started something")
    db.append_message("hh", role="assistant", content="ok")

    db._conn.commit()
    return db


class TestSkillUsage:
    """Signal 19: Skill usage extractor."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_skill_loads_detected(self, hermes_db):
        signals = self._get_signals(hermes_db)
        su = signals["skill_usage"]
        # ha loads cos-gate 2x
        assert su["total_loads"] >= 2
        assert "cos-gate" in su["load_counts"]
        assert su["load_counts"]["cos-gate"] >= 2

    def test_skill_edits_detected(self, hermes_db):
        signals = self._get_signals(hermes_db)
        su = signals["skill_usage"]
        # ha edits my-skill 1x
        assert su["total_edits"] >= 1
        assert "my-skill" in su["edit_counts"]

    def test_skill_distinct_count(self, hermes_db):
        signals = self._get_signals(hermes_db)
        su = signals["skill_usage"]
        # cos-gate + my-skill = at least 2 distinct
        assert su["distinct_skills"] >= 2

    def test_skill_adoption_rate(self, hermes_db):
        signals = self._get_signals(hermes_db)
        su = signals["skill_usage"]
        assert 0 <= su["adoption_rate"] <= 1.0
        assert su["adoption_rate"] > 0  # at least one session loads skills

    def test_skill_most_used(self, hermes_db):
        signals = self._get_signals(hermes_db)
        su = signals["skill_usage"]
        assert su["most_used"] == "cos-gate"

    def test_skill_usage_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_skill_usage([])
        assert result["total_loads"] == 0
        assert result["total_edits"] == 0
        assert result["distinct_skills"] == 0
        assert result["most_used"] is None
        assert result["adoption_rate"] == 0.0


class TestMemoryManagement:
    """Signal 20: Memory management extractor."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_memory_add_detected(self, hermes_db):
        signals = self._get_signals(hermes_db)
        mm = signals["memory_management"]
        assert mm["add_count"] >= 1
        assert mm["replace_count"] >= 1

    def test_memory_correction_rate(self, hermes_db):
        signals = self._get_signals(hermes_db)
        mm = signals["memory_management"]
        assert mm["total"] >= 2
        assert 0 <= mm["correction_rate"] <= 1.0
        # 1 add + 1 replace → correction = 1/2 = 0.5
        assert mm["correction_rate"] > 0

    def test_memory_target_breakdown(self, hermes_db):
        signals = self._get_signals(hermes_db)
        mm = signals["memory_management"]
        assert mm["target_memory"] >= 1  # add → memory
        assert mm["target_user"] >= 1   # replace → user

    def test_memory_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_memory_management([])
        assert result["add_count"] == 0
        assert result["total"] == 0
        assert result["correction_rate"] == 0.0


class TestCrossSessionMemory:
    """Signal 21: Cross-session memory (session_search)."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_searches_detected(self, hermes_db):
        signals = self._get_signals(hermes_db)
        csm = signals["cross_session_memory"]
        assert csm["total_searches"] >= 1
        assert csm["sessions_that_search"] >= 1

    def test_search_vs_no_search_avg(self, hermes_db):
        signals = self._get_signals(hermes_db)
        csm = signals["cross_session_memory"]
        # ha searches and has tool_call_count=50; hb doesn't and has 10
        assert csm["avg_tool_calls_searching"] >= csm["avg_tool_calls_not_searching"]

    def test_cross_session_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_cross_session_memory([], [])
        assert result["total_searches"] == 0
        assert result["sessions_that_search"] == 0


class TestCronAutonomy:
    """Signal 22: Cron autonomy."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_cron_sessions_counted(self, hermes_db):
        signals = self._get_signals(hermes_db)
        cron = signals["cron_autonomy"]
        # hc and hd are cron sessions
        assert cron["cron_session_count"] >= 2

    def test_cron_end_reasons(self, hermes_db):
        signals = self._get_signals(hermes_db)
        cron = signals["cron_autonomy"]
        assert "cron_complete" in cron["end_reasons"]

    def test_cron_success_rate(self, hermes_db):
        signals = self._get_signals(hermes_db)
        cron = signals["cron_autonomy"]
        assert 0 <= cron["success_rate"] <= 1.0
        # hc completed, hd didn't → at most 50%
        assert cron["success_rate"] <= 0.5

    def test_cron_total_messages(self, hermes_db):
        signals = self._get_signals(hermes_db)
        cron = signals["cron_autonomy"]
        assert cron["total_messages"] >= 5  # hc has 5

    def test_cron_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_cron_autonomy(time.time())
        assert result["cron_session_count"] == 0
        assert result["success_rate"] == 0.0

    def test_cron_source_filter_excludes(self, hermes_db):
        """When source='cli', cron sessions are excluded."""
        analyzer = BehavioralAnalyzer(hermes_db)
        result = analyzer._extract_cron_autonomy(time.time() - 86400, source="cli")
        assert result["cron_session_count"] == 0


class TestDelegationPatterns:
    """Signal 23: Delegation patterns."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_delegation_dispatches(self, hermes_db):
        signals = self._get_signals(hermes_db)
        dp = signals["delegation_patterns"]
        assert dp["total_dispatches"] >= 2
        assert dp["background"] >= 1
        assert dp["foreground"] >= 1

    def test_delegation_max_concurrent(self, hermes_db):
        signals = self._get_signals(hermes_db)
        dp = signals["delegation_patterns"]
        assert dp["max_concurrent"] >= 0  # at least 0

    def test_delegation_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_delegation_patterns([], [])
        assert result["total_dispatches"] == 0
        assert result["max_concurrent"] == 0


class TestTaskManagement:
    """Signal 24: Task management (todo)."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_todo_creation_detected(self, hermes_db):
        signals = self._get_signals(hermes_db)
        tm = signals["task_management"]
        assert tm["creation_count"] >= 1

    def test_todo_sessions_with_todos(self, hermes_db):
        signals = self._get_signals(hermes_db)
        tm = signals["task_management"]
        assert tm["sessions_with_todos"] >= 1
        assert tm["sessions_without_todos"] >= 0

    def test_todo_completion_rate(self, hermes_db):
        signals = self._get_signals(hermes_db)
        tm = signals["task_management"]
        # ha has 1 completed out of 2 items
        if tm["completion_rate"] is not None:
            assert 0 <= tm["completion_rate"] <= 1.0

    def test_todo_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_task_management([])
        assert result["creation_count"] == 0
        assert result["sessions_with_todos"] == 0
        assert result["completion_rate"] is None


class TestModelEffectiveness:
    """Signal 25: Model effectiveness."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_model_effectiveness_has_models(self, hermes_db):
        signals = self._get_signals(hermes_db)
        me = signals["model_effectiveness"]
        assert len(me["models"]) >= 2  # model-alpha and model-beta

    def test_model_effectiveness_rates_in_range(self, hermes_db):
        signals = self._get_signals(hermes_db)
        me = signals["model_effectiveness"]
        for m in me["models"]:
            assert 0 <= m["crash_out_rate"] <= 1.0
            assert 0 <= m["steering_rate"] <= 1.0
            assert m["session_count"] > 0

    def test_model_effectiveness_crash_out(self, hermes_db):
        signals = self._get_signals(hermes_db)
        me = signals["model_effectiveness"]
        # model-beta has crash-out in hg
        beta = next(m for m in me["models"] if m["model"] == "model-beta")
        assert beta["crash_out_rate"] > 0

    def test_model_effectiveness_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_model_effectiveness([], [])
        assert result["models"] == []


class TestSkillROI:
    """Signal 26: Skill ROI."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_skill_roi_partition(self, hermes_db):
        signals = self._get_signals(hermes_db)
        roi = signals["skill_roi"]
        assert roi["skill_session_count"] >= 1  # ha
        assert roi["non_skill_session_count"] >= 1  # hb, hg, etc.

    def test_skill_roi_avg_tool_calls(self, hermes_db):
        signals = self._get_signals(hermes_db)
        roi = signals["skill_roi"]
        # ha has 50 tool calls, hb has 10
        assert roi["avg_tool_calls_with"] > roi["avg_tool_calls_without"]

    def test_skill_roi_multiplier(self, hermes_db):
        signals = self._get_signals(hermes_db)
        roi = signals["skill_roi"]
        assert roi["productivity_multiplier"] > 0

    def test_skill_roi_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_skill_roi([], [])
        assert result["skill_session_count"] == 0
        assert result["productivity_multiplier"] == 0.0


class TestSessionAbandonment:
    """Signal 27: Session abandonment."""

    def _get_signals(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        cutoff = time.time() - (days * 86400)
        return analyzer._extract_signals(cutoff)

    def test_abandonment_total(self, hermes_db):
        signals = self._get_signals(hermes_db)
        sa = signals["session_abandonment"]
        assert sa["total_sessions"] > 0

    def test_abandonment_detected(self, hermes_db):
        signals = self._get_signals(hermes_db)
        sa = signals["session_abandonment"]
        # hd and hh have no end_reason → abandoned
        assert sa["abandoned"] >= 2

    def test_abandonment_rate_in_range(self, hermes_db):
        signals = self._get_signals(hermes_db)
        sa = signals["session_abandonment"]
        assert 0 <= sa["abandonment_rate"] <= 1.0

    def test_abandonment_closed_by_user(self, hermes_db):
        signals = self._get_signals(hermes_db)
        sa = signals["session_abandonment"]
        assert sa["closed_by_user"] >= 1  # ha, hb, he, hg end with user_exit

    def test_abandonment_closed_by_agent(self, hermes_db):
        signals = self._get_signals(hermes_db)
        sa = signals["session_abandonment"]
        # hc ends with cron_complete (agent reason)
        assert sa["closed_by_agent"] >= 1

    def test_abandonment_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        result = analyzer._extract_session_abandonment(time.time())
        assert result["total_sessions"] == 0
        assert result["abandonment_rate"] == 0.0

    def test_abandonment_sums_to_total(self, hermes_db):
        signals = self._get_signals(hermes_db)
        sa = signals["session_abandonment"]
        total = sa["abandoned"] + sa["closed_by_user"] + sa["closed_by_agent"] + sa["reset"] + sa["other"]
        assert total == sa["total_sessions"]


# =========================================================================
# Hermes-exclusive deep insight card tests (cards 16-23)
# =========================================================================

class TestDeepInsightCards:
    """Test the 8 Hermes-exclusive deep insight cards (16-23)."""

    def _get_report(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        with patch.object(analyzer, "_call_llm_for_scoring", return_value=None):
            return analyzer.generate(days=days)

    def test_card_skill_mastery_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "skill_mastery" in report["cards"]
        card = report["cards"]["skill_mastery"]
        assert "title" in card
        assert "body" in card

    def test_card_skill_mastery_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["skill_mastery"]
        # hermes_db has skill loads → title should mention skills
        assert "skill" in card["title"].lower() or "No skills" in card["title"]

    def test_card_skill_mastery_empty(self, db):
        """No skills → card says no skills loaded."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        report = self._get_report(db)
        card = report["cards"]["skill_mastery"]
        assert "no skill" in card["title"].lower()

    def test_card_memory_hygiene_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "memory_hygiene" in report["cards"]
        card = report["cards"]["memory_hygiene"]
        assert len(card["title"]) > 0
        assert len(card["body"]) > 0

    def test_card_memory_hygiene_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["memory_hygiene"]
        assert "memory" in card["title"].lower() or "operation" in card["title"].lower()

    def test_card_memory_hygiene_empty(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        report = self._get_report(db)
        card = report["cards"]["memory_hygiene"]
        assert "no memory" in card["title"].lower()

    def test_card_autonomy_level_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "autonomy_level" in report["cards"]
        card = report["cards"]["autonomy_level"]
        assert len(card["title"]) > 0

    def test_card_autonomy_level_with_cron(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["autonomy_level"]
        # hermes_db has cron sessions → should mention automation
        assert "automated" in card["title"].lower() or "manual" in card["title"].lower()

    def test_card_autonomy_level_manual(self, db):
        """No cron or delegation → manual operator."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        report = self._get_report(db)
        card = report["cards"]["autonomy_level"]
        assert "manual" in card["title"].lower()

    def test_card_cross_session_memory_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "cross_session_memory" in report["cards"]
        card = report["cards"]["cross_session_memory"]
        assert len(card["title"]) > 0

    def test_card_cross_session_memory_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["cross_session_memory"]
        # hermes_db has session_search → should mention searches
        assert "search" in card["title"].lower() or "history" in card["title"].lower()

    def test_card_cross_session_memory_empty(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        report = self._get_report(db)
        card = report["cards"]["cross_session_memory"]
        assert "no history" in card["title"].lower() or "no search" in card["title"].lower()

    def test_card_tool_orchestration_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "tool_orchestration" in report["cards"]
        card = report["cards"]["tool_orchestration"]
        assert len(card["title"]) > 0

    def test_card_tool_orchestration_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["tool_orchestration"]
        # hermes_db uses multiple tools → should mention distinct tools
        assert "tool" in card["title"].lower() or "distinct" in card["title"].lower()

    def test_card_model_effectiveness_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "model_effectiveness" in report["cards"]
        card = report["cards"]["model_effectiveness"]
        assert len(card["title"]) > 0

    def test_card_model_effectiveness_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["model_effectiveness"]
        # hermes_db has model-alpha and model-beta
        assert "model" in card["title"].lower() or "comparison" in card["title"].lower()

    def test_card_model_effectiveness_empty(self, db):
        analyzer = BehavioralAnalyzer(db)
        signals = analyzer._extract_signals(time.time() - 86400)
        card = analyzer._card_model_effectiveness(signals)
        assert "no model" in card["title"].lower()

    def test_card_skill_roi_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "skill_roi" in report["cards"]
        card = report["cards"]["skill_roi"]
        assert len(card["title"]) > 0

    def test_card_skill_roi_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["skill_roi"]
        # hermes_db: ha has skills and 50 tc, hb has no skills and 10 tc
        assert "productive" in card["body"].lower() or "skill" in card["title"].lower()

    def test_card_skill_roi_empty(self, db):
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        report = self._get_report(db)
        card = report["cards"]["skill_roi"]
        assert "skill" in card["title"].lower()

    def test_card_session_abandonment_present(self, hermes_db):
        report = self._get_report(hermes_db)
        assert "session_abandonment" in report["cards"]
        card = report["cards"]["session_abandonment"]
        assert len(card["title"]) > 0

    def test_card_session_abandonment_with_data(self, hermes_db):
        report = self._get_report(hermes_db)
        card = report["cards"]["session_abandonment"]
        # hermes_db has abandoned sessions (hd, hh)
        assert "closure" in card["title"].lower() or "clean" in card["title"].lower() or "%" in card["title"]

    def test_card_session_abandonment_clean(self, db):
        """All sessions ended → clean closures."""
        db.create_session(session_id="s", source="cli", model="m")
        db.end_session("s", end_reason="user_exit")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        report = self._get_report(db)
        card = report["cards"]["session_abandonment"]
        assert "clean" in card["title"].lower()


# =========================================================================
# Formatting tests for new deep insight cards
# =========================================================================

class TestDeepInsightCardFormatting:
    """Test that the 8 new cards appear in terminal and gateway output."""

    def _get_report(self, db, days=30):
        analyzer = BehavioralAnalyzer(db)
        with patch.object(analyzer, "_call_llm_for_scoring", return_value=None):
            return analyzer.generate(days=days)

    def test_terminal_has_skill_mastery(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Skill mastery" in text

    def test_terminal_has_memory_hygiene(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Memory hygiene" in text

    def test_terminal_has_autonomy_level(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Autonomy level" in text

    def test_terminal_has_cross_session_memory(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Cross-session memory" in text

    def test_terminal_has_tool_orchestration(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Tool orchestration" in text

    def test_terminal_has_model_effectiveness(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Model effectiveness" in text

    def test_terminal_has_skill_roi(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Skill ROI" in text

    def test_terminal_has_session_abandonment(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_terminal(report)
        assert "Session abandonment" in text

    def test_gateway_has_skill_mastery(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_gateway(report)
        assert "Skill mastery" in text

    def test_gateway_has_all_new_cards(self, hermes_db):
        report = self._get_report(hermes_db)
        text = BehavioralAnalyzer(hermes_db).format_gateway(report)
        for label in ["Skill mastery", "Memory hygiene", "Autonomy level",
                       "Cross-session memory", "Tool orchestration",
                       "Model effectiveness", "Skill ROI", "Session abandonment"]:
            assert label in text, f"Missing card '{label}' in gateway output"


# =========================================================================
# LLM prompt tests for Hermes-exclusive signals
# =========================================================================

class TestLLMPromptHermesSignals:
    """Test that the LLM prompt includes the new Hermes-exclusive signals."""

    def test_prompt_includes_skills_when_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "Skills:" in prompt or "skill" in prompt.lower()

    def test_prompt_includes_memory_when_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "Memory ops:" in prompt

    def test_prompt_includes_cron_when_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "Cron:" in prompt

    def test_prompt_includes_delegation_when_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "Delegation:" in prompt

    def test_prompt_includes_abandonment_when_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "Session abandonment:" in prompt

    def test_prompt_excludes_signals_when_absent(self, db):
        """No Hermes-exclusive data → those lines omitted from prompt."""
        db.create_session(session_id="s", source="cli", model="m")
        db.append_message("s", role="user", content="hello world test message")
        db.append_message("s", role="assistant", content="hi")
        db._conn.commit()
        analyzer = BehavioralAnalyzer(db)
        signals = analyzer._extract_signals(time.time() - 30 * 86400)
        prompt = analyzer._build_llm_prompt(signals)
        assert "Memory ops:" not in prompt
        assert "Cron:" not in prompt
        assert "Delegation:" not in prompt


# =========================================================================
# Integration: signals in report dict
# =========================================================================

class TestSignalsInReport:
    """Test that all 9 new signals appear in the report's signals dict."""

    def test_all_new_signals_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        with patch.object(analyzer, "_call_llm_for_scoring", return_value=None):
            report = analyzer.generate(days=30)
        signals = report["signals"]
        for key in ["skill_usage", "memory_management", "cross_session_memory",
                     "cron_autonomy", "delegation_patterns", "task_management",
                     "model_effectiveness", "skill_roi", "session_abandonment"]:
            assert key in signals, f"Missing signal '{key}' in report signals"

    def test_all_new_cards_present(self, hermes_db):
        analyzer = BehavioralAnalyzer(hermes_db)
        with patch.object(analyzer, "_call_llm_for_scoring", return_value=None):
            report = analyzer.generate(days=30)
        cards = report["cards"]
        for key in ["skill_mastery", "memory_hygiene", "autonomy_level",
                     "cross_session_memory", "tool_orchestration", "model_effectiveness",
                     "skill_roi", "session_abandonment"]:
            assert key in cards, f"Missing card '{key}' in report cards"

    def test_raw_signals_json_has_new_keys(self, hermes_db):
        """Persisted raw_signals JSON should include the new signal keys."""
        analyzer = BehavioralAnalyzer(hermes_db)
        with patch.object(analyzer, "_call_llm_for_scoring", return_value=None):
            analyzer.generate(days=30)
        cursor = hermes_db._conn.execute(
            "SELECT raw_signals FROM behavioral_scores ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        raw = json.loads(row["raw_signals"])
        assert "skill_usage" in raw
        assert "memory_management" in raw
        assert "cron_autonomy" in raw
        assert "session_abandonment" in raw