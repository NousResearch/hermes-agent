"""Tests for the SIE skill-usage telemetry aggregation in InsightsEngine.

Covers the AC#8 surface: the _SkillUsageAggregator helper, the
skill_usage section in InsightsEngine.generate(), and the format_terminal
rendering.
"""

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.insights import InsightsEngine, _SkillUsageAggregator


def _ts(dt: datetime) -> str:
    """Format a datetime as the ISO-8601 'Z' suffix the telemetry uses."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_jsonl(path: Path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            if isinstance(row, str):
                fh.write(row + "\n")
            else:
                fh.write(json.dumps(row) + "\n")


def _make_db():
    """Build a SessionDB-like shim with the minimum surface InsightsEngine uses.

    The skill_usage section is independent from the DB path, so the DB only
    needs to satisfy the InsightsEngine constructor (db._conn) and return
    zero sessions for the rest of the report.
    """
    class _DBShim:
        def __init__(self, conn):
            self._conn = conn

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    # Minimal schema so the existing _get_* queries do not error out.
    conn.executescript("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT,
            model TEXT,
            started_at REAL,
            ended_at REAL,
            message_count INTEGER,
            tool_call_count INTEGER,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT
        );
        CREATE TABLE messages (
            session_id TEXT,
            role TEXT,
            tool_name TEXT,
            tool_calls TEXT,
            timestamp REAL,
            content TEXT
        );
    """)
    return _DBShim(conn)


def test_happy_path_mixed_init_complete(tmp_path):
    now = datetime.now(timezone.utc)
    p = tmp_path / "skill-usage.jsonl"
    rows = [
        # paired init + complete (success)
        {"ts": _ts(now - timedelta(hours=2)), "phase": "init", "skill": "review", "agent": "reid",
         "outcome": "in_flight", "dispatch_id": "d1", "started_at": _ts(now - timedelta(hours=2))},
        {"ts": _ts(now - timedelta(hours=2) + timedelta(seconds=2)), "phase": "complete", "skill": "review",
         "agent": "reid", "outcome": "success", "dispatch_id": "d1", "duration_s": 2.0},
        # paired init + complete (failure)
        {"ts": _ts(now - timedelta(hours=1)), "phase": "init", "skill": "review", "agent": "reid",
         "outcome": "in_flight", "dispatch_id": "d2", "started_at": _ts(now - timedelta(hours=1))},
        {"ts": _ts(now - timedelta(hours=1) + timedelta(seconds=4)), "phase": "complete", "skill": "review",
         "agent": "reid", "outcome": "failure", "dispatch_id": "d2", "duration_s": 4.0},
        # single complete (a different skill, different agent)
        {"ts": _ts(now - timedelta(minutes=30)), "phase": "complete", "skill": "coding",
         "agent": "grant", "outcome": "success", "dispatch_id": "d3", "duration_s": 1.0},
    ]
    _write_jsonl(p, rows)

    agg = _SkillUsageAggregator(str(p)).aggregate(days=7, now_ts=now.timestamp())
    assert agg["available"] is True
    assert agg["total_dispatches"] == 3
    by_skill = {r["skill"]: r for r in agg["by_skill"]}
    assert by_skill["review"]["total"] == 2
    assert by_skill["review"]["success"] == 1
    assert by_skill["review"]["failure"] == 1
    assert by_skill["review"]["success_rate"] == 0.5
    assert by_skill["review"]["avg_duration_s"] == 3.0
    assert by_skill["coding"]["total"] == 1
    by_agent = {r["agent"]: r for r in agg["by_agent"]}
    assert by_agent["reid"]["total"] == 2
    assert by_agent["grant"]["total"] == 1
    assert agg["unjoined_in_flight"] == 0


def test_missing_file_graceful(tmp_path):
    p = tmp_path / "does-not-exist.jsonl"
    agg = _SkillUsageAggregator(str(p)).aggregate(days=30)
    assert agg["available"] is False
    assert "not found" in agg["reason"]
    assert agg["total_dispatches"] == 0
    assert agg["by_skill"] == []


def test_malformed_rows_skipped(tmp_path):
    now = datetime.now(timezone.utc)
    p = tmp_path / "skill-usage.jsonl"
    rows = [
        "not valid json",
        "{\"partial\": ",  # broken
        "",  # empty line
        {"ts": _ts(now - timedelta(hours=1)), "phase": "complete", "skill": "x",
         "agent": "a", "outcome": "success", "dispatch_id": "ok", "duration_s": 1.0},
        {"ts": "nonsense-timestamp", "phase": "complete", "skill": "y",
         "agent": "a", "outcome": "success", "dispatch_id": "bad", "duration_s": 1.0},
    ]
    _write_jsonl(p, rows)

    agg = _SkillUsageAggregator(str(p)).aggregate(days=30, now_ts=now.timestamp())
    assert agg["available"] is True
    # Only the well-formed in-window row counts.
    assert agg["total_dispatches"] == 1
    assert agg["by_skill"][0]["skill"] == "x"


def test_window_filtering(tmp_path):
    now = datetime.now(timezone.utc)
    p = tmp_path / "skill-usage.jsonl"
    rows = [
        # outside window (60 days ago)
        {"ts": _ts(now - timedelta(days=60)), "phase": "complete", "skill": "old",
         "agent": "reid", "outcome": "success", "dispatch_id": "old1", "duration_s": 1.0},
        # inside window
        {"ts": _ts(now - timedelta(days=1)), "phase": "complete", "skill": "new",
         "agent": "reid", "outcome": "success", "dispatch_id": "new1", "duration_s": 1.0},
    ]
    _write_jsonl(p, rows)

    agg = _SkillUsageAggregator(str(p)).aggregate(days=7, now_ts=now.timestamp())
    assert agg["total_dispatches"] == 1
    assert agg["by_skill"][0]["skill"] == "new"


def test_unjoined_in_flight_detection(tmp_path):
    now = datetime.now(timezone.utc)
    p = tmp_path / "skill-usage.jsonl"
    rows = [
        # Init older than 30 min with no matching complete = stuck
        {"ts": _ts(now - timedelta(hours=2)), "phase": "init", "skill": "stuck", "agent": "reid",
         "outcome": "in_flight", "dispatch_id": "stuck1",
         "started_at": _ts(now - timedelta(hours=2))},
        # Init within 30 min with no complete = NOT stuck (still ramping)
        {"ts": _ts(now - timedelta(minutes=5)), "phase": "init", "skill": "fresh", "agent": "reid",
         "outcome": "in_flight", "dispatch_id": "fresh1",
         "started_at": _ts(now - timedelta(minutes=5))},
        # Init paired with complete = NOT stuck
        {"ts": _ts(now - timedelta(hours=3)), "phase": "init", "skill": "done", "agent": "reid",
         "outcome": "in_flight", "dispatch_id": "done1",
         "started_at": _ts(now - timedelta(hours=3))},
        {"ts": _ts(now - timedelta(hours=3) + timedelta(seconds=5)), "phase": "complete",
         "skill": "done", "agent": "reid", "outcome": "success", "dispatch_id": "done1",
         "duration_s": 5.0},
    ]
    _write_jsonl(p, rows)

    agg = _SkillUsageAggregator(str(p)).aggregate(days=30, now_ts=now.timestamp())
    assert agg["unjoined_in_flight"] == 1


def test_top_failure_skills_ranking(tmp_path):
    now = datetime.now(timezone.utc)
    p = tmp_path / "skill-usage.jsonl"
    rows = []
    # skill A: 3 failures, 1 success
    for i in range(3):
        rows.append({"ts": _ts(now - timedelta(hours=1 + i)), "phase": "complete",
                     "skill": "skill-a", "agent": "reid", "outcome": "failure",
                     "dispatch_id": f"a-{i}", "duration_s": 1.0})
    rows.append({"ts": _ts(now - timedelta(hours=5)), "phase": "complete",
                 "skill": "skill-a", "agent": "reid", "outcome": "success",
                 "dispatch_id": "a-s", "duration_s": 1.0})
    # skill B: 1 error
    rows.append({"ts": _ts(now - timedelta(hours=1)), "phase": "complete",
                 "skill": "skill-b", "agent": "reid", "outcome": "error",
                 "dispatch_id": "b-1", "duration_s": 1.0})
    _write_jsonl(p, rows)

    agg = _SkillUsageAggregator(str(p)).aggregate(days=7, now_ts=now.timestamp())
    top = agg["top_failure_skills"]
    assert top[0] == {"skill": "skill-a", "failure_count": 3}
    assert top[1] == {"skill": "skill-b", "failure_count": 1}


def test_engine_integration_and_terminal_format(tmp_path):
    now = datetime.now(timezone.utc)
    p = tmp_path / "skill-usage.jsonl"
    rows = [
        {"ts": _ts(now - timedelta(hours=1)), "phase": "complete", "skill": "review",
         "agent": "reid", "outcome": "success", "dispatch_id": "d1", "duration_s": 2.5},
        {"ts": _ts(now - timedelta(hours=1)), "phase": "complete", "skill": "coding",
         "agent": "grant", "outcome": "failure", "dispatch_id": "d2", "duration_s": 3.5},
    ]
    _write_jsonl(p, rows)

    db = _make_db()
    engine = InsightsEngine(db, skill_usage_path=str(p))
    report = engine.generate(days=7)
    # AC-1: skill_usage key present
    assert "skill_usage" in report
    sie = report["skill_usage"]
    assert sie["available"] is True
    assert sie["total_dispatches"] == 2
    # AC-2: format_terminal renders the new section header
    rendered = engine.format_terminal(report)
    assert "## Skill Usage" in rendered
    assert "review" in rendered
    # HR-8: no em dashes in the new section. Locate the section and scan.
    section_start = rendered.index("## Skill Usage")
    new_section = rendered[section_start:]
    assert "\u2014" not in new_section, "Em dash found in Skill Usage section (HR-8)"


def test_signature_default_path_uses_expanduser(tmp_path, monkeypatch):
    """Default path resolution respects HOME via expanduser."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    db = _make_db()
    engine = InsightsEngine(db)  # no skill_usage_path
    # Expanded path should start with the fake home.
    assert engine.skill_usage_path.startswith(str(fake_home))
    assert engine.skill_usage_path.endswith("sie/analytics/skill-usage.jsonl")


def test_unreadable_file_graceful(tmp_path, monkeypatch):
    p = tmp_path / "skill-usage.jsonl"
    p.write_text("{}\n")

    # Force open() to raise OSError when called with this path.
    real_open = open

    def fake_open(path, *args, **kwargs):
        if str(path) == str(p):
            raise OSError("permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    agg = _SkillUsageAggregator(str(p)).aggregate(days=30)
    assert agg["available"] is False
    assert "unreadable" in agg["reason"]
