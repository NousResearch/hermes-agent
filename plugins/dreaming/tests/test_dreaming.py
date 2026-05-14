"""Tests for the Dreaming plugin.

Covers: Candidate/DreamReport data structures, Light/REM/Deep phases,
scoring, noise filtering, memory file I/O, and the full cycle integration.
"""

import json
import sqlite3
import tempfile
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and reset config cache for each test."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import plugins.dreaming as d
    d._reset_config_cache()


@pytest.fixture
def hermes_home(tmp_path):
    """Return the isolated HERMES_HOME path."""
    return tmp_path


@pytest.fixture
def sessions_db(tmp_path, monkeypatch):
    """Create a minimal sessions.db with test data."""
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )
    """)

    now = datetime.now(tz=timezone.utc)

    sessions = [
        ("s1", "Project planning", (now - timedelta(hours=2)).isoformat()),
        ("s2", "Bug fix session", (now - timedelta(hours=5)).isoformat()),
        ("s3", "Old session", (now - timedelta(days=10)).isoformat()),
    ]
    for sid, title, ts in sessions:
        conn.execute("INSERT INTO sessions VALUES (?, ?, ?)", (sid, title, ts))

    messages = [
        ("s1", "user", "I want to build a Reddit automation tool for Aurex marketing.", now),
        ("s1", "assistant", "Great idea! Let's use the ACE pattern for that.", now),
        ("s1", "user", "The Reddit automation should handle account creation and posting.", now),
        ("s2", "user", "I'm frustrated with the gateway crashing after the Hermes update.", now - timedelta(hours=5)),
        ("s2", "assistant", "Let me check the gateway logs for you.", now - timedelta(hours=5)),
        ("s2", "user", "The gateway crash is related to the bot token already in use error.", now - timedelta(hours=5)),
    ]
    for i, (sid, role, content, ts) in enumerate(messages):
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (i, sid, role, content, ts.isoformat()),
        )

    conn.commit()
    conn.close()

    # Patch _db_path to use our test DB
    import plugins.dreaming as d
    monkeypatch.setattr(d, "_db_path", lambda: db_path)
    return db_path


# ---------------------------------------------------------------------------
# Data structure tests
# ---------------------------------------------------------------------------


class TestCandidate:
    def test_creation(self):
        from plugins.dreaming import Candidate
        c = Candidate("Test memory", source="s1")
        assert c.text == "Test memory"
        assert c.source == "s1"
        assert c.frequency == 1
        assert c.score == 0.0

    def test_whitespace_stripped(self):
        from plugins.dreaming import Candidate
        c = Candidate("  spaced out  ")
        assert c.text == "spaced out"


class TestDreamReport:
    def test_empty_report(self):
        from plugins.dreaming import DreamReport
        r = DreamReport()
        md = r.to_markdown()
        assert "Dream Cycle" in md
        assert "0 candidates staged" in md

    def test_report_with_data(self):
        from plugins.dreaming import DreamReport
        r = DreamReport()
        r.light_count = 10
        r.rem_themes = ["'aurex' appeared 3 times"]
        r.promoted = ["User is working on Aurex marketing"]
        r.skipped = ["low score entry"]
        r.narrative = "Test narrative"
        md = r.to_markdown()
        assert "10 candidates" in md
        assert "aurex" in md
        assert "1 promoted" in md
        assert "Test narrative" in md


# ---------------------------------------------------------------------------
# Noise filter tests
# ---------------------------------------------------------------------------


class TestNoiseFilter:
    @pytest.mark.parametrize("text", [
        "ok",
        "yes",
        "thanks",
        "thank you",
        "sure",
        "got it",
        "nice",
        "cool",
        "great",
        "perfect",
        "http://example.com",
        "https://test.org",
        "www.google.com",
        "```python",
        "# heading",
        "- list item",
        "1. numbered",
        "> quote",
        "Error: something failed",
        "Traceback (most recent call last)",
        "[[audio_as_voice]]",
        "MEDIA:/tmp/image.png",
        "TICK_OK",
        "HEARTBEAT_OK",
        "ab",  # too short
        "   ",  # only whitespace
    ])
    def test_is_noise_true(self, text):
        from plugins.dreaming import _is_noise
        assert _is_noise(text) is True, f"Expected noise: {text!r}"

    @pytest.mark.parametrize("text", [
        "I want to build a Reddit automation tool",
        "The gateway is crashing after the update",
        "Mina prefers direct results over explanations",
        "Aurex is an AI language learning app with 7 pillars",
    ])
    def test_is_noise_false(self, text):
        from plugins.dreaming import _is_noise
        assert _is_noise(text) is False, f"Expected not noise: {text!r}"


# ---------------------------------------------------------------------------
# Light phase tests
# ---------------------------------------------------------------------------


class TestLightPhase:
    def test_extracts_candidates(self, sessions_db):
        from plugins.dreaming import _light_phase, _recent_sessions
        sessions = _recent_sessions(days=7)
        candidates = _light_phase(sessions, max_candidates=50)
        assert len(candidates) > 0

    def test_deduplicates(self, sessions_db):
        from plugins.dreaming import _light_phase, _recent_sessions
        sessions = _recent_sessions(days=7)
        candidates = _light_phase(sessions, max_candidates=50)
        texts = [c.text.lower() for c in candidates]
        assert len(texts) == len(set(texts))

    def test_respects_max_candidates(self, sessions_db):
        from plugins.dreaming import _light_phase, _recent_sessions
        sessions = _recent_sessions(days=7)
        candidates = _light_phase(sessions, max_candidates=3)
        assert len(candidates) <= 3

    def test_sorts_by_frequency(self, sessions_db):
        from plugins.dreaming import _light_phase, _recent_sessions
        sessions = _recent_sessions(days=7)
        candidates = _light_phase(sessions, max_candidates=50)
        for i in range(len(candidates) - 1):
            assert candidates[i].frequency >= candidates[i + 1].frequency

    def test_filters_noise(self, sessions_db):
        from plugins.dreaming import _light_phase, _recent_sessions
        sessions = _recent_sessions(days=7)
        candidates = _light_phase(sessions, max_candidates=50)
        from plugins.dreaming import _is_noise
        for c in candidates:
            assert not _is_noise(c.text)

    def test_empty_sessions(self, tmp_path, monkeypatch):
        from plugins.dreaming import _light_phase, _recent_sessions
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sessions (session_id TEXT, title TEXT, updated_at TEXT)")
        conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, created_at TEXT)")
        conn.commit()
        conn.close()
        import plugins.dreaming as d
        monkeypatch.setattr(d, "_db_path", lambda: db_path)
        sessions = _recent_sessions(days=7)
        candidates = _light_phase(sessions)
        assert candidates == []


# ---------------------------------------------------------------------------
# REM phase tests
# ---------------------------------------------------------------------------


class TestREMPhase:
    def test_returns_themes(self):
        from plugins.dreaming import Candidate, _rem_phase
        candidates = [
            Candidate("Working on Aurex marketing automation"),
            Candidate("Aurex has 7 pillars for language learning"),
            Candidate("Need to fix the gateway crash"),
        ]
        themes, narrative = _rem_phase(candidates, [])
        assert isinstance(themes, list)
        assert isinstance(narrative, str)
        assert "3 candidates" in narrative

    def test_empty_candidates(self):
        from plugins.dreaming import _rem_phase
        themes, narrative = _rem_phase([], [])
        assert themes == []
        assert "No significant" in narrative


# ---------------------------------------------------------------------------
# Deep phase tests
# ---------------------------------------------------------------------------


class TestDeepPhase:
    def test_scores_and_promotes(self):
        from plugins.dreaming import Candidate, _deep_phase
        now = datetime.now(tz=timezone.utc)
        c1 = Candidate("User is working on Aurex marketing with Reddit automation", ts=now)
        c1.frequency = 3
        c2 = Candidate("Random one-off comment", ts=now)
        c2.frequency = 1
        promoted, skipped = _deep_phase([c1, c2], existing="", threshold=0.3, min_recall=2)
        assert any("Aurex" in p for p in promoted)

    def test_respects_threshold(self):
        from plugins.dreaming import Candidate, _deep_phase
        c = Candidate("Short", ts=datetime.now(tz=timezone.utc))
        c.frequency = 1
        promoted, skipped = _deep_phase([c], existing="", threshold=0.9, min_recall=1)
        assert len(promoted) == 0
        assert len(skipped) == 1

    def test_respects_min_recall(self):
        from plugins.dreaming import Candidate, _deep_phase
        c = Candidate("This is a detailed memory about something important", ts=datetime.now(tz=timezone.utc))
        c.frequency = 1
        promoted, skipped = _deep_phase([c], existing="", threshold=0.1, min_recall=3)
        assert len(promoted) == 0

    def test_deduplicates_against_existing_memory(self):
        from plugins.dreaming import Candidate, _deep_phase
        c = Candidate("User prefers direct results", ts=datetime.now(tz=timezone.utc))
        c.frequency = 2
        existing = "User prefers direct results over long explanations and detailed reports"
        promoted, skipped = _deep_phase([c], existing=existing, threshold=0.6, min_recall=2)
        # Should be skipped because it's already in memory (consolidation penalty)
        assert len(promoted) == 0


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestScoring:
    def test_high_frequency_scores_higher(self):
        from plugins.dreaming import Candidate, _score
        now = datetime.now(tz=timezone.utc)
        c_high = Candidate("Detailed memory about project work and preferences", ts=now)
        c_high.frequency = 5
        c_low = Candidate("Detailed memory about project work and preferences", ts=now)
        c_low.frequency = 1
        s_high = _score(c_high, 10, "")
        s_low = _score(c_low, 10, "")
        assert s_high > s_low

    def test_recent_scores_higher(self):
        from plugins.dreaming import Candidate, _score
        now = datetime.now(tz=timezone.utc)
        c_recent = Candidate("Memory about recent work", ts=now)
        c_recent.frequency = 2
        c_old = Candidate("Memory about recent work", ts=now - timedelta(days=30))
        c_old.frequency = 2
        s_recent = _score(c_recent, 10, "")
        s_old = _score(c_old, 10, "")
        assert s_recent > s_old

    def test_existing_memory_penalizes(self):
        from plugins.dreaming import Candidate, _score
        now = datetime.now(tz=timezone.utc)
        c = Candidate("User prefers direct results", ts=now)
        c.frequency = 3
        s_fresh = _score(c, 10, "")
        c2 = Candidate("User prefers direct results", ts=now)
        c2.frequency = 3
        s_dup = _score(c2, 10, "User prefers direct results over long explanations")
        assert s_fresh > s_dup


# ---------------------------------------------------------------------------
# Memory file I/O tests
# ---------------------------------------------------------------------------


class TestMemoryIO:
    def test_append_creates_file(self, tmp_path, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "_memory_path", lambda: tmp_path / "MEMORY.md")
        monkeypatch.setattr(d, "_dreams_path", lambda: tmp_path / "DREAMS.md")
        d._append_memory(["Test memory entry"])
        p = tmp_path / "MEMORY.md"
        assert p.exists()
        content = p.read_text()
        assert "Test memory entry" in content

    def test_append_deduplicates(self, tmp_path, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "_memory_path", lambda: tmp_path / "MEMORY.md")
        monkeypatch.setattr(d, "_dreams_path", lambda: tmp_path / "DREAMS.md")
        d._append_memory(["Unique memory"])
        d._append_memory(["Unique memory"])
        content = (tmp_path / "MEMORY.md").read_text()
        assert content.count("Unique memory") == 1

    def test_append_preserves_existing(self, tmp_path, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "_memory_path", lambda: tmp_path / "MEMORY.md")
        monkeypatch.setattr(d, "_dreams_path", lambda: tmp_path / "DREAMS.md")
        existing = tmp_path / "MEMORY.md"
        existing.write_text("# MEMORY.md\n\n- Existing entry\n")
        d._append_memory(["New entry"])
        content = existing.read_text()
        assert "Existing entry" in content
        assert "New entry" in content

    def test_dreams_file(self, tmp_path, monkeypatch):
        from plugins.dreaming import DreamReport, _append_dreams
        import plugins.dreaming as d
        monkeypatch.setattr(d, "_memory_path", lambda: tmp_path / "MEMORY.md")
        monkeypatch.setattr(d, "_dreams_path", lambda: tmp_path / "DREAMS.md")
        r = DreamReport()
        r.light_count = 5
        r.narrative = "Test"
        _append_dreams(r)
        p = tmp_path / "DREAMS.md"
        assert p.exists()
        assert "Test" in p.read_text()


# ---------------------------------------------------------------------------
# Full cycle integration test
# ---------------------------------------------------------------------------


class TestFullCycle:
    def test_runs_end_to_end(self, sessions_db, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "is_enabled", lambda: True)
        report = d.run_cycle(force=True)
        assert report is not None
        assert report.light_count > 0

    def test_writes_output_files(self, sessions_db, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "is_enabled", lambda: True)
        d.run_cycle(force=True)
        assert d._dreams_path().exists()

    def test_respects_force_flag(self, sessions_db, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "is_enabled", lambda: True)
        report = d.run_cycle(force=True)
        assert report is not None

    def test_disabled_returns_none(self, sessions_db, monkeypatch):
        import plugins.dreaming as d
        monkeypatch.setattr(d, "is_enabled", lambda: False)
        report = d.run_cycle(force=False)
        assert report is None


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_values(self):
        from plugins.dreaming import _cfg, DEFAULT_FREQUENCY, DEFAULT_QUIET_MINUTES
        # Without config, should return defaults
        assert _cfg("nonexistent", "default") == "default"

    def test_is_enabled_defaults_false(self, monkeypatch):
        from plugins.dreaming import is_enabled
        # Without explicit config, should be False (opt-in)
        assert is_enabled() is False


# ---------------------------------------------------------------------------
# Session store tests
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_last_activity(self, sessions_db):
        from plugins.dreaming import _last_user_activity
        result = _last_user_activity()
        assert result is not None
        assert isinstance(result, datetime)

    def test_is_quiet_no_activity(self, tmp_path, monkeypatch):
        from plugins.dreaming import _is_quiet
        import plugins.dreaming as d
        # Empty DB
        db_path = tmp_path / "sessions.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE sessions (session_id TEXT, title TEXT, updated_at TEXT)")
        conn.commit()
        conn.close()
        monkeypatch.setattr(d, "_db_path", lambda: db_path)
        assert _is_quiet() is True

    def test_recent_sessions_filters_by_date(self, sessions_db):
        from plugins.dreaming import _recent_sessions
        # Default 7-day lookback should include s1, s2 but not s3 (10 days old)
        sessions = _recent_sessions(days=7)
        ids = {s["session_id"] for s in sessions}
        assert "s1" in ids
        assert "s2" in ids
        assert "s3" not in ids
