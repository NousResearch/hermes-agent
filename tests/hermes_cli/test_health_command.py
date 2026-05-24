"""
Tests for the 'hermes health' CLI command.

Mocks DB and filesystem calls; verifies output sections and content.
"""

import io
import sqlite3
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import hermes_cli.health as health_mod


# ---------------------------------------------------------------------------
# Unit tests for pure helper functions
# ---------------------------------------------------------------------------

class TestFmtBytes:
    def test_bytes(self):
        assert health_mod._fmt_bytes(500) == "500 B"

    def test_kilobytes(self):
        assert "KB" in health_mod._fmt_bytes(2048)

    def test_megabytes(self):
        result = health_mod._fmt_bytes(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.0" in result

    def test_gigabytes(self):
        result = health_mod._fmt_bytes(2 * 1024 ** 3)
        assert "GB" in result


class TestRelativeTime:
    def test_just_now(self):
        assert health_mod._relative_time(time.time() - 10) == "just now"

    def test_minutes_ago(self):
        assert "m ago" in health_mod._relative_time(time.time() - 300)

    def test_hours_ago(self):
        assert "h ago" in health_mod._relative_time(time.time() - 7200)

    def test_days_ago(self):
        assert "days ago" in health_mod._relative_time(time.time() - 86400 * 10)

    def test_never(self):
        assert health_mod._relative_time(None) == "never"

    def test_iso_string(self):
        from datetime import datetime, timezone, timedelta
        ts = datetime.now(timezone.utc) - timedelta(minutes=5)
        result = health_mod._relative_time(ts.isoformat())
        assert "m ago" in result


class TestDbInfo:
    def test_missing_db_returns_zeros(self, tmp_path):
        result = health_mod._db_info(tmp_path / "nonexistent.db")
        assert result["total_sessions"] == 0
        assert result["total_messages"] == 0
        assert result["db_size"] == 0

    def test_real_db(self, tmp_path):
        db_path = tmp_path / "state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE sessions (id TEXT PRIMARY KEY, parent_session_id TEXT)"
        )
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO sessions VALUES ('s1', NULL)")
        conn.execute("INSERT INTO sessions VALUES ('s2', 's1')")
        conn.execute("INSERT INTO messages VALUES (1)")
        conn.execute("INSERT INTO messages VALUES (2)")
        conn.execute("INSERT INTO messages VALUES (3)")
        conn.commit()
        conn.close()

        result = health_mod._db_info(db_path)
        assert result["total_sessions"] == 2
        assert result["active_chains"] == 1       # s1 has no parent
        assert result["compressed_sessions"] == 1  # s2 has parent
        assert result["total_messages"] == 3
        assert result["db_size"] > 0
        assert result["error"] is None

    def test_wal_file_detected(self, tmp_path):
        db_path = tmp_path / "state.db"
        db_path.write_bytes(b"\x00" * 1024)
        wal_path = tmp_path / "state.db-wal"
        wal_path.write_bytes(b"\x00" * 6_000_000)  # 6 MB

        result = health_mod._db_info(db_path)
        assert result["wal_size"] == 6_000_000


class TestRecommendations:
    def test_no_issues(self):
        db = {"wal_size": 0}
        ckpt = {"repos_over_50_commits": 0}
        assert health_mod._recommendations(db, ckpt) == []

    def test_wal_too_large(self):
        db = {"wal_size": 10 * 1024 * 1024}  # 10 MB
        ckpt = {"repos_over_50_commits": 0}
        recs = health_mod._recommendations(db, ckpt)
        assert len(recs) == 1
        assert "WAL file" in recs[0]
        assert "hermes db vacuum" in recs[0]

    def test_checkpoint_repos_over_limit(self):
        db = {"wal_size": 0}
        ckpt = {"repos_over_50_commits": 3}
        recs = health_mod._recommendations(db, ckpt)
        assert len(recs) == 1
        assert "checkpoint" in recs[0].lower()
        assert "hermes checkpoints prune" in recs[0]

    def test_both_issues(self):
        db = {"wal_size": 10 * 1024 * 1024}
        ckpt = {"repos_over_50_commits": 2}
        recs = health_mod._recommendations(db, ckpt)
        assert len(recs) == 2


# ---------------------------------------------------------------------------
# Integration tests for run_health output format
# ---------------------------------------------------------------------------

def _make_db_dict(**overrides):
    base = dict(
        db_size=44_040_192,
        wal_size=0,
        page_count=100,
        page_size=4096,
        total_sessions=1247,
        active_chains=38,
        compressed_sessions=892,
        total_messages=47832,
        error=None,
    )
    base.update(overrides)
    return base


def _make_ckpt_dict(**overrides):
    base = dict(
        repo_count=14,
        total_size=2_254_857_830,
        oldest_mtime=time.time() - 86400 * 45,
        repos_over_50_commits=0,
    )
    base.update(overrides)
    return base


def _make_cron_dict(**overrides):
    base = dict(
        active_jobs=5,
        failed_jobs=0,
        last_run_ts=time.time() - 720,
        error=None,
    )
    base.update(overrides)
    return base


def _capture_run_health(monkeypatch, tmp_path, db_dict=None, ckpt_dict=None, cron_dict=None):
    """Run run_health with mocked helpers; return captured stdout."""
    db_dict = db_dict or _make_db_dict()
    ckpt_dict = ckpt_dict or _make_ckpt_dict()
    cron_dict = cron_dict or _make_cron_dict()

    monkeypatch.setattr(health_mod, "_db_info", lambda p: db_dict)
    monkeypatch.setattr(health_mod, "_checkpoint_info", lambda p: ckpt_dict)
    monkeypatch.setattr(health_mod, "_cron_info", lambda: cron_dict)
    monkeypatch.setattr(health_mod, "_get_hermes_home_path", lambda: tmp_path)
    monkeypatch.setattr(health_mod, "_get_checkpoint_base", lambda: tmp_path / "checkpoints")

    args = MagicMock()
    buf = io.StringIO()
    orig = sys.stdout
    sys.stdout = buf
    try:
        health_mod.run_health(args)
    finally:
        sys.stdout = orig
    return buf.getvalue()


class TestRunHealth:
    def test_database_section_present(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Database" in out

    def test_sessions_line_present(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Sessions:" in out
        assert "1247" in out

    def test_active_chains_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "38 active chains" in out

    def test_compressed_sessions_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "892 compressed" in out

    def test_messages_line_present(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Messages:" in out
        assert "47832" in out

    def test_checkpoints_section_present(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Checkpoints" in out
        assert "14 repos" in out

    def test_checkpoint_oldest_age(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Oldest:" in out
        assert "days ago" in out

    def test_cron_section_present(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Cron" in out
        assert "5 jobs active" in out

    def test_cron_last_run_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "last run:" in out
        assert "m ago" in out

    def test_no_recommendations_when_healthy(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "Recommendations" not in out

    def test_wal_recommendation_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            db_dict=_make_db_dict(wal_size=10 * 1024 * 1024),
        )
        assert "Recommendations" in out
        assert "WAL file" in out
        assert "hermes db vacuum" in out

    def test_checkpoint_recommendation_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            ckpt_dict=_make_ckpt_dict(repos_over_50_commits=3),
        )
        assert "Recommendations" in out
        assert "hermes checkpoints prune" in out

    def test_failed_cron_jobs_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            cron_dict=_make_cron_dict(failed_jobs=1),
        )
        assert "1 failed" in out

    def test_cron_error_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            cron_dict=dict(error="no cron module", active_jobs=0, failed_jobs=0, last_run_ts=None),
        )
        assert "unavailable" in out

    def test_db_error_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            db_dict=dict(
                error="disk I/O error", db_size=0, wal_size=0,
                total_sessions=0, active_chains=0, compressed_sessions=0,
                total_messages=0,
            ),
        )
        assert "error" in out.lower()

    def test_no_checkpoints(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            ckpt_dict=dict(repo_count=0, total_size=0, oldest_mtime=None, repos_over_50_commits=0),
        )
        assert "No checkpoint repos found" in out

    def test_wal_size_shown_in_db_line(self, monkeypatch, tmp_path):
        out = _capture_run_health(
            monkeypatch, tmp_path,
            db_dict=_make_db_dict(wal_size=8 * 1024 * 1024),
        )
        assert "WAL:" in out

    def test_db_size_shown(self, monkeypatch, tmp_path):
        out = _capture_run_health(monkeypatch, tmp_path)
        assert "state.db:" in out
        # 44 MB
        assert "MB" in out


class TestHealthSubparserRegistered:
    """Verify 'hermes health' cmd_health function exists in main module."""

    def test_cmd_health_defined_in_main(self):
        from hermes_cli import main as main_mod
        assert hasattr(main_mod, "cmd_health"), "cmd_health must be defined in main.py"
        assert callable(main_mod.cmd_health)
