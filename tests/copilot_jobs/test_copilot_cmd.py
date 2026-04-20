"""Tests for hermes_cli.copilot_cmd — slash command parsing and dispatch."""

import io
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    """Provide a fresh SessionDB with HERMES_HOME pointed at tmp_path."""
    db_path = tmp_path / ".hermes" / "state.db"
    db_path.parent.mkdir(parents=True)
    _db = SessionDB(db_path=db_path)
    # Prevent handlers' finally: db.close() from killing our connection.
    _real_close = _db.close
    _db.close = lambda: None
    yield _db
    _real_close()


@pytest.fixture(autouse=True)
def _patch_get_db(db, monkeypatch):
    """Patch _get_db in copilot_cmd to use the test DB."""
    monkeypatch.setattr(
        "hermes_cli.copilot_cmd._get_db", lambda: db
    )


def _capture_slash(cmd: str) -> str:
    """Run handle_copilot_slash and capture combined stdout+stderr."""
    from hermes_cli.copilot_cmd import handle_copilot_slash
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf
    try:
        handle_copilot_slash(cmd)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return buf.getvalue()


class TestSlashList:
    def test_empty_list(self):
        out = _capture_slash("/copilot list")
        assert "No copilot jobs found" in out

    def test_list_shows_job(self, db):
        db.create_copilot_job(
            job_id="cj_test1", repo_slug="my-repo", repo_path="/test"
        )
        out = _capture_slash("/copilot list")
        assert "cj_test1" in out
        assert "my-repo" in out

    def test_default_subcommand_is_list(self, db):
        """Bare /copilot with no subcommand should show the job list."""
        db.create_copilot_job(
            job_id="cj_bare", repo_slug="bare-repo", repo_path="/test"
        )
        out = _capture_slash("/copilot")
        assert "cj_bare" in out

    def test_list_state_filter(self, db):
        db.create_copilot_job(
            job_id="cj_p", repo_slug="repo-a", repo_path="/a"
        )
        db.create_copilot_job(
            job_id="cj_r", repo_slug="repo-b", repo_path="/b"
        )
        db.transition_copilot_job("cj_r", "running", event_type="test")

        out = _capture_slash("/copilot list --state running")
        assert "cj_r" in out
        assert "cj_p" not in out


class TestSlashLaunchNoAuto:
    def test_no_auto_creates_pending(self, db):
        out = _capture_slash(
            "/copilot launch --no-auto --repo test-repo --repo-path /test Fix bug"
        )
        assert "pending" in out
        assert "Manual launch required" in out

        job = db.list_copilot_jobs(state="pending")
        assert len(job) == 1
        assert job[0]["repo_slug"] == "test-repo"

    def test_no_auto_records_prompt(self, db):
        _capture_slash(
            "/copilot launch --no-auto --repo r --repo-path /p Refactor auth module"
        )
        jobs = db.list_copilot_jobs()
        assert "Refactor auth module" in (jobs[0].get("prompt") or "")


class TestSlashLaunchDryRun:
    def test_dry_run_auto_launches(self, db):
        out = _capture_slash(
            "/copilot launch --dry-run --repo dr-repo --repo-path /dr Do something"
        )
        assert "running" in out
        assert "pid 0" in out

        jobs = db.list_copilot_jobs(state="running")
        assert len(jobs) == 1
        assert jobs[0]["copilot_session_id"].startswith("dry-run-")

    def test_model_flag(self, db):
        out = _capture_slash(
            "/copilot launch --dry-run --model gpt-5 --repo m-repo --repo-path /m Test"
        )
        assert "running" in out


class TestSlashShow:
    def test_show_existing(self, db):
        db.create_copilot_job(
            job_id="cj_show", repo_slug="show-repo", repo_path="/show"
        )
        out = _capture_slash("/copilot show cj_show")
        assert "cj_show" in out
        assert "show-repo" in out
        assert "pending" in out

    def test_show_nonexistent(self):
        out = _capture_slash("/copilot show cj_nope")
        assert "not found" in out.lower()


class TestSlashActivate:
    def test_activate_with_session_id(self, db):
        db.create_copilot_job(
            job_id="cj_act", repo_slug="act-repo", repo_path="/act"
        )
        out = _capture_slash(
            "/copilot activate cj_act --session-id ses_999"
        )
        assert "activated" in out

        job = db.get_copilot_job("cj_act")
        assert job["state"] == "running"
        assert job["copilot_session_id"] == "ses_999"


class TestSlashTakeover:
    def test_takeover(self, db):
        db.create_copilot_job(
            job_id="cj_take", repo_slug="take-repo", repo_path="/take"
        )
        out = _capture_slash("/copilot takeover cj_take")
        assert "human" in out

        job = db.get_copilot_job("cj_take")
        assert job["owner"] == "human"


class TestSlashIdleAndClose:
    def test_idle_then_close(self, db):
        db.create_copilot_job(
            job_id="cj_ic", repo_slug="ic-repo", repo_path="/ic"
        )
        db.transition_copilot_job("cj_ic", "running", event_type="test")

        out = _capture_slash("/copilot idle cj_ic")
        assert "idle" in out.lower()

        out = _capture_slash("/copilot close cj_ic")
        assert "closed" in out.lower()

        job = db.get_copilot_job("cj_ic")
        assert job["state"] == "closed"


class TestSlashReap:
    def test_reap_nothing(self):
        out = _capture_slash("/copilot reap")
        assert "nothing to do" in out.lower()

    def test_reap_stale_pending(self, db):
        db.create_copilot_job(
            job_id="cj_stale", repo_slug="stale-repo", repo_path="/stale"
        )
        db._conn.execute(
            "UPDATE copilot_jobs SET created_at = ? WHERE id = ?",
            (time.time() - 7200, "cj_stale"),
        )
        db._conn.commit()

        out = _capture_slash("/copilot reap")
        assert "stale" in out.lower() or "pending" in out.lower()

        job = db.get_copilot_job("cj_stale")
        assert job["state"] == "closed"


class TestSlashErrorPaths:
    def test_duplicate_repo_guard(self, db):
        db.create_copilot_job(
            job_id="cj_dup", repo_slug="dup-repo", repo_path="/dup"
        )
        out = _capture_slash(
            "/copilot launch --no-auto --repo dup-repo --repo-path /dup Again"
        )
        assert "already exists" in out.lower()

    def test_empty_launch(self):
        out = _capture_slash("/copilot launch")
        assert "required" in out.lower()

    def test_close_running_fails(self, db):
        db.create_copilot_job(
            job_id="cj_crf", repo_slug="crf-repo", repo_path="/crf"
        )
        db.transition_copilot_job("cj_crf", "running", event_type="test")

        out = _capture_slash("/copilot close cj_crf")
        assert "invalid transition" in out.lower()

    def test_unknown_subcommand_shows_help(self):
        out = _capture_slash("/copilot foobar")
        assert "usage" in out.lower()
        assert "launch" in out
        assert "reap" in out
