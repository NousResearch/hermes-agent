"""Tests for hermes_cli.copilot_cmd — slash command parsing and dispatch."""

import io
import sys
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
            job_id="cj_r", repo_slug="repo-a", repo_path="/a"
        )
        db.create_copilot_job(
            job_id="cj_d", repo_slug="repo-b", repo_path="/b"
        )
        db.finish_copilot_job("cj_d", state="done", exit_code=0)

        out = _capture_slash("/copilot list --state running")
        assert "cj_r" in out
        assert "cj_d" not in out


class TestSlashLaunchDryRun:
    def test_dry_run_launches(self, db):
        out = _capture_slash(
            "/copilot launch --dry-run --repo dr-repo --repo-path /dr Do something"
        )
        assert "done" in out.lower() or "dry-run" in out.lower()

        jobs = db.list_copilot_jobs(state="done")
        assert len(jobs) == 1
        assert jobs[0]["copilot_session_id"].startswith("dry-run-")

    def test_model_flag(self, db):
        out = _capture_slash(
            "/copilot launch --dry-run --model gpt-5 --repo m-repo --repo-path /m Test"
        )
        assert "done" in out.lower() or "dry-run" in out.lower()


class TestSlashShow:
    def test_show_existing(self, db):
        db.create_copilot_job(
            job_id="cj_show", repo_slug="show-repo", repo_path="/show"
        )
        out = _capture_slash("/copilot show cj_show")
        assert "cj_show" in out
        assert "show-repo" in out

    def test_show_nonexistent(self):
        out = _capture_slash("/copilot show cj_nope")
        assert "not found" in out.lower()


class TestSlashErrorPaths:
    def test_empty_launch(self):
        out = _capture_slash("/copilot launch")
        assert "required" in out.lower()

    def test_unknown_subcommand_shows_help(self):
        out = _capture_slash("/copilot foobar")
        assert "usage" in out.lower()
        assert "launch" in out
