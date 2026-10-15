"""Cron sessions stamp the job's workdir on their session row (#108205).

The sidebar groups sessions by cwd prefix; a cron run's row was left NULL even when the
job ran inside a repo workdir (``_launch_cwd_for_session`` records no cwd for cron
source), so every cron session filed under __no_project__. The scheduler — the owner of
cron-session finalization — stamps the job's workdir before closing the session, with the
same before-end_session ordering as the title write (#50536).
"""

from __future__ import annotations

from unittest.mock import patch

from cron.scheduler import run_job
from hermes_state import SessionDB

_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


class _FakeCronAgent:
    """Stand-in for the cron AIAgent: creates its session row on the first turn exactly
    like AIAgent's lazy create (source 'cron', no cwd — _launch_cwd_for_session records
    none for cron source)."""

    def __init__(self, *args, session_id=None, session_db=None, **kwargs):
        self.session_id = session_id
        self.session_db = session_db

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        if self.session_db is not None:
            self.session_db.create_session(self.session_id, source="cron")
        return {"final_response": "ok"}


def _run_job_with_real_db(job, db, tmp_path):
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
         patch("run_agent.AIAgent", _FakeCronAgent):
        return run_job(job)


def _cron_rows(db):
    return [dict(r) for r in db._read_all("SELECT * FROM sessions WHERE id LIKE 'cron_%'")]


def test_run_job_stamps_workdir_on_cron_session_row(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    repo = tmp_path / "repo"
    repo.mkdir()
    job = {"id": "workdir-job", "name": "workdir job", "prompt": "hello", "workdir": str(repo)}

    try:
        success, _output, _final, error = _run_job_with_real_db(job, db, tmp_path)

        assert success is True, error
        rows = _cron_rows(db)
        assert len(rows) == 1
        assert rows[0]["cwd"] == str(repo)
    finally:
        db.close()


def test_run_job_without_workdir_leaves_cron_session_cwd_null(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    job = {"id": "no-workdir-job", "name": "no workdir", "prompt": "hello"}

    try:
        success, _output, _final, error = _run_job_with_real_db(job, db, tmp_path)

        assert success is True, error
        rows = _cron_rows(db)
        assert len(rows) == 1
        assert rows[0]["cwd"] is None
    finally:
        db.close()
