"""A cron session row carries the run's reasoning config, so a chat that continues it resumes at
the job's effort.

The gateway owner reserves the cron session row before the agent runs. The turn lease then finds
the row and skips the agent's lazy create, which is the write that persisted the constructor's
``reasoning_config``. Desktop resume (``_stored_session_runtime_overrides``) restores effort only
from that field, so a ``reasoning_effort: low`` job continued at the global level instead.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from cron.scheduler import run_job
from hermes_state import SessionDB

_RUNTIME = {
    "api_key": "test-key",
    "base_url": "https://example.invalid/v1",
    "provider": "openrouter",
    "api_mode": "chat_completions",
}


class _ReservedRowCronAgent:
    """Stand-in for the cron AIAgent when the owner already reserved its row: the turn lease
    marks the row as created, so the agent writes no session row of its own."""

    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        return {"final_response": "ok"}


def _run_owned_job(job, db, tmp_path, session_id):
    from gateway import session_cron
    token = session_cron._execution.set((SimpleNamespace(db=db), session_id, job["id"], None))
    try:
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state_registry.acquire", return_value=db), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
             patch("run_agent.AIAgent", _ReservedRowCronAgent):
            return run_job(job)
    finally:
        session_cron._execution.reset(token)


def test_owner_reserved_cron_row_keeps_the_jobs_reasoning_effort(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    session_id = "cron_low-effort-job_owned"
    db.create_session(session_id, source="cron")  # the owner's reservation
    job = {"id": "low-effort-job", "name": "low effort", "prompt": "hello", "reasoning_effort": "low"}
    try:
        success, _output, _final, error = _run_owned_job(job, db, tmp_path, session_id)

        assert success is True, error
        row = db.get_session(session_id)
        assert row is not None
        assert json.loads(row["model_config"])["reasoning_config"] == {"enabled": True, "effort": "low"}
    finally:
        db.close()
