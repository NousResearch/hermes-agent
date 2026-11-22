"""A cron session row carries the run's reasoning config, so a chat that continues it resumes at
the job's effort.

The gateway owner reserves the cron session row before the agent runs. The turn lease then finds
the row and skips the agent's lazy create, which is the write that persisted the constructor's
``reasoning_config``. Desktop resume (``_stored_session_runtime_overrides``) restores effort only
from that field, so a ``reasoning_effort: low`` job continued at the global level instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from cron.scheduler import run_job
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope

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
    token = session_cron._execution.set((SimpleNamespace(persistence=db), session_id, job["id"], None))
    try:
        with patch("cron.scheduler._hermes_home", tmp_path), \
             patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=_RUNTIME), \
             patch("run_agent.AIAgent", _ReservedRowCronAgent):
            return run_job(job)
    finally:
        session_cron._execution.reset(token)


def test_owner_reserved_cron_row_keeps_the_jobs_reasoning_effort(tmp_path):
    db = open_persistence(profile_scope(home=tmp_path))
    session_id = "cron_low-effort-job_owned"
    db.sessions.create(NewSession(session_id=session_id, source="cron"))  # the owner's reservation
    job = {"id": "low-effort-job", "name": "low effort", "prompt": "hello", "reasoning_effort": "low"}
    try:
        success, _output, _final, error = _run_owned_job(job, db, tmp_path, session_id)

        assert success is True, error
        assert db.sessions.get(session_id) is not None
        assert db.sessions.get_model_config_value(session_id, "reasoning_config") == {
            "enabled": True, "effort": "low"}
    finally:
        db.close()
