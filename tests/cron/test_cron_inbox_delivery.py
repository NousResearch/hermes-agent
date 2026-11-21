"""The ``inbox`` deliver target: it passes the scheduler's delivery preflight (filing needs no adapter),
and the Desktop/Web default files only jobs that have a run session to file."""
import json
from types import SimpleNamespace

import pytest

from cron.scheduler_preflight import _preflight_check_delivery


def test_preflight_lets_an_inbox_job_run():
    assert _preflight_check_delivery({"id": "j1", "deliver": "inbox"}) is None
    assert _preflight_check_delivery({"id": "j1", "deliver": "local,inbox", "failure_deliver": "inbox"}) is None


@pytest.mark.parametrize("no_agent", [False, True])
def test_a_desktop_chat_defaults_only_agent_jobs_to_the_inbox(tmp_path, monkeypatch, no_agent):
    """A script-only fire writes no run session, so filing it into the Inbox would deliver nothing."""
    from cron import inbox
    from gateway import session_policy
    from tools.cronjob_tools import cronjob
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr(session_policy, "active_policy", lambda: SimpleNamespace(source="gui"))
    monkeypatch.setattr(inbox, "_in_bot_chat", lambda: False)
    script = tmp_path / "probe.py"
    script.write_text("print('ok')\n")
    monkeypatch.setattr("tools.cronjob_tools._validate_cron_script_path", lambda *a, **k: None)
    created = json.loads(cronjob(action="create", schedule="every 1h", name="probe", prompt="Check",
                                 **({"no_agent": True, "script": str(script)} if no_agent else {})))
    assert created["success"] is True, created
    from cron.jobs import load_jobs
    job = next(job for job in load_jobs() if job["id"] == created["job_id"])
    assert (job.get("deliver") == "inbox") is (not no_agent)
