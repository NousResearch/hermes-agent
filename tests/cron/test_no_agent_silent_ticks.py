"""A silent no_agent script fire must not recover an existing failure (#140178)."""

import pytest

from cron import incidents
from cron import scheduler as sched


@pytest.mark.parametrize(
    ("stdout", "neutral"),
    [
        ("", True),
        ('{"wakeAgent": false}', True),
        ("healthy script output", False),
    ],
)
def test_no_agent_script_recovers_only_with_output(monkeypatch, tmp_path, stdout, neutral):
    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "executions.db")
    incident_id, _ = incidents.upsert_incident("watchdog", "script failed")
    sched._mark_incident_alerted(incident_id)

    monkeypatch.setattr(
        sched, "_run_job_script_with_claim_heartbeat",
        lambda *_args, **_kwargs: (True, stdout),
    )
    job = {"id": "watchdog", "name": "watchdog", "script": "watchdog.py", "no_agent": True}
    success, _output, final_response, error = sched._run_no_agent_job(
        job, job["id"], job["name"], None,
    )
    assert success is True
    assert error is None

    suppressed = job.pop("_suppressed_tick", False)
    assert suppressed is neutral
    if neutral:
        assert final_response == sched.SILENT_MARKER

    sched._compose_run_delivery(
        job, success=success, error=error, final_response=final_response,
        output_file=None, suppressed=suppressed,
    )
    assert incidents.get_incident(incident_id)["state"] == (
        "alerted" if neutral else "resolved"
    )
