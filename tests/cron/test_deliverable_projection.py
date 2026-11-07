"""Cron marks the deliverable while retaining full run archives (#53383, #62511)."""

import pytest

from cron import scheduler, scheduler_delivery, scheduler_prompt
from tests.cron import test_scheduler as scheduler_cases

_QUOTED_MARKER_REPORT = """Status: green

```text
FINAL_CRON_OUTPUT: literal example
```
Next: deploy"""


@pytest.mark.parametrize("response, expected", [
    ("Working notes.\nFINAL_CRON_OUTPUT:\nPublic report", "Public report"),
    (f"Working notes.\n\nFINAL_CRON_OUTPUT:\n{_QUOTED_MARKER_REPORT}", _QUOTED_MARKER_REPORT),
    ("Notes.\nFINAL_CRON_OUTPUT: Title\n- body", "Title\n- body"),
    ("Report\n---\nBody", "Report\n---\nBody"),
    ("Notes.\nFinal answer:\nReport", "Notes.\nFinal answer:\nReport"),
    ("Notes.\nFINAL_CRON_OUTPUT:\n", "Notes.\nFINAL_CRON_OUTPUT:"),
    ("Notes.\nfinal_cron_output: Report", "Report"),
    ("", ""), (None, ""),
])
def test_only_the_explicit_boundary_projects_delivery(response, expected):
    assert scheduler_delivery._extract_cron_deliverable_response(response) == expected


@pytest.mark.parametrize("tail", ["Public report", "[SILENT]", _QUOTED_MARKER_REPORT])
def test_run_archive_and_queued_delivery_keep_their_own_projections(tmp_path, monkeypatch, tail):
    job = {"id": "deliverable-test", "name": "test", "prompt": "Report now", "deliver": "local"}
    response = f"Working notes; [SILENT] was considered.\nFINAL_CRON_OUTPUT:\n{tail}"
    with scheduler_cases.TestRunJobSessionPersistence()._run_job_patches(tmp_path) as (db, agent):
        agent.return_value.run_conversation.return_value = {"final_response": response}
        success, output, returned, error = scheduler_cases._run_owned_job(job, tmp_path, db)
    assert success and error is None and returned == response
    assert response in output
    home = tmp_path / "archive-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    queued = []

    def enqueue(execution_id, queued_job, content, **kwargs):
        queued.append(content)
        return {"status": "queued"}

    monkeypatch.setattr("cron.delivery_queue.enqueue", enqueue)
    delivery_job = {**job, "deliver": "telegram:123", "execution_id": "fixture-execution"}
    delivery = scheduler._RunDelivery(delivery_job, success=True, error=None)
    scheduler._save_compose_deliver(delivery, scheduler._FireOwnership(delivery_job), response, output,
                                  adapters=None, loop=None, verbose=False, execution_token=None)
    archives = list((home / "cron" / "output" / job["id"]).glob("*.md"))
    assert len(archives) == 1 and archives[0].read_text(encoding="utf-8") == output
    if tail == "[SILENT]":
        assert queued == []
        assert scheduler_prompt._archive_answer(output) is None
    else:
        assert queued == [tail]
        assert scheduler_prompt._archive_answer(output) == response
    assert "FINAL_CRON_OUTPUT:" in scheduler_prompt._build_job_prompt(job)
