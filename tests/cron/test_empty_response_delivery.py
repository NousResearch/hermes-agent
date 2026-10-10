"""Empty agent responses use the real scheduler failure path and persisted status."""
from unittest.mock import patch

import pytest

from cron import scheduler
from cron.jobs import create_job, get_job


@pytest.mark.parametrize("response", ["", " \n\t "])
def test_empty_output_persists_failure_and_routes_notice(response):
    job = create_job(prompt="Report", schedule="every 1h", name="empty-output", deliver="local")
    with patch.object(scheduler, "run_job", return_value=(True, "execution output", response, None)), \
         patch.object(scheduler, "_deliver_result", return_value=None) as deliver:
        assert scheduler.run_one_job(job) is True
    stored = get_job(job["id"])
    assert stored["last_status"] == "error"
    assert "empty response" in stored["last_error"]
    deliver.assert_called_once()
    assert deliver.call_args.kwargs["for_failure"] is True
    assert "empty response" in deliver.call_args.args[1]


@pytest.mark.parametrize("no_agent", [False, True])
def test_intentional_silence_remains_successful(no_agent):
    kwargs = {"no_agent": True, "script": "unused.py"} if no_agent else {}
    job = create_job(prompt="Report", schedule="every 1h", name="silent-output", deliver="local", **kwargs)
    with patch.object(scheduler, "run_job", return_value=(True, "execution output", "[SILENT]", None)), \
         patch.object(scheduler, "_deliver_result", return_value=None) as deliver:
        assert scheduler.run_one_job(job) is True
    assert get_job(job["id"])["last_status"] == "ok"
    deliver.assert_not_called()
