"""Editing a job's delivery from inside a cron run must not borrow the running job's target."""

import json

import pytest

from tests.cron.test_cron_created_delivery import _create, _enter_cron_context, _exit_cron_context


@pytest.fixture
def temp_cron_home(tmp_path):
    from cron import jobs as cron_jobs

    with cron_jobs.use_cron_store(tmp_path):
        cron_jobs.ensure_dirs()
        yield tmp_path


def _update(job_id, **kw):
    from tools.cronjob_tools import cronjob

    return json.loads(cronjob(action="update", job_id=job_id, **kw))


@pytest.mark.parametrize("running", [(), ("telegram", "-100999")])
def test_reapplying_origin_from_a_cron_run_keeps_it(temp_cron_home, running):
    from cron.jobs import get_job

    created = _create(deliver="origin")  # normal shell: literal origin, no captured origin
    assert get_job(created["job_id"])["deliver"] == "origin"
    tokens, extra = _enter_cron_context(*running)
    try:
        result = _update(created["job_id"], deliver="origin", failure_deliver="origin")
    finally:
        _exit_cron_context(tokens, extra)
    assert result["success"] is True
    job = get_job(created["job_id"])
    assert (job["deliver"], job["failure_deliver"]) == ("origin", "origin")


def test_job_with_captured_origin_keeps_it(temp_cron_home):
    from cron.jobs import get_job, update_job

    created = _create(deliver="local")
    update_job(created["job_id"], {"origin": {"platform": "telegram", "chat_id": "555"}})
    tokens, extra = _enter_cron_context("telegram", "-100999")
    try:
        _update(created["job_id"], deliver="origin")
    finally:
        _exit_cron_context(tokens, extra)
    assert get_job(created["job_id"])["deliver"] == "origin"
