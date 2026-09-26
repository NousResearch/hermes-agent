"""A cron run that delivers nothing must not be recorded as plain success.

An agent may answer with the silence marker to say "nothing to report". The
run itself succeeded, so before this fix it was stored with last_status="ok"
— identical to a run that did real work. A job that silently stopped
producing output therefore looked healthy in ``hermes cron list`` and in any
health check reading last_status. The distinct ``no_op`` status names that
outcome without turning it into a failure.
"""

import pytest

import cron.scheduler as s


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json/executions don't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_silent_status_constant_is_distinct():
    assert s.SILENT_STATUS not in ("ok", "error")


def test_run_delivery_silent_response_defaults_false():
    """The flag must default off — only the explicit silence-detection branch in
    _save_compose_deliver sets it, never any other should_deliver=False path
    (fire-claim loss, unresolved delivery target, an auto-retry hold, ...)."""
    d = s._RunDelivery(job={"id": "j"}, success=True, error=None)
    assert d.silent_response is False


def test_mark_job_run_accepts_explicit_status(temp_home):
    """An explicit status wins over the success/failure default; callers that pass
    none keep the old ok/error behaviour."""
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="Test", schedule="every 1h")
    mark_job_run(job["id"], True, status=s.SILENT_STATUS)
    assert get_job(job["id"])["last_status"] == s.SILENT_STATUS

    job_ok = create_job(prompt="Test", schedule="every 1h")
    mark_job_run(job_ok["id"], True)
    assert get_job(job_ok["id"])["last_status"] == "ok"

    job_err = create_job(prompt="Test", schedule="every 1h")
    mark_job_run(job_err["id"], False, "boom")
    updated_err = get_job(job_err["id"])
    assert updated_err["last_status"] == "error"
    assert updated_err["last_error"] == "boom"


def test_silent_agent_response_records_no_op_status(temp_home, monkeypatch):
    """End to end through run_one_job: an agent response that is nothing but the
    silence marker must not be delivered, and must land as last_status="no_op" —
    not "ok", which would be indistinguishable from a run that did real work."""
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="silent-job")
    assert claim_job_for_fire(job["id"]) is True
    job = get_job(job["id"])

    delivered = []
    monkeypatch.setattr(
        s, "run_job",
        lambda job, **kwargs: (True, "output text", s.SILENT_MARKER, None))
    monkeypatch.setattr(
        s, "_deliver_result",
        lambda job, content, **kwargs: delivered.append(content))

    assert s.run_one_job(job) is True

    assert delivered == [], "a silent response must never be delivered"
    record = get_job(job["id"])
    assert record["last_status"] == s.SILENT_STATUS
    assert record["last_status"] != "ok"
    # A silent no-op is not a failure: it must not count against the streak.
    assert record["failure_streak"] == 0
