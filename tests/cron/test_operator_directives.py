import json
from pathlib import Path

import pytest

from cron.jobs import (
    CronDirectiveError,
    append_directive_event,
    cancel_operator_directive,
    consume_operator_directive_for_run,
    create_operator_directive,
    list_directive_events,
)
from cron.scheduler import _build_job_prompt


@pytest.fixture(autouse=True)
def _cron_paths(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")
    monkeypatch.setattr("cron.jobs.DIRECTIVES_FILE", cron_dir / "directives.json")
    monkeypatch.setattr("cron.jobs.DIRECTIVE_EVENTS_FILE", cron_dir / "directive_events.jsonl")
    yield cron_dir


def _job(job_id="job-a"):
    return {
        "id": job_id,
        "name": "Directive test",
        "prompt": "Pick the next autonomous slice.",
        "schedule_display": "manual",
    }


def test_valid_directive_is_created_consumed_injected_and_audited(_cron_paths):
    directive = create_operator_directive(
        "job-a",
        "Slice source: Operator-named: backend deployed-SHA exposure.",
        "operator-test",
        ttl_seconds=600,
    )

    consumed = consume_operator_directive_for_run("job-a", "cron_job-a_1")
    assert consumed["directive_id"] == directive["directive_id"]
    assert consumed["status"] == "consumed"
    assert consumed["consumed_run_id"] == "cron_job-a_1"

    prompt = _build_job_prompt(_job(), operator_directive=consumed["directive_text"])
    assert "## Operator Directive" in prompt
    assert "backend deployed-SHA exposure" in prompt
    assert prompt.index("## Operator Directive") < prompt.index("Pick the next autonomous slice.")

    events = list_directive_events(job_id="job-a")
    assert [event["event_type"] for event in events] == ["created", "consumed"]
    assert (_cron_paths / "directive_events.jsonl").exists()


def test_consumed_directive_cannot_be_reused_by_id():
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)
    consume_operator_directive_for_run("job-a", "cron_job-a_1")

    with pytest.raises(CronDirectiveError) as exc:
        consume_operator_directive_for_run(
            "job-a",
            "cron_job-a_2",
            directive_id=directive["directive_id"],
        )

    assert "not pending" in str(exc.value)


def test_expired_directive_fails_closed():
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)
    store_path = Path("unused")
    from cron import jobs as cron_jobs

    store_path = cron_jobs.DIRECTIVES_FILE
    data = json.loads(store_path.read_text(encoding="utf-8"))
    data["directives"][0]["expires_at"] = "2000-01-01T00:00:00+00:00"
    store_path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(CronDirectiveError) as exc:
        consume_operator_directive_for_run("job-a", "cron_job-a_1")

    assert exc.value.event_type == "expired"
    events = list_directive_events(job_id="job-a")
    assert events[-1]["event_type"] == "expired"
    assert events[-1]["directive_id"] == directive["directive_id"]


def test_cancelled_directive_fails_closed():
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)
    cancel_operator_directive("job-a")

    with pytest.raises(CronDirectiveError) as exc:
        consume_operator_directive_for_run("job-a", "cron_job-a_1")

    assert exc.value.event_type == "cancelled"
    events = list_directive_events(job_id="job-a")
    assert events[-1]["event_type"] == "cancelled"
    assert events[-1]["directive_id"] == directive["directive_id"]


def test_mismatched_job_id_fails_closed():
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)

    with pytest.raises(CronDirectiveError) as exc:
        consume_operator_directive_for_run(
            "job-b",
            "cron_job-b_1",
            directive_id=directive["directive_id"],
        )

    assert exc.value.event_type == "mismatch"
    events = list_directive_events()
    assert events[-1]["event_type"] == "mismatch"
    assert events[-1]["context"]["actual_job_id"] == "job-b"


def test_missing_explicit_supervised_directive_fails_closed():
    with pytest.raises(CronDirectiveError) as exc:
        consume_operator_directive_for_run(
            "job-a",
            "cron_job-a_1",
            directive_id="missing-directive",
        )

    assert exc.value.event_type == "missing-in-supervised-context"
    events = list_directive_events()
    assert events[-1]["event_type"] == "missing-in-supervised-context"


def test_unparseable_store_fails_closed_and_logs_parse_failed(_cron_paths):
    from cron import jobs as cron_jobs

    cron_jobs.ensure_dirs()
    cron_jobs.DIRECTIVES_FILE.write_text("{not-json", encoding="utf-8")

    with pytest.raises(CronDirectiveError) as exc:
        consume_operator_directive_for_run("job-a", "cron_job-a_1")

    assert exc.value.event_type == "parse-failed"
    events = list_directive_events()
    assert events[-1]["event_type"] == "parse-failed"


def test_autonomous_no_directive_remains_unchanged():
    assert consume_operator_directive_for_run("job-a", "cron_job-a_1") is None
    baseline = _build_job_prompt(_job())
    unchanged = _build_job_prompt(_job(), operator_directive=None)
    assert unchanged == baseline
    assert "## Operator Directive" not in unchanged


def test_supervised_directive_forces_prompt_when_script_output_empty():
    job = {**_job(), "script": "noop.py"}
    autonomous = _build_job_prompt(job, prerun_script=(True, ""), operator_directive=None)
    supervised = _build_job_prompt(
        job,
        prerun_script=(True, ""),
        operator_directive="Run the operator-named slice anyway.",
    )

    assert autonomous is None
    assert supervised is not None
    assert "Run the operator-named slice anyway." in supervised


def test_atomic_append_event_file_is_writable(_cron_paths):
    append_directive_event("invalid", "job-a", "directive-x", {"reason": "dry-check"})
    events_file = _cron_paths / "directive_events.jsonl"
    assert events_file.exists()
    assert json.loads(events_file.read_text(encoding="utf-8").strip())["event_type"] == "invalid"
