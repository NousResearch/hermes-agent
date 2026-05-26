import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
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
from cron.scheduler import _build_job_prompt, run_job


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


def test_concurrent_consumers_cannot_both_consume():
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)

    def consume_once(run_id):
        return consume_operator_directive_for_run("job-a", run_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume_once, ["cron_job-a_1", "cron_job-a_2"]))

    consumed = [result for result in results if result is not None]
    misses = [result for result in results if result is None]
    assert len(consumed) == 1
    assert len(misses) == 1
    assert consumed[0]["directive_id"] == directive["directive_id"]
    assert consumed[0]["status"] == "consumed"

    events = list_directive_events(job_id="job-a", event_type="consumed")
    assert len(events) == 1
    assert events[0]["directive_id"] == directive["directive_id"]


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


def test_consumed_directive_prompt_assembly_failure_audited_once(monkeypatch):
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)
    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        types.SimpleNamespace(AIAgent=object),
    )

    def fail_build(*args, **kwargs):
        raise RuntimeError("prompt assembly exploded")

    monkeypatch.setattr("cron.scheduler._build_job_prompt", fail_build)

    success, output, final_response, error = run_job(_job())

    assert success is False
    assert final_response == ""
    assert "prompt assembly exploded" in error
    assert "prompt assembly exploded" in output
    events = list_directive_events(job_id="job-a", event_type="consumed-but-runner-failed")
    assert len(events) == 1
    assert events[0]["directive_id"] == directive["directive_id"]


def test_consumed_directive_prerun_script_failure_audited_once(monkeypatch):
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)
    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        types.SimpleNamespace(AIAgent=object),
    )

    def fail_script(*args, **kwargs):
        raise RuntimeError("pre-run script exploded")

    monkeypatch.setattr("cron.scheduler._run_job_script", fail_script)

    success, output, final_response, error = run_job({**_job(), "script": "noop.py"})

    assert success is False
    assert final_response == ""
    assert "pre-run script exploded" in error
    assert "pre-run script exploded" in output
    events = list_directive_events(job_id="job-a", event_type="consumed-but-runner-failed")
    assert len(events) == 1
    assert events[0]["directive_id"] == directive["directive_id"]


def test_consumed_directive_agent_execution_failure_audited_once(monkeypatch):
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)

    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run_conversation(self, prompt):
            raise RuntimeError("agent execution exploded")

        def close(self):
            pass

    class FakeSessionDB:
        def end_session(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class FakeAuthError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "run_agent",
        types.SimpleNamespace(AIAgent=FakeAgent),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_state",
        types.SimpleNamespace(SessionDB=FakeSessionDB),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        types.SimpleNamespace(
            resolve_runtime_provider=lambda **kwargs: {
                "provider": "test",
                "api_key": "test-key",
                "base_url": "http://127.0.0.1",
                "api_mode": "chat_completions",
            },
            format_runtime_provider_error=lambda exc: str(exc),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.auth",
        types.SimpleNamespace(AuthError=FakeAuthError),
    )
    monkeypatch.setattr("cron.scheduler._deliver_result", lambda *args, **kwargs: None)

    success, output, final_response, error = run_job(_job())

    assert success is False
    assert final_response == ""
    assert "agent execution exploded" in error
    assert "agent execution exploded" in output

    consumed_events = list_directive_events(job_id="job-a", event_type="consumed")
    assert len(consumed_events) == 1
    assert consumed_events[0]["directive_id"] == directive["directive_id"]

    failure_events = list_directive_events(job_id="job-a", event_type="consumed-but-runner-failed")
    assert len(failure_events) == 1
    assert failure_events[0]["directive_id"] == directive["directive_id"]


def test_no_agent_job_with_manual_directive_fails_closed_before_script(monkeypatch):
    directive = create_operator_directive("job-a", "Do the named slice.", "operator-test", ttl_seconds=600)

    def script_should_not_run(*args, **kwargs):
        raise AssertionError("no_agent script should not run when directive is present")

    monkeypatch.setattr("cron.scheduler._run_job_script", script_should_not_run)
    success, output, final_response, error = run_job({**_job(), "no_agent": True, "script": "noop.py"})

    assert success is False
    assert final_response == ""
    assert "no_agent jobs cannot consume directives" in error
    assert "BLOCKED" in output
    events = list_directive_events(job_id="job-a", event_type="invalid")
    assert len(events) == 1
    assert events[0]["directive_id"] == directive["directive_id"]
    assert events[0]["context"]["reason"] == "no_agent_jobs_do_not_support_operator_directives"


def test_atomic_append_event_file_is_writable(_cron_paths):
    append_directive_event("invalid", "job-a", "directive-x", {"reason": "dry-check"})
    events_file = _cron_paths / "directive_events.jsonl"
    assert events_file.exists()
    assert json.loads(events_file.read_text(encoding="utf-8").strip())["event_type"] == "invalid"
