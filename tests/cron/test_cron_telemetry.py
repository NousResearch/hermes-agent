import json
import os

import cron.scheduler as scheduler


def test_no_agent_cron_run_emits_completion_event(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "ok.sh"
    script.write_text("printf 'hello from watchdog'\n", encoding="utf-8")
    script.chmod(0o700)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(scheduler, "_hermes_home", hermes_home)
    monkeypatch.setattr(scheduler, "_SCRIPT_TIMEOUT", 5)

    job = {
        "id": "raw-job-id-123",
        "name": "Sensitive prompt-like job name",
        "no_agent": True,
        "script": "ok.sh",
        "schedule": "every 1h",
    }

    success, _doc, final_response, error = scheduler.run_job(job)

    assert success is True
    assert final_response == "hello from watchdog"
    assert error is None

    event_files = list((hermes_home / "ops" / "events").glob("*.jsonl"))
    assert len(event_files) == 1
    lines = event_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])

    assert event["event_type"] == "cron_run"
    assert event["source"] == "cron.scheduler"
    assert event["status"] == "ok"
    assert event["payload"]["mode"] == "no_agent"
    assert event["payload"]["has_script"] is True
    assert event["payload"]["delivery_state"] == "non_empty"
    assert event["payload"]["final_response_chars"] == len(final_response)
    assert event["payload"]["error_fingerprint"] is None
    assert event["payload"]["job_id_hash"]

    raw_line = lines[0]
    assert "raw-job-id-123" not in raw_line
    assert "Sensitive prompt-like job name" not in raw_line
    assert "hello from watchdog" not in raw_line


def test_no_agent_cron_failure_emits_error_event(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    scripts_dir = hermes_home / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / "fail.sh"
    script.write_text("echo 'boom' >&2\nexit 7\n", encoding="utf-8")
    script.chmod(0o700)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(scheduler, "_hermes_home", hermes_home)
    monkeypatch.setattr(scheduler, "_SCRIPT_TIMEOUT", 5)

    success, _doc, _final_response, error = scheduler.run_job(
        {
            "id": "failing-job-id",
            "no_agent": True,
            "script": "fail.sh",
            "schedule": "every 1h",
        }
    )

    assert success is False
    assert error

    event_file = next((hermes_home / "ops" / "events").glob("*.jsonl"))
    event = json.loads(event_file.read_text(encoding="utf-8").strip())

    assert event["event_type"] == "cron_run"
    assert event["status"] == "error"
    assert event["payload"]["mode"] == "no_agent"
    assert event["payload"]["error_fingerprint"]
    assert "boom" not in json.dumps(event, ensure_ascii=False)
