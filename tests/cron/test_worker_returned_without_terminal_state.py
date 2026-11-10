"""A restart-safe worker must never end a fire without a durable outcome.

``_run_external_worker_payload`` returns falsy when ``run_one_job`` stops without
recording its own terminal state (an interrupted fire). Before this guard the
attempt stayed ``claimed``/``running``: the parent could only log "whether side
effects ran is unknown", and a later restart filed it under a dead-owner sweep it
never caused. The worker also deleted its own stderr capture on the way out, so the
only diagnostic for *why* the run stopped was destroyed.
"""

import json


def test_falsy_run_settles_the_attempt_and_keeps_stderr(tmp_path, monkeypatch):
    from cron import executions, scheduler

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", home / "cron" / "executions.db")
    monkeypatch.setattr(scheduler, "load_config_readonly", dict)

    record = executions.create_execution("job-interrupted", source="direct")
    assert executions.mark_execution_handoff_pending(record["id"]) is not None

    payload_path = tmp_path / "payload.json"
    payload_path.write_text(
        json.dumps(
            {
                "job": {
                    "id": "job-interrupted",
                    "execution_id": record["id"],
                    "prompt": "Reply OK",
                    "deliver": "local",
                },
                "profile_home": str(home),
                "multiplex_active": False,
            }
        ),
        encoding="utf-8",
    )
    ack_path = tmp_path / "external-workers" / "worker.ack.json"
    stderr_path = ack_path.with_suffix(".stderr")
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.write_text("interrupted mid-flight\n", encoding="utf-8")

    monkeypatch.setattr(scheduler, "run_one_job", lambda *args, **kwargs: False)

    assert scheduler._run_external_worker_payload(payload_path, ack_path) is False

    row = executions.get_execution(record["id"])
    assert row["status"] == "unknown", row
    assert "without a durable terminal state" in (row["error"] or ""), row
    assert row["finished_at"], row
    # The diagnostic survives a run that recorded nothing.
    assert stderr_path.exists(), "worker deleted the only explanation for the stop"
