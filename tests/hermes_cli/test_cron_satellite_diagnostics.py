from datetime import datetime, timedelta, timezone
import json
import os
import time

import pytest


@pytest.fixture
def profile(tmp_path, monkeypatch):
    from cron import jobs

    root = tmp_path / "home"
    home = root / "profiles" / "probe"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "probe")
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr("hermes_constants.get_default_hermes_root", lambda: root)
    monkeypatch.setattr(jobs, "CRON_DIR", home / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", home / "cron/jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", home / "cron/output")
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [])
    # Model the default gateway's identity, not merely a live pytest PID. The satellite has no lock.
    monkeypatch.setattr(
        "gateway.status.is_gateway_runtime_lock_active",
        lambda lock_path=None: lock_path == root / "gateway.lock",
    )
    monkeypatch.setattr("gateway.status._read_process_cmdline", lambda pid: "hermes gateway run")
    root.joinpath("gateway.pid").write_text(json.dumps({"pid": os.getpid()}))
    root.joinpath("config.yaml").write_text("gateway:\n  multiplex_profiles: true\n")
    return root


@pytest.mark.parametrize("mode", ["missing", "fresh", "stale", "disabled", "excluded", "local", "external", "unrelated_pid"])
def test_status_preserves_profile_health_contract(profile, capsys, monkeypatch, mode):
    from cron import jobs
    from hermes_cli import cron

    if mode in {"fresh", "stale", "local"}:
        jobs.record_ticker_heartbeat(success=True)
    if mode == "stale":
        (jobs.CRON_DIR / "ticker_heartbeat").write_text(str(time.time() - 3600))
    if mode == "disabled":
        profile.joinpath("config.yaml").write_text("gateway:\n  multiplex_profiles: false\n")
    if mode == "excluded":
        profile.joinpath("gateway_state.json").write_text(json.dumps({"served_profiles": ["other"]}))
    if mode == "local":
        monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda: [os.getpid()])
    if mode == "external":
        monkeypatch.setattr(cron, "_active_cron_provider_name", lambda: "managed-test")
    if mode == "unrelated_pid":
        monkeypatch.setattr("gateway.status._read_process_cmdline", lambda pid: "python -m pytest")

    cron.cron_status()
    output = capsys.readouterr().out
    assert ("Scheduler host: default-profile multiplexer" in output) == (mode in {"missing", "fresh", "stale"})
    assert ("will fire automatically" in output) == (mode in {"fresh", "local"})
    if mode in {"missing", "stale"}:
        assert "hermes --profile default gateway restart" in output
    if mode == "missing":
        assert "has not reported a heartbeat" in output
    if mode == "stale":
        assert "STALLED" in output
    if mode in {"disabled", "excluded"}:
        assert "24/7" not in output
        assert "hermes gateway install" in output
        assert "sudo hermes gateway install --system" in output
        assert "hermes gateway run" in output
        assert "hermes --profile default gateway restart" in output
    if mode == "external":
        assert "managed scheduler" in output
        assert "STALLED" not in output


@pytest.mark.parametrize("dispatch", ["catch_up", "late", "forward_error"])
def test_doctor_reports_persisted_dispatch_health(profile, capsys, dispatch):
    from cron import jobs
    from hermes_cli.cron import cron_doctor

    job = jobs.create_job(prompt="probe", schedule="every 1h")
    if dispatch == "forward_error":
        jobs.note_fire_forward_failure(job["id"], "loopback unavailable")
    else:
        records = jobs.load_jobs()
        delay = timedelta(hours=5) if dispatch == "catch_up" else timedelta(minutes=5)
        records[0]["next_run_at"] = (datetime.now(timezone.utc) - delay).isoformat()
        jobs.save_jobs(records)
        assert len(jobs.get_due_jobs()) == 1
        persisted = jobs.get_job(job["id"])
        assert persisted is not None
        assert persisted["last_dispatch"]["kind"] == dispatch
    assert cron_doctor() == 1
    output = capsys.readouterr().out
    expected = {"catch_up": "catch-up", "late": "last fire was late", "forward_error": "loopback unavailable"}
    assert expected[dispatch] in output
    assert "scheduler was not running" not in output
    if dispatch == "forward_error":
        jobs.mark_job_run(job["id"], success=True)
        assert cron_doctor() == 0
