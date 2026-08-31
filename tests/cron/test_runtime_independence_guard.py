"""Fail-closed cron runtime-independence guard across core entry points."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from hermes_time import now as hermes_now


@pytest.fixture
def guarded_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "cron").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "config.yaml").write_text(
        "cron:\n  provider: builtin\n",
        encoding="utf-8",
    )
    return tmp_path


def write_guard(home, *, enforce, jobs=None, epoch="epoch-1", fence=None):
    payload = {
        "schema_version": 1,
        "enforce": enforce,
        "epoch": epoch,
        "jobs": jobs or {},
    }
    if fence is not None:
        payload["fence"] = fence
    (home / "cron" / "runtime-independence.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (home / "cron" / "runtime-independence.json").chmod(0o600)


def approve_job(home, job, *, epoch="epoch-1"):
    from cron.runtime_independence import compute_execution_contract_digest

    digest = compute_execution_contract_digest(job, home)
    write_guard(
        home,
        enforce=True,
        epoch=epoch,
        jobs={
            job["id"]: {
                "status": "approved",
                "epoch": epoch,
                "execution_digest": digest,
                "dependency_status": "independent",
            }
        },
    )


def test_missing_guard_is_backward_compatible_until_enforcement(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="safe", schedule="every 5m", name="default-off")

    assert claim_job_for_fire(job["id"]) is True


def test_enable_resume_and_trigger_share_fail_closed_guard(guarded_home):
    from cron.jobs import create_job, pause_job, resume_job, trigger_job, update_job

    job = create_job(prompt="safe", schedule="every 5m", name="blocked")
    pause_job(job["id"])
    write_guard(guarded_home, enforce=True)

    with pytest.raises(ValueError, match="runtime independence"):
        resume_job(job["id"])
    with pytest.raises(ValueError, match="runtime independence"):
        trigger_job(job["id"])
    with pytest.raises(ValueError, match="runtime independence"):
        update_job(job["id"], {"enabled": True, "state": "scheduled"})


def test_approved_digest_allows_claim_and_due_scan(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job, get_due_jobs, update_job

    job = create_job(prompt="safe", schedule="every 5m", name="approved")
    due_at = (hermes_now() - timedelta(minutes=1)).isoformat()
    update_job(job["id"], {"next_run_at": due_at})
    current = {**job, "next_run_at": due_at}
    approve_job(guarded_home, current)

    due = get_due_jobs()
    assert [row["id"] for row in due] == [job["id"]]
    assert claim_job_for_fire(job["id"], force=True) is True


def test_blocked_due_scan_does_not_persist_oneshot_run_claim(guarded_home):
    from cron.jobs import create_job, get_due_jobs, list_jobs, update_job

    job = create_job(
        prompt="safe",
        schedule=(hermes_now() + timedelta(hours=1)).isoformat(),
        name="blocked-once",
    )
    update_job(
        job["id"],
        {"next_run_at": (hermes_now() - timedelta(minutes=1)).isoformat()},
    )
    write_guard(guarded_home, enforce=True)

    assert get_due_jobs() == []
    stored = next(row for row in list_jobs(include_disabled=True) if row["id"] == job["id"])
    assert "run_claim" not in stored


def test_wrapper_drift_blocks_next_claim(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    script = guarded_home / "scripts" / "task.py"
    script.write_text('print("v1")\n', encoding="utf-8")
    job = create_job(
        prompt="",
        schedule="every 5m",
        name="scripted",
        script="task.py",
        no_agent=True,
    )
    approve_job(guarded_home, job)
    script.write_text('print("v2")\n', encoding="utf-8")

    assert claim_job_for_fire(job["id"], force=True) is False


def test_tool_or_config_drift_blocks_next_claim(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="use queue tool", schedule="every 5m", name="agent")
    approve_job(guarded_home, job)
    (guarded_home / "config.yaml").write_text(
        "cron:\n  provider: builtin\n"
        "mcp_servers:\n"
        "  queue:\n"
        "    command: /opt/new-queue-adapter.py\n",
        encoding="utf-8",
    )

    assert claim_job_for_fire(job["id"], force=True) is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "new-model"),
        ("model_snapshot", "snap-model"),
        ("provider_snapshot", "snap-provider"),
        ("context_from", ["upstream-job"]),
        ("reasoning_effort", "high"),
    ],
)
def test_execution_digest_covers_runtime_job_overrides(guarded_home, field, value):
    from cron.runtime_independence import compute_execution_contract_digest

    job = {"id": "job-1", "prompt": "safe"}
    baseline = compute_execution_contract_digest(job, guarded_home)

    assert compute_execution_contract_digest({**job, field: value}, guarded_home) != baseline


def test_execution_digest_tracks_repeat_policy_not_completion_progress(guarded_home):
    from cron.runtime_independence import compute_execution_contract_digest

    job = {"id": "job-1", "prompt": "safe", "repeat": {"times": 3, "completed": 0}}
    baseline = compute_execution_contract_digest(job, guarded_home)

    assert (
        compute_execution_contract_digest(
            {**job, "repeat": {"times": 3, "completed": 1}}, guarded_home
        )
        == baseline
    )
    assert (
        compute_execution_contract_digest(
            {**job, "repeat": {"times": 4, "completed": 0}}, guarded_home
        )
        != baseline
    )


def test_non_boolean_enforcement_value_fails_closed(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="safe", schedule="every 5m", name="invalid-enforce")
    write_guard(guarded_home, enforce="true")

    assert claim_job_for_fire(job["id"], force=True) is False


def test_external_scheduler_provider_is_always_held(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="safe", schedule="every 5m", name="external")
    (guarded_home / "config.yaml").write_text(
        "cron:\n  provider: chronos\n",
        encoding="utf-8",
    )
    approve_job(guarded_home, job)

    assert claim_job_for_fire(job["id"], force=True) is False


def test_malformed_guard_blocks_when_file_exists(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="safe", schedule="every 5m", name="malformed")
    (guarded_home / "cron" / "runtime-independence.json").write_text(
        "{not-json",
        encoding="utf-8",
    )
    (guarded_home / "cron" / "runtime-independence.json").chmod(0o600)

    assert claim_job_for_fire(job["id"], force=True) is False


def test_direct_scheduler_execution_stops_before_script(guarded_home, monkeypatch):
    from cron import scheduler
    from cron.jobs import create_job

    script = guarded_home / "scripts" / "task.py"
    script.write_text('print("must-not-run")\n', encoding="utf-8")
    job = create_job(
        prompt="",
        schedule="every 5m",
        name="direct",
        script="task.py",
        no_agent=True,
    )
    write_guard(guarded_home, enforce=True)
    calls = []
    monkeypatch.setattr(
        scheduler,
        "_run_job_script_with_claim_heartbeat",
        lambda *args, **kwargs: calls.append(args) or (True, "unexpected"),
    )

    success, _output, _response, error = scheduler.run_job(job)

    assert not success
    assert "runtime independence" in str(error)
    assert calls == []


def test_public_or_symlinked_attestation_fails_closed(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="safe", schedule="every 5m", name="permissions")
    write_guard(guarded_home, enforce=True)
    attestation = guarded_home / "cron" / "runtime-independence.json"
    attestation.chmod(0o644)
    assert claim_job_for_fire(job["id"], force=True) is False

    attestation.unlink()
    target = guarded_home / "outside-attestation.json"
    target.write_text('{"schema_version": 1, "enforce": false}', encoding="utf-8")
    target.chmod(0o600)
    attestation.symlink_to(target)
    assert claim_job_for_fire(job["id"], force=True) is False


def test_active_and_expired_fences_require_explicit_release(guarded_home):
    from cron.jobs import claim_job_for_fire, create_job
    from cron.runtime_independence import compute_execution_contract_digest

    job = create_job(prompt="safe", schedule="every 5m", name="fenced")
    entry = {
        job["id"]: {
            "status": "approved",
            "epoch": "epoch-1",
            "execution_digest": compute_execution_contract_digest(job, guarded_home),
            "dependency_status": "independent",
        }
    }
    write_guard(
        guarded_home,
        enforce=True,
        jobs=entry,
        fence={
            "state": "active",
            "epoch": "epoch-1",
            "expires_at": "2999-01-01T00:00:00Z",
        },
    )
    assert claim_job_for_fire(job["id"], force=True) is False

    write_guard(
        guarded_home,
        enforce=True,
        jobs=entry,
        fence={
            "state": "active",
            "epoch": "epoch-1",
            "expires_at": "2000-01-01T00:00:00Z",
        },
    )
    assert claim_job_for_fire(job["id"], force=True) is False

    write_guard(
        guarded_home,
        enforce=True,
        jobs=entry,
        fence={
            "state": "released",
            "epoch": "epoch-1",
            "released_at": "2026-08-31T00:00:00Z",
        },
    )
    assert claim_job_for_fire(job["id"], force=True) is True
