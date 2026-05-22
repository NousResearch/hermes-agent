"""Tests for read-only Runtime mirrors, health snapshots, and sandbox cleanup."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _runtime_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def test_dashboard_mirror_is_read_only_and_omits_private_job_body():
    from agent_runtime import dashboard_mirror, db

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime dashboard", public_ref="HP-88", now=1_700_000_000)
        done_job = db.create_job(conn, run_id=run_id, role="scribe", title="Document result", now=1_700_000_020)
        claim = db.claim_next_job(conn, lease_owner="test-daemon", lease_ttl_seconds=60, now=1_700_000_025)
        assert claim is not None and claim.job_id == done_job
        db.complete_job(
            conn,
            done_job,
            summary="documented",
            now=1_700_000_030,
            lease_owner=claim.lease_owner,
            attempt_id=claim.attempt_id,
        )
        ready_job = db.create_job(
            conn,
            run_id=run_id,
            role="explorer",
            title="Read-only discovery",
            body="PRIVATE PROMPT BODY MUST NOT LEAK",
            now=1_700_000_035,
        )
        before = db.doctor_status(conn)

        snapshot = dashboard_mirror.build_dashboard_snapshot(conn, now=1_700_000_040)
        after = db.doctor_status(conn)

    assert before == after
    assert snapshot["success"] is True
    assert snapshot["mirror_only"] is True
    assert snapshot["source"] == "runtime_sqlite"
    assert snapshot["counts"]["jobs_by_status"]["ready"] == 1
    assert snapshot["counts"]["jobs_by_status"]["succeeded"] == 1
    assert snapshot["runs"][0]["id"] == run_id
    assert snapshot["runs"][0]["progress"]["source"] == "runtime_job_status_counts"
    assert any(card["id"] == ready_job and card["lane"] == "ready" for card in snapshot["cards"])
    assert "PRIVATE PROMPT BODY MUST NOT LEAK" not in json.dumps(snapshot, ensure_ascii=False)
    assert all(card.get("mirror_only") is True for card in snapshot["cards"])


def test_dashboard_mirror_redacts_secret_like_values_from_display_fields():
    from agent_runtime import dashboard_mirror, db

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(
            conn,
            title="Runtime api_key=SUPERSECRET OPENAI_API_KEY=ENVSECRET title",
            objective="password: hunter2 MY_PASSWORD=ENVHUNTER AWS_SECRET_ACCESS_KEY=AWSFAKE should not leak",
            owner_source="ACCESS_TOKEN=owner-secret GITHUB_TOKEN=gh-secret DATABASE_URL=dbfake",
            public_ref="HP-88",
            now=1_700_000_000,
        )
        db.create_job(conn, run_id=run_id, role="explorer", title="Check sk-abcdefghijklmnop", now=1_700_000_010)
        snapshot = dashboard_mirror.build_dashboard_snapshot(conn, now=1_700_000_020)

    encoded = json.dumps(snapshot, ensure_ascii=False)
    assert "SUPERSECRET" not in encoded
    assert "hunter2" not in encoded
    assert "owner-secret" not in encoded
    assert "ENVSECRET" not in encoded
    assert "ENVHUNTER" not in encoded
    assert "gh-secret" not in encoded
    assert "AWSFAKE" not in encoded
    assert "dbfake" not in encoded
    assert "sk-abcdefghijklmnop" not in encoded
    assert "[REDACTED]" in encoded


def test_health_snapshot_redacts_secret_like_lease_owner():
    from agent_runtime import db, observability

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime health", now=1_700_000_000)
        db.create_job(conn, run_id=run_id, role="explorer", title="Stale lease", now=1_700_000_010)
        claim = db.claim_next_job(conn, lease_owner="token=LEASESECRET OPENAI_API_KEY=LEASEENVSECRET", lease_ttl_seconds=60, now=1_700_000_020)
        assert claim is not None
        snapshot = observability.build_health_snapshot(
            conn,
            service_status={"ActiveState": "active", "SubState": "running", "NRestarts": "0"},
            now=1_700_000_200,
            stale_lease_seconds=30,
        )

    encoded = json.dumps(snapshot, ensure_ascii=False)
    assert "LEASESECRET" not in encoded
    assert "LEASEENVSECRET" not in encoded
    assert "token=[REDACTED]" in encoded


def test_probe_runtime_service_supplies_root_user_bus_when_env_missing(monkeypatch):
    from agent_runtime import observability

    calls = []

    class Result:
        returncode = 0
        stdout = "ActiveState=active\nSubState=running\nNRestarts=0\n"
        stderr = ""

    def fake_run(argv, *, capture_output, text, check, timeout, env=None):
        calls.append({"argv": argv, "env": dict(env or {})})
        return Result()

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(os, "getuid", lambda: 0)
    monkeypatch.setattr(Path, "is_dir", lambda self: str(self) == "/run/user/0")
    monkeypatch.setattr(observability.subprocess, "run", fake_run)

    status = observability.probe_runtime_service()

    assert status["ActiveState"] == "active"
    assert calls[0]["env"]["XDG_RUNTIME_DIR"] == "/run/user/0"


def test_health_snapshot_alerts_on_inactive_service_and_stale_lease_without_mutating_db():
    from agent_runtime import db, observability

    db.init_db()
    with db.connect() as conn:
        run_id = db.create_run(conn, title="Runtime health", now=1_700_000_000)
        job_id = db.create_job(conn, run_id=run_id, role="explorer", title="Stale lease", now=1_700_000_010)
        claim = db.claim_next_job(conn, lease_owner="daemon", lease_ttl_seconds=60, now=1_700_000_020)
        assert claim is not None and claim.job_id == job_id
        before = db.get_job(conn, job_id).to_dict()

        snapshot = observability.build_health_snapshot(
            conn,
            service_status={"ActiveState": "inactive", "SubState": "dead", "NRestarts": "2"},
            now=1_700_000_200,
            stale_lease_seconds=30,
        )
        after = db.get_job(conn, job_id).to_dict()

    assert before == after
    assert snapshot["status"] == "critical"
    codes = {alert["code"] for alert in snapshot["alerts"]}
    assert "runtime_service_not_active" in codes
    assert "runtime_job_lease_expired" in codes
    assert "runtime_service_restarted" in codes
    assert snapshot["service"]["active_state"] == "inactive"


def test_sandbox_cleanup_policy_is_dry_run_by_default_and_rejects_unsafe_parent(tmp_path):
    from agent_runtime import cleanup

    parent = tmp_path / "tmp"
    parent.mkdir()
    old = parent / "hermes-agent-runtime-workers-job_old-att_old-abc"
    old.mkdir()
    (old / "context.json").write_text("{}", encoding="utf-8")
    young = parent / "hermes-agent-runtime-workers-job_young-att_young-def"
    young.mkdir()
    other = parent / "unrelated"
    other.mkdir()
    old_time = 1_700_000_000 - 10_000
    young_time = 1_700_000_000 - 10
    os.utime(old, (old_time, old_time))
    os.utime(young, (young_time, young_time))

    dry = cleanup.cleanup_worker_sandboxes(parent=parent, max_age_seconds=3600, now=1_700_000_000, execute=False)

    assert dry["executed"] is False
    assert dry["candidates"] == 1
    assert dry["removed"] == []
    assert old.exists()
    assert young.exists()
    assert other.exists()

    executed = cleanup.cleanup_worker_sandboxes(parent=parent, max_age_seconds=3600, now=1_700_000_000, execute=True)

    assert executed["executed"] is True
    assert executed["candidates"] == 1
    assert len(executed["removed"]) == 1
    assert not old.exists()
    assert young.exists()
    assert other.exists()

    with pytest.raises(ValueError, match="unsafe"):
        cleanup.cleanup_worker_sandboxes(parent=Path("/"), execute=False)


def test_sandbox_cleanup_rejects_parent_with_symlink_component(tmp_path):
    from agent_runtime import cleanup

    real = tmp_path / "real"
    real.mkdir()
    nested = real / "nested"
    nested.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        cleanup.cleanup_worker_sandboxes(parent=link / "nested", execute=False)
