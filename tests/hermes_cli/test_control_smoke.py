from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import control_db as cp
from hermes_cli.control import _dispatch_statutepm_wave, _readiness, _run_statutepm_live_smoke, _run_statutepm_smoke, _sample_payload, register_subparser
from hermes_cli.control_runtime import resolve_control_target
from hermes_cli.control_worker import AGENT_WORKER_TERMINATE_GRACE_S, run_deterministic_dispatch
from hermes_cli.statutepm_flow import PM_CHILD_TIMEOUT_GRACE_S, StatutePMFlow


def test_control_smoke_statutepm_isolated_root(tmp_path):
    result = _run_statutepm_smoke(tmp_path / ".hermes", (tmp_path / ".hermes" / "control-plane" / "control.db").resolve())
    assert result["ok"] is True
    assert result["spawned"]


def test_readiness_temp_root_reports_implementation_ready(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "statute-worker")
    monkeypatch.setattr("hermes_cli.control.worker_spawnability_status", lambda *_args, **_kwargs: {"status": "dry_run_ok", "command": [], "returncode": 0, "stderr": ""})
    monkeypatch.setattr("hermes_cli.control.help_parse_status", lambda *_args, **_kwargs: {"ok": True, "returncode": 0, "stderr": "", "command": []})
    target = resolve_control_target(root=tmp_path / ".hermes")
    result = _readiness(type("Args", (), {"live_check": False})(), target)
    assert result["implementation_ready"] is True
    assert result["live_ready"] is False
    assert result["profile_mapping"] == {"statutepm": "nj-statutes-pm"}
    assert result["agent_worker_ready"] is True


def test_live_smoke_helper_verifies_temp_root_without_real_subprocess(tmp_path):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
    finally:
        conn.close()
    target = resolve_control_target(root=root)

    def fake_spawn(child_id, payload, child_root, parent_id):
        run_deterministic_dispatch(root=child_root, profile_id="statute-worker", instance_id="statute-worker:smoke", dispatch_id=child_id)
        return 7

    result = _run_statutepm_live_smoke(
        target,
        smoke_tag="unit-smoke",
        idempotency_key="unit-smoke-v1",
        deterministic=True,
        spawn_child=fake_spawn,
    )
    assert result["ok"] is True
    assert result["verification"]["child_result_exists"] is True
    assert result["verification"]["artifacts_contained"] is True
    assert result["verification"]["artifact_files_exist"] is True


def test_readiness_live_check_reports_lease_health_from_isolated_home(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in {"default", "nj-statutes-pm", "statute-worker"})
    monkeypatch.setattr("hermes_cli.control.worker_spawnability_status", lambda *_args, **_kwargs: {"status": "dry_run_ok", "command": [], "returncode": 0, "stderr": ""})
    monkeypatch.setattr("hermes_cli.control.help_parse_status", lambda *_args, **_kwargs: {"ok": True, "returncode": 0, "stderr": "", "command": []})
    target = resolve_control_target(live=True)
    result = _readiness(type("Args", (), {"live_check": True})(), target)
    assert result["live_ready"] is True
    assert result["deterministic_operational_ready"] is True
    assert result["cutover_state"] in {"already_control_db", "safe_to_cutover_deterministic"}
    assert result["seeded_instance_leases"]["default:bootstrap"]["live"] is True
    assert result["runtime_profiles"]["nj-statutes-pm"] == "present"


def test_readiness_live_check_treats_missing_live_admin_as_not_ready(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        cp.mark_instance_offline(conn, "default:bootstrap")
        cp.mark_instance_offline(conn, "statutepm:bootstrap")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in {"default", "nj-statutes-pm", "statute-worker"})
    monkeypatch.setattr("hermes_cli.control.worker_spawnability_status", lambda *_args, **_kwargs: {"status": "dry_run_ok", "command": [], "returncode": 0, "stderr": ""})
    monkeypatch.setattr("hermes_cli.control.help_parse_status", lambda *_args, **_kwargs: {"ok": True, "returncode": 0, "stderr": "", "command": []})
    result = _readiness(type("Args", (), {"live_check": True})(), resolve_control_target(live=True))
    assert result["live_ready"] is False
    assert result["live_admin_available"] is False
    assert result["seeded_instance_leases"]["default:bootstrap"]["live"] is False
    assert "bootstrap_command" not in result
    assert "no live admin control-plane instance" in result["reasons"]


def test_readiness_live_check_does_not_count_null_admin_lease(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        conn.execute("UPDATE cp_profile_instances SET lease_expires_at_ms=NULL WHERE instance_id='default:bootstrap'")
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in {"default", "nj-statutes-pm", "statute-worker"})
    monkeypatch.setattr("hermes_cli.control.worker_spawnability_status", lambda *_args, **_kwargs: {"status": "dry_run_ok", "command": [], "returncode": 0, "stderr": ""})
    monkeypatch.setattr("hermes_cli.control.help_parse_status", lambda *_args, **_kwargs: {"ok": True, "returncode": 0, "stderr": "", "command": []})
    result = _readiness(type("Args", (), {"live_check": True})(), resolve_control_target(live=True))
    assert result["live_admin_available"] is False
    assert result["live_ready"] is False
    assert "no live admin control-plane instance" in result["reasons"]


def _wave_args(payload, key="wave-1", **overrides):
    values = {
        "payload_json": json.dumps(payload),
        "idempotency_key": key,
        "supervise": True,
        "admin_profile": "default",
        "pm_profile_id": "statutepm",
        "pm_runtime_profile": "nj-statutes-pm",
        "worker_profile": "statute-worker",
        "supervisor_instance_id": None,
        "pm_instance_id": None,
        "supervisor_lease_ms": 3_600_000,
        "poll_interval_s": 0,
        "child_timeout_s": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_statutepm_wave_supervises_exact_dispatch_and_offlines_finite_instances(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in {"nj-statutes-pm", "statute-worker"})
    target = resolve_control_target(root=root)
    payload = _sample_payload(repo)
    conn = cp.connect(root=root)
    try:
        boot = cp.bootstrap_statutepm_policies(conn, seed_instances=True)
        old = cp.create_dispatch_from_instance(conn, sender_instance_id=boot["instances"]["default"], receiver_profile="statutepm", payload=payload, idempotency_key="older")
    finally:
        conn.close()

    spawned: list[str] = []

    def fake_spawn(child_id, child_payload, child_root, parent_id):
        spawned.append(child_id)
        run_deterministic_dispatch(root=child_root, profile_id="statute-worker", instance_id="statute-worker:wave", dispatch_id=child_id)
        return 11

    result = _dispatch_statutepm_wave(_wave_args(payload, key="new-wave"), target, spawn_child=fake_spawn)
    assert result["status"] == "completed"
    assert result["parent_status"] == "completed"
    assert result["supervisor_offline"] is True
    assert spawned
    spawned_count = len(spawned)
    repeat = _dispatch_statutepm_wave(_wave_args(payload, key="new-wave"), target, spawn_child=fake_spawn)
    assert repeat["parent_dispatch_id"] == result["parent_dispatch_id"]
    assert repeat["parent_status"] == "completed"
    assert repeat["supervision"]["already_terminal"] is True
    assert len(spawned) == spawned_count

    conn = cp.connect(root=root)
    try:
        rows = {r["instance_id"]: dict(r) for r in conn.execute("SELECT * FROM cp_profile_instances").fetchall()}
        assert rows[result["supervisor_instance_id"]]["status"] == "offline"
        assert rows[result["pm_instance_id"]]["status"] == "offline"
        assert conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (old,)).fetchone()["status"] == "pending"
        assert conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (result["parent_dispatch_id"],)).fetchone()["status"] == "completed"
    finally:
        conn.close()


def test_statutepm_wave_supervised_status_mirrors_action_required_child(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in {"nj-statutes-pm", "statute-worker"})
    target = resolve_control_target(root=root)
    payload = _sample_payload(repo)
    action_result = {
        "schema": "control_result_v1",
        "status": "action_required",
        "summary": "child needs CodeRabbit auth",
        "artifacts": [],
        "tests": [],
        "blockers": [{"kind": "auth", "message": "CodeRabbit auth required"}],
    }

    def fake_spawn(child_id, child_payload, child_root, parent_id):
        conn = cp.connect(root=child_root)
        try:
            cp.register_instance(conn, "statute-worker", instance_id="statute-worker:wave-action")
            ok, epoch = cp.claim_dispatch_by_id(conn, dispatch_id=child_id, instance_id="statute-worker:wave-action")
            assert ok and epoch == 1
            cp.advance_dispatch(conn, child_id, instance_id="statute-worker:wave-action", lease_epoch=epoch, status="running")
            cp.record_result(conn, dispatch_id=child_id, instance_id="statute-worker:wave-action", lease_epoch=epoch, result=action_result)
            cp.advance_dispatch(conn, child_id, instance_id="statute-worker:wave-action", lease_epoch=epoch, status="blocked", last_error="child needs CodeRabbit auth")
        finally:
            conn.close()
        return 12

    result = _dispatch_statutepm_wave(_wave_args(payload, key="action-wave"), target, spawn_child=fake_spawn)
    assert result["status"] == "action_required"
    assert result["parent_status"] == "blocked"
    assert result["supervision"]["status"] == "action_required"
    repeat = _dispatch_statutepm_wave(_wave_args(payload, key="action-wave"), target, spawn_child=fake_spawn)
    assert repeat["parent_dispatch_id"] == result["parent_dispatch_id"]
    assert repeat["status"] == "action_required"
    assert repeat["parent_status"] == "blocked"
    assert repeat["supervision"]["already_terminal"] is True

    conn = cp.connect(root=root)
    try:
        parent = conn.execute("SELECT status FROM cp_dispatches WHERE dispatch_id=?", (result["parent_dispatch_id"],)).fetchone()
        assert parent is not None
        assert parent["status"] == "blocked"
        latest_row = cp.get_latest_dispatch_result(conn, result["parent_dispatch_id"])
        assert latest_row is not None
        latest = latest_row["result"]
        assert latest["status"] == "action_required"
        assert latest["summary"] == "child needs CodeRabbit auth"
    finally:
        conn.close()


def test_statutepm_wave_rejects_bootstrap_instances_before_dispatch(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in {"nj-statutes-pm", "statute-worker"})
    try:
        _dispatch_statutepm_wave(_wave_args(_sample_payload(repo), supervisor_instance_id="default:bootstrap"), resolve_control_target(root=root))
    except SystemExit as exc:
        assert "seeded bootstrap" in str(exc)
    else:
        raise AssertionError("expected bootstrap instance rejection")
    conn = cp.connect(root=root)
    try:
        assert conn.execute("SELECT COUNT(*) FROM cp_dispatches").fetchone()[0] == 0
    finally:
        conn.close()


def _control_parser():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    register_subparser(subparsers)
    return parser


def test_statutepm_cli_child_timeout_defaults_match_agent_worker_default():
    parser = _control_parser()

    worker = parser.parse_args(["control", "worker", "run", "disp_child", "--profile-id", "statute-worker", "--instance-id", "statute-worker:test", "--handler", "agent"])
    pm = parser.parse_args(["control", "pm", "run", "--once"])
    wave = parser.parse_args(["control", "wave", "dispatch-statutepm", "--payload-json", "{}", "--idempotency-key", "key"])

    assert worker.soft_timeout_s == 600.0
    assert worker.hard_timeout_s == 3000.0
    assert worker.timeout_s is None
    assert pm.child_soft_timeout_s == worker.soft_timeout_s
    assert pm.child_hard_timeout_s == worker.hard_timeout_s
    assert wave.child_soft_timeout_s == worker.soft_timeout_s
    assert wave.child_hard_timeout_s == worker.hard_timeout_s

    alias = parser.parse_args(["control", "worker", "run", "disp_child", "--profile-id", "statute-worker", "--instance-id", "statute-worker:test", "--handler", "agent", "--timeout-s", "1200"])
    explicit = parser.parse_args(["control", "worker", "run", "disp_child", "--profile-id", "statute-worker", "--instance-id", "statute-worker:test", "--handler", "agent", "--hard-timeout-s", "1200"])
    assert alias.timeout_s == 1200.0
    assert alias.hard_timeout_s_explicit is False
    assert explicit.hard_timeout_s == 1200.0
    assert explicit.hard_timeout_s_explicit is True


def test_statutepm_parent_deadline_and_lease_include_child_hard_timeout_and_graces(tmp_path):
    flow = StatutePMFlow(root=tmp_path / ".hermes", pm_instance_id="statutepm:test", poll_interval_s=2.0, child_soft_timeout_s=1.0, child_hard_timeout_s=10.0)

    assert flow._child_deadline_s(now_s=100.0) == 100.0 + 10.0 + AGENT_WORKER_TERMINATE_GRACE_S + PM_CHILD_TIMEOUT_GRACE_S
    assert flow._parent_lease_ms() > int((10.0 + AGENT_WORKER_TERMINATE_GRACE_S + PM_CHILD_TIMEOUT_GRACE_S) * 1000)
    assert flow._parent_lease_ms() >= int((10.0 + AGENT_WORKER_TERMINATE_GRACE_S + PM_CHILD_TIMEOUT_GRACE_S + 2.0 + 30.0) * 1000)
