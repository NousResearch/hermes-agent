from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hermes_cli import control_db as cp
from hermes_cli.control import _dispatch_statutepm_wave, _readiness, _run_statutepm_live_smoke, _run_statutepm_smoke, _sample_payload
from hermes_cli.control_runtime import resolve_control_target
from hermes_cli.control_worker import run_deterministic_dispatch


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


def test_readiness_live_check_treats_expired_bootstrap_leases_as_diagnostic(tmp_path, monkeypatch):
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
    assert result["live_ready"] is True
    assert result["seeded_instance_leases"]["default:bootstrap"]["live"] is False
    assert "bootstrap_command" not in result
    assert not any("bootstrap lease not live" in reason for reason in result["reasons"])


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
    assert result["status"] == "supervised"
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
