from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli import control_db as cp
from hermes_cli.control_runtime import resolve_control_target


def run_cli(*args: str, check: bool = True, env: dict[str, str] | None = None):
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)
    return subprocess.run([sys.executable, "-m", "hermes_cli.main", "control", *args], text=True, capture_output=True, check=check, env=proc_env)


def _profile_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    (home / "profiles" / "nj-statutes-pm").mkdir(parents=True)
    (home / "profiles" / "statute-worker").mkdir(parents=True)
    return {"HERMES_HOME": str(home)}


def test_control_root_target_resolver_isolated_from_live(tmp_path, monkeypatch):
    root = tmp_path / "root"
    env_root = tmp_path / "env"
    monkeypatch.setenv("HERMES_CONTROL_ROOT", str(env_root))
    assert resolve_control_target().root == env_root
    assert resolve_control_target(root=root).root == root
    assert resolve_control_target(root=root).db_path != cp.control_db_path().resolve()


def test_live_mutation_commands_require_live_or_root(tmp_path, monkeypatch):
    live_root = tmp_path / "live"
    monkeypatch.setenv("HERMES_HOME", str(live_root))
    target = resolve_control_target()
    assert target.live is True
    assert resolve_control_target(root=tmp_path / "isolated").live is False


def test_bootstrap_route_check_and_denied_path(tmp_path):
    root = tmp_path / "cp"
    out = run_cli("bootstrap-statutepm", "--root", str(root), "--seed-instances", "--pm-profile-id", "statutepm").stdout
    data = json.loads(out)
    assert data["instances"]["default"] == "default:bootstrap"
    legacy = json.loads(run_cli("bootstrap-statutepm", "--root", str(root), "--pm-profile", "statutepm").stdout)
    assert legacy["profiles"]["statutepm"]["status"] == "existing"
    assert run_cli("route", "check", "--root", str(root), "--sender", "default", "--receiver", "statutepm", "--kind", "dispatch", "--capability", "dispatch", "--strict").stdout.strip() == "allow"
    assert run_cli("route", "check", "--root", str(root), "--sender", "statute-worker", "--receiver", "default", "--kind", "dispatch", "--capability", "dispatch").stdout.strip() == "deny"


def test_dispatch_and_message_cli_use_sender_instance(tmp_path):
    root = tmp_path / "cp"
    run_cli("bootstrap-statutepm", "--root", str(root), "--seed-instances")
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = {
        "schema": "statute_dispatch_v1",
        "silo": "statute",
        "repo_root": str(repo),
        "allowed_paths": [str(repo)],
        "task_type": "generic",
        "task_permissions": ["read"],
        "parent_dispatch_id": None,
        "instructions": "work",
        "constraints": {"no_live_db_mutation": True, "no_push": True},
    }
    created = json.loads(run_cli("dispatch", "create", "--root", str(root), "--sender-instance-id", "default:bootstrap", "--receiver", "statutepm", "--payload-json", json.dumps(payload)).stdout)
    assert created["dispatch_id"].startswith("disp_")
    msg = json.loads(run_cli("message", "create", "--root", str(root), "--sender-instance-id", "default:bootstrap", "--receiver", "statutepm", "--kind", "instruction", "--body", "go").stdout)
    assert msg["message_id"].startswith("msg_")
    ack = json.loads(run_cli("message", "ack", msg["message_id"], "--root", str(root), "--actor-instance-id", "statutepm:bootstrap", "--reason", "seen").stdout)
    assert ack["status"] == "acknowledged"
    assert ack["changed"] is True
    ack2 = json.loads(run_cli("message", "ack", msg["message_id"], "--root", str(root), "--actor-instance-id", "statutepm:bootstrap").stdout)
    assert ack2["changed"] is False
    refused = run_cli("message", "cancel", msg["message_id"], "--root", str(root), "--actor-instance-id", "statutepm:bootstrap", check=False)
    assert refused.returncode != 0


def test_mode_requires_live_admin_actor_for_control_db(tmp_path):
    root = tmp_path / "cp"
    run_cli("bootstrap-statutepm", "--root", str(root), "--seed-instances")
    assert run_cli("mode", "control_db", "--root", str(root), "--actor-profile", "default", "--actor-instance-id", "default:bootstrap").stdout.strip() == "control_db"


def test_pm_run_once_idle_honors_mapping(tmp_path):
    root = tmp_path / "cp"
    env = _profile_env(tmp_path)
    run_cli("bootstrap-statutepm", "--root", str(root), "--seed-instances", env=env)
    result = json.loads(
        run_cli(
            "pm",
            "run",
            "--root",
            str(root),
            "--pm-profile-id",
            "statutepm",
            "--pm-instance-id",
            "statutepm:bootstrap",
            "--pm-runtime-profile",
            "nj-statutes-pm",
            "--worker-profile",
            "statute-worker",
            "--once",
            env=env,
        ).stdout
    )
    assert result["status"] == "idle"
    assert result["pm_runtime_profile"] == "nj-statutes-pm"


def test_pm_run_refuses_live_target_without_live(tmp_path):
    env = _profile_env(tmp_path)
    result = run_cli("pm", "run", "--once", env=env, check=False)
    assert result.returncode != 0
    assert "refusing to mutate live control DB without --live" in result.stderr


def test_live_smoke_requires_live_before_touching_db(tmp_path):
    env = _profile_env(tmp_path)
    result = run_cli("live-smoke", "statutepm", "--deterministic", "--smoke-tag", "x", "--idempotency-key", "x", env=env, check=False)
    assert result.returncode != 0
    assert "refusing to mutate live control DB without --live" in result.stderr


def test_message_transition_requires_live_flag_for_live_root(tmp_path):
    env = _profile_env(tmp_path)
    result = run_cli("message", "ack", "msg_missing", "--actor-instance-id", "default:bootstrap", env=env, check=False)
    assert result.returncode != 0
    assert "refusing to mutate live control DB without --live" in result.stderr


def test_live_message_admin_transition_renews_expired_bootstrap_admin_once(tmp_path):
    env = _profile_env(tmp_path)
    home = Path(env["HERMES_HOME"])
    run_cli("bootstrap-statutepm", "--live", "--seed-instances", env=env)
    msg = json.loads(
        run_cli(
            "message",
            "create",
            "--live",
            "--sender-instance-id",
            "statutepm:bootstrap",
            "--receiver",
            "default",
            "--kind",
            "action_required",
            "--body",
            "stale administrative action",
            env=env,
        ).stdout
    )
    conn = cp.connect(root=home)
    try:
        conn.execute(
            "UPDATE cp_profile_instances SET heartbeat_at_ms=0, lease_expires_at_ms=1 WHERE instance_id='default:bootstrap'"
        )
        conn.commit()
    finally:
        conn.close()

    result = json.loads(
        run_cli(
            "message",
            "resolve",
            msg["message_id"],
            "--live",
            "--actor-type",
            "admin",
            "--actor-profile",
            "default",
            "--actor-instance-id",
            "default:bootstrap",
            "--reason",
            "stale and superseded",
            env=env,
        ).stdout
    )

    assert result["status"] == "resolved"
    assert result["changed"] is True
    assert result["admin_lease_renewed"] is True


def test_admin_lease_command_renews_existing_live_bootstrap_admin(tmp_path):
    env = _profile_env(tmp_path)
    home = Path(env["HERMES_HOME"])
    run_cli("bootstrap-statutepm", "--live", "--seed-instances", env=env)
    conn = cp.connect(root=home)
    try:
        conn.execute("UPDATE cp_profile_instances SET heartbeat_at_ms=0, lease_expires_at_ms=1 WHERE instance_id='default:bootstrap'")
        conn.commit()
    finally:
        conn.close()

    result = json.loads(
        run_cli(
            "admin",
            "lease",
            "--live",
            "--profile",
            "default",
            "--instance-id",
            "default:bootstrap",
            env=env,
        ).stdout
    )

    assert result["instance_id"] == "default:bootstrap"
    assert result["status"] == "online"
    assert result["lease_expires_at_ms"] > result["heartbeat_at_ms"]


def test_new_control_help_surfaces_parse():
    assert run_cli("pm", "run", "--help").returncode == 0
    assert run_cli("live-smoke", "--help").returncode == 0
    assert run_cli("wave", "dispatch-statutepm", "--help").returncode == 0


def test_v3_cli_status_blocker_supervision_and_runtime_mapping(tmp_path):
    root = tmp_path / "cp"
    run_cli("bootstrap-statutepm", "--root", str(root), "--seed-instances")
    repo = tmp_path / "repo"
    repo.mkdir()
    payload = {
        "schema": "statute_dispatch_v1",
        "silo": "statute",
        "repo_root": str(repo),
        "allowed_paths": [str(repo)],
        "task_type": "generic",
        "task_permissions": ["read"],
        "parent_dispatch_id": None,
        "instructions": "work",
        "constraints": {"no_live_db_mutation": True, "no_push": True},
    }
    created = json.loads(run_cli(
        "dispatch", "create", "--root", str(root), "--sender-instance-id", "statutepm:bootstrap",
        "--receiver", "statute-worker", "--payload-json", json.dumps(payload), "--dispatch-schema", "statute_dispatch_v1",
    ).stdout)
    dispatch_id = created["dispatch_id"]
    run_cli("heartbeat", "--root", str(root), "statute-worker", "--instance-id", "statute-worker:bootstrap")
    claimed = json.loads(run_cli("dispatch", "claim", dispatch_id, "--root", str(root), "--instance-id", "statute-worker:bootstrap").stdout)
    assert claimed["lease_epoch"] == 1
    status = json.loads(run_cli("status", "emit", "--root", str(root), "--instance-id", "statute-worker:bootstrap", "--dispatch-id", dispatch_id, "--status", "running", "--summary", "running").stdout)
    assert status["event_id"].startswith("evt_")
    blocker = json.loads(run_cli("blocker", "open", "--root", str(root), "--dispatch-id", dispatch_id, "--instance-id", "statute-worker:bootstrap", "--severity", "blocked", "--kind", "missing_context", "--summary", "need input", "--response-profile", "default").stdout)
    assert blocker["blocker_id"].startswith("blk_")
    resolved = json.loads(run_cli("blocker", "resolve", blocker["blocker_id"], "--root", str(root), "--resolver-instance-id", "default:bootstrap", "--resolution-json", '{"summary":"ok"}').stdout)
    assert resolved["resolved"] is True
    sup = json.loads(run_cli("supervision", "start", "--root", str(root), "--actor-instance-id", "default:bootstrap", "--scope-json", '{"dry_run":true}').stdout)
    assert sup["run_id"].startswith("sup_")
    finished = json.loads(run_cli("supervision", "finish", sup["run_id"], "--root", str(root), "--status", "completed", "--findings-json", '[{"code":"ok"}]').stdout)
    assert finished["finished"] is True
    mapped = json.loads(run_cli("runtime", "map", "--root", str(root), "statute-worker", "statute-worker").stdout)
    assert mapped["runtime_profile"] == "statute-worker"
    watchdog = json.loads(run_cli("watchdog", "run", "--root", str(root), "--actor-instance-id", "default:bootstrap", "--dry-run").stdout)
    assert watchdog["dry_run"] is True
    assert watchdog["run_id"].startswith("sup_")


def test_watchdog_apply_marks_expired_worker_instances_offline_without_reaping_rows(tmp_path):
    root = tmp_path / "cp"
    run_cli("bootstrap-statutepm", "--root", str(root), "--seed-instances")
    conn = cp.connect(root=root)
    try:
        cp.register_instance(conn, "statute-worker", instance_id="statute-worker:expired", lease_ms=-1)
    finally:
        conn.close()

    watchdog = json.loads(run_cli("watchdog", "run", "--root", str(root), "--actor-instance-id", "default:bootstrap", "--apply").stdout)
    assert watchdog["dry_run"] is False
    cleanup = next(action for action in watchdog["actions"] if action["action"] == "mark_expired_worker_instances_offline")
    assert cleanup["instance_ids"] == ["statute-worker:expired"]

    conn = cp.connect(root=root)
    try:
        row = conn.execute("SELECT status FROM cp_profile_instances WHERE instance_id='statute-worker:expired'").fetchone()
        assert row is not None
        assert row["status"] == "offline"
    finally:
        conn.close()
