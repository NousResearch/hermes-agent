from __future__ import annotations

import pytest

from hermes_cli import control_db as cp
from hermes_cli.control_runtime import (
    runtime_profile_for_control_profile,
    validate_pm_runtime_mapping,
    profile_id,
    register_runtime_instance,
    resolve_control_target,
    worker_spawnability_status,
)


def test_control_target_precedence_and_live_guard(monkeypatch, tmp_path):
    env_root = tmp_path / "env"
    arg_root = tmp_path / "arg"
    monkeypatch.setenv("HERMES_CONTROL_ROOT", str(env_root))
    assert resolve_control_target().root == env_root
    assert resolve_control_target(root=arg_root).root == arg_root
    assert resolve_control_target(live=True).db_path == cp.control_db_path().resolve()


def test_profile_id_prefers_control_env(monkeypatch):
    monkeypatch.setenv("HERMES_PROFILE", "legacy")
    monkeypatch.setenv("HERMES_PROFILE_NAME", "name")
    monkeypatch.setenv("HERMES_PROFILE_ID", "control")
    assert profile_id() == "control"


def test_runtime_registration_blocks_ambient_pm_admin(tmp_path):
    root = tmp_path / ".hermes"
    conn = cp.connect(root=root)
    try:
        cp.bootstrap_statutepm_policies(conn)
        with pytest.raises(PermissionError):
            register_runtime_instance(conn, profile="statutepm", instance="statutepm:ambient")
        seeded = cp.bootstrap_statutepm_policies(conn, seed_instances=True)["instances"]["statutepm"]
        assert register_runtime_instance(conn, profile="statutepm", instance=seeded) == seeded
        assert register_runtime_instance(conn, profile="statute-worker", instance="statute-worker:1") == "statute-worker:1"
    finally:
        conn.close()


def test_statutepm_runtime_mapping_requires_mapped_profile(monkeypatch):
    monkeypatch.setattr("hermes_cli.control_runtime.runtime_profile_presence", lambda name: "present" if name == "nj-statutes-pm" else "missing")
    assert runtime_profile_for_control_profile("statutepm") == "nj-statutes-pm"
    assert validate_pm_runtime_mapping("statutepm") == "nj-statutes-pm"
    with pytest.raises(PermissionError):
        validate_pm_runtime_mapping("statutepm", "missing-runtime")


def test_worker_spawnability_uses_dry_run_help(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/bin/hermes")

    def fake_help(args, **kwargs):
        assert args[:3] == ["/bin/hermes", "-p", "statute-worker"]
        assert args[-1] == "--help"
        return {"ok": True, "returncode": 0, "stderr": ""}

    monkeypatch.setattr("hermes_cli.control_runtime.help_parse_status", fake_help)
    assert worker_spawnability_status("statute-worker")["status"] == "dry_run_ok"
