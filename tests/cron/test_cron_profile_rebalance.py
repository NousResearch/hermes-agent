import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

for module_name in ("cron_profile_assignment_audit", "cron_profile_rebalance"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / f"{module_name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

rebalance = sys.modules["cron_profile_rebalance"]


def write_jobs(home: Path, jobs):
    jobs_file = home / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)
    jobs_file.write_text(json.dumps({"jobs": jobs, "updated_at": "old"}), encoding="utf-8")


def read_jobs(home: Path):
    return json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]


def make_script(home: Path, name: str, content: str = "print('ok')\n", mode: int | None = None) -> Path:
    path = home / "scripts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if mode is not None:
        path.chmod(mode)
    return path


def test_manifest_marks_script_copies_shadow_jobs_gateways_and_default_keeps(tmp_path):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead_check.py")
    write_jobs(
        default_home,
        [
            {"id": "j1", "name": "crm-lead-check", "script": "lead_check.py", "no_agent": True},
            {"id": "j2", "name": "crm-daily-scan", "prompt": "scan crm"},
            {"id": "j3", "name": "backup-daily", "script": "backup.sh", "no_agent": True},
        ],
    )

    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    by_id = {move.job_id: move for move in manifest.moves}

    assert by_id["j1"].action == "stage"
    assert by_id["j1"].script_copy["relative"] == "lead_check.py"
    assert by_id["j1"].default_action == "disable_after_verification"
    assert by_id["j2"].shadow_required is True
    assert by_id["j2"].default_action == "leave_active_shadow"
    assert by_id["j3"].action == "keep"
    assert manifest.gateway_profiles == ["rva-leads"]


def test_manifest_blocks_missing_skills_unsafe_scripts_and_collisions(tmp_path):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead_check.py")
    write_jobs(
        default_home,
        [
            {"id": "lead", "name": "crm-lead-check", "script": "lead_check.py", "no_agent": True},
            {"id": "tax", "name": "tax-research-archive-lint", "script": "../escape.py", "no_agent": True, "skills": ["tax-skill"]},
            {"id": "sms", "name": "ringcentral-sms-opt-out-watchdog", "script": "lead_check.py", "no_agent": True},
        ],
    )
    target = default_home / "profiles" / "rva-leads"
    write_jobs(target, [{"id": "lead", "name": "unrelated-existing"}])
    (target / "cron" / "output" / "sms").mkdir(parents=True)

    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    by_id = {move.job_id: move for move in manifest.moves}

    assert "id_collision" in by_id["lead"].blockers
    assert "source_script_outside_profile_scripts" in by_id["tax"].blockers
    assert "missing_target_skills" in by_id["tax"].blockers
    assert "output_history_collision" in by_id["sms"].blockers


def test_apply_stage_copies_scripts_preserves_mode_and_leaves_source_active(tmp_path):
    default_home = tmp_path / ".hermes"
    script = make_script(default_home, "lead_check.sh", "#!/usr/bin/env bash\necho ok\n", 0o755)
    write_jobs(default_home, [{"id": "j1", "name": "crm-lead-check", "script": "lead_check.sh", "no_agent": True}])

    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    result = rebalance.apply_stage(manifest, default_home=default_home)
    second_result = rebalance.apply_stage(manifest, default_home=default_home)

    target_script = default_home / "profiles" / "rva-leads" / "scripts" / "lead_check.sh"
    assert target_script.exists()
    assert target_script.read_text(encoding="utf-8") == script.read_text(encoding="utf-8")
    assert target_script.stat().st_mode & stat.S_IXUSR
    assert result["staged_jobs"] == ["j1"]
    assert second_result["staged_jobs"] == []
    target_jobs = read_jobs(default_home / "profiles" / "rva-leads")
    assert len(target_jobs) == 1
    assert target_jobs[0]["enabled"] is False
    assert target_jobs[0]["paused_at"]
    assert target_jobs[0]["next_run_at"] is None
    assert target_jobs[0]["profile_migration"]["source_job_id"] == "j1"
    assert read_jobs(default_home)[0]["enabled"] is True


def test_apply_stage_skips_source_job_removed_after_manifest(tmp_path):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead_check.py")
    write_jobs(default_home, [{"id": "j1", "name": "crm-lead-check", "script": "lead_check.py", "no_agent": True}])
    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    write_jobs(default_home, [])

    result = rebalance.apply_stage(manifest, default_home=default_home)

    assert result["staged_jobs"] == []
    assert result["skipped"] == {"j1": "source_job_missing"}


def test_cutover_enables_verified_script_job_and_removes_source_when_requested(tmp_path):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead_check.py")
    write_jobs(default_home, [{"id": "j1", "name": "crm-lead-check", "script": "lead_check.py", "no_agent": True}])
    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    rebalance.apply_stage(manifest, default_home=default_home)

    result = rebalance.apply_cutover(
        manifest,
        default_home=default_home,
        verified_job_ids={"j1"},
        remove_source=True,
    )

    assert result["cutover"] == ["j1"]
    assert result["backups"]["default"] is not None
    assert result["backups"]["rva-leads"] is not None
    assert read_jobs(default_home) == []
    target_job = read_jobs(default_home / "profiles" / "rva-leads")[0]
    assert target_job["enabled"] is True
    assert target_job["state"] == "scheduled"
    assert target_job["next_run_at"] is None
    assert "paused_at" not in target_job
    assert "cutover_at" in target_job["profile_migration"]


def test_cutover_keeps_agent_shadow_until_accepted(tmp_path):
    default_home = tmp_path / ".hermes"
    write_jobs(default_home, [{"id": "agent", "name": "crm-daily-scan", "prompt": "scan crm"}])
    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    rebalance.apply_stage(manifest, default_home=default_home)

    blocked = rebalance.apply_cutover(manifest, default_home=default_home, verified_job_ids={"agent"})
    assert blocked["cutover"] == []
    assert blocked["skipped"] == {"agent": "agent_shadow_not_accepted"}
    assert read_jobs(default_home)[0]["enabled"] is True

    accepted = rebalance.apply_cutover(
        manifest,
        default_home=default_home,
        verified_job_ids={"agent"},
        accepted_agent_job_ids={"agent"},
    )
    assert accepted["cutover"] == ["agent"]
    assert accepted["backups"]["default"] is not None
    assert read_jobs(default_home)[0]["enabled"] is False
    assert read_jobs(default_home)[0]["paused_at"]
    assert read_jobs(default_home / "profiles" / "rva-leads")[0]["enabled"] is True


def test_stage_creates_only_receiving_profile_dirs(tmp_path):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead.py")
    make_script(default_home, "content.py")
    write_jobs(
        default_home,
        [
            {"id": "lead", "name": "crm-lead-check", "script": "lead.py", "no_agent": True},
            {"id": "content", "name": "content-topic-miner", "script": "content.py", "no_agent": True},
            {"id": "default", "name": "backup-daily", "script": "backup.sh", "no_agent": True},
        ],
    )

    manifest = rebalance.build_manifest(default_home, default_home=default_home)
    rebalance.apply_stage(manifest, default_home=default_home)

    assert (default_home / "profiles" / "rva-leads" / "cron").exists()
    assert (default_home / "profiles" / "rva-profit-pulse" / "cron").exists()
    assert not (default_home / "profiles" / "cpa-tax-researcher").exists()
    assert not (default_home / "profiles" / "rva-firm-ops").exists()
    assert not (default_home / "profiles" / "personal").exists()
    assert not (default_home / "profiles" / "rva-dev").exists()


def test_manifest_blocks_jobs_without_ids(tmp_path):
    default_home = tmp_path / ".hermes"
    write_jobs(default_home, [{"name": "crm-lead-check", "no_agent": True}])

    manifest = rebalance.build_manifest(default_home, default_home=default_home)

    assert manifest.moves[0].action == "blocked"
    assert manifest.moves[0].blockers == ["missing_job_id"]


def test_target_profile_filter_limits_manifest_and_stage(tmp_path):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead.py")
    make_script(default_home, "content.py")
    write_jobs(
        default_home,
        [
            {"id": "lead", "name": "crm-lead-check", "script": "lead.py", "no_agent": True},
            {"id": "content", "name": "content-topic-miner", "script": "content.py", "no_agent": True},
        ],
    )

    manifest = rebalance.build_manifest(default_home, default_home=default_home, target_profiles={"rva-leads"})
    result = rebalance.apply_stage(manifest, default_home=default_home)

    assert [move.target_profile for move in manifest.moves] == ["rva-leads"]
    assert result["staged_jobs"] == ["lead"]
    assert (default_home / "profiles" / "rva-leads" / "cron").exists()
    assert not (default_home / "profiles" / "rva-profit-pulse").exists()


def test_rebalance_main_json_smoke_and_rejects_stage_cutover_combo(tmp_path, capsys):
    default_home = tmp_path / ".hermes"
    make_script(default_home, "lead.py")
    write_jobs(default_home, [{"id": "lead", "name": "crm-lead-check", "script": "lead.py", "no_agent": True}])

    assert rebalance.main(["--source-home", str(default_home), "--default-home", str(default_home), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_home"] == "<profile:default>"
    assert payload["moves"][0]["script_copy"]["source"] == "<default:scripts/lead.py>"

    with pytest.raises(SystemExit):
        rebalance.main([
            "--source-home",
            str(default_home),
            "--default-home",
            str(default_home),
            "--apply-stage",
            "--cutover-verified",
            "lead",
        ])
