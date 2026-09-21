import json
import subprocess

import pytest


def _task(kb, **overrides):
    data = dict(
        id="t_test",
        title="test",
        body=None,
        assignee="octacon",
        status="ready",
        priority=0,
        created_by=None,
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    data.update(overrides)
    return kb.Task(**data)


def test_validate_forced_skills_accepts_profile_visible_skill(tmp_path):
    from hermes_cli.kanban_db import validate_forced_skills_visible

    skill_dir = tmp_path / "skills" / "devops" / "kanban-worker"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: kanban-worker\n---\n", encoding="utf-8")

    assert validate_forced_skills_visible(["kanban-worker"], str(tmp_path)) == []


def test_validate_forced_skills_rejects_missing_skill(tmp_path):
    from hermes_cli.kanban_db import validate_forced_skills_visible

    missing = validate_forced_skills_visible(["missing-skill"], str(tmp_path))

    assert missing == ["missing-skill"]


def test_default_spawn_blocks_missing_forced_skill_before_worker_start(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    task = _task(
        kb,
        id="t_missing",
        title="missing forced skill",
        status="ready",
        assignee="octacon",
        created_at=1,
        skills=["missing-skill"],
    )

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda profile: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda profile: profile)

    def fail_popen(*args, **kwargs):  # pragma: no cover - should never be reached
        raise AssertionError("worker should not start when forced skill is invisible")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)

    with pytest.raises(RuntimeError, match="Forced skill.*missing-skill"):
        kb._default_spawn(task, str(tmp_path / "workspace"), board="ops")


def test_default_spawn_records_dispatch_event_when_feature_enabled(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb

    skill_dir = tmp_path / "skills" / "extra"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: extra\n---\n", encoding="utf-8")

    task = _task(
        kb,
        id="t_dispatch",
        title="dispatch",
        status="running",
        assignee="octacon",
        created_at=1,
        current_run_id=42,
        skills=["extra"],
    )

    recorded = []
    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda profile: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda profile: profile)
    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: tmp_path / "kanban.db")
    monkeypatch.setattr(kb, "workspaces_root", lambda board=None: tmp_path / "workspaces")
    monkeypatch.setattr(kb, "worker_logs_dir", lambda board=None: tmp_path / "logs")
    monkeypatch.setattr(kb, "worker_log_rotation_config", lambda: (0, 0))
    monkeypatch.setattr(kb, "_rotate_worker_log", lambda *a, **k: None)
    monkeypatch.setattr(kb, "record_event_if_enabled", lambda **kwargs: recorded.append(kwargs) or "evt")

    class DummyProc:
        pid = 1234

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: DummyProc())

    assert kb._default_spawn(task, str(tmp_path / "workspace"), board="ops") == 1234
    assert recorded
    assert recorded[0]["event_type"] == "kanban.worker.dispatched"
    assert recorded[0]["target_profile"] == "octacon"
    assert recorded[0]["payload"]["skills"][0] == "extra"
