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


def _write_skill(profile_home, name, category="devops"):
    skill_dir = profile_home / "skills" / category / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\n---\n", encoding="utf-8")


def _write_config(profile_home, enabled_skills=None, always_skills=None):
    cfg = {}
    if enabled_skills is not None or always_skills is not None:
        skills = {}
        if enabled_skills is not None:
            skills["enabled_skills"] = enabled_skills
        if always_skills is not None:
            skills["always_skills"] = always_skills
        cfg["skills"] = skills
    (profile_home / "config.yaml").write_text(
        json.dumps(cfg), encoding="utf-8"
    )


def test_missing_forced_skills_visible_but_not_enabled(monkeypatch, tmp_path):
    """A visible-but-not-enabled skill must be reported as missing."""
    from hermes_cli import kanban_db as kb

    _write_skill(tmp_path, "test-driven-development")
    _write_config(tmp_path, enabled_skills=["avoid-ai-writing"])

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda p: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda p: p)

    missing = kb._missing_worker_forced_skills(
        "octacon", ["test-driven-development"]
    )
    assert missing == ["test-driven-development"]


def test_missing_forced_skills_enabled_ok(monkeypatch, tmp_path):
    """A visible AND enabled skill must NOT be reported as missing."""
    from hermes_cli import kanban_db as kb

    _write_skill(tmp_path, "test-driven-development")
    _write_config(tmp_path, enabled_skills=["test-driven-development"])

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda p: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda p: p)

    missing = kb._missing_worker_forced_skills(
        "octacon", ["test-driven-development"]
    )
    assert missing == []


def test_missing_forced_skills_always_skills_ok(monkeypatch, tmp_path):
    """always_skills are implicitly enabled even if not in enabled_skills."""
    from hermes_cli import kanban_db as kb

    _write_skill(tmp_path, "test-driven-development")
    _write_config(
        tmp_path,
        enabled_skills=["avoid-ai-writing"],
        always_skills=["test-driven-development"],
    )

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda p: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda p: p)

    missing = kb._missing_worker_forced_skills(
        "octacon", ["test-driven-development"]
    )
    assert missing == []


def test_missing_forced_skills_no_allowlist_ok(monkeypatch, tmp_path):
    """No enabled_skills key configured → unrestricted (back-compat)."""
    from hermes_cli import kanban_db as kb

    _write_skill(tmp_path, "test-driven-development")
    _write_config(tmp_path)  # no skills section

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda p: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda p: p)

    missing = kb._missing_worker_forced_skills(
        "octacon", ["test-driven-development"]
    )
    assert missing == []


def test_missing_forced_skills_empty_allowlist_blocks(monkeypatch, tmp_path):
    """A configured-but-empty allowlist denies everything except always_skills."""
    from hermes_cli import kanban_db as kb

    _write_skill(tmp_path, "test-driven-development")
    _write_config(tmp_path, enabled_skills=[])

    monkeypatch.setattr("hermes_cli.profiles.resolve_profile_env", lambda p: str(tmp_path))
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda p: p)

    missing = kb._missing_worker_forced_skills(
        "octacon", ["test-driven-development"]
    )
    assert missing == ["test-driven-development"]


def test_child_cli_records_forced_skill_rejected_event(monkeypatch, tmp_path):
    """The child CLI hard-fail path records a forced_skill_rejected event."""
    from hermes_cli import kanban_db as kb

    # Build a real board DB with a running task (insert directly so we can
    # pin the id and current_run_id).
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    conn.execute(
        "INSERT INTO tasks (id, title, assignee, status, created_at, "
        "current_run_id, skills) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "t_hardfail",
            "hard fail",
            "octacon",
            "running",
            1,
            7,
            json.dumps(["test-driven-development"]),
        ),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_hardfail")
    monkeypatch.setattr(kb, "kanban_db_path", lambda board=None: db_path)
    monkeypatch.setattr(kb, "get_current_board", lambda: "ops")

    # Instantiate the CLI object and call the recorder directly.
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._record_forced_skill_rejected(["test-driven-development"])

    conn = kb.connect(db_path)
    events = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id='t_hardfail'"
    ).fetchall()
    conn.close()

    kinds = [e["kind"] for e in events]
    assert "forced_skill_rejected" in kinds
    ev = next(e for e in events if e["kind"] == "forced_skill_rejected")
    payload = json.loads(ev["payload"])
    assert payload["reason"] == "child_cli_hard_fail"
    assert payload["missing_skills"] == ["test-driven-development"]
    assert payload["assignee"] == "octacon"
