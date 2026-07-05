from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _profile_with_skills(home: Path, name: str, skills: list[str]) -> Path:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text("toolsets:\n  - terminal\n", encoding="utf-8")
    for skill in skills:
        skill_dir = profile_dir / "skills" / skill
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill}\n---\n# {skill}\n", encoding="utf-8"
        )
    return profile_dir


def test_create_task_rejects_skill_missing_from_actual_target_profile(kanban_home):
    _profile_with_skills(kanban_home, "worker", ["present-skill"])
    default_only = kanban_home / "skills" / "default-only"
    default_only.mkdir(parents=True)
    (default_only / "SKILL.md").write_text("---\nname: default-only\n---\n", encoding="utf-8")

    conn = kb.connect()
    try:
        with pytest.raises(ValueError) as exc:
            kb.create_task(
                conn,
                title="invalid worker skill",
                assignee="worker",
                skills=["present-skill", "default-only", "missing-skill"],
            )

        msg = str(exc.value)
        assert "worker" in msg
        assert "default-only" in msg
        assert "missing-skill" in msg
        assert "present-skill" not in msg
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
    finally:
        conn.close()


def test_dispatch_blocks_legacy_invalid_skills_before_spawn_without_retry_loss(
    kanban_home, all_assignees_spawnable
):
    _profile_with_skills(kanban_home, "worker", ["present-skill"])
    spawned = []

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="legacy invalid skill card",
            assignee="worker",
            skills=["present-skill"],
        )
        conn.execute(
            "UPDATE tasks SET skills = ? WHERE id = ?",
            (json.dumps(["present-skill", "missing-skill"]), tid),
        )
        conn.commit()

        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda task, workspace: spawned.append(task.id) or 98765,
        )

        assert spawned == []
        assert tid in res.spawn_blocked_missing_skills
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 0
        assert task.last_failure_error
        assert tid in task.last_failure_error
        assert "worker" in task.last_failure_error
        assert "missing-skill" in task.last_failure_error
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.outcome == "spawn_blocked_missing_skills"
    finally:
        conn.close()


def test_rerouted_replacement_cards_revalidate_skills_for_new_assignee(kanban_home):
    _profile_with_skills(kanban_home, "source-worker", ["source-only"])
    _profile_with_skills(kanban_home, "target-worker", ["target-only"])

    conn = kb.connect()
    try:
        source = kb.create_task(
            conn,
            title="source card",
            assignee="source-worker",
            skills=["source-only"],
        )
        with pytest.raises(ValueError) as exc:
            kb.create_task(
                conn,
                title="replacement card",
                assignee="target-worker",
                parents=[source],
                skills=["source-only"],
            )

        msg = str(exc.value)
        assert "target-worker" in msg
        assert "source-only" in msg
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = ?", ("replacement card",)
        ).fetchone()[0] == 0
    finally:
        conn.close()
