"""Regression for #156561: a loaded task skill must not hide a missing review skill.

A Kanban worker only raises when *all* requested skills fail to load. A task's
custom skill can therefore mask a missing sdlc-review at worker startup.
The review dispatcher must check its forced skill independently, not treat
"at least one skill loaded" as sufficient for a review.
"""

from pathlib import Path

import pytest

from agent.skill_commands import build_preloaded_skills_prompt
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def test_loaded_task_skill_does_not_mask_missing_forced_review_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # The task's requested skill works; only the mandated reviewer skill is absent.
    task_skill = home / "skills" / "devops" / "specialist"
    task_skill.mkdir(parents=True)
    (task_skill / "SKILL.md").write_text(
        "---\nname: specialist\ndescription: Task-specific expertise.\n---\n\n"
        "# Specialist\n\nHandle this task's specific context.\n",
        encoding="utf-8",
    )
    _, loaded, missing = build_preloaded_skills_prompt(["specialist", "sdlc-review"])
    assert "specialist" in loaded
    assert "sdlc-review" in missing

    # Pin the true review_dispatch default in this isolated test.
    import hermes_cli.config as config
    monkeypatch.setattr(config, "load_config", lambda *a, **kw: {"kanban": {"review_dispatch": True}})
    kb.init_db()
    spawned = []

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="review with specialist", assignee="default", skills=["specialist"],
        )
        assert kb.claim_task(conn, tid) is not None
        assert kb.request_review(
            conn, tid, summary="ready for review",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )

        def spawn(task, workspace, board=None):
            spawned.append(task)
            return 4242

        result = kbd.dispatch_once(conn, spawn_fn=spawn)
        task = kb.get_task(conn, tid)

    assert spawned == [], "having a task skill cannot license review without sdlc-review"
    assert result.skills_unavailable == [(tid, "sdlc-review")]
    assert task.status == "blocked"
    assert "kanban_skills_unavailable: sdlc-review" in (task.last_failure_error or "")
