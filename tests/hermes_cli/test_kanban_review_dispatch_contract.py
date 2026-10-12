from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_review_dispatch_uses_reviewer_profile_model_not_implementer_override(
    kanban_home, tmp_path,
):
    workspace = tmp_path / "implementation"
    workspace.mkdir()
    captured = {}

    def fake_spawn(task, review_workspace, board=None):
        captured["task"] = task
        return 42

    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review candidate",
            assignee="reviewer",
            workspace_kind="dir",
            workspace_path=str(workspace),
            model_override="grok-implementer",
            provider_override="builder-provider",
        )
        conn.execute(
            "UPDATE tasks SET status = 'review', claim_lock = NULL, claim_expires = NULL "
            "WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        result = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        durable = kb.get_task(conn, task_id)

    assert result.spawned
    assert durable is not None
    assert captured["task"].model_override is None
    assert captured["task"].provider_override is None
    assert durable.model_override == "grok-implementer"
    assert durable.provider_override == "builder-provider"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_review_dispatch_uses_reviewer_owned_detached_worktree(
    kanban_home, tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    repo.joinpath("candidate.txt").write_text("candidate\n", encoding="utf-8")
    repo.joinpath(".gitignore").write_text("*.review-cache\n", encoding="utf-8")
    _git(repo, "add", "candidate.txt", ".gitignore")
    _git(repo, "commit", "-m", "candidate")
    candidate_sha = _git(repo, "rev-parse", "HEAD")
    implementation = tmp_path / "implementer-worktree"
    _git(repo, "worktree", "add", "--detach", str(implementation), candidate_sha)

    captured = {}
    board = "review-contract"
    kb.create_board(board)

    def fake_spawn(task, review_workspace, board=None):
        captured["workspace"] = Path(review_workspace)
        return 42

    with kbc.connect(board=board) as conn:
        task_id = kb.create_task(
            conn,
            title="review candidate",
            assignee="reviewer",
            workspace_kind="dir",
            workspace_path=str(implementation),
        )
        conn.execute(
            "UPDATE tasks SET status = 'review', claim_lock = NULL, claim_expires = NULL "
            "WHERE id = ?",
            (task_id,),
        )
        conn.commit()
        result = kbd.dispatch_once(conn, spawn_fn=fake_spawn, board=board)
        persisted = kb.get_task(conn, task_id)

    review_workspace = captured["workspace"]
    assert persisted is not None
    assert result.spawned
    assert review_workspace != implementation
    assert "reviewer" in review_workspace.name
    assert _git(review_workspace, "rev-parse", "HEAD") == candidate_sha
    assert _git(review_workspace, "rev-parse", "--is-inside-work-tree") == "true"
    assert persisted.workspace_path == str(implementation)

    _git(review_workspace, "switch", "-c", "accidental-review-branch")
    ignored_artifact = review_workspace / "stale.review-cache"
    ignored_artifact.write_text("stale\n", encoding="utf-8")
    reused = kbw.resolve_review_workspace(persisted, board=board)
    attached = subprocess.run(
        ["git", "-C", str(reused), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        text=True,
    )
    assert reused == review_workspace
    assert attached.returncode != 0
    assert not ignored_artifact.exists()

    with kbc.connect(board=board) as conn:
        running_review = kb.get_task(conn, task_id)
        assert running_review is not None
        assert kb.complete_task(
            conn,
            task_id,
            summary="review approved",
            metadata={"review_outcome": "approved"},
            expected_run_id=running_review.current_run_id,
        )

    assert not review_workspace.exists()
    assert str(review_workspace) not in _git(repo, "worktree", "list", "--porcelain")
