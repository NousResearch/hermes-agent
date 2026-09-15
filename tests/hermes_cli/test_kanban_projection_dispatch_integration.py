"""End-to-end projection materialization through the dispatcher claim path."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_dispatch_materializes_bounded_projection_and_rederives(kanban_home, monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "allowed.txt").write_text("v1")
    (repo / "secret.txt").write_text("secret")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=x", "-c", "user.email=x@y", "commit", "-qm", "v1"], check=True)
    base_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    projection = {"repo": str(repo), "base_sha": base_sha,
                  "ceiling": {"read": ["allowed.txt"], "write": ["allowed.txt"]}}
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda _name: True)
    spawns = []
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="project", assignee="alice", workspace_kind="projected",
            workspace_path=str(repo), scope_paths=["allowed.txt"],
            body=json.dumps({"projection": projection}),
        )
        result = kbd.dispatch_once(conn, spawn_fn=lambda task, workspace, board=None: spawns.append(workspace) or 1)
        assert result.spawned and spawns
        projected = Path(spawns[0])
        assert (projected / "allowed.txt").read_text() == "v1"
        assert not (projected / "secret.txt").exists()
        # Traversal is rejected before archive extraction, not clipped into the tree.
        conn.execute("UPDATE tasks SET scope_paths=? WHERE id=?", (json.dumps(["../secret.txt"]), task_id))
        with pytest.raises(ValueError):
            kbd._projected_workspace(kb.get_task(conn, task_id), board=None)

        # Change the pinned commit and re-claim: the old projection is replaced.
        (repo / "allowed.txt").write_text("v2")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=x", "-c", "user.email=x@y", "commit", "-qm", "v2"], check=True)
        new_sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        projection["base_sha"] = new_sha
        conn.execute("UPDATE tasks SET status='ready', workspace_path=?, scope_paths=? WHERE id=?", (str(projected), json.dumps(["allowed.txt"]), task_id))
        projection_body = json.dumps({"projection": projection})
        conn.execute("UPDATE tasks SET body=? WHERE id=?", (projection_body, task_id))
        monkeypatch.setattr(kbd._kb, "read_board_metadata", lambda board=None: {"dispatch_enabled": True})
        # Task body remains the durable source of projection inputs.
        task = kb.get_task(conn, task_id)
        fresh = kbd._projected_workspace(task, board=None)
        assert Path(fresh).joinpath("allowed.txt").read_text() == "v2"
