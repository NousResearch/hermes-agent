"""Tests for the manual todo steering mutation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import projects_db
from hermes_cli.roadmaps_writer import (
    InvalidRoadmapTodoTransitionError,
    RoadmapTodoNotFoundError,
    RoadmapsWriter,
)


def seed(path: Path) -> None:
    conn = projects_db.connect(path)
    conn.execute("INSERT INTO projects(id, slug, name, created_at) VALUES ('p', 'p', 'P', 1)")
    conn.execute(
        "INSERT INTO roadmaps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", "Roadmap", None, "in_progress", 1, "a", "a", 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "validated", "src", None, "a", 1, None),
    )
    conn.execute(
        "INSERT INTO roadmap_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "n1", None, "step", "Node", None, "in_progress", 10, "agent", None, 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_todos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "t-open", "n1", "Open todo", "open", 0, 1, 1),
    )
    conn.execute(
        "INSERT INTO roadmap_todos VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("profile", "p", "r", 1, "t-done", "n1", "Done todo", "done", 1, 1, 1),
    )
    conn.commit()
    conn.close()


def todo_state(path: Path, todo_id: str) -> str:
    conn = projects_db.connect(path)
    row = conn.execute("SELECT state FROM roadmap_todos WHERE todo_id=?", (todo_id,)).fetchone()
    conn.close()
    return row["state"]


def test_update_todo_transitions_open_to_done(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path)
    result = RoadmapsWriter(path).update_todo("profile", "p", "r", "t-open", "user", "done", 1)
    assert result["success"] is True
    assert result["todo"]["state"] == "done"
    assert result["before"] == {"state": "open"}
    assert todo_state(path, "t-open") == "done"


def test_update_todo_rejects_transition_from_done(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path)
    with pytest.raises(InvalidRoadmapTodoTransitionError):
        RoadmapsWriter(path).update_todo("profile", "p", "r", "t-done", "user", "open", 1)
    assert todo_state(path, "t-done") == "done"


def test_update_todo_rejects_unknown_todo_and_bad_state(tmp_path: Path):
    path = tmp_path / "projects.db"
    seed(path)
    writer = RoadmapsWriter(path)
    with pytest.raises(RoadmapTodoNotFoundError):
        writer.update_todo("profile", "p", "r", "t-missing", "user", "done", 1)
    with pytest.raises(ValueError, match="state must be"):
        writer.update_todo("profile", "p", "r", "t-open", "user", "finished", 1)
