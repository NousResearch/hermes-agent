"""Task-skills normalization at card creation: joined strings and toolset names
never reach a card (a slash-joined pseudo-skill persisted to ``tasks.skills``
aborts the dispatched worker at boot, on every retry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_skills import _normalize_task_skills


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.mark.parametrize("joined, refusal", [
    ("github-pr-workflow,github-code-review", "cannot contain comma"),
    ("github-pr-workflow/github-code-review", "cannot contain '/'"),
    ("skills/github-code-review", "cannot contain '/'"),
    ("github-code-review/", "cannot contain '/'"),
])
def test_refuses_joined_strings(joined, refusal):
    """A joined multi-name string must never land in one argv slot: it persists to
    the card verbatim and the dispatched worker then fails to resolve it at boot."""
    with pytest.raises(ValueError, match=refusal):
        _normalize_task_skills([joined])


def test_slash_refusal_names_the_fix():
    """The '/' refusal must be actionable — name separate names, not a path form."""
    with pytest.raises(ValueError, match="pass separate names"):
        _normalize_task_skills(["github-pr-workflow/github-code-review"])


def test_rejects_toolset_names():
    with pytest.raises(ValueError, match="are toolset names, not skill name"):
        _normalize_task_skills(["web", "terminal"])


def test_strips_and_dedupes():
    assert _normalize_task_skills([" a ", "", "b", "a"]) == ["a", "b"]
    assert _normalize_task_skills(None) is None


def test_create_task_refuses_slash_joined_skills_before_persist(kanban_home):
    """The refusal must fire at the persistence boundary: no card may be created
    carrying a slash-joined pseudo-skill for a worker to crash on later."""
    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="cannot contain '/'"):
            kb.create_task(conn, title="poisoned card", assignee="worker-a",
                           skills=["github-pr-workflow/github-code-review"])
        assert kb.list_tasks(conn) == []
