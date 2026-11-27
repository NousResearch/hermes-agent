"""Regression: kanban_create must not persist literal 'null' option strings.

A caller (an LLM tool call) stringified its absent optional fields as the
literal ``"null"``/``"None"``; create_task stored them verbatim and the card
could never spawn — resolve_workspace raised on
scratch + workspace_path='null', the dispatcher gave up after 2 failures,
and the card sat ``blocked`` with no owner action. The create path now
coerces those spellings to None before persisting, so the card spawns
like any other scratch task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_nullish_optional_fields_stored_as_null_and_scratch_resolves(
    kanban_home: Path,
) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="nullish options",
            assignee="worker",
            workspace_kind="scratch",
            workspace_path="null",
            tenant="None",
            branch_name="null",
            project_id="None",
            model_override="null",
            provider_override="None",
            skills=["null", "None", "", "real-skill"],
        )
        row = conn.execute(
            "SELECT workspace_path, tenant, branch_name, project_id, "
            "model_override, provider_override, skills FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert row["workspace_path"] is None
        assert row["tenant"] is None
        assert row["branch_name"] is None
        assert row["project_id"] is None
        assert row["model_override"] is None
        assert row["provider_override"] is None
        assert json.loads(row["skills"]) == ["real-skill"]

        # The original failure: a truthy junk
        # workspace_path on a scratch task raised "non-absolute
        # workspace_path" at spawn. Resolved (and created) instead.
        task = kb.get_task(conn, tid)
        assert task is not None
        p = kbw.resolve_workspace(task)
        assert p.is_dir() and p.name == task.id


def test_legit_values_are_not_coerced(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="legit values",
            assignee="worker",
            model_override="nullm-overRide",
        )
        row = conn.execute(
            "SELECT model_override FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["model_override"] == "nullm-overRide"
