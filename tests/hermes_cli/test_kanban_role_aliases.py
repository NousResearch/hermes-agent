from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn(*args, **kwargs):
    return 12345


def test_create_task_canonicalizes_legacy_role_assignee_aliases(kanban_home: Path) -> None:
    with kb.connect() as conn:
        operator_id = kb.create_task(conn, title="ops lane", assignee="operator")
        researcher_id = kb.create_task(conn, title="research lane", assignee="researcher")
        builder_id = kb.create_task(conn, title="build lane", assignee="builder")
        steward_id = kb.create_task(conn, title="steward lane", assignee="steward")

        assert kb.get_task(conn, operator_id).assignee == "winston"
        assert kb.get_task(conn, researcher_id).assignee == "brennan"
        assert kb.get_task(conn, builder_id).assignee == "stark"
        assert kb.get_task(conn, steward_id).assignee == "pepper"


def test_dispatch_rewrites_existing_role_alias_rows_before_spawn(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "winston")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy operator row", assignee="default")
        conn.execute("UPDATE tasks SET assignee = 'operator' WHERE id = ?", (task_id,))
        conn.commit()

        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn)

        assert res.spawned == [(task_id, "winston", str(kb.resolve_workspace(kb.get_task(conn, task_id))))]
        assert kb.get_task(conn, task_id).assignee == "winston"
        assigned = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'assigned' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(assigned["payload"])
        assert payload == {
            "assignee": "winston",
            "source": "kanban.role_alias",
            "alias": "operator",
        }


def test_default_assignee_uses_role_alias_before_profile_validation(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "brennan")

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="unassigned")
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            default_assignee="researcher",
        )

        assert res.auto_assigned_default == [task_id]
        assert res.spawned[0][1] == "brennan"
        assert kb.get_task(conn, task_id).assignee == "brennan"


def test_decomposer_routes_role_name_choice_to_live_profile_instead_of_default() -> None:
    from hermes_cli.kanban_decompose import _normalize_assignee_choice

    assert _normalize_assignee_choice(
        "builder",
        default_assignee="default",
        valid_names={"default", "stark"},
    ) == "stark"
