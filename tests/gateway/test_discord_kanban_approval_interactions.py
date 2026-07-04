from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_discord_approvals import build_custom_id
from plugins.platforms.discord.adapter import handle_kanban_approval_interaction


class ResponseRecorder:
    def __init__(self):
        self.sent = []
        self.edits = []

    async def send_message(self, content, **kwargs):
        self.sent.append({"content": content, **kwargs})

    async def edit_message(self, **kwargs):
        self.edits.append(kwargs)


def _interaction(action, *, task_id="t_gate", user_id="111", display_name="Matthew", role_ids=(), content="approval prompt"):
    button = SimpleNamespace(disabled=False)
    return SimpleNamespace(
        data={"custom_id": build_custom_id(task_id, action)},
        user=SimpleNamespace(
            id=user_id,
            display_name=display_name,
            roles=[SimpleNamespace(id=role_id) for role_id in role_ids],
        ),
        message=SimpleNamespace(
            content=content,
            components=[SimpleNamespace(children=[button])],
        ),
        response=ResponseRecorder(),
        _button=button,
    )


@pytest.fixture
def gate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    with kb.connect_closing(db_path=db_path) as conn:
        task_id = kb.create_task(conn, title="Human gate", assignee="forge", body="", initial_status="blocked")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    return db_path, task_id


@pytest.mark.asyncio
async def test_kanban_approval_interaction_allowed_user_approves_and_disables_buttons(gate_db):
    db_path, task_id = gate_db
    interaction = _interaction("approve", task_id=task_id, user_id="111", display_name="Matthew")

    handled = await handle_kanban_approval_interaction(interaction, {"111"}, set())

    assert handled is True
    assert interaction.response.sent == []
    assert interaction.response.edits
    assert interaction.response.edits[0]["view"] is None
    assert "Approved by Matthew" in interaction.response.edits[0]["content"]
    assert interaction._button.disabled is True
    with kb.connect_closing(db_path=db_path) as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"] == "ready"
        comments = kb.list_comments(conn, task_id)
    assert comments[0].author == "discord:Matthew (111)"
    assert "APPROVED via Discord approval button" in comments[0].body


@pytest.mark.asyncio
async def test_kanban_approval_interaction_allowed_role_can_deny(gate_db):
    db_path, task_id = gate_db
    interaction = _interaction("deny", task_id=task_id, user_id="222", display_name="Reviewer", role_ids=[42])

    handled = await handle_kanban_approval_interaction(interaction, set(), {42})

    assert handled is True
    assert "Denied by Reviewer" in interaction.response.edits[0]["content"]
    assert interaction._button.disabled is True
    with kb.connect_closing(db_path=db_path) as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"] == "blocked"
        comments = kb.list_comments(conn, task_id)
    assert "DENIED via Discord approval button" in comments[0].body


@pytest.mark.asyncio
async def test_kanban_approval_interaction_unauthorized_user_gets_ephemeral_error(gate_db):
    db_path, task_id = gate_db
    interaction = _interaction("approve", task_id=task_id, user_id="999", display_name="Intruder")

    handled = await handle_kanban_approval_interaction(interaction, {"111"}, {42})

    assert handled is True
    assert interaction.response.edits == []
    assert interaction.response.sent == [{"content": "You're not authorized to approve Kanban gates~", "ephemeral": True}]
    assert interaction._button.disabled is False
    with kb.connect_closing(db_path=db_path) as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"] == "blocked"
        assert kb.list_comments(conn, task_id) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected_label", "expected_comment", "expected_status"),
    [
        ("approve", "Approved", "APPROVED", "ready"),
        ("deny", "Denied", "DENIED", "blocked"),
        ("needs_changes", "Needs changes requested", "NEEDS CHANGES", "blocked"),
    ],
)
async def test_kanban_approval_interaction_records_all_decision_outcomes(tmp_path, monkeypatch, action, expected_label, expected_comment, expected_status):
    db_path = tmp_path / f"{action}.db"
    with kb.connect_closing(db_path=db_path) as conn:
        task_id = kb.create_task(conn, title="Human gate", assignee="forge", body="", initial_status="blocked")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    interaction = _interaction(action, task_id=task_id, user_id="111", display_name="Matthew")

    handled = await handle_kanban_approval_interaction(interaction, {"111"}, set())

    assert handled is True
    assert expected_label in interaction.response.edits[0]["content"]
    assert interaction.response.sent == []
    assert interaction._button.disabled is True
    with kb.connect_closing(db_path=db_path) as conn:
        assert conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()["status"] == expected_status
        comments = kb.list_comments(conn, task_id)
    assert expected_comment in comments[0].body
