"""Native group detail reaches the existing fenced approval decision path."""

import sqlite3
import json

import pytest

from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_room_messaging_files as files
from gateway.choice_picker import ChoicePage
from tests.gateway.test_group_native_menu_navigation import english, menu_for, token
from tests.gateway.test_hosted_room_file_access import file_state
from tests.gateway.test_hosted_room_messaging_files import consumer


def pending(state, request_id="approval-1", *, member_id="ops", command="rm -rf ./obsolete-draft"):
    approvals.persist_pending_approval(
        state.db, room_id="room-1", member_id=member_id,
        action={
            "kind": "approval", "authority_gateway_id": state.authority,
            "authority_epoch": 1, "task_id": "task-1", "execution_generation": 1,
            "request_id": request_id,
            "approval": {"description": "Remove obsolete draft", "command": command,
                         "choices": ["once", "deny"]},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["once", "deny"])
async def test_group_detail_opens_native_approval_and_journals_one_exact_decision(consumer, decision):
    state, _, _ = consumer
    menu = await menu_for(consumer)
    page = await menu.room_page()
    assert not any(action[0] == "approvals" for action in menu.actions.values())
    pending(state)
    page = await menu.room_page()
    assert "waiting for your approval" in page.title
    page = await menu.choose("chat", token(menu, page, "approvals"))
    assert "Approval needed · Files" in page.title
    assert "rm -rf ./obsolete-draft" in page.title
    assert all(choice["full_width"] for choice in page.choices)
    assert not approvals.list_pending_approval_commands(state.db, room_id="room-1")
    index = 0 if decision == "once" else 1
    value = page.choices[index]["value"]
    result = await menu.choose("chat", value)
    assert isinstance(result, ChoicePage)
    assert "Decision sent for ops." in result.title
    commands = approvals.list_pending_approval_commands(state.db, room_id="room-1")
    assert [(row["choice"], row["request_id"]) for row in commands] == [(decision, "approval-1")]
    await menu.choose("chat", value)
    assert len(approvals.list_pending_approval_commands(state.db, room_id="room-1")) == 1
    result = await menu.choose("chat", token(menu, result, "room"))
    assert isinstance(result, ChoicePage)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [4, 5, 8])
async def test_every_long_approval_has_its_own_complete_actionable_page(consumer, count):
    state, _, _ = consumer
    members = [{"member_id": f"bot{i}", "profile": f"bot{i}", "handle": f"bot{i}"}
               for i in range(1, count + 1)]
    with sqlite3.connect(state.db) as conn:
        conn.execute("UPDATE hosted_rooms SET members_json=? WHERE room_id='room-1'",
                     (json.dumps(members),))
    for i, member in enumerate(members, 1):
        pending(state, f"approval-{i}", member_id=member["member_id"],
                command="printf '" + "x" * 470 + f" COMMAND{i}END'")
    menu = await menu_for(consumer)
    page = await menu.approval_page()
    for i in range(1, count + 1):
        assert len(page.title) <= 2048
        assert f"Request {i} of {count}" in page.title
        assert f"COMMAND{i}END" in page.title
        decisions = [c for c in page.choices if menu.actions[c["value"]][0] == "approval_decision"]
        assert [c["label"] for c in decisions] == ["✓ Allow once", "✕ Deny"]
        if i < count:
            next_page = next(c for c in page.choices if c["label"].endswith(files.text("approval_next")))
            page = await menu.choose("chat", next_page["value"])
    assert not approvals.list_pending_approval_commands(state.db, room_id="room-1")
    await menu.choose("chat", decisions[0]["value"])
    result = approvals.list_pending_approval_commands(state.db, room_id="room-1")
    assert [(r["member_id"], r["request_id"], r["choice"]) for r in result] == [
        (f"bot{count}", f"approval-{count}", "once"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["wrong_chat", "permission", "authority", "request", "expired"])
async def test_nested_approval_never_applies_a_stale_or_unauthorized_choice(consumer, monkeypatch, change):
    state, runner, _ = consumer
    menu = await menu_for(consumer)
    pending(state)
    page = await menu.room_page()
    page = await menu.choose("chat", token(menu, page, "approvals"))
    selected = page.choices[0]["value"]
    if change == "permission":
        monkeypatch.setattr(runner, "_can_approve_group_chats", lambda event: False)
    elif change == "authority":
        with sqlite3.connect(state.db) as conn:
            conn.execute("UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='room-1'")
    elif change == "request":
        pending(state, "approval-2")
    elif change == "expired":
        menu.deadline = 0
    await menu.choose("other" if change == "wrong_chat" else "chat", selected)
    assert not approvals.list_pending_approval_commands(state.db, room_id="room-1")
