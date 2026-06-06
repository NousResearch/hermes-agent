"""Discord Kanban review-required approval card tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, KanbanReviewCardView


def _make_adapter(*, allowed_users=None, allowed_roles=None):
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = DiscordAdapter(config)
    adapter._client = MagicMock()
    adapter._allowed_user_ids = set(allowed_users or [])
    adapter._allowed_role_ids = set(allowed_roles or [])
    return adapter


def _make_interaction(*, user_id="42", display_name="Tester", roles=None):
    user = SimpleNamespace(
        id=user_id,
        display_name=display_name,
        roles=[SimpleNamespace(id=r) for r in (roles or [])],
    )
    embed = MagicMock()
    embed.color = None
    embed.set_footer = MagicMock()
    message = SimpleNamespace(embeds=[embed])
    return SimpleNamespace(
        user=user,
        message=message,
        response=SimpleNamespace(
            edit_message=AsyncMock(),
            send_message=AsyncMock(),
        ),
        followup=SimpleNamespace(send=AsyncMock()),
    )


@pytest.mark.asyncio
async def test_send_kanban_review_card_posts_embed_and_buttons():
    adapter = _make_adapter(allowed_users={"42"})
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=999)))
    adapter._client.get_channel.return_value = channel

    result = await adapter.send_kanban_review_card(
        "123",
        task_id="t_review123",
        title="Verify deployment",
        reason="review-required: tests passed, approve merge",
        board="main",
        pending_since=1717600000,
        assignee="codex",
        metadata={"thread_id": "456"},
    )

    assert result.success is True
    adapter._client.get_channel.assert_called_once_with(456)
    kwargs = channel.send.call_args.kwargs
    embed = kwargs["embed"]
    view = kwargs["view"]
    assert embed.title == "🚨 Review needed"
    assert "Verify deployment" in embed.description
    assert any(f["name"] == "Reason" and "review-required" in f["value"] for f in embed.fields)
    assert isinstance(view, KanbanReviewCardView)
    assert [child.label for child in view.children] == ["✅ Approve", "📋 Details", "⏸ Keep blocked"]


@pytest.mark.asyncio
async def test_review_card_approve_marks_resolved_and_disables_buttons(monkeypatch):
    view = KanbanReviewCardView(
        task_id="t_review123",
        board="main",
        allowed_user_ids={"42"},
    )
    monkeypatch.setattr(view, "_approve_sync", lambda actor: (True, "approved"))
    interaction = _make_interaction(user_id="42", display_name="Aniketan")

    await view._on_approve(interaction)

    assert view.resolved is True
    assert all(child.disabled for child in view.children)
    interaction.response.edit_message.assert_awaited_once()
    interaction.followup.send.assert_awaited_once_with("approved")


def test_review_card_approve_sync_marks_real_kanban_task_done(tmp_path, monkeypatch):
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    from hermes_cli import kanban_db as kb

    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="Needs human review",
            body="Worker handoff",
            assignee="codex",
            created_by="pytest",
            initial_status="blocked",
        )
    finally:
        conn.close()

    view = KanbanReviewCardView(
        task_id=task_id,
        board=None,
        allowed_user_ids={"42"},
    )

    ok, text = view._approve_sync("Aniketan")

    assert ok is True
    assert "marked it done" in text
    conn = kb.connect()
    try:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "done"
        comments = kb.list_comments(conn, task_id)
        assert any("Approved via Discord" in c.body for c in comments)
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_review_card_rejects_unauthorized_click(monkeypatch):
    view = KanbanReviewCardView(
        task_id="t_review123",
        board="main",
        allowed_user_ids={"42"},
    )
    called = False

    def fake_approve(actor):
        nonlocal called
        called = True
        return True, "approved"

    monkeypatch.setattr(view, "_approve_sync", fake_approve)
    interaction = _make_interaction(user_id="99", display_name="Rando")

    await view._on_approve(interaction)

    assert called is False
    interaction.response.send_message.assert_awaited_once()
    assert "not authorized" in interaction.response.send_message.call_args.args[0]
