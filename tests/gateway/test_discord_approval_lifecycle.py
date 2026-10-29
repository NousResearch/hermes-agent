"""Approval cards must release their view and preserve terminal state across edits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.i18n import t
from gateway.config import PlatformConfig
from plugins.platforms.discord import adapter as discord_adapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


async def _pending_card(monkeypatch):
    adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    session = "agent:other:discord:555"
    entry = _ApprovalEntry({"request_id": "request-1", "command": "ls"})
    monkeypatch.setitem(approval._gateway_queues, session, [entry])
    message = SimpleNamespace(id=1234, edit=AsyncMock(), embeds=[])
    sent = {}

    async def send(**kwargs):
        sent.update(kwargs)
        message.embeds = [kwargs["embed"]]
        return message

    channel = SimpleNamespace(send=send, get_partial_message=lambda _id: message)
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    await adapter.send_exec_approval(
        chat_id="555", command="ls", session_key=session,
        request_id="request-1",
    )
    return adapter, sent["view"], entry, message


@pytest.mark.asyncio
@pytest.mark.parametrize("edit_fails", [False, True])
async def test_click_stops_view_even_if_interaction_edit_fails(monkeypatch, edit_fails):
    adapter, view, entry, message = await _pending_card(monkeypatch)
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    edit = AsyncMock(side_effect=RuntimeError("Discord unavailable") if edit_fails else None)
    interaction = SimpleNamespace(
        message=message, user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(edit_message=edit),
    )
    if edit_fails:
        with pytest.raises(RuntimeError, match="Discord unavailable"):
            await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")
    else:
        await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")
    assert entry.result == "once"
    assert view.is_finished()
    footer = message.embeds[0].footer
    entry.observers[0]("resolved")
    await asyncio.sleep(0)
    await asyncio.gather(*adapter._background_tasks)
    assert view.is_finished()
    if edit_fails:
        # The recorded answer must not leave a pending-looking card: it is stamped through the message.
        message.edit.assert_awaited_once()
        assert message.edit.await_args.kwargs["view"] is None
        footer = message.embeds[0].footer
        assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
            "platform.discord.approval.already_resolved")
    else:
        assert message.embeds[0].footer == footer
        message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_observer_that_fires_before_a_failed_click_edit_still_sees_the_card_retired(monkeypatch):
    """The waiter usually wakes before Discord rejects the edit, so the observer runs while the click
    still owns the card and skips it; the click itself must then retire the card."""
    adapter, view, entry, message = await _pending_card(monkeypatch)
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))

    async def edit_message(**_kwargs):
        entry.observers[0]("resolved")
        await asyncio.sleep(0)
        await asyncio.gather(*adapter._background_tasks)
        message.edit.assert_not_awaited()  # the observer skipped: the click owned the edit
        raise RuntimeError("Discord unavailable")

    interaction = SimpleNamespace(
        message=message, user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(edit_message=edit_message))

    with pytest.raises(RuntimeError, match="Discord unavailable"):
        await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")

    message.edit.assert_awaited_once()
    assert message.edit.await_args.kwargs["view"] is None
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
        "platform.discord.approval.already_resolved")


@pytest.mark.asyncio
@pytest.mark.parametrize("ui_timeout_first", [True, False])
async def test_ui_timeout_does_not_end_wait_or_overwrite_later_answer(monkeypatch, ui_timeout_first):
    _adapter, view, entry, message = await _pending_card(monkeypatch)
    if ui_timeout_first:
        await view.on_timeout()
        assert not entry.event.is_set()
        assert entry.result is None
        assert approval._gateway_queues[view.session_key] == [entry]
    assert approval.resolve_gateway_approval(view.session_key, "once", request_id=view.request_id) == 1
    await view.retire("resolved")
    if not ui_timeout_first:
        await view.on_timeout()
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
        "platform.discord.approval.already_resolved")
    assert view.is_finished()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,footer_key", [
    ("resolved", "platform.discord.approval.already_resolved"),
    ("timeout", "platform.discord.prompt.expired_footer"),
    ("interrupted", "platform.discord.slash_confirm.resolved_cancel"),
    ("session_closed", "platform.discord.slash_confirm.resolved_cancel"),
    ("notify_failed", "platform.discord.slash_confirm.resolved_cancel"),
])
async def test_external_settlement_retires_card_and_tracks_edit(monkeypatch, reason, footer_key):
    adapter, view, entry, message = await _pending_card(monkeypatch)
    pending_color = message.embeds[0].color
    entered, release = asyncio.Event(), asyncio.Event()

    async def edit(**_kwargs):
        entered.set()
        await release.wait()

    message.edit.side_effect = edit
    entry.observers[0](reason)
    await asyncio.wait_for(entered.wait(), 5)
    assert view.is_finished()
    assert len(adapter._background_tasks) == 1
    task = next(iter(adapter._background_tasks))
    kwargs = message.edit.await_args.kwargs
    assert kwargs["view"] is None
    embed = kwargs["embed"]
    footer = embed.footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(footer_key)
    assert embed.color != pending_color
    release.set()
    await task
    await asyncio.sleep(0)
    assert not adapter._background_tasks


@pytest.mark.asyncio
async def test_inflight_ui_expiry_cannot_overwrite_backend_answer(monkeypatch):
    _adapter, view, _entry, message = await _pending_card(monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    displayed = []

    async def edit(**kwargs):
        footer = kwargs["embed"].footer
        text = footer["text"] if isinstance(footer, dict) else footer.text
        if not entered.is_set():
            entered.set()
            await release.wait()
        displayed.append(text)

    message.edit.side_effect = edit
    ui_task = asyncio.create_task(view.on_timeout())
    await asyncio.wait_for(entered.wait(), 5)
    backend_task = asyncio.create_task(view.retire("resolved"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(ui_task, backend_task)
    assert displayed == [t("platform.discord.prompt.expired_footer"),
                         t("platform.discord.approval.already_resolved")]


@pytest.mark.asyncio
@pytest.mark.parametrize("notice_last", [False, True])
async def test_timeout_edits_cannot_restore_buttons_or_pending_embed(monkeypatch, notice_last):
    adapter, view, _entry, message = await _pending_card(monkeypatch)
    pending_color = message.embeds[0].color
    entered = {name: asyncio.Event() for name in ("notice", "retire")}
    release = {name: asyncio.Event() for name in entered}
    state = {"content": "command", "view": view, "embed": message.embeds[0]}

    async def edit(**kwargs):
        name = "notice" if "content" in kwargs else "retire"
        entered[name].set()
        await release[name].wait()
        state.update(kwargs)

    message.edit.side_effect = edit
    notice_task = asyncio.create_task(adapter.expire_exec_approval("555", "1234", "Timed out"))
    retire_task = asyncio.create_task(view.retire("timeout"))
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 5)
    first, last = ("retire", "notice") if notice_last else ("notice", "retire")
    tasks = {"notice": notice_task, "retire": retire_task}
    release[first].set()
    await tasks[first]
    release[last].set()
    await tasks[last]
    assert state["content"] == "Timed out"
    assert state["view"] is None
    assert state["embed"].color != pending_color
    assert view.is_finished()


@pytest.mark.asyncio
async def test_external_answer_then_stale_click_preserves_resolved_observer(monkeypatch):
    """A no-op stale click must not suppress the externally resolved observer."""
    adapter = discord_adapter.DiscordAdapter(PlatformConfig(enabled=True, token="test-token"))
    monkeypatch.setattr(discord_adapter.discord.Color, "dark_grey", lambda: 8, raising=False)
    adapter._allowed_user_ids = {"42"}
    session = "agent:other:discord:555"
    entry = _ApprovalEntry({"request_id": "request-1", "command": "ls"})
    monkeypatch.setitem(approval._gateway_queues, session, [entry])
    message = SimpleNamespace(id=1234, edit=AsyncMock(), embeds=[])
    sent = {}

    async def send(**kwargs):
        sent.update(kwargs)
        message.embeds = [kwargs["embed"]]
        return message

    channel = SimpleNamespace(send=send, get_partial_message=lambda _id: message)
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel)
    await adapter.send_exec_approval(
        chat_id="555", command="ls", session_key=session,
        request_id="request-1",
    )
    view = sent["view"]

    async def interaction_edit(**kwargs):
        message.embeds = [kwargs["embed"]]

    interaction = SimpleNamespace(
        message=message,
        user=SimpleNamespace(id=42, display_name="tester"),
        response=SimpleNamespace(
            edit_message=AsyncMock(side_effect=interaction_edit),
            send_message=AsyncMock(),
        ),
    )

    assert approval.resolve_gateway_approval(session, "once", request_id="request-1") == 1
    assert entry.result == "once"
    entry.observers[0]("resolved")  # the waiter thread schedules the resolved observer
    assert not view.resolved, "the resolved observer has not run; the old button is still active"

    await view._resolve(interaction, "once", 2, "platform.discord.approval.resolved_once")
    await asyncio.sleep(0)  # let the observer's call_soon_threadsafe callback enqueue retirement
    await asyncio.gather(*adapter._background_tasks)

    footer = message.embeds[0].footer
    footer_text = footer["text"] if isinstance(footer, dict) else footer.text
    assert footer_text == t("platform.discord.approval.already_resolved")
    # The stale click only answers its clicker; the card edit belongs to the observer, so the
    # two edits can't race and leave "command was not run" on top.
    interaction.response.edit_message.assert_not_awaited()
    interaction.response.send_message.assert_awaited_once_with(
        t("platform.discord.approval.no_longer_pending"), ephemeral=True)


@pytest.mark.asyncio
async def test_failed_click_keeps_a_pending_card_live(monkeypatch):
    """A click whose lookup raises recorded nothing: the card stays clickable, only the clicker
    hears about it, and the request's observer still settles the card later."""
    adapter, view, entry, message = await _pending_card(monkeypatch)
    session = view.session_key
    monkeypatch.setattr(view, "_gate", AsyncMock(return_value=True))
    real_resolve = approval.resolve_gateway_approval
    monkeypatch.setattr(approval, "resolve_gateway_approval", lambda *a, **k: 1 / 0)
    interaction = SimpleNamespace(
        user=SimpleNamespace(display_name="tester"),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock()))

    await view._resolve(interaction, "once", 1, "gateway.exec_approval.action_once")

    interaction.response.send_message.assert_awaited_once_with(
        t("platform.discord.approval.click_failed"), ephemeral=True)
    interaction.response.edit_message.assert_not_awaited()
    assert not view.resolved and not view.is_finished()
    assert approval._gateway_queues[session] == [entry]  # still pending, nothing recorded

    assert real_resolve(session, "once", request_id="request-1") == 1  # answered by text later
    entry.observers[0]("resolved")
    await asyncio.sleep(0)
    await asyncio.gather(*adapter._background_tasks)
    footer = message.embeds[0].footer
    assert (footer["text"] if isinstance(footer, dict) else footer.text) == t(
        "platform.discord.approval.already_resolved")
    assert view.is_finished()
