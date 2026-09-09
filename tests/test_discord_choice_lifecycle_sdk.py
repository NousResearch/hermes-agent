"""Exercise Discord's real view retirement independently of gateway SDK doubles."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.choice_picker import ChoiceProgress
from plugins.platforms.discord.choice_picker import define_choice_picker_view

discord = pytest.importorskip("discord")


def make_view(callback, *, reusable=True):
    view_type = define_choice_picker_view(
        discord_sdk=discord,
        component_check_auth=lambda *args: True,
        truncate_component_text=lambda text, limit: text[:limit],
        logger=logging.getLogger(__name__),
        max_options=25,
        field_limit=100,
    )
    view = view_type(
        [{"label": "Get file", "value": "get"}],
        callback,
        {"1"},
        requester_user_id="1",
        reusable=reusable,
        channel_id="101",
    )
    view._message = SimpleNamespace(id=10, edit=AsyncMock())
    if reusable:
        view.arm_expiry()
    interaction = SimpleNamespace(
        user=SimpleNamespace(id=1),
        message=SimpleNamespace(id=10),
        channel_id=101,
        data={"values": [view.children[0].options[0].value]},
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        edit_original_response=AsyncMock(),
    )
    return view, interaction


@pytest.mark.asyncio
@pytest.mark.parametrize("reusable", [False, True])
async def test_real_sdk_one_shot_and_reusable_selection(reusable):
    callback = AsyncMock(return_value="Done")
    view, interaction = make_view(callback, reusable=reusable)
    try:
        await view._on_select(interaction)
        callback.assert_awaited_once_with("101", "get")
        assert view.resolved and view.is_finished()
    finally:
        view._stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["stop", "timeout-dispatch"])
async def test_sdk_retirement_before_deferred_work_refuses_entry(retirement):
    complete = AsyncMock(return_value="Sent")
    callback = AsyncMock(return_value=ChoiceProgress("Sending", complete))
    view, interaction = make_view(callback)
    entered, release = asyncio.Event(), asyncio.Event()

    async def feedback(**kwargs):
        entered.set()
        await release.wait()

    interaction.edit_original_response.side_effect = feedback
    task = asyncio.create_task(view._on_select(interaction))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        callback.assert_awaited_once()
        complete.assert_not_awaited()
        if retirement == "stop":
            view.stop()
            release.set()
        else:
            # SDK retirement precedes its queued async on_timeout callback.
            release.set()
            view._dispatch_timeout()
        assert view.is_finished() and not view.resolved
        await asyncio.wait_for(task, 2)
        complete.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
        view._stop()


@pytest.mark.asyncio
async def test_late_progress_edit_restores_completed_expiry_feedback():
    complete = AsyncMock(return_value="Sent")
    view, interaction = make_view(
        AsyncMock(return_value=ChoiceProgress("Sending", complete))
    )
    entered, release, expired = asyncio.Event(), asyncio.Event(), asyncio.Event()
    visible = []

    async def feedback(**kwargs):
        entered.set()
        await release.wait()
        visible.append(kwargs["embed"].description)

    async def expired_feedback(**kwargs):
        visible.append(kwargs["embed"].description)
        expired.set()

    interaction.edit_original_response.side_effect = feedback
    view._message.edit.side_effect = expired_feedback
    task = asyncio.create_task(view._on_select(interaction))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        view._dispatch_timeout()
        await asyncio.wait_for(expired.wait(), 5)
        assert view.resolved and view.is_finished()
        assert "expired" in visible[-1].lower()
        release.set()
        await asyncio.wait_for(task, 5)
        complete.assert_not_awaited()
        assert "expired" in visible[-1].lower()
        assert "Sending" in visible
    finally:
        release.set()
        await asyncio.wait_for(task, 5)
        view._stop()
