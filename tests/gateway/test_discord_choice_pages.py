"""Real Discord picker/view behavior with a fake channel and interaction API."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.choice_picker import ChoicePage, ChoiceProgress
from plugins.platforms.discord import adapter as module


def page(value="older"):
    return ChoicePage(
        "Files", [{"label": "report.pdf · Writer · 06:41 · 210 KB", "value": value}]
    )


async def picker(callback, *, reusable=True, choices=None):
    a = object.__new__(module.DiscordAdapter)
    message = SimpleNamespace(id=10, edit=AsyncMock())
    channel = SimpleNamespace(send=AsyncMock(return_value=message))
    a._client = SimpleNamespace(
        get_channel=lambda target: channel,
        fetch_channel=AsyncMock(return_value=channel),
    )
    a._allowed_user_ids = {"1", "2"}
    a._allowed_role_ids = set()
    result = await a.send_choice_picker(
        "100",
        "Files",
        choices or page().choices,
        "session",
        callback,
        {"choice_pages": reusable, "requester_user_id": "1", "thread_id": "101"},
    )
    assert result.success
    view = channel.send.await_args.kwargs["view"]
    view.stop = Mock()
    return view, channel


def interaction(view, action=None, *, user=1, message_id=10, channel_id=101):
    return SimpleNamespace(
        user=SimpleNamespace(id=user, roles=[]),
        message=SimpleNamespace(id=message_id),
        channel_id=channel_id,
        data={"values": [action or view.children[0].options[0].value]},
        response=SimpleNamespace(defer=AsyncMock(), send_message=AsyncMock()),
        edit_original_response=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_next_back_and_stale_revision_keep_original_message():
    callback = AsyncMock(side_effect=[page("newer"), page("cancel"), "Cancelled"])
    view, channel = await picker(callback)
    old = view.children[0].options[0].value
    first = interaction(view)
    await view._on_select(first)
    first.response.defer.assert_awaited_once()
    assert not view.resolved
    assert view.children[0].options[0].value != old
    await view._on_select(interaction(view, old))
    assert callback.await_count == 1
    await view._on_select(interaction(view))
    last = interaction(view)
    await view._on_select(last)
    assert [call.args for call in callback.await_args_list] == [
        ("101", "older"),
        ("101", "newer"),
        ("101", "cancel"),
    ]
    assert view.resolved and not view.children
    assert (
        last.edit_original_response.await_args.kwargs["embed"].description
        == "Cancelled"
    )
    assert channel.send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{"user": 2}, {"message_id": 9}, {"channel_id": 100}])
async def test_exact_requester_message_and_thread(scope):
    callback = AsyncMock(return_value=page())
    view, _ = await picker(callback)
    await view._on_select(interaction(view, **scope))
    callback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reusable", [False, True])
async def test_ack_and_busy_claim_precede_callback(reusable):
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(*args):
        first.response.defer.assert_awaited_once()
        entered.set()
        await release.wait()
        return "Done"

    view, _ = await picker(callback, reusable=reusable)
    first = interaction(view)
    action = first.data["values"][0]
    task = asyncio.create_task(view._on_select(first))
    await entered.wait()
    duplicate = interaction(view, action)
    await view._on_select(duplicate)
    duplicate.response.defer.assert_awaited_once()
    duplicate.edit_original_response.assert_not_awaited()
    release.set()
    await task
    assert view.resolved


@pytest.mark.asyncio
async def test_timeout_during_callback_discards_late_page():
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(*args):
        entered.set()
        await release.wait()
        return page("late")

    view, _ = await picker(callback)
    first = interaction(view)
    task = asyncio.create_task(view._on_select(first))
    await entered.wait()
    await view.on_timeout()
    release.set()
    await task
    assert view.resolved and not view.children
    first.edit_original_response.assert_not_awaited()
    assert "expired" in view._message.edit.await_args.kwargs["embed"].description


@pytest.mark.asyncio
async def test_expired_tap_never_calls_consumer():
    callback = AsyncMock(return_value=page())
    view, _ = await picker(callback)
    tap = interaction(view)
    view.expires_at = time.monotonic() - 1
    await view._on_select(tap)
    callback.assert_not_awaited()
    assert view.resolved


@pytest.mark.asyncio
async def test_progress_is_edited_before_delivery_and_then_sent():
    async def deliver():
        assert (
            tap.edit_original_response.await_args.kwargs["embed"].description
            == "Sending"
        )
        assert not view.children
        return "Sent"

    view, _ = await picker(AsyncMock(return_value=ChoiceProgress("Sending", deliver)))
    tap = interaction(view)
    await view._on_select(tap)
    assert [
        call.kwargs["embed"].description
        for call in tap.edit_original_response.await_args_list
    ] == ["Sending", "Sent"]
    assert view.resolved


@pytest.mark.asyncio
async def test_names_fit_but_values_are_not_truncated():
    value = "x" * 900
    callback = AsyncMock(return_value="Done")
    view, _ = await picker(callback, choices=[{"label": "研究" * 100, "value": value}])
    assert len(view.children[0].options[0].label) <= 100
    assert len(view.children[0].options[0].value) <= 100
    await view._on_select(interaction(view))
    callback.assert_awaited_once_with("101", value)


@pytest.mark.asyncio
async def test_invalid_payload_does_not_claim_picker():
    callback = AsyncMock(return_value="Done")
    view, _ = await picker(callback)
    tap = interaction(view)
    tap.data = {"values": []}
    await view._on_select(tap)
    assert not view.busy and not view.resolved
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_stops_view_and_fixed_expiry_timer():
    entered = asyncio.Event()

    async def callback(*args):
        entered.set()
        await asyncio.Event().wait()

    view, _ = await picker(callback)
    handle = view._expiry_handle
    task = asyncio.create_task(view._on_select(interaction(view)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert view.resolved and not view.children and handle.cancelled()


@pytest.mark.asyncio
async def test_failed_progress_edit_never_starts_work():
    complete = AsyncMock(return_value="Sent")
    view, _ = await picker(AsyncMock(return_value=ChoiceProgress("Sending", complete)))
    tap = interaction(view)
    tap.edit_original_response.side_effect = [RuntimeError("API failed"), None]
    await view._on_select(tap)
    complete.assert_not_awaited()
    assert view.resolved and not view.children


@pytest.mark.asyncio
async def test_fixed_expiry_fires_without_another_interaction(monkeypatch):
    from plugins.platforms.discord import choice_picker

    monkeypatch.setattr(choice_picker, "PAGE_TIMEOUT_SECONDS", 0.01)
    view, _ = await picker(AsyncMock(return_value=page()))
    expired = asyncio.Event()
    view._message.edit.side_effect = lambda **kwargs: expired.set()
    await asyncio.wait_for(expired.wait(), timeout=2)
    assert view.resolved and not view.children


@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["timeout", "sdk-stop"])
async def test_expired_view_cannot_start_work_after_delayed_ack(retirement):
    entered, release = asyncio.Event(), asyncio.Event()
    callback = AsyncMock(return_value="Done")
    view, _ = await picker(callback)
    event = interaction(view)

    async def delayed_ack():
        entered.set()
        await release.wait()

    event.response.defer.side_effect = delayed_ack
    task = asyncio.create_task(view._on_select(event))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if retirement == "timeout":
            await view.on_timeout()
        else:
            view.is_finished = lambda: True
        callback.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
        await view.on_timeout()
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_journey_deadline_starts_after_send_success(monkeypatch):
    from plugins.platforms.discord import choice_picker

    clock = [100.0]
    monkeypatch.setattr(
        choice_picker, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    adapter = object.__new__(module.DiscordAdapter)
    message = SimpleNamespace(id=10, edit=AsyncMock())

    async def slow_send(**kwargs):
        clock[0] = 160.0
        return message

    channel = SimpleNamespace(send=AsyncMock(side_effect=slow_send))
    adapter._client = SimpleNamespace(get_channel=lambda _target: channel)
    adapter._allowed_user_ids = {"1"}
    adapter._allowed_role_ids = set()
    result = await adapter.send_choice_picker(
        "100",
        "Files",
        page().choices,
        "session",
        AsyncMock(return_value="Done"),
        {"choice_pages": True, "requester_user_id": "1"},
    )
    assert result.success
    view = channel.send.await_args.kwargs["view"]
    view.stop = Mock()
    try:
        assert view.expires_at == 160.0 + choice_picker.PAGE_TIMEOUT_SECONDS
    finally:
        await view.on_timeout()
