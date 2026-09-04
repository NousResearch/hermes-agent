"""Real Matrix reaction ingress with fake message/reaction APIs, no live room."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.choice_picker import ChoicePage, ChoiceProgress
from gateway.platforms.base import SendResult
from plugins.platforms.matrix.choice_picker import (
    cancel_choice_pages,
    expire_choice_page,
)
from tests.gateway.test_matrix import _make_adapter


def page(value="older", count=1):
    return ChoicePage(
        "Files",
        [{"label": "資料" * 100, "value": f"{value}-{i}"} for i in range(count)],
    )


@pytest.fixture
def adapter():
    a = _make_adapter()
    a._client = object()
    a._allowed_user_ids = {"@owner:example.org", "@other:example.org"}
    a._approval_require_sender = False
    a._is_self_sender = lambda sender: False
    a._is_duplicate_event = lambda event_id: False
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append((chat_id, content, kwargs))
        return SendResult(success=True, message_id=f"$message-{len(sent)}")

    a.send = AsyncMock(side_effect=send)
    a._send_reaction = AsyncMock(
        side_effect=lambda *args: f"$reaction-{a._send_reaction.await_count}"
    )
    a.redact_message = AsyncMock(return_value=SendResult(success=True))
    a.send_read_receipt = AsyncMock(return_value=True)
    a.edit_message = AsyncMock(return_value=SendResult(success=True))
    yield a
    cancel_choice_pages(a)


async def send(adapter, callback, *, reusable=True, count=1):
    p = page(count=count)
    result = await adapter.send_choice_picker(
        "!room:example.org",
        p.title,
        p.choices,
        "session",
        callback,
        {
            "choice_pages": reusable,
            "requester_user_id": "@owner:example.org",
            "thread_id": "$thread",
        },
    )
    assert result.success
    return result.message_id


def reaction(
    adapter, target, *, sender="@owner:example.org", room="!room:example.org", key=None
):
    prompt = adapter._choice_picker_prompts_by_event.get(target)
    return SimpleNamespace(
        sender=sender,
        event_id="$tap",
        room_id=room,
        content={
            "m.relates_to": {
                "event_id": target,
                "key": key or next(iter(prompt.choices)),
            }
        },
    )


@pytest.mark.asyncio
async def test_next_back_are_fresh_messages_and_old_slots_do_not_retarget(adapter):
    callback = AsyncMock(side_effect=[page("newer"), page("cancel"), "Cancelled"])
    first = await send(adapter, callback)
    original_deadline = adapter._choice_picker_prompts_by_event[first].expires_at
    old = reaction(adapter, first)
    await adapter._on_reaction(old)
    second = next(iter(adapter._choice_picker_prompts_by_event))
    assert second != first
    assert (
        adapter._choice_picker_prompts_by_event[second].expires_at == original_deadline
    )
    await adapter._on_reaction(old)
    assert callback.await_count == 1
    await adapter._on_reaction(reaction(adapter, second))
    third = next(iter(adapter._choice_picker_prompts_by_event))
    await adapter._on_reaction(reaction(adapter, third))
    assert [call.args[1] for call in callback.await_args_list] == [
        "older-0",
        "newer-0",
        "cancel-0",
    ]
    assert not adapter._choice_picker_prompts_by_event
    adapter.edit_message.assert_not_awaited()
    assert adapter.send.await_args.args[1] == "Cancelled"
    assert all(
        call.kwargs["metadata"]["thread_id"] == "$thread"
        for call in adapter.send.await_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope", [{"sender": "@other:example.org"}, {"room": "!other:example.org"}]
)
async def test_pages_preserve_requester_even_when_approval_sender_setting_is_off(
    adapter, scope
):
    callback = AsyncMock(return_value=page())
    target = await send(adapter, callback)
    await adapter._on_reaction(reaction(adapter, target, **scope))
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_claim_precedes_async_auth_and_read_receipt_precedes_work(adapter):
    entered, release = asyncio.Event(), asyncio.Event()

    async def auth(*args):
        entered.set()
        await release.wait()
        return True

    adapter._validate_matrix_prompt_reactor = AsyncMock(side_effect=auth)

    async def callback(*args):
        adapter.send_read_receipt.assert_awaited_once()
        return "Sent"

    consumer = AsyncMock(side_effect=callback)
    target = await send(adapter, consumer)
    event = reaction(adapter, target)
    task = asyncio.create_task(adapter._on_reaction(event))
    await entered.wait()
    await adapter._on_reaction(event)
    assert adapter._validate_matrix_prompt_reactor.await_count == 1
    release.set()
    await task
    consumer.assert_awaited_once()


@pytest.mark.asyncio
async def test_expiry_gives_feedback_and_removes_reactions(adapter):
    callback = AsyncMock(return_value=page())
    target = await send(adapter, callback)
    event = reaction(adapter, target)
    adapter._choice_picker_prompts_by_event[target].expires_at = time.monotonic() - 1
    await adapter._on_reaction(event)
    assert not adapter._choice_picker_prompts_by_event
    assert "expired" in adapter.send.await_args.args[1]
    assert adapter.redact_message.await_count == 1
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_twelve_slots_include_navigation_and_thirteen_refuses(adapter):
    target = await send(adapter, AsyncMock(), count=12)
    assert len(adapter._choice_picker_prompts_by_event[target].choices) == 12
    assert adapter._send_reaction.await_count == 12
    lines = adapter.send.await_args.args[1].splitlines()[2:-2]
    assert len(lines) == 12 and all(len(line) < 120 for line in lines)
    result = await adapter.send_choice_picker(
        "!room",
        "Files",
        [{"value": str(i)} for i in range(13)],
        "s",
        AsyncMock(),
        {"choice_pages": True, "requester_user_id": "@owner"},
    )
    assert not result.success


@pytest.mark.asyncio
async def test_progress_uses_short_messages_before_and_after_delivery(adapter):
    async def deliver():
        assert adapter.send.await_args.args[1] == "Sending"
        return "Sent"

    target = await send(
        adapter, AsyncMock(return_value=ChoiceProgress("Sending", deliver))
    )
    await adapter._on_reaction(reaction(adapter, target))
    assert [call.args[1] for call in adapter.send.await_args_list][1:] == [
        "Sending",
        "Sent",
    ]
    adapter.edit_message.assert_not_awaited()
    assert not adapter._choice_picker_prompts_by_event


@pytest.mark.asyncio
async def test_one_shot_keeps_existing_confirmation_and_consumes_prompt(adapter):
    callback = AsyncMock(return_value="Setting applied")
    target = await send(adapter, callback, reusable=False)
    event = reaction(adapter, target)
    await adapter._on_reaction(event)
    await adapter._on_reaction(event)
    callback.assert_awaited_once_with("!room:example.org", "older-0")
    assert adapter.send.await_args.args[1] == "Setting applied"
    assert adapter.send.await_args.kwargs == {"reply_to": target}


@pytest.mark.asyncio
async def test_expiry_while_callback_waits_does_not_open_late_page(adapter):
    entered, release = asyncio.Event(), asyncio.Event()

    async def callback(*args):
        entered.set()
        await release.wait()
        return page("late")

    target = await send(adapter, callback)
    prompt = adapter._choice_picker_prompts_by_event[target]
    task = asyncio.create_task(adapter._on_reaction(reaction(adapter, target)))
    await entered.wait()
    await expire_choice_page(adapter, prompt)
    release.set()
    await task
    assert not adapter._choice_picker_prompts_by_event
    assert adapter.send.await_count == 2
    assert "expired" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
async def test_one_shot_invalid_reaction_keeps_existing_feedback(adapter):
    callback = AsyncMock(return_value="Done")
    target = await send(adapter, callback, reusable=False)
    await adapter._on_reaction(reaction(adapter, target, key="invalid"))
    assert "not one of the available choices" in adapter.send.await_args.args[1]
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_callback_clears_reactions_and_state(adapter):
    entered = asyncio.Event()

    async def callback(*args):
        entered.set()
        await asyncio.Event().wait()

    target = await send(adapter, callback)
    task = asyncio.create_task(adapter._on_reaction(reaction(adapter, target)))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not adapter._choice_picker_prompts_by_event
    assert adapter.redact_message.await_count == 1


@pytest.mark.asyncio
async def test_failed_progress_send_never_starts_work(adapter):
    complete = AsyncMock(return_value="Sent")
    target = await send(
        adapter, AsyncMock(return_value=ChoiceProgress("Sending", complete))
    )
    event = reaction(adapter, target)
    adapter.send.side_effect = [
        SendResult(success=False),
        SendResult(success=True, message_id="$error"),
    ]
    await adapter._on_reaction(event)
    complete.assert_not_awaited()
    assert not adapter._choice_picker_prompts_by_event


@pytest.mark.asyncio
async def test_page_labels_are_inert_but_callback_value_is_unchanged(adapter):
    label = "/group @room [report](https://example.org) **bold**"
    callback = AsyncMock(return_value="Done")
    result = await adapter.send_choice_picker(
        "!room:example.org",
        "Files",
        [{"label": label, "value": label}],
        "session",
        callback,
        {"choice_pages": True, "requester_user_id": "@owner:example.org"},
    )
    content = adapter.send.await_args.args[1]
    assert "@room" not in content and "/group" not in content
    assert "\\[report\\]" in content
    adapter._allow_room_mentions = True
    rendered = adapter._build_text_message_content(content)
    assert "m.mentions" not in rendered
    await adapter._on_reaction(reaction(adapter, result.message_id))
    callback.assert_awaited_once_with("!room:example.org", label)


@pytest.mark.asyncio
async def test_page_must_fit_one_message_instead_of_splitting_reaction_slots(adapter):
    adapter.max_message_length = 500
    p = page(count=12)
    result = await adapter.send_choice_picker(
        "!room:example.org",
        p.title,
        p.choices,
        "session",
        AsyncMock(),
        {"choice_pages": True, "requester_user_id": "@owner:example.org"},
    )
    assert not result.success
    adapter.send.assert_not_awaited()
    assert not adapter._choice_picker_prompts_by_event


@pytest.mark.asyncio
async def test_expired_page_cannot_start_work_after_read_receipt(adapter):
    entered, release = asyncio.Event(), asyncio.Event()
    callback = AsyncMock(return_value="Done")
    target = await send(adapter, callback)
    prompt = adapter._choice_picker_prompts_by_event[target]

    async def receipt(*args):
        entered.set()
        await release.wait()
        return True

    adapter.send_read_receipt.side_effect = receipt
    task = asyncio.create_task(adapter._on_reaction(reaction(adapter, target)))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await expire_choice_page(adapter, prompt)
        callback.assert_not_awaited()
    finally:
        release.set()
        await asyncio.wait_for(task, 2)
    callback.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [False, True])
async def test_cancelled_seed_cleans_the_new_prompt_and_known_reactions(
    adapter, replacement
):
    entered, release = asyncio.Event(), asyncio.Event()
    if replacement:
        target = await send(adapter, AsyncMock(return_value=page(count=2)))
    calls = 0

    async def seed(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "$known-seed"
        entered.set()
        await release.wait()
        return "$late-seed"

    adapter._send_reaction.side_effect = seed
    operation = (
        adapter._on_reaction(reaction(adapter, target))
        if replacement
        else send(adapter, AsyncMock(), count=2)
    )
    task = asyncio.create_task(operation)
    await asyncio.wait_for(entered.wait(), 2)
    prompts = list(adapter._choice_picker_prompts_by_event.values())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert not adapter._choice_picker_prompts_by_event
    assert all(prompt.resolved and prompt.expiry_handle is None for prompt in prompts)
    assert any(
        call.args[1] == "$known-seed" for call in adapter.redact_message.await_args_list
    )
