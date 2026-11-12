"""A QQBot approval card resolves the request it was rendered for, not the session's oldest (#124974).

Drives the real ``hermes_agent.tools.approval`` queue through ``send_exec_approval``'s keyboard and the adapter's
``_default_interaction_dispatch``; nothing on the approval side is faked.
"""

import pytest

from hermes_agent.gateway.config import PlatformConfig
from plugins.platforms.qqbot.keyboards import build_approval_keyboard, parse_approval_button_data, parse_interaction_event

SESSION = "agent:main:qqbot:dm:u-42"


def _adapter():
    from plugins.platforms.qqbot.adapter import QQAdapter
    return QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))


def _queue(n):
    """``n`` pending exec approvals in SESSION, each on its own waiting thread; returns their entries."""
    from hermes_agent.tools import approval
    entries = []
    for i in range(n):
        from hermes_agent.tools.approval_gateway_wait import _ApprovalEntry
        entry = _ApprovalEntry({"command": f"cmd-{i}", "description": "d"})
        with approval._lock:
            approval._gateway_queues.setdefault(SESSION, []).append(entry)
        entries.append(entry)
    return entries


def _tap(button_data):
    return parse_interaction_event({"id": "i", "chat_type": 2, "user_openid": "u-42",
                                    "data": {"resolved": {"button_data": button_data}}})


async def _sent_card(adapter, request_id):
    """The keyboard ``send_exec_approval`` hands to the QQ transport."""
    sent = []

    async def capture(chat_id, content, keyboard, reply_to=None):
        sent.append(keyboard)
        from hermes_agent.gateway.platforms.base import SendResult
        return SendResult(success=True, message_id="m1")

    adapter.send_with_keyboard = capture
    await adapter.send_exec_approval("u-42", "rm -rf x", SESSION, request_id=request_id)
    return sent[0]


@pytest.mark.asyncio
async def test_tap_on_the_newest_card_resolves_the_newest_request():
    from hermes_agent.tools import approval
    old, new = _queue(2)
    try:
        adapter = _adapter()
        card = await _sent_card(adapter, new.data["request_id"])
        allow = card.content.rows[0].buttons[0].action.data

        await adapter._default_interaction_dispatch(_tap(allow))

        assert new.event.is_set() and new.result == "once"
        assert not old.event.is_set(), "the tap answered the session's oldest approval instead"
    finally:
        with approval._lock:
            approval._gateway_queues.pop(SESSION, None)


@pytest.mark.asyncio
async def test_card_of_a_settled_request_resolves_nothing():
    from hermes_agent.tools import approval
    (pending,) = _queue(1)
    try:
        stale = build_approval_keyboard(SESSION, request_id="0" * 32).content.rows[0].buttons[0].action.data
        await _adapter()._default_interaction_dispatch(_tap(stale))
        assert not pending.event.is_set()
    finally:
        with approval._lock:
            approval._gateway_queues.pop(SESSION, None)


def test_legacy_card_whose_session_ends_in_hex_keeps_its_whole_session():
    """QQ openids are 32 hex: a card without a request id must not have its session tail read as one."""
    from plugins.platforms.qqbot.keyboards import approval_button_request_id
    hex_session = "agent:main:qqbot:dm:" + "ab" * 16
    assert parse_approval_button_data(f"approve:{hex_session}:deny") == (hex_session, "deny")
    assert approval_button_request_id(f"approve:{hex_session}:deny") is None
