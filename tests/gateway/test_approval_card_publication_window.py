"""A card tap can approve only a request whose card was rendered for it: an id-less tap (a card from an
older build) settles nothing, and a freshly queued request is Deny-only from the moment it is enqueued,
so no tap path can approve it in the window before its renderer has decided what the card may offer."""

import asyncio
import threading

import pytest

from hermes_agent.gateway.config import PlatformConfig

SK = "agent:main:qqbot:dm:u-42"


def _qq_tap(button_data):
    from plugins.platforms.qqbot.adapter import QQAdapter
    from plugins.platforms.qqbot.keyboards import parse_interaction_event

    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))
    asyncio.run(adapter._default_interaction_dispatch(parse_interaction_event(
        {"id": "i", "chat_type": 2, "user_openid": "u-42", "data": {"resolved": {"button_data": button_data}}})))


def _publish_and_hold():
    """A real gateway wait whose notifier is held before any rendering: the request is published."""
    from tools import approval
    from tools.approval_gateway_wait import _await_gateway_decision

    entered, release, out = threading.Event(), threading.Event(), {}

    def notify(data):
        out["request_id"] = data["request_id"]
        entered.set()
        assert release.wait(10)

    def wait():
        out["decision"] = _await_gateway_decision(SK, notify, {"command": "rm -rf /tmp/x"})

    thread = threading.Thread(target=wait)
    thread.start()
    assert entered.wait(10)
    return approval, thread, release, out


def _finish(approval, thread, release, out):
    if approval.has_blocking_approval(SK):
        approval.resolve_gateway_approval(SK, "deny", request_id=out["request_id"])
    release.set()
    thread.join(12)
    assert not thread.is_alive()


def test_an_old_id_less_card_tap_never_approves_a_newly_published_request():
    approval, thread, release, out = _publish_and_hold()
    try:
        _qq_tap(f"approve:{SK}:allow-once")  # the legacy card format: no request id
        assert approval.list_gateway_approvals(SK), "an id-less tap settled the new request"
    finally:
        _finish(approval, thread, release, out)
    assert out["decision"]["choice"] == "deny"


def test_a_published_request_is_deny_only_before_its_card_is_rendered():
    approval, thread, release, out = _publish_and_hold()
    try:
        (pending,) = approval._gateway_queues[SK]
        assert pending.card_deny_only is True
        assert approval.resolve_gateway_approval_card(SK, "once", out["request_id"]) == 0
    finally:
        _finish(approval, thread, release, out)


@pytest.mark.asyncio
async def test_a_card_that_shows_the_whole_command_lifts_the_mark(monkeypatch):
    from hermes_agent.gateway.platforms.base import SendResult
    from plugins.platforms.qqbot.adapter import QQAdapter
    from tools import approval
    from tools.approval_gateway_wait import _ApprovalEntry

    entry = _ApprovalEntry({"request_id": "r1"})
    entry.card_deny_only = True  # as every request is created
    monkeypatch.setitem(approval._gateway_queues, SK, [entry])
    adapter = QQAdapter(PlatformConfig(enabled=True, extra={"app_id": "a", "client_secret": "b"}))

    async def send_with_keyboard(*_a, **_k):
        return SendResult(success=True, message_id="card")

    adapter.send_with_keyboard = send_with_keyboard
    await adapter.send_exec_approval(chat_id="u-42", command="ls", session_key=SK, request_id="r1")
    assert approval.resolve_gateway_approval_card(SK, "once", "r1") == 1 and entry.result == "once"
