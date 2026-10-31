"""Email must not turn routine lease waits into permanent messages (#146451)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.run import _sanitize_gateway_final_response
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext


WAIT_MESSAGES = [
    "⏳ Another Hermes process is using this session; "
    "waiting for it to finish before starting your turn...",
    "⏳ Still waiting for the other Hermes process on this session (196s)...",
    "⏳ Still waiting for the other Hermes process on this session (211s)...",
]
TIMEOUT_MESSAGE = (
    "⏳ Another Hermes process kept this session busy too long. Your message was not "
    "processed - wait for the other process to finish, then send it again."
)


def _status_sink(platform, adapter=None):
    if adapter is None:
        adapter = SimpleNamespace(send=AsyncMock())
    ctx = TurnContext(
        source=SessionSource(platform=platform, chat_id="offline"), user_config={}
    )
    ctx._status_adapter = adapter
    ctx._status_chat_id = "offline"
    ctx._status_thread_metadata = None
    holder = SimpleNamespace(
        _ctx=ctx,
        _status_live=lambda: True,
        _schedule=lambda coro, *_args: asyncio.run(coro),
    )
    return holder, adapter.send


def test_email_waits_stay_silent_but_actionable_status_and_final_survive(monkeypatch):
    from plugins.platforms.email.adapter import EmailAdapter

    monkeypatch.setenv("EMAIL_ADDRESS", "hermes@example.test")
    adapter = EmailAdapter(PlatformConfig(enabled=True))
    smtp_send = Mock()
    monkeypatch.setattr(adapter, "_smtp_send", smtp_send)
    adapter.send = AsyncMock(wraps=adapter.send)
    holder, send = _status_sink(Platform.EMAIL, adapter)
    for message in WAIT_MESSAGES:
        TurnRunner._status_callback_sync(holder, "lifecycle", message)
    send.assert_not_awaited()
    smtp_send.assert_not_called()

    for kind, message in (("lifecycle", TIMEOUT_MESSAGE), ("warn", "Session turn lease lost; stopping to protect the transcript.")):
        TurnRunner._status_callback_sync(holder, kind, message)
    assert [call.args[1] for call in send.await_args_list] == [
        TIMEOUT_MESSAGE,
        "Session turn lease lost; stopping to protect the transcript.",
    ]
    final = "Your request is complete."
    result = asyncio.run(adapter.send(
        "offline", _sanitize_gateway_final_response(Platform.EMAIL, final)
    ))
    assert result.success
    bodies = [
        call.args[0].get_payload(0).get_payload(decode=True).decode("utf-8")
        for call in smtp_send.call_args_list
    ]
    assert bodies == [
        TIMEOUT_MESSAGE,
        "Session turn lease lost; stopping to protect the transcript.",
        final,
    ]


@pytest.mark.parametrize("platform", [Platform.TELEGRAM, Platform.SLACK])
def test_other_transports_keep_lease_wait_status(platform):
    holder, send = _status_sink(platform)
    for message in WAIT_MESSAGES:
        TurnRunner._status_callback_sync(holder, "lifecycle", message)
    assert [call.args[1] for call in send.await_args_list] == WAIT_MESSAGES
