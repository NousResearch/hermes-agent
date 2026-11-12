"""A requester-refusal toast on an expired Telegram callback query is swallowed like every other
callback answer (151004's expired-query guard x 144799's requester binding)."""

import pytest

from tests.gateway.test_approval_requester_binding import pending_from, write_approvers


@pytest.fixture(autouse=True)
def _clean():
    from hermes_agent.tools import approval
    approval._gateway_queues.clear()
    yield
    approval._gateway_queues.clear()


@pytest.mark.asyncio
async def test_refused_tap_on_an_expired_query_does_not_raise(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from telegram.error import BadRequest
    from tests.gateway.test_telegram_approval_buttons import _make_adapter

    write_approvers([])
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "*")
    session = "agent:main:telegram:group:1:9"
    alice = pending_from("alice", "Alice", session)
    adapter = _make_adapter()
    adapter._approval_state = {4: session}
    adapter._approval_request_ids = {4: alice.data["request_id"]}
    query = AsyncMock(data="ea:once:4", message=MagicMock(chat_id=1, text_html="x"),
                      from_user=MagicMock(first_name="Bob", id="bob"))
    query.answer.side_effect = BadRequest("Query is too old and response timeout expired or query id is invalid")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    assert query.answer.await_args.kwargs.get("show_alert") is True
    assert not alice.event.is_set()
    assert 4 in adapter._approval_state  # Alice's card still answers
