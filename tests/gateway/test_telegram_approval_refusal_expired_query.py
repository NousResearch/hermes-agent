"""A requester-refusal alert on an expired Telegram callback query is dropped, not raised.

Telegram refuses ``answerCallbackQuery`` once the query is older than its window. A refused tap on
an old card must leave the requester's approval pending without failing the update in PTB.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_approval_requester_binding import pending_from, write_approvers

STALE = "Bad Request: Query is too old and response timeout expired or query id is invalid"


def _error(text: str) -> Exception:
    """A PTB ``BadRequest`` when the library is importable, else a text-only stand-in."""
    try:
        from telegram.error import BadRequest
    except ImportError:
        return RuntimeError(text)
    return BadRequest(text)


@pytest.fixture(autouse=True)
def _clean():
    from tools import approval
    approval._gateway_queues.clear()
    yield
    approval._gateway_queues.clear()


def _refused_tap(monkeypatch, answer_error):
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
    query.answer.side_effect = answer_error
    return adapter, query, alice


@pytest.mark.asyncio
async def test_refused_tap_on_an_expired_query_does_not_raise(monkeypatch):
    adapter, query, alice = _refused_tap(monkeypatch, _error(STALE))

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    assert query.answer.await_args.kwargs.get("show_alert") is True
    assert not alice.event.is_set()
    assert 4 in adapter._approval_state  # Alice's card still answers


@pytest.mark.asyncio
async def test_refusal_alert_failing_for_another_reason_still_raises(monkeypatch):
    adapter, query, _alice = _refused_tap(monkeypatch, _error("Bad Request: message is not modified"))

    with pytest.raises(Exception, match="not modified"):
        await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())
