"""A Telegram turn can answer ``browser_vault`` one-time-code prompts by replying to a ForceReply card.

Telegram has no private form a bot can open. The author's reply is taken off the inbound path before
batching or the agent see it, deleted, and handed to the waiting browser tool; it never becomes a turn.
"""
import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run  # import-time probes run before the per-test home I/O guard
from agent.vault_backends import unlock
from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_discord_vault_code_modal import _enter_code_page
from tests.waiting import wait_until

AUTHOR, OTHER, CHAT, CODE = 111, 222, 111, "954145"


@pytest.fixture(autouse=True)
def _no_callbacks():
    for setter in (unlock.set_unlock_prompt_callback, unlock.set_save_login_prompt_callback,
                   unlock.set_code_prompt_callback):
        setter(None)
    yield
    unlock.set_code_prompt_callback(None)


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="t", extra={"allow_from": [str(AUTHOR), str(OTHER)]}))
    card = SimpleNamespace(message_id=700, chat_id=CHAT, message_thread_id=None)
    adapter._bot = MagicMock(send_message=AsyncMock(return_value=card), delete_message=AsyncMock(),
                             edit_message_text=AsyncMock())
    adapter._message_handler = AsyncMock()
    return adapter


def _turn(loop, adapter):
    ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.TELEGRAM, user_id=str(AUTHOR)), _status_adapter=adapter,
                          _status_chat_id=str(CHAT), _status_thread_metadata={}, session_key="agent:main:telegram:dm:111")
    return SimpleNamespace(_ctx=ctx, _approval_owner=None,
                           _schedule=lambda coro, _msg: asyncio.run_coroutine_threadsafe(coro, loop))


def _update(text, *, user=AUTHOR, chat_type="private", reply_to=None, message_id=900):
    msg = SimpleNamespace(
        message_id=message_id, text=text, caption=None, entities=[], caption_entities=[], message_thread_id=None,
        is_topic_message=False, chat_id=CHAT, chat=SimpleNamespace(id=CHAT, type=chat_type, title=None, is_forum=False),
        from_user=SimpleNamespace(id=user, full_name="u", first_name="u", is_bot=False),
        reply_to_message=SimpleNamespace(message_id=reply_to) if reply_to else None, date=None, media_group_id=None)
    return SimpleNamespace(update_id=message_id, message=msg, effective_message=msg)


def test_author_replies_with_the_code_it_is_deleted_and_never_a_turn_or_a_log_line(loop, caplog):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools import browser_vault_tool
    adapter = _adapter()
    seen, out = {}, {}
    p1, p2, p3 = _enter_code_page(seen)

    def run():
        with p1, p2, p3, vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
            out["raw"] = browser_vault_tool.browser_vault_enter_code(task_id="t")
    caplog.set_level(logging.DEBUG)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    wait_until(lambda: adapter._bot.send_message.await_count, what="vault code card")
    card = adapter._bot.send_message.call_args.kwargs
    assert "login.example.com" in card["text"]
    assert card["reply_markup"] is not None  # ForceReply aimed at the author

    # The bot's own debug log of the polled update would carry the code: filtered while a reply is pending.
    logging.getLogger("telegram.Bot").debug("Call to Bot API endpoint `getUpdates` finished with return value `%s`",
                                            json.dumps([{"message": {"text": CODE}}]))
    asyncio.run_coroutine_threadsafe(
        adapter._handle_text_message(_update(f" {CODE} ", reply_to=700), SimpleNamespace()), loop).result(5)
    worker.join(10)

    result = json.loads(out["raw"])
    assert result["success"] is True and result["source"] == "user"
    assert f'"value": "{CODE}"' in seen["expr"]
    adapter._bot.delete_message.assert_awaited_once()
    assert adapter._bot.delete_message.call_args.kwargs["message_id"] == 900
    adapter._message_handler.assert_not_awaited()  # never a turn
    assert not adapter._pending_text_batches
    assert CODE not in repr(adapter._bot.mock_calls) and CODE not in out["raw"] and CODE not in caplog.text
    assert unlock.get_code_prompt_callback() is None


def test_another_member_or_an_unrelated_group_message_is_not_taken_as_the_code(loop, monkeypatch):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    monkeypatch.setattr("gateway.session_vault_prompt.CODE_TIMEOUT_S", 2)
    adapter = _adapter()
    out = {}
    with vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
        worker = threading.Thread(target=propagate_context_to_thread(
            lambda: out.update(code=unlock.get_code_prompt_callback()("login.example.com", ""))), daemon=True)
        worker.start()
        wait_until(lambda: adapter._bot.send_message.await_count, what="vault code card")
        for update in (_update("000000", user=OTHER, chat_type="group", reply_to=700),
                       _update("hello", chat_type="group", message_id=901)):
            assert asyncio.run_coroutine_threadsafe(
                adapter.consume_vault_code_reply(update.message), loop).result(5) is False
        assert "code" not in out
        worker.join(5)
    assert out["code"] == ""  # timed out: neither message answered it
    adapter._bot.delete_message.assert_not_awaited()


def test_a_late_reply_to_an_ended_card_is_still_deleted_and_never_a_turn(loop, monkeypatch):
    """After the request times out the card is forgotten as a request, not as a code card: a late reply
    carrying the code must not fall through to the agent as chat text."""
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    monkeypatch.setattr("gateway.session_vault_prompt.CODE_TIMEOUT_S", 1)
    adapter = _adapter()
    out = {}
    with vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
        worker = threading.Thread(target=propagate_context_to_thread(
            lambda: out.update(code=unlock.get_code_prompt_callback()("login.example.com", ""))), daemon=True)
        worker.start()
        worker.join(5)
    assert out["code"] == ""
    wait_until(lambda: not adapter._vault_code_replies(), what="card retired by its timer")
    asyncio.run_coroutine_threadsafe(
        adapter._handle_text_message(_update(CODE, chat_type="group", reply_to=700), SimpleNamespace()), loop).result(5)
    adapter._message_handler.assert_not_awaited()
    assert not adapter._pending_text_batches
    adapter._bot.delete_message.assert_awaited_once()


def test_each_card_has_its_own_timer_and_an_old_timer_never_ends_a_newer_card(loop):
    adapter = _adapter()
    cards = iter([SimpleNamespace(message_id=700, chat_id=CHAT, message_thread_id=None),
                  SimpleNamespace(message_id=701, chat_id=CHAT, message_thread_id=None)])
    adapter._bot.send_message = AsyncMock(side_effect=lambda **_kw: next(cards))
    got = []

    async def post_two():
        await adapter.send_vault_code_prompt(str(CHAT), "a.example", "", user_id=str(AUTHOR), submit=lambda c: got.append(c) is None,
                                             timeout_s=0.2)
        await adapter.send_vault_code_prompt(str(CHAT), "b.example", "", user_id=str(AUTHOR), submit=lambda c: got.append(c) is None,
                                             timeout_s=30)
    asyncio.run_coroutine_threadsafe(post_two(), loop).result(5)
    wait_until(lambda: (str(CHAT), "700") not in adapter._vault_code_replies(), what="first card's timer")
    assert (str(CHAT), "701") in adapter._vault_code_replies()  # the newer card survived the older timer
    assert asyncio.run_coroutine_threadsafe(
        adapter.consume_vault_code_reply(_update(CODE, reply_to=701).message), loop).result(5) is True
    assert got == [CODE]


def test_only_the_card_author_is_captured_whatever_their_reply_points_at(loop):
    """Private chat: the author's code sent as a reply to some other message still answers the live card.
    Group: another member's reply to an ended card is their own message and is left alone."""
    adapter = _adapter()
    got = []

    async def post():
        await adapter.send_vault_code_prompt(str(CHAT), "a.example", "", user_id=str(AUTHOR),
                                             submit=lambda c: got.append(c) is None, timeout_s=30)
    asyncio.run_coroutine_threadsafe(post(), loop).result(5)
    consume = adapter.consume_vault_code_reply
    assert asyncio.run_coroutine_threadsafe(consume(_update(CODE, reply_to=555).message), loop).result(5) is True
    assert got == [CODE]
    # The card has ended now; another member's reply to it is ordinary chat, never deleted.
    other = _update("what is this?", user=OTHER, chat_type="group", reply_to=700, message_id=902).message
    assert asyncio.run_coroutine_threadsafe(consume(other), loop).result(5) is False
    assert adapter._bot.delete_message.await_count == 1
