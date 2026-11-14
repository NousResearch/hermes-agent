"""A Slack turn can answer ``browser_vault`` one-time-code prompts through a button and a private modal.

Same contract as the Discord card (``test_discord_vault_code_modal.py``, #141844): the card's button opens
a modal for the turn's author only, and the code reaches the page without being posted, echoed or logged
(salvage of #130569).
"""
import asyncio
import json
import logging
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_slack_clarify_buttons import _attach_auth_runner, _make_adapter
from tests.gateway.test_discord_vault_code_modal import _enter_code_page
from tests.waiting import wait_until

import gateway.run  # import-time probes run before the per-test home I/O guard
from agent.vault_backends import unlock
from gateway.config import Platform
from plugins.platforms.slack.adapter_vault_code import OPEN_ACTION, _BLOCK_ID, _INPUT_ACTION

AUTHOR, OTHER, CODE = "U_AUTHOR", "U_OTHER", "954145"


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
    adapter = _make_adapter()
    _attach_auth_runner(adapter)
    client = adapter._team_clients["T1"]
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.222", "channel": "C1"})
    client.chat_update, client.chat_postEphemeral, client.views_open = AsyncMock(), AsyncMock(), AsyncMock()
    return adapter, client


def _turn(loop, adapter):
    ctx = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.SLACK, user_id=AUTHOR), _status_adapter=adapter,
        _status_chat_id="C1", _status_thread_metadata={"thread_id": "100.1", "scope_id": "T1"},
        session_key="agent:main:slack:group:C1:100.1")
    return SimpleNamespace(_ctx=ctx, _approval_owner=None,
                           _schedule=lambda coro, _msg: asyncio.run_coroutine_threadsafe(coro, loop))


def _click(user_id, token):
    body = {"team": {"id": "T1"}, "user": {"id": user_id, "name": "u"}, "channel": {"id": "C1"},
            "message": {"ts": "111.222", "thread_ts": "100.1"}, "trigger_id": "trig-1"}
    return body, {"action_id": OPEN_ACTION, "value": token}


def _submission(user_id, view, code):
    view = dict(view, state={"values": {_BLOCK_ID: {_INPUT_ACTION: {"value": code}}}})
    return {"team": {"id": "T1"}, "user": {"id": user_id}, "view": view}, view


def _run(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(5)


def test_author_enters_the_code_through_the_modal_and_it_is_never_posted_or_logged(loop, caplog):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools import browser_vault_tool
    adapter, client = _adapter()
    seen, out = {}, {}
    p1, p2, p3 = _enter_code_page(seen)

    def run():
        with p1, p2, p3, vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
            out["raw"] = browser_vault_tool.browser_vault_enter_code(task_id="t")
    caplog.set_level(logging.DEBUG)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    wait_until(lambda: client.chat_postMessage.await_count, what="vault code card")
    posted = client.chat_postMessage.call_args.kwargs
    assert posted["thread_ts"] == "100.1" and "login.example.com" in posted["blocks"][0]["text"]["text"]
    token = posted["blocks"][1]["elements"][0]["value"]

    # Another allowlisted member can neither open the modal nor submit into it.
    _run(loop, adapter._handle_vault_code_open(AsyncMock(), *_click(OTHER, token)))
    client.views_open.assert_not_awaited()
    assert client.chat_postEphemeral.call_args.kwargs["user"] == OTHER
    _run(loop, adapter._handle_vault_code_open(AsyncMock(), *_click(AUTHOR, token)))
    view = client.views_open.call_args.kwargs["view"]
    forged_ack = AsyncMock()
    _run(loop, adapter._handle_vault_code_submit(forged_ack, *_submission(OTHER, view, "000000")))
    assert forged_ack.call_args.kwargs["response_action"] == "errors" and "raw" not in out

    ack = AsyncMock()
    _run(loop, adapter._handle_vault_code_submit(ack, *_submission(AUTHOR, view, f" {CODE} ")))
    worker.join(10)
    ack.assert_awaited_once_with()
    result = json.loads(out["raw"])
    assert result["success"] is True and result["source"] == "user"
    assert f'"value": "{CODE}"' in seen["expr"]
    wait_until(lambda: client.chat_update.await_count, what="card retired")
    sent = repr(client.mock_calls)
    assert CODE not in sent and CODE not in out["raw"] and CODE not in caplog.text
    assert all(block["type"] != "actions" for block in client.chat_update.call_args.kwargs["blocks"])
    assert unlock.get_code_prompt_callback() is None


def test_a_late_submit_after_the_turn_ends_enters_nothing(loop):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    adapter, client = _adapter()
    out = {}
    with vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
        worker = threading.Thread(target=propagate_context_to_thread(
            lambda: out.update(code=unlock.get_code_prompt_callback()("login.example.com", ""))), daemon=True)
        worker.start()
        wait_until(lambda: client.chat_postMessage.await_count, what="vault code card")
    worker.join(5)
    assert out["code"] == ""
    token = client.chat_postMessage.call_args.kwargs["blocks"][1]["elements"][0]["value"]
    _run(loop, adapter._handle_vault_code_open(AsyncMock(), *_click(AUTHOR, token)))
    view = client.views_open.call_args.kwargs["view"]
    ack = AsyncMock()
    _run(loop, adapter._handle_vault_code_submit(ack, *_submission(AUTHOR, view, CODE)))
    assert ack.call_args.kwargs["response_action"] == "errors"
    assert CODE not in repr(client.mock_calls)
