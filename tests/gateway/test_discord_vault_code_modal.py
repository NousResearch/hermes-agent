"""A Discord turn can answer ``browser_vault`` one-time-code prompts through a button and a modal (#141844).

``vault_prompt_scope`` installed the vault callbacks only for local-viewer turns, so on Discord
``browser_vault action=enter_code`` returned ``prompt_unavailable`` and an SMS- or email-only sign-in could
not finish from chat. A messaging turn whose adapter implements ``send_vault_code_prompt`` now gets the
code prompt: the card's button opens a modal for the turn's author only, and the code reaches the page
without ever being posted, echoed or logged.
"""
import asyncio
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run  # import-time checkout probes run at collection, before the per-test home I/O guard
from agent.vault_backends import unlock
from gateway.config import Platform, PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter, VaultCodeView, _t_discord
from tests.waiting import wait_until

AUTHOR, OTHER, CODE = "4242", "7777", "954145"


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
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="t", extra={}))
    adapter._client = MagicMock()
    adapter._allowed_user_ids = {AUTHOR, OTHER}
    adapter._allowed_role_ids = set()
    message = SimpleNamespace(id=99, embeds=[], edit=AsyncMock())
    channel = SimpleNamespace(send=AsyncMock(return_value=message))
    adapter._resolve_channel = AsyncMock(return_value=channel)
    return adapter, channel


def _turn(loop, adapter, platform=Platform.DISCORD, user_id=AUTHOR):
    ctx = SimpleNamespace(
        source=SimpleNamespace(platform=platform, user_id=user_id), _status_adapter=adapter,
        _status_chat_id="c1", _status_thread_metadata={"thread_id": "t1"}, session_key="agent:main:discord:dm:c1")
    return SimpleNamespace(_ctx=ctx, _approval_owner=None,
                           _schedule=lambda coro, _msg, loop_=None: asyncio.run_coroutine_threadsafe(coro, loop))


def _interaction(user_id):
    response = SimpleNamespace(send_message=AsyncMock(), send_modal=AsyncMock(), defer=AsyncMock())
    return SimpleNamespace(user=SimpleNamespace(id=user_id, display_name="u", roles=[]), response=response,
                           channel_id=1, guild=None, message=None)


def _run(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(5)


def _ask_in_background(turn, agent=None):
    """Run the turn's code prompt on a worker (the agent thread blocks there); returns (thread, out)."""
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    out, ready, release = {}, threading.Event(), threading.Event()

    def run():
        with vault_prompt_scope(turn, agent or SimpleNamespace()):
            worker = threading.Thread(target=propagate_context_to_thread(
                lambda: out.update(code=unlock.get_code_prompt_callback()("login.example.com", ""))), daemon=True)
            worker.start()
            ready.set()
            worker.join(10)
            release.wait(10)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(5)
    return thread, out, release


def _sent_view(channel):
    wait_until(lambda: channel.send.await_count, what="vault code card")
    return channel.send.call_args.kwargs["view"]


def _enter_code_page(seen):
    """Patches for a page with one OTP field; ``seen['expr']`` is the JS that writes the code."""
    from tools import browser_vault_tool
    controls = [{"index": 0, "type": "text", "name": "otp", "label": "Code", "autocomplete": "one-time-code"}]

    def fake_eval(_task, expr):
        return {"success": True, "result": json.dumps(controls) if "querySelectorAll" in expr
                else "https://login.example.com/2fa"}

    def fake_secret(_task, expr):
        seen["expr"] = expr
        return {"success": True, "result": json.dumps({"filled": 1})}
    return (patch.object(browser_vault_tool, "_focus_bound_origin", lambda *a, **k: None),
            patch.object(browser_vault_tool, "_eval_js", side_effect=fake_eval),
            patch.object(browser_vault_tool, "_eval_js_secret", side_effect=fake_secret))


def test_author_enters_the_code_through_the_modal_and_it_is_never_posted_or_logged(loop, caplog):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools import browser_vault_tool
    adapter, channel = _adapter()
    seen, out = {}, {}
    p1, p2, p3 = _enter_code_page(seen)

    def run():
        with p1, p2, p3, vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
            out["raw"] = browser_vault_tool.browser_vault_enter_code(task_id="t")
    caplog.set_level(logging.DEBUG)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    view = _sent_view(channel)
    assert isinstance(view, VaultCodeView)
    adapter._resolve_channel.assert_awaited_with("t1")
    assert "login.example.com" in channel.send.call_args.kwargs["content"]

    click = _interaction(AUTHOR)
    _run(loop, view.enter_code(click, None))
    modal = click.response.send_modal.call_args.args[0]
    modal.code.value = f" {CODE} "
    submit = _interaction(AUTHOR)
    _run(loop, modal.on_submit(submit))
    worker.join(10)

    result = json.loads(out["raw"])
    assert result["success"] is True and result["source"] == "user"
    assert f'"value": "{CODE}"' in seen["expr"]  # the code went to the page
    posted = [repr(c) for c in channel.send.call_args_list + submit.response.send_message.call_args_list
              + view._message.edit.call_args_list]
    assert not any(CODE in text for text in posted + [out["raw"], caplog.text])
    assert submit.response.send_message.call_args.kwargs["ephemeral"] is True
    assert unlock.get_code_prompt_callback() is None  # restored when the turn ended


def test_another_allowlisted_user_cannot_open_or_submit_the_modal(loop):
    adapter, channel = _adapter()
    thread, out, release = _ask_in_background(_turn(loop, adapter))
    view = _sent_view(channel)

    stranger = _interaction(OTHER)
    _run(loop, view.enter_code(stranger, None))
    stranger.response.send_modal.assert_not_awaited()
    assert stranger.response.send_message.call_args.kwargs["ephemeral"] is True
    forged = _interaction(OTHER)
    _run(loop, view.submit_code(forged, CODE))
    assert not view.resolved and "code" not in out

    owner = _interaction(AUTHOR)
    _run(loop, view.submit_code(owner, CODE))
    release.set()
    thread.join(10)
    assert out["code"] == CODE


def test_interrupt_releases_the_wait_and_the_tool_reports_code_declined(loop, monkeypatch):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools import browser_vault_tool
    monkeypatch.setattr("gateway.session_vault_prompt.CODE_TIMEOUT_S", 30)
    adapter, channel = _adapter()
    p1, p2, p3 = _enter_code_page({})
    started = time.monotonic()
    with p1, p2, p3, vault_prompt_scope(_turn(loop, adapter), SimpleNamespace(_interrupt_requested=True)):
        out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
    assert time.monotonic() - started < 10  # released by /stop, not by the 30 s card timeout
    assert channel.send.await_count == 1
    assert out["error_type"] == "code_declined"


def test_unanswered_card_times_out_and_a_late_submit_enters_nothing(loop, monkeypatch):
    monkeypatch.setattr("gateway.session_vault_prompt.CODE_TIMEOUT_S", 1)
    adapter, channel = _adapter()
    thread, out, release = _ask_in_background(_turn(loop, adapter))
    view = _sent_view(channel)
    wait_until(lambda: "code" in out, what="timeout")
    assert out["code"] == ""
    late = _interaction(AUTHOR)
    _run(loop, view.submit_code(late, CODE))
    assert out["code"] == ""
    # The author is told the request ended, not that a code is being entered.
    assert late.response.send_message.call_args.args[0] == _t_discord("platform.discord.vault_code.ended", 2000)
    assert CODE not in repr(late.response.send_message.call_args)
    release.set()
    thread.join(10)


def test_turn_end_releases_a_tool_thread_still_waiting_on_the_card(loop):
    """A tool worker abandoned by its batch deadline can outlive the turn: the scope's exit wakes it."""
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools.thread_context import propagate_context_to_thread
    adapter, channel = _adapter()
    out = {}
    with vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
        worker = threading.Thread(target=propagate_context_to_thread(
            lambda: out.update(code=unlock.get_code_prompt_callback()("login.example.com", ""))), daemon=True)
        worker.start()
        _sent_view(channel)
        assert "code" not in out
    worker.join(5)
    assert not worker.is_alive() and out["code"] == ""


def test_card_that_cannot_be_sent_answers_at_once(loop):
    from gateway.session_vault_prompt import vault_prompt_scope
    adapter, channel = _adapter()
    channel.send.side_effect = RuntimeError("Missing Access")
    with vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
        assert unlock.get_code_prompt_callback()("login.example.com", "") == ""


@pytest.mark.parametrize("case", ["adapter_without_code_card", "turn_without_author", "cron"])
def test_surfaces_that_cannot_ask_still_report_prompt_unavailable(loop, monkeypatch, case):
    from gateway.session_vault_prompt import vault_prompt_scope
    from tools import browser_vault_tool
    adapter, channel = _adapter()
    turn = _turn(loop, adapter, user_id="" if case == "turn_without_author" else AUTHOR)
    if case == "adapter_without_code_card":
        turn._ctx._status_adapter = SimpleNamespace(send=AsyncMock())  # e.g. a webhook-style text adapter
    if case == "cron":
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    p1, p2, p3 = _enter_code_page({})
    with p1, p2, p3, vault_prompt_scope(turn, SimpleNamespace()):
        out = json.loads(browser_vault_tool.browser_vault_enter_code(task_id="t"))
        assert unlock.get_unlock_prompt_callback() is None and unlock.get_save_login_prompt_callback() is None
    assert out["error_type"] == "prompt_unavailable"
    channel.send.assert_not_awaited()


def test_a_code_prompt_alone_does_not_make_unlock_or_save_login_answerable(loop):
    """Only the code prompt is offered on a chat platform: unlock and save_login keep their old answer."""
    from gateway.session_vault_prompt import vault_prompt_scope
    adapter, _ = _adapter()
    with vault_prompt_scope(_turn(loop, adapter), SimpleNamespace()):
        assert unlock.can_prompt_here(unlock.get_code_prompt_callback()) is True
        assert unlock.can_prompt_here() is False
