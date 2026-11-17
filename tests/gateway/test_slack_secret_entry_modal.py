"""A Slack turn can collect a skill's missing secret through a button and a private modal, opt-in.

Before, a skill whose ``required_environment_variables`` were missing got only the "secure secret entry is
not supported over messaging" hint on every chat platform. With ``platforms.slack.extra.secret_entry_modal``
on, the turn's author presses the card's button, types the value in a modal, and it lands in the owning
profile's ``.env``; it is never posted, echoed, logged or returned to the model. Off (the default), the
legacy hint stays.
"""
import asyncio
import logging
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.gateway.test_slack_clarify_buttons import _attach_auth_runner, _make_adapter
from tests.waiting import wait_until

import gateway.run  # import-time probes run before the per-test home I/O guard
from gateway.config import Platform
from plugins.platforms.slack.adapter_private_entry import SECRET

AUTHOR, OTHER, VALUE = "U_AUTHOR", "U_OTHER", "sk-live-9c1e7Q4kz"


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def _adapter(enabled):
    adapter = _make_adapter()
    adapter.config.extra["secret_entry_modal"] = enabled
    _attach_auth_runner(adapter)
    client = adapter._team_clients["T1"]
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.222", "channel": "C1"})
    client.chat_update, client.chat_postEphemeral, client.views_open = AsyncMock(), AsyncMock(), AsyncMock()
    return adapter, client


def _turn(loop, adapter):
    ctx = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.SLACK, user_id=AUTHOR), _status_adapter=adapter,
        _status_chat_id="C1", _status_thread_metadata={"thread_id": "100.1", "scope_id": "T1"},
        session_key="agent:b:slack:group:C1:100.1")
    return SimpleNamespace(_ctx=ctx, _approval_owner=None,
                           _schedule=lambda coro, _msg: asyncio.run_coroutine_threadsafe(coro, loop))


def _click(user_id, token):
    body = {"team": {"id": "T1"}, "user": {"id": user_id, "name": "u"}, "channel": {"id": "C1"},
            "message": {"ts": "111.222", "thread_ts": "100.1"}, "trigger_id": "trig-1"}
    return body, {"action_id": SECRET.open_action, "value": token}


def _submission(user_id, view, value):
    view = dict(view, state={"values": {SECRET.block_id: {SECRET.input_action: {"value": value}}}})
    return {"team": {"id": "T1"}, "user": {"id": user_id}, "view": view}, view


def _run(loop, coro):
    return asyncio.run_coroutine_threadsafe(coro, loop).result(5)


def _profiles(tmp_path, monkeypatch):
    """A multiplexed host: the launch (default) profile and the served profile ``b`` the turn runs in."""
    default_home = tmp_path / ".hermes"
    prof_b = default_home / "profiles" / "b"
    prof_b.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("FOO_KEY", raising=False)
    return default_home, prof_b


def _skill_needs_foo_key(loop, adapter, prof_b, out):
    """The agent thread of profile b's turn loads a skill whose FOO_KEY is missing."""
    from agent import secret_scope
    from gateway.session_secret_prompt import secret_prompt_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.skills_tool_setup import _capture_required_environment_variables

    def run():
        secret_scope.set_multiplex_active(True)
        home_token = set_hermes_home_override(str(prof_b))
        scope_token = secret_scope.set_secret_scope({})
        try:
            with secret_prompt_scope(_turn(loop, adapter), SimpleNamespace(_interrupt_requested=False)):
                out.update(_capture_required_environment_variables(
                    "foo-skill", [{"name": "FOO_KEY", "prompt": "Paste your Foo API key"}]))
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)
            secret_scope.set_multiplex_active(False)
    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    return worker


def test_author_saves_the_secret_to_the_owning_profiles_env_through_the_modal(loop, tmp_path, monkeypatch, caplog):
    default_home, prof_b = _profiles(tmp_path, monkeypatch)
    adapter, client = _adapter(True)
    out = {}
    caplog.set_level(logging.DEBUG)
    worker = _skill_needs_foo_key(loop, adapter, prof_b, out)
    wait_until(lambda: client.chat_postMessage.await_count, what="secret card")
    posted = client.chat_postMessage.call_args.kwargs
    assert posted["thread_ts"] == "100.1" and "FOO_KEY" in posted["blocks"][0]["text"]["text"]
    token = posted["blocks"][1]["elements"][0]["value"]

    # Another allowlisted member can neither open the modal nor submit into it.
    _run(loop, adapter._handle_entry_open(AsyncMock(), *_click(OTHER, token)))
    client.views_open.assert_not_awaited()
    assert client.chat_postEphemeral.call_args.kwargs["user"] == OTHER
    _run(loop, adapter._handle_entry_open(AsyncMock(), *_click(AUTHOR, token)))
    view = client.views_open.call_args.kwargs["view"]
    forged_ack = AsyncMock()
    _run(loop, adapter._handle_entry_submit(forged_ack, *_submission(OTHER, view, "forged")))
    assert forged_ack.call_args.kwargs["response_action"] == "errors" and not out

    ack = AsyncMock()
    _run(loop, adapter._handle_entry_submit(ack, *_submission(AUTHOR, view, f" {VALUE} ")))
    worker.join(10)
    ack.assert_awaited_once_with()
    assert out["missing_names"] == [] and out["gateway_setup_hint"] is None
    assert f"FOO_KEY={VALUE}" in (prof_b / ".env").read_text(encoding="utf-8-sig")
    # The launch profile never sees profile b's key: not in its .env, not in the shared process env.
    assert not (default_home / ".env").exists() and "FOO_KEY" not in os.environ
    wait_until(lambda: client.chat_update.await_count, what="card retired")
    assert all(block["type"] != "actions" for block in client.chat_update.call_args.kwargs["blocks"])
    assert VALUE not in repr(client.mock_calls) and VALUE not in repr(out) and VALUE not in caplog.text


def test_flag_off_keeps_the_messaging_refusal_hint(loop, tmp_path, monkeypatch):
    _default_home, prof_b = _profiles(tmp_path, monkeypatch)
    adapter, client = _adapter(False)
    out = {}
    _skill_needs_foo_key(loop, adapter, prof_b, out).join(10)
    assert out["missing_names"] == ["FOO_KEY"]
    assert "not supported over messaging" in out["gateway_setup_hint"]
    client.chat_postMessage.assert_not_awaited()
    assert not (prof_b / ".env").exists()
