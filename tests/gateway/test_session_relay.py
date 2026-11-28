"""Session relay (gateway/session_relay.py): config gate, allowlist, and session-key contract.

The contract that matters: the relayed turn's source must key to the SAME session a human reply
in that thread resolves to, or the relay starts a session nobody continues.
"""
import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from gateway import session_relay as sr
from gateway.config import Platform
from gateway.session import SessionSource, build_session_key

PARENT, THREAD, NEW_THREAD, GUILD, USER = "100", "200", "300", "900", "111"
ENABLED = {"gateway": {"session_relay": {"enabled": True, "allowed_parents": [PARENT]}}}


def _thread(thread_id):
    t = MagicMock(spec=discord.Thread)
    t.id, t.parent_id, t.name = int(thread_id), int(PARENT), "t"
    t.guild = SimpleNamespace(id=int(GUILD))
    t.add_user = AsyncMock()
    return t


def _build_source(**kw):
    return SessionSource(platform=Platform.DISCORD, chat_id=kw["chat_id"], chat_type=kw["chat_type"],
                         user_id=kw["user_id"], user_name=kw.get("user_name"), thread_id=kw["thread_id"],
                         guild_id=kw.get("guild_id"), parent_chat_id=kw.get("parent_chat_id"))


def _runner():
    threads = {THREAD: _thread(THREAD), NEW_THREAD: _thread(NEW_THREAD)}
    client = SimpleNamespace(get_channel=lambda i: threads.get(str(i)), fetch_channel=AsyncMock())
    adapter = SimpleNamespace(_client=client, build_source=_build_source,
                              create_handoff_thread=AsyncMock(return_value=NEW_THREAD),
                              _threads=SimpleNamespace(mark_async=AsyncMock()))
    runner = SimpleNamespace(adapters={Platform.DISCORD: adapter},
                             _session_key_for_source=lambda s: build_session_key(s),
                             _resolve_profile_home_for_source=lambda s: "/nonexistent")
    return runner, adapter, threads


@contextlib.asynccontextmanager
async def _null_scope(_home):
    yield


def _run(runner, body, *, config=ENABLED, route_wake=False, paused=False):
    # Stub the module's gateway seams: importing gateway.run bootstraps the real runtime.
    internal = AsyncMock()
    with patch.object(sr, "_load_config", return_value=config), \
            patch.object(sr, "_profile_scope", _null_scope), \
            patch.object(sr, "_route_wake", AsyncMock(return_value=route_wake)) as rw, \
            patch.object(sr, "_admit_internal", internal), \
            patch.object(sr, "_paused", return_value=paused):
        result = asyncio.run(sr.relay(runner, body))
    return result, rw, internal


def _human_reply(thread_id):
    return SessionSource(platform=Platform.DISCORD, chat_id=thread_id, chat_type="thread", user_id=USER,
                         thread_id=thread_id, guild_id=GUILD, parent_chat_id=PARENT)


def test_settings_default_off_and_parse_comma_list():
    assert sr.relay_settings({}) == (False, set())
    on = {"gateway": {"session_relay": {"enabled": "true", "allowed_parents": "1, 2"}}}
    assert sr.relay_settings(on) == (True, {"1", "2"})
    # ``gateway.relay`` is the relay platform's block, never read as this gate.
    assert sr.relay_settings({"gateway": {"relay": {"enabled": True}}}) == (False, set())


def test_frame_marks_text_as_relayed():
    framed = sr.frame_text("hello", "abc")
    assert framed.startswith("[Relayed from session abc; not typed by the user]")
    assert framed.endswith("\nhello")


def test_disabled_refuses_before_touching_discord():
    runner, adapter, _ = _runner()
    with pytest.raises(sr.RelayError) as exc:
        _run(runner, {"parent_chat_id": PARENT, "title": "x", "text": "hi"}, config={})
    assert exc.value.code == "relay_disabled"
    adapter.create_handoff_thread.assert_not_awaited()


def test_paused_refuses():
    runner, _, _ = _runner()
    with pytest.raises(sr.RelayError) as exc:
        _run(runner, {"thread_id": THREAD, "text": "hi"}, paused=True)
    assert exc.value.code == "relay_paused"


def test_parent_outside_allowlist_is_refused():
    runner, adapter, _ = _runner()
    with pytest.raises(sr.RelayError) as exc:
        _run(runner, {"parent_chat_id": "555", "title": "x", "text": "hi"})
    assert exc.value.code == "relay_forbidden"
    adapter.create_handoff_thread.assert_not_awaited()


def test_new_thread_runs_first_turn_in_the_session_a_reply_continues():
    runner, adapter, threads = _runner()
    result, _rw, internal = _run(runner, {"parent_chat_id": PARENT, "title": "topic", "text": "ctx",
                                         "user_id": USER, "from_session": "s1"})
    adapter.create_handoff_thread.assert_awaited_once_with(PARENT, "topic")
    threads[NEW_THREAD].add_user.assert_awaited_once()
    adapter._threads.mark_async.assert_awaited_once_with(NEW_THREAD)
    assert result["created"] is True and result["mode"] == "direct" and result["user_added"] is True
    assert internal.await_args is not None
    event = internal.await_args.args[1]
    assert event.internal is True
    assert event.text.startswith("[Relayed from session s1")
    assert build_session_key(event.source) == build_session_key(_human_reply(NEW_THREAD))
    assert result["session_key"] == build_session_key(_human_reply(NEW_THREAD))


def test_existing_thread_is_queued_behind_the_session_fifo():
    runner, adapter, _ = _runner()
    result, rw, internal = _run(runner, {"thread_id": THREAD, "text": "ctx"}, route_wake=True)
    assert result["mode"] == "queued" and result["created"] is False
    internal.assert_not_awaited()
    adapter.create_handoff_thread.assert_not_awaited()
    assert rw.await_args is not None
    source = rw.await_args.args[2]
    assert build_session_key(source) == build_session_key(_human_reply(THREAD))
    assert build_session_key(source, profile="berkshire") == build_session_key(
        _human_reply(THREAD), profile="berkshire")


def test_profile_prefixed_request_is_refused_before_the_body_is_read():
    from gateway.platforms import api_server_session_relay as route
    from gateway.platforms.api_server import _api_request_profile

    adapter = SimpleNamespace(_check_auth=lambda _r: None, _read_json_body=AsyncMock(),
                              gateway_runner=object())
    token = _api_request_profile.set("berkshire")
    try:
        response = asyncio.run(route.handle_relay(adapter, SimpleNamespace(app={})))
    finally:
        _api_request_profile.reset(token)
    assert response.status == 403
    adapter._read_json_body.assert_not_awaited()


def test_rejects_non_numeric_ids_and_missing_text():
    runner, _, _ = _runner()
    for body in ({"thread_id": "abc", "text": "x"}, {"thread_id": THREAD, "text": "  "},
                 {"parent_chat_id": PARENT, "text": "x"}):
        with pytest.raises(sr.RelayError):
            _run(runner, body)
