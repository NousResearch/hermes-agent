"""Queued turns use their own tool identity without changing the cached prompt."""

import asyncio
import importlib
import json
from unittest.mock import AsyncMock, call

import pytest

from agent.runtime_cwd import scoped_session_cwd, set_session_cwd, reset_session_cwd
from agent.secret_scope import get_secret
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.session_identity import replace_source
from gateway.session_context import (
    clear_session_vars, get_session_env, get_session_transport, set_session_vars,
)
from hermes_constants import get_hermes_home
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.registry import registry


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_queued_tool_context_restores_outer_identity_and_profile(tmp_path, monkeypatch, failure):
    importlib.import_module("tools.matrix_followup_tool")
    importlib.import_module("tools.matrix_reaction_tool")
    homes = {profile: tmp_path / profile for profile in ("a", "b")}
    adapters = {}
    sources = {}
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    runner._draining = False
    runner._gateway_loop = asyncio.get_running_loop()
    runner._profile_adapters = {}
    runner._resolve_profile_home_for_source = lambda source: homes[source.profile]
    for profile, requester in (("a", "@alice:test"), ("b", "@bob:test")):
        homes[profile].mkdir()
        (homes[profile] / ".env").write_text(f"TEST_OWNER={profile}\n", encoding="utf-8")
        adapter = MatrixAdapter(PlatformConfig(enabled=True))
        adapter._reactions_enabled = False
        adapter._joined_rooms.add("!room:test")
        adapter._is_dm_room = AsyncMock(return_value=False)
        adapter.set_authorization_check(
            lambda user, _chat_type, _chat_id: user in {"@alice:test", "@bob:test"}
        )
        adapter.add_reaction = AsyncMock(return_value={"success": True})
        adapters[profile] = adapter
        runner._profile_adapters[profile] = {Platform.MATRIX: adapter}
        sources[profile] = SessionSource(
            platform=Platform.MATRIX, chat_id="!room:test", chat_type="group",
            thread_id="$thread", user_id=requester, profile=profile,
            message_id=f"${profile}-inbound",
        )
    runner._pending_event_audio_paths = lambda _event: []
    runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="Queued request")
    runner._is_goal_continuation_event = lambda _event: False
    runner._pinned_channel_inputs = lambda _key, prompt, source, **_kwargs: (prompt, source)
    runner._run_agent_deliver_first_response = AsyncMock()
    runner._refresh_agent_cache_message_count = AsyncMock()
    runner._clear_streaming_tts_turn_completed = lambda *_args: None
    async def execute(message, context_prompt, history, source, session_id, **kwargs):
        configured = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_followup", {"enabled": True},
        ))
        reacted = json.loads(await asyncio.to_thread(
            registry.dispatch, "matrix_reaction", {"action": "react", "emoji": "👍"},
        ))
        bound_adapter, _ = get_session_transport()
        choice = bound_adapter._reaction_followup_actions[get_session_env("HERMES_SESSION_KEY")]
        observation = (
            get_hermes_home(), get_secret("TEST_OWNER"), context_prompt,
            choice.requester, choice.session_id, configured, reacted,
        )
        assert observation == (
            homes[source.profile], source.profile, "Pinned session prompt", source.user_id,
            session_id, {"success": True, "enabled": True, "emoji": []}, {"success": True},
        )
        if failure:
            raise RuntimeError("queued execution failed")
        return {"final_response": "Queued answer", "messages": history}

    monkeypatch.setattr(runner, "_run_agent_inner", execute)
    source = sources["a"]
    queued_sources = [replace_source(source, user_id="@bob:test", message_id="$bob-inbound"),
                      sources["b"], source]
    key = runner._session_key_for_source(source)
    turn = TurnContext(
        source=source, session_key=key, session_id="sid", run_generation=1,
        _interrupt_depth=0, history=[], _status_thread_metadata={},
        context_prompt="Pinned session prompt",
    )
    tokens = set_session_vars(
        platform="matrix", chat_id=source.chat_id, user_id=source.user_id,
        thread_id=source.thread_id, profile="a", session_key=key,
        session_id="sid", message_id=source.message_id, transport_adapter=adapters["a"],
        transport_loop=asyncio.get_running_loop(),
    )
    cwd_token = set_session_cwd("/outer/workspace")
    try:
        with _profile_runtime_scope(homes["a"]):
            outer = (get_session_env("HERMES_SESSION_USER_ID"), get_session_transport(),
                     scoped_session_cwd(), get_hermes_home(), get_secret("TEST_OWNER"))
            for queued_source in queued_sources:
                profile = queued_source.profile
                queued_key = runner._session_key_for_source(queued_source)
                adapter = adapters[profile]
                adapter._active_sessions[queued_key] = asyncio.Event()
                event = MessageEvent(text="Queued request", source=queued_source,
                                     message_id=queued_source.message_id)
                adapter._pending_messages[key] = event
                pending_event, pending = await runner._run_agent_drain_pending(
                    {"final_response": "First answer"}, adapter, source, key,
                )
                operation = runner._run_agent_queued_followup(
                    turn, adapter, pending, pending_event, "First answer",
                    {"final_response": "First answer", "messages": []}, None,
                )
                if failure:
                    with pytest.raises(RuntimeError, match="queued execution failed"):
                        await operation
                else:
                    await operation
                assert (get_session_env("HERMES_SESSION_USER_ID"), get_session_transport(),
                        scoped_session_cwd(), get_hermes_home(), get_secret("TEST_OWNER")) == outer
            assert adapters["a"].add_reaction.await_args_list == [
                call(chat_id="!room:test", emoji="👍", message_id="$bob-inbound"),
                call(chat_id="!room:test", emoji="👍", message_id="$a-inbound"),
            ]
            adapters["b"].add_reaction.assert_awaited_once_with(
                chat_id="!room:test", emoji="👍", message_id="$b-inbound",
            )
    finally:
        reset_session_cwd(cwd_token)
        clear_session_vars(tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['fifo-photo', 'busy-start', 'telegram-grace', 'recursion-cap'])
async def test_existing_pending_turn_keeps_a_conflicting_reply_separate(route, monkeypatch):
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.turn_context import TurnContext

    runner = GatewayRunner(config=GatewayConfig())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='1234:dummy'))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='12345', chat_type='dm', user_id='sender')
    key = 'key'
    monkeypatch.setattr(runner, '_delivery_adapter_for', lambda actual_source: adapter)
    first = MessageEvent(text='one', source=source, message_id='m-one', message_type=MessageType.PHOTO,
                         media_urls=['one.png'], media_types=['image/png'],
                         reply_to_message_id='q-one', reply_to_text='quoted one')
    second = MessageEvent(text='two', source=source, message_id='m-two', message_type=MessageType.PHOTO,
                          media_urls=['two.png'], media_types=['image/png'],
                          reply_to_message_id='q-two', reply_to_text='quoted two')
    adapter._pending_messages[key] = first
    if route == 'fifo-photo':
        runner._queue_or_replace_pending_event(key, second)
    elif route == 'busy-start':
        runner._hm_merge_pending_for_source(source, key, second, merge_text=True)
    elif route == 'telegram-grace':
        import time
        first.message_type = second.message_type = MessageType.TEXT
        first.media_urls = second.media_urls = []
        first.media_types = second.media_types = []
        runner._session_state(key).turn.started_ts = time.time()
        assert runner._hm_busy_telegram_grace_queue(second, source, key, 'interrupt')
    else:
        context = TurnContext(source=source, session_id='session', session_key=key, history=[],
                              _interrupt_depth=runner._MAX_INTERRUPT_DEPTH)
        await runner._run_agent_queued_followup(context, adapter, second.text, second, 'done', {}, None)
    events = [adapter._pending_messages[key], *(runner._overflow_queue(key) or ())]
    expected = [('one', 'q-one', 'quoted one'), ('two', 'q-two', 'quoted two')]
    if route == 'recursion-cap':
        expected.reverse()
    assert [(event.text, event.reply_to_message_id, event.reply_to_text) for event in events] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_kind", ["telegram", "raft"])
@pytest.mark.parametrize("kind", [MessageType.TEXT, MessageType.PHOTO])
@pytest.mark.parametrize("withdraw", [False, True])
async def test_standalone_busy_inputs_keep_each_reply_target(adapter_kind, kind, withdraw):
    from gateway.platforms.base_pending import pending_part
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from plugins.platforms.raft.adapter import RaftAdapter

    adapter_type = TelegramAdapter if adapter_kind == "telegram" else RaftAdapter
    adapter = adapter_type(PlatformConfig(enabled=True, token="1234:dummy"))
    adapter._busy_text_mode = "interrupt"
    adapter._busy_text_debounce_seconds = 0
    source = SessionSource(platform=adapter.platform, chat_id="chat", chat_type="dm", user_id="sender")
    events = [MessageEvent(text=text, source=source, message_id=f"m-{text}", message_type=kind,
                           reply_to_message_id=f"q-{text}", reply_to_text=f"quoted {text}",
                           reply_to_author_id="author", reply_to_author_name="Quoted author",
                           reply_to_is_own_message=False, reply_to_author_authorized=True,
                           media_urls=[text + ".png"] if kind == MessageType.PHOTO else [],
                           media_types=["image/png"] if kind == MessageType.PHOTO else [])
              for text in (("one", "two", "three", "four") if withdraw else ("one", "two", "three"))]
    expected = [pending_part(event) for event in events]
    entered, release = asyncio.Event(), asyncio.Event()
    consumed = []

    async def model(event):
        consumed.append(pending_part(event))
        if event.message_id == "m-one":
            entered.set()
            await release.wait()

    adapter.set_message_handler(model)
    try:
        await adapter.handle_message(events[0])
        await asyncio.wait_for(entered.wait(), 2)
        for event in events[1:]:
            await adapter.handle_message(event)
        if withdraw:
            assert adapter.withdraw_pending_message("m-three", chat_id="chat", sender_id="sender")
            expected.pop(2)
        release.set()
        while tasks := list(adapter._background_tasks):
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert consumed == expected
    finally:
        release.set()
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["idle", "queued"])
@pytest.mark.parametrize("quoted", [False, True])
async def test_idle_and_queued_intake_report_the_actual_reply_context(route, quoted, caplog):
    import logging
    from gateway.turn_context import TurnContext

    runner = GatewayRunner(config=GatewayConfig())
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", user_id="sender", user_name="Sender")
    event = MessageEvent(text="request\ntext", source=source, message_id="message",
                         reply_to_message_id="quote" if quoted else None,
                         reply_to_text="quoted\n" + "x" * 100 if quoted else None)
    caplog.set_level(logging.INFO, logger="gateway.run")
    if route == "idle":
        runner._hmwa_resolve_session = AsyncMock(return_value=None)
        await runner._handle_message_with_agent(event, source, "key", 1)
        message = event.text
    else:
        opening_source = replace_source(source, user_id="previous", user_name="Previous")
        turn = TurnContext(source=opening_source, session_key="key", session_id="session", history=[])
        runner._strict_session_current = AsyncMock(return_value=True)
        runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="prepared\nrequest")
        runner._is_goal_continuation_event = lambda _event: False
        runner._pinned_channel_inputs = lambda _key, prompt, actual_source, **_kwargs: (prompt, actual_source)
        runner._persist_prompt_pins = AsyncMock()
        runner._refresh_agent_cache_message_count = AsyncMock()
        runner._delivery_adapter_for = lambda _source: None
        runner._intake_adapter_for = lambda _source: None
        runner._run_agent = AsyncMock(return_value={"final_response": "answer", "messages": []})
        await runner._run_agent_queued_followup(
            turn, None, event.text, event, "answer", {"interrupted": True, "messages": []}, None,
        )
        message = "prepared\nrequest"
    records = [(record.name, record.levelno, record.msg, record.args)
               for record in caplog.records if record.msg.startswith("inbound message:")]
    assert records == [("gateway.run", logging.INFO,
                        "inbound message: platform=%s user=%s chat=%s msg=%r reply_to_id=%s reply_to_text=%r queued=%s",
                        ("telegram", "Sender", "chat", message[:80].replace("\n", " "),
                         event.reply_to_message_id, (event.reply_to_text or "")[:80].replace("\n", " "),
                         route == "queued"))]
