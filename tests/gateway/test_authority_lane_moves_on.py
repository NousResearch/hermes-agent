"""After a chat moves to a new conversation, an app view of its old one keeps working, off the chat.

Teknium's ruling: a messaging chat's ``/new`` must not lock Desktop / Webapp / mobile users out of
the conversation they have open, and the chat must not get that conversation's replies once it is on
another one. A Discord chat's conversation A is open in an app; the chat sends ``/new`` and then a
message, which runs in its new conversation B. The app's prompt on A runs in A on A's own key, and
the chat adapter receives nothing from it; B's turns and the chat's replies are untouched. A
``/resume`` of A on the chat brings A's replies back to the chat. Real GatewayRunner, DiscordAdapter
(its ``send`` recorded), SessionAuthority, drain and session resolver over a real SessionDB; the
agent run is a stand-in that answers with the session and route it was given.
"""
import asyncio

import pytest

from tests.waiting import async_wait_until


async def _gateway(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    sent = []

    async def send(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)
        return SendResult(success=True, message_id='s')
    monkeypatch.setattr(adapter, 'send', send)
    runner.adapters = {Platform.DISCORD: adapter}
    monkeypatch.setattr(runner, '_read_user_config', lambda: {'approvals': {'destructive_slash_confirm': False}})
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    ran, gates = [], {}

    async def agent_turn(event, source, session_entry, session_key, *args):
        # The stand-in agent: it answers with where it ran, so a reply in the chat names its source.
        ran.append((event.text, session_entry.session_id, session_key))
        if event.text in gates:
            await gates[event.text].wait()
        reply = f'reply to {event.text!r} from {session_entry.session_id}'
        execution_result.get()['result'] = {'final_response': reply, 'messages': []}
        return reply, None
    monkeypatch.setattr(runner, '_hmwa_prepare_turn', agent_turn)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    return runner, adapter, authority, source, sent, ran, gates


async def _chat(adapter, source, sent, text):
    from gateway.platforms.event import MessageEvent
    before = len(sent)
    await adapter.handle_message(MessageEvent(text=text, source=source, message_id=text))
    await async_wait_until(lambda: len(sent) > before, what=f'the reply to {text!r}')


def _app():
    from gateway.session_contract import Principal
    return Principal('app-viewer', 'default', frozenset(
        {'session:read', 'session:submit', 'session:control', 'session:approve', 'session:respond'}), 'viewer')


async def _app_turn(authority, ref, text):
    """The app's prompt, settled: ``(status, outcome)`` of its admission."""
    from gateway.session_contract import Submission
    from hermes_state_runtime import list_session_admissions
    receipt = await authority.submit(_app(), Submission(text, ref, {'text': text}, 'queue'))

    def settled():
        rows = [row for row in list_session_admissions(authority.db, session_id=ref.session_id, pending_only=False)
                if row['admission_id'] == receipt.admission_id and row['status'] == 'terminal']
        return rows and rows[0]
    row = await async_wait_until(settled, timeout=10, what=f'the app turn {text!r}')
    await asyncio.wait_for(authority.sessions[ref.session_id].task, 10)  # its delivery runs after the settlement
    return row['status'], row['outcome']


@pytest.mark.asyncio
async def test_the_app_keeps_its_old_conversation_and_the_chat_gets_none_of_its_replies(monkeypatch):
    runner, adapter, authority, source, sent, ran, _gates = await _gateway(monkeypatch)
    lane = runner.session_store._generate_session_key(source)
    await _chat(adapter, source, sent, 'hello')
    a = runner.session_store.peek_session_id(lane)
    ref_a = authority.register(source)
    await _chat(adapter, source, sent, '/new')
    await _chat(adapter, source, sent, 'hello again')
    b = runner.session_store.peek_session_id(lane)
    assert b != a
    chat_replies = len(sent)

    assert await _app_turn(authority, ref_a, 'from the app') == ('terminal', 'completed')
    text, ran_in, key = ran[-1]
    assert (text, ran_in) == ('from the app', a)  # in A, on A's own key, not the chat's
    assert key != lane and runner.session_store.peek_session_id(lane) == b
    assert len(sent) == chat_replies, f'the chat got the old conversation\'s reply: {sent[chat_replies:]}'

    await _chat(adapter, source, sent, 'still here')  # the chat keeps its new conversation
    assert ran[-1][1:] == (b, lane) and sent[-1] == f"reply to 'still here' from {b}"


@pytest.mark.asyncio
async def test_the_old_conversation_runs_beside_the_chats_turn_and_resume_brings_its_replies_back(monkeypatch):
    runner, adapter, authority, source, sent, ran, gates = await _gateway(monkeypatch)
    lane = runner.session_store._generate_session_key(source)
    await _chat(adapter, source, sent, 'hello')
    a = runner.session_store.peek_session_id(lane)
    ref_a = authority.register(source)
    authority.db.set_session_title(a, 'First chat')
    await _chat(adapter, source, sent, '/new')

    from gateway.platforms.event import MessageEvent
    gates['long chat turn'] = asyncio.Event()
    await adapter.handle_message(MessageEvent(text='long chat turn', source=source, message_id='long'))
    await async_wait_until(lambda: ran and ran[-1][0] == 'long chat turn', what="B's turn to start")
    b = ran[-1][1]
    # While B's turn runs, the app's prompt and Stop on A act on A alone.
    assert await _app_turn(authority, ref_a, 'meanwhile') == ('terminal', 'completed')
    assert ran[-1][:2] == ('meanwhile', a)
    await authority.interrupt(_app(), ref_a, authority._handle(ref_a).execution_generation)
    gates['long chat turn'].set()
    await async_wait_until(lambda: sent and sent[-1] == f"reply to 'long chat turn' from {b}", what="B's reply")
    assert not any('meanwhile' in text for text in sent)

    await _chat(adapter, source, sent, '/resume First chat')
    chat_replies = len(sent)
    assert await _app_turn(authority, ref_a, 'after resume') == ('terminal', 'completed')
    assert ran[-1] == ('after resume', a, lane)  # back on the chat's route
    assert sent[chat_replies:] == [f"reply to 'after resume' from {a}"]


@pytest.mark.asyncio
async def test_the_old_conversation_never_drives_the_chats_topic_or_claims_a_reset(monkeypatch):
    """Review on #144840: a detached turn's compression must not rewrite the chat's Telegram topic
    binding (the chat would fall back into A), and its context exhaustion has no route to reset."""
    from gateway.session_detached import detach
    from gateway.session_identity import replace_source
    runner, adapter, authority, source, sent, _ran, _gates = await _gateway(monkeypatch)
    await _chat(adapter, source, sent, 'hello')
    ref_a = authority.register(source)
    await _chat(adapter, source, sent, '/new')
    detached = authority.sessions[ref_a.session_id].source
    assert detached.detached_route  # /new detached A
    monkeypatch.setattr(runner, '_is_telegram_dm', lambda source: True)
    monkeypatch.setattr(runner, '_telegram_topic_mode_enabled', lambda source: True)
    topic = replace_source(source, thread_id='9')
    assert runner._is_telegram_topic_lane(topic)
    live = authority.sessions[ref_a.session_id]
    live.source = replace_source(topic, detached_route=None)
    live.route = runner.session_store._generate_session_key(topic)
    detach(authority, live, ref_a.session_id)
    assert not runner._is_telegram_topic_lane(live.source)

    entry = runner.session_store.get_or_create_session(detached)
    response, kept = await runner._hmwa_compression_exhaustion_reset(
        {'compression_exhausted': True}, 'too long', entry, entry.session_key, detached)
    # The app viewer is told how to go on (the guidance a failed overflow turn gives), never "reset".
    from gateway.run import _context_overflow_reply
    assert (response, kept.session_id) == (f'too long\n\n{_context_overflow_reply()}', ref_a.session_id)


@pytest.mark.asyncio
async def test_an_app_prompt_queued_for_the_old_conversation_runs_after_a_restart(monkeypatch):
    """Review on #144840: the routing index names only the chat's current conversation, so boot
    recovery of queued input never reached a detached one; its queued app prompt now runs."""
    from gateway.config import GatewayConfig
    from gateway.run import GatewayRunner
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Submission
    runner, adapter, authority, source, sent, _ran, _gates = await _gateway(monkeypatch)
    await _chat(adapter, source, sent, 'hello')
    ref_a = authority.register(source)
    await _chat(adapter, source, sent, '/new')
    await _chat(adapter, source, sent, 'hello again')
    authority._schedule = lambda ref: None  # accepted, then the gateway stops before running it
    await authority.submit(_app(), Submission('queued', ref_a, {'text': 'queued over the restart'}, 'queue'))

    restarted = GatewayRunner(GatewayConfig())
    restarted.adapters = dict(runner.adapters)
    cold = await initialize_session_authority(restarted, profile_id='default', instance_id='restarted')
    scheduled = []
    cold._schedule = scheduled.append
    outcome = await recover_profile_native_sessions(restarted, cold)
    assert outcome.get(ref_a.session_id) == 'detached' and scheduled == [ref_a]
    assert cold.sessions[ref_a.session_id].detached
