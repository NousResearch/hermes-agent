"""Persisted hosted policies decline a question without a reply channel; local ones still ask."""
import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest

from tools import clarify_gateway as clarify_mod


@pytest.fixture
def ask(owner, tmp_path, monkeypatch):
    """Drive the real delivery chain (native send, text fallback, wait) for one choice question."""
    from gateway.run_turn_runner_clarify_delivery import _clarify_send_then_wait, text_fallback_coro
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import build_policy

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(clarify_mod, 'get_clarify_timeout', lambda: 10)
    schedule = lambda coro: asyncio.run_coroutine_threadsafe(coro, loop)

    def run(source, on_sent=None, request=None):
        adapter = LocalSessionAdapter(owner)
        policy = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'm'}, {})
        adapter.policies['chat'] = policy if source == 'cli' else replace(policy, source=source, platform=source)
        if request is not None:
            adapter.policies['chat'] = replace(adapter.policies['chat'], request_json=json.dumps(request))
        kwargs = dict(chat_id='chat', question='Which?', choices=['one', 'two'], clarify_id='q-' + source,
                      session_key='route-' + source)
        clarify_mod.register(clarify_id=kwargs['clarify_id'], session_key=kwargs['session_key'],
                             question='Which?', choices=['one', 'two'])
        sent = schedule(adapter.send_clarify(**kwargs))
        if on_sent is not None:
            sent.add_done_callback(lambda _: on_sent(kwargs['session_key']))
        def fallback():
            coro = text_fallback_coro(adapter, **kwargs)
            return None if coro is None else schedule(coro)
        started = time.monotonic()
        result = _clarify_send_then_wait(sent, clarify_id=kwargs['clarify_id'], session_key=kwargs['session_key'],
                                         clarify_mod=clarify_mod, fallback=fallback)
        return result, time.monotonic() - started

    yield run
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)


def test_hosted_clarify_is_declined_without_waiting(ask):
    from gateway.run_turn_runner_clarify_delivery import UNDELIVERED_DECLINED
    (response, answered), elapsed = ask('bot_room')
    assert (response, answered) == (UNDELIVERED_DECLINED, False)
    assert elapsed < 5, 'the member waited for an answer nobody can submit'


def test_local_choice_prompt_still_accepts_a_typed_answer(ask):
    outcomes = []
    (response, answered), _ = ask('cli', on_sent=lambda key: outcomes.append(
        clarify_mod.attempt_text_response_for_session(key, 'something else entirely')))
    assert outcomes == [clarify_mod.TEXT_RESOLVED]
    assert (response, answered) == ('something else entirely', True)


def test_clarify_without_an_answering_viewer_is_declined(ask, owner):
    """N8: the decline keys on who can answer, not on one source string. A forwarded A2A peer
    never can; a producer turn (Bot Chat DM) can only while a viewer holding session:respond is
    attached. A viewer's own turn keeps waiting for its author (the local test above)."""
    from gateway.run_turn_runner_clarify_delivery import UNDELIVERED_DECLINED
    from gateway.session_authority import LiveSession
    from gateway.session_contract import Principal
    from hermes_state_runtime import admit_session_input, claim_session_input
    forwarded = {'source': 'a2a', 'a2a_identity': ['dev', '', 'peer', 'ctx']}
    (response, answered), elapsed = ask('a2a', request=forwarded)
    assert (response, answered) == (UNDELIVERED_DECLINED, False) and elapsed < 5
    owner.db.create_session('chat', 'local')
    owner.sessions['chat'] = live = LiveSession(object(), 'route-chat')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='automation:route-chat', session_id='chat',
                        request_id='bot:dm', payload={'text': 'hi', 'local_automation_v1': {'identity': 'bot:dm'}})
    assert claim_session_input(owner.db, epoch=owner.epoch, session_id='chat')['status'] == 'started'
    (response, answered), elapsed = ask('cli')
    assert (response, answered) == (UNDELIVERED_DECLINED, False), 'a Bot DM waited for an unseen question'
    assert elapsed < 5
    live.subscribers['viewer'] = Principal('uid', 'default', frozenset({'session:read', 'session:respond'}), 't')
    outcomes = []
    (response, answered), _ = ask('cli', on_sent=lambda key: outcomes.append(
        clarify_mod.attempt_text_response_for_session(key, 'blue')))
    assert (response, answered) == ('blue', True)
