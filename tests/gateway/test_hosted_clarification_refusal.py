"""Persisted hosted policies decline a question without a reply channel; local ones still ask."""
import asyncio
import json
import threading
import time
from dataclasses import replace

import pytest

from tools import clarify_gateway as clarify_mod
from tests.fixtures.storage_seed import seed_session


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
    from gateway.run_turn_runner_clarify_delivery import UNDELIVERED_NO_VIEWER
    (response, answered), elapsed = ask('bot_room')
    assert (response, answered) == (UNDELIVERED_NO_VIEWER, False)
    assert elapsed < 5, 'the member waited for an answer nobody can submit'


def test_local_choice_prompt_still_accepts_a_typed_answer(ask):
    outcomes = []
    (response, answered), _ = ask('cli', on_sent=lambda key: outcomes.append(
        clarify_mod.attempt_text_response('q-cli', 'something else entirely')))
    assert outcomes == [clarify_mod.TEXT_RESOLVED]
    assert (response, answered) == ('something else entirely', True)


def test_clarify_without_an_answering_viewer_is_declined(ask, owner):
    """N8: the decline keys on who can answer, not on one source string. A forwarded A2A peer
    never can; a producer turn (Bot Chat DM) can only while a viewer holding session:respond is
    attached. A viewer's own turn keeps waiting for its author (the local test above)."""
    from gateway.run_turn_runner_clarify_delivery import UNDELIVERED_NO_VIEWER
    from gateway.session_authority import LiveSession
    from gateway.session_contract import SessionActor
    forwarded = {'source': 'a2a', 'a2a_identity': ['dev', '', 'peer', 'ctx']}
    (response, answered), elapsed = ask('a2a', request=forwarded)
    assert (response, answered) == (UNDELIVERED_NO_VIEWER, False) and elapsed < 5
    seed_session(owner.persistence, 'chat', 'local')
    owner.sessions['chat'] = live = LiveSession(object(), 'route-chat')
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id='automation:route-chat', session_id='chat',
                                       request_id='bot:dm', payload={'text': 'hi', 'local_automation_v1': {'identity': 'bot:dm'}})
    assert owner.persistence.admissions.claim(epoch=owner.epoch, session_id='chat').status == 'started'
    (response, answered), elapsed = ask('cli')
    assert (response, answered) == (UNDELIVERED_NO_VIEWER, False), 'a Bot DM waited for an unseen question'
    assert elapsed < 5
    live.subscribers['viewer'] = SessionActor('uid', 'default', frozenset({'session:read', 'session:respond'}), 't')
    outcomes = []
    (response, answered), _ = ask('cli', on_sent=lambda key: outcomes.append(
        clarify_mod.attempt_text_response('q-cli', 'blue')))
    assert (response, answered) == ('blue', True)


def test_automation_clarify_with_no_viewer_is_not_reported_as_a_connector_decline(owner, tmp_path, monkeypatch, caplog):
    """#159160: a wake turn (Bot DM, cron, kanban) whose question nobody attached can answer is
    refused by the local adapter itself. No connector or relay is involved, so neither the log nor
    the model may read it as a connector egress decline ("destination refused"): the model is told
    no viewer can answer, the decline names the automation turn and the viewers attached (attach
    state is never persisted, so the log is the only record), and the turn is still released at once."""
    import logging
    from types import SimpleNamespace

    from gateway import run_turn_runner_clarify_delivery as delivery
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_authority import LiveSession
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import build_policy

    adapter = LocalSessionAdapter(owner)
    adapter.policies['chat'] = build_policy({'source': 'cli', 'cwd': str(tmp_path), 'model': 'm'}, {})
    seed_session(owner.persistence, 'chat', 'local')
    owner.sessions['chat'] = LiveSession(object(), 'route-chat')
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id='automation:route-chat', session_id='chat',
                                       request_id='bot:dm', payload={'text': 'hi', 'local_automation_v1': {'identity': 'bot:dm'}})
    wake = owner.persistence.admissions.claim(epoch=owner.epoch, session_id='chat')
    assert wake.status == 'started'

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(clarify_mod, 'get_clarify_timeout', lambda: 30)
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(_status_adapter=adapter, _status_chat_id='chat', _status_thread_metadata=None,
                                  session_key='route-chat', stream_consumer_holder=[None], _loop_for_step=loop)
    runner._close_native_stream_boundary = lambda *a, **k: None
    try:
        with caplog.at_level(logging.INFO):
            started = time.monotonic()
            reply = runner._clarify_callback_sync(
                [{'qid': 'q0', 'question': 'Which?', 'choices': ['one', 'two'], 'multi_select': False}])
            elapsed = time.monotonic() - started
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)

    logged = [record.getMessage() for record in caplog.records]
    assert not [line for line in logged if 'egress' in line.lower()], logged
    assert any(wake.admission_id in line and '0 viewer(s) attached' in line for line in logged), logged
    assert reply == {'answers': {}, 'outcome': 'undelivered', 'notice': delivery.UNDELIVERED_NO_VIEWER}
    assert elapsed < 5, 'the wake turn waited for a question nobody can see'
    assert clarify_mod.get_pending_for_session('route-chat', include_choice_prompts=True) is None
