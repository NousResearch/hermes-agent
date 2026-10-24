"""Exercise presentation through the authenticated dispatcher and native entry.
No socket, model provider or external action is needed.
"""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_contract import SessionRef
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tools import clarify_gateway as cm


@pytest.fixture
def live_question(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('s', source='test')
        epoch = begin_runtime_epoch(db, instance_id='current')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id=str(tmp_path),
                                     instance_id='current', db=db, epoch=epoch)
        connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
        source = SimpleNamespace(platform=Platform.LOCAL, user_id=connection.actor.subject)
        live = LiveSession(source, 'presentation-route')
        authority.sessions['s'] = live
        live.subscribers['viewer'] = connection.actor
        admit_session_input(db, epoch=epoch, principal_id='human', session_id='s',
                            request_id='input', payload={'text': 'question'})
        generation = claim_session_input(db, epoch=epoch, session_id='s')['generation']
        entry = cm.register('presentation', live.route, 'Choose', ['a', 'b'])
        authority.register_clarify('s', generation, entry)
        params = dict(session_id='s', execution_generation=generation, prompt_id=entry.clarify_id, state='presented')
        yield authority, connection, live, entry, params
        cm.clear_session(live.route)


async def dispatch(connection, params):
    return await connection.dispatch({'id': 1, 'method': 'clarify.presentation', 'params': params})


@pytest.mark.asyncio
async def test_exact_open_prompt_records_without_answering(live_question):
    authority, connection, _live, entry, params = live_question
    result = await dispatch(connection, params)
    assert result.get('result') == {'status': 'recorded', 'prompt_id': entry.clarify_id}, result
    assert entry.response is None and not entry.event.is_set()
    assert (await dispatch(connection, params))['result']['status'] == 'already_recorded'
    await authority.detach(connection.actor, 'viewer')
    assert entry.presentation_invalidated


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['generation', 'prompt', 'route', 'membership', 'transport', 'kind', 'remote', 'profile'])
async def test_uncertain_or_foreign_receipt_cannot_qualify(live_question, change):
    authority, connection, live, entry, params = live_question
    changes = {
        'generation': lambda: params.update(execution_generation=params['execution_generation'] + 1),
        'prompt': lambda: params.update(prompt_id='other'),
        'route': lambda: setattr(entry, 'session_key', 'different-route'),
        'membership': live.subscribers.clear,
        'transport': lambda: authority.events.pop(connection.actor.transport_id),
        'kind': lambda: live.controls.pending[entry.clarify_id][1].update(kind='approval'),
        'remote': lambda: live.controls.remote_responders.update({entry.clarify_id: lambda *args: None}),
        'profile': lambda: params.update(profile='unserved'),
    }
    changes[change]()
    result = await dispatch(connection, params)
    assert 'error' in result or result['result']['status'] in {'ineligible', 'already_resolved'}, result
    assert entry.presented_at is None and entry.response is None


@pytest.mark.asyncio
async def test_overflow_withdraws_presenter_and_refuses_replay(live_question):
    authority, connection, _live, entry, params = live_question
    assert (await dispatch(connection, params))['result']['status'] == 'recorded'
    authority._retire_overflowed('s', connection.transport)
    assert entry.presentation_invalidated
    assert 'error' in await dispatch(connection, params)


@pytest.mark.parametrize('evidence', ['presented', 'unknown', 'interaction'])
def test_authority_waiter_and_real_tool_result_keep_partial_answers(live_question, monkeypatch, evidence):
    import asyncio
    import json
    from concurrent.futures import Future
    from gateway.platforms.base import SendResult
    from gateway.run_turn_runner import TurnRunner
    from tools.clarify_tool import clarify_tool

    authority, connection, live, _initial, params = live_question
    cm.clear_session(live.route)
    now = SimpleNamespace(value=0.0)
    monkeypatch.setattr(cm, 'time', SimpleNamespace(monotonic=lambda: now.value))
    monkeypatch.setattr(cm, 'get_clarify_timeout', lambda: 10)
    real_register = cm.register
    attempted = []

    def register(**kwargs):
        entry = real_register(**kwargs)
        monkeypatch.setattr(entry.event, 'wait', lambda timeout: setattr(now, 'value', now.value + timeout) or False)
        attempted.append(entry)
        return entry

    monkeypatch.setattr(cm, 'register', register)

    class Adapter:
        async def send_clarify(self, **kwargs):
            return SendResult(success=True)

        def pause_typing_for_chat(self, chat_id):
            pass

        def resume_typing_for_chat(self, chat_id):
            pass

    turn = object.__new__(TurnRunner)
    turn._approval_owner = (authority, 's', params['execution_generation'])
    turn._ctx = SimpleNamespace(source=live.source, _status_adapter=Adapter(), _status_chat_id='s',
                                _status_thread_metadata={}, session_key=live.route, stream_consumer_holder=[None])
    turn._close_native_stream_boundary = lambda *args, **kwargs: None

    def schedule(coro, label):
        async def send():
            result = await coro
            entry = attempted[-1]
            if len(attempted) == 1:
                await authority.respond(connection.actor,
                    SessionRef(authority.profile_id, 's'),
                    params['execution_generation'], entry.clarify_id, {'answer': 'b'}, kind='clarify')
            elif evidence != 'unknown':
                receipt = dict(params, prompt_id=entry.clarify_id)
                assert (await dispatch(connection, receipt))['result']['status'] == 'recorded'
                if evidence == 'interaction':
                    await dispatch(connection, dict(receipt, state='interaction'))
            return result

        future = Future()
        future.set_result(asyncio.run(send()))
        return future

    turn._schedule = schedule
    result = json.loads(clarify_tool([{'question': f'Question {i}', 'choices': ['a', 'b']} for i in range(3)],
                                   callback=turn._clarify_callback_sync))
    assert len(attempted) == 2
    assert result['outcome'] == 'timed_out'
    assert [row['user_response'] for row in result['responses']] == ['b', None, None]
    assert 'question 2' in result['notice'].lower()
    assert 'not attempted' in result['notice']
    expected = {'presented': 'full native wait of 10 seconds elapsed',
                'unknown': 'presentation unknown', 'interaction': 'presentation invalidated'}
    assert expected[evidence] in result['notice']


@pytest.mark.asyncio
async def test_shared_transport_keeps_equal_session_ids_in_their_profile(tmp_path, monkeypatch):
    from contextlib import ExitStack
    from pathlib import Path
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    root = tmp_path / '.hermes'
    monkeypatch.setenv('HERMES_HOME', str(root))
    homes = [root, root / 'profiles' / 'secondary']
    runner = SimpleNamespace(_draining=False, session_authorities=[])
    entries = []
    with ExitStack() as stack:
        for index, home in enumerate(homes):
            home.mkdir(parents=True, exist_ok=True)
            db = stack.enter_context(SessionDB(db_path=home / 'state.db'))
            db.create_session('same', source='test')
            epoch = begin_runtime_epoch(db, instance_id='current')
            authority = SessionAuthority(runner, profile_id=str(home), instance_id='current', db=db, epoch=epoch)
            runner.session_authorities.append(authority)
            authority.sessions['same'] = LiveSession(SimpleNamespace(platform=Platform.LOCAL, user_id='uid:human'), f'route-{index}')
            admit_session_input(db, epoch=epoch, principal_id='uid:human', session_id='same',
                                request_id='input', payload={'text': 'question'})
            generation = claim_session_input(db, epoch=epoch, session_id='same')['generation']
            entry = cm.register(f'profile-{index}', f'route-{index}', 'Choose', ['a', 'b'])
            entries.append(entry)
            authority.register_clarify('same', generation, entry)
        connection = AuthorityConnection(runner.session_authorities[0], object(),
            {'user_id': 'human', 'profile_scope': 'host'})
        secondary = runner.session_authorities[1]
        params = dict(session_id='same', execution_generation=generation, prompt_id=entries[1].clarify_id,
                      state='presented', profile='secondary')
        try:
            # Routing alone is not an attachment and cannot grant presentation authority.
            assert (await dispatch(connection, params))['error']['message'] == 'permission_denied'
            sibling = connection._siblings[str(homes[1])]
            secondary.sessions['same'].source.user_id = sibling.actor.subject
            secondary.sessions['same'].subscribers['viewer'] = sibling.actor
            assert (await dispatch(connection, params))['result']['status'] == 'recorded'
            assert entries[0].presented_at is None and entries[1].presented_at is not None
            wrong = dict(params, profile='default', prompt_id=entries[1].clarify_id)
            assert 'error' in await dispatch(connection, wrong)
        finally:
            for entry in entries:
                cm.clear_session(entry.session_key)
