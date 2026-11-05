"""Structured results are claim-owned and visible only after settlement."""
import pytest

from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_result_survives_restart_without_reexecuting(tmp_path):
    from gateway.session_authority import admission_row
    from gateway.session_results import retain_result, admission_result
    db = storage_handle(tmp_path)
    seed_session(db, 'api-session', 'api_server')
    epoch = db.runtime.begin_epoch(instance_id='first')
    admitted, _ = db.admissions.admit(epoch=epoch, principal_id='api', session_id='api-session',
                                      request_id='retry', payload={'text': 'hello'})
    row = admission_row(db.admissions.claim(epoch=epoch, session_id='api-session'))
    result = {'final_response': 'reply', 'messages': [], 'usage': {'input_tokens': 7, 'output_tokens': 3}}
    retain_result(storage_handle(tmp_path), epoch=epoch, row=row, result=result)
    assert admission_result(storage_handle(tmp_path), admitted.admission_id) == result
    db.close()
    db = storage_handle(tmp_path)
    try:
        epoch = db.runtime.begin_epoch(instance_id='second')
        db.admissions.recover(epoch=epoch)
        retried, _ = db.admissions.admit(epoch=epoch, principal_id='api', session_id='api-session',
                                         request_id='retry', payload={'text': 'hello'})
        assert retried.admission_id == admitted.admission_id
        assert admission_result(storage_handle(tmp_path), retried.admission_id) == result
        assert db.admissions.claim(epoch=epoch, session_id='api-session') is None
    finally:
        db.close()


def test_stale_result_cannot_overwrite_settled_or_unknown_claim(tmp_path):
    from gateway.session_authority import admission_row
    from gateway.session_results import retain_result, admission_result
    store = storage_handle(tmp_path)
    try:
        seed_session(store, 'api-session', 'api_server')
        epoch = store.runtime.begin_epoch(instance_id='first')
        store.admissions.admit(epoch=epoch, principal_id='api', session_id='api-session',
                               request_id='one', payload={'text': 'hello'})
        row = admission_row(store.admissions.claim(epoch=epoch, session_id='api-session'))
        retain_result(store, epoch=epoch, row=row, result={'final_response': 'first'})
        with pytest.raises(AuthorityRefusal, match='stale_generation'):
            retain_result(store, epoch=epoch, row=row, result={'final_response': 'late'})
        assert admission_result(store, row['admission_id']) == {'final_response': 'first'}
        newer = store.runtime.begin_epoch(instance_id='second')
        with pytest.raises(AuthorityRefusal, match='stale_epoch'):
            retain_result(store, epoch=epoch, row=row, result={})
        assert newer != epoch
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('fails', [True, False], ids=['agent-raised', 'plain-reply'])
async def test_a_turn_that_failed_before_running_settles_failed_on_every_surface(tmp_path, monkeypatch, fails):
    """A non-API admission whose agent turn raised before producing a TurnRunner result (agent
    initialization failure) must not settle ``completed`` with the apology as its output: ACP would
    answer the editor ``end_turn`` and a finite CLI run would exit 0. A handler reply that is not a
    failure (a notice, a command answer) still completes."""
    import asyncio
    from types import SimpleNamespace
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef, Submission

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = storage_handle(tmp_path)
    seed_session(db, 's', 'telegram')
    gateway = object.__new__(GatewayRunner)

    async def stop_typing(event, source):
        return None
    gateway._hmwa_stop_typing_for_turn = stop_typing
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='human')

    async def handle(event):
        if not fails:
            return 'Noted.'
        # The production except-body of the agent turn (agent construction raised).
        return await gateway._hmwa_agent_error_reply(RuntimeError('provider init failed'), event, source, None,
                                                     'k', gateway._PreparedTurn([], '', None, None, None, None))
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner',
                                 epoch=db.runtime.begin_epoch(instance_id='owner'), persistence=db)
    authority.sessions['s'] = LiveSession(source, 'route')
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    try:
        receipt = await authority.submit(SessionActor('human', 'p', frozenset({'session:submit'}), 't'),
                                         Submission('r1', SessionRef('p', 's'), {'text': 'hello'}, 'queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        outcome = 'failed' if fails else 'completed'
        assert db.admissions.get(admission_id=receipt.admission_id).outcome == outcome
        complete = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert complete[-1]['outcome'] == outcome
        assert complete[-1]['status'] == ('error' if fails else 'complete')
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('text,outcome', [('do the work', 'failed'), ('/status', 'completed')],
                         ids=['refused-prompt', 'command-answer'])
async def test_a_finite_prompt_answered_without_running_settles_failed(tmp_path, monkeypatch, text, outcome):
    """``hermes chat -q`` exits on the committed outcome. A handler that answers a finite prompt
    with a notice and never runs the turn (drain gate, lease timeout, any stub refusal) did not do
    the asked work, so the admission settles failed and the CLI exits non-zero; a finite slash
    command's reply IS its answer and still completes."""
    import asyncio
    from types import SimpleNamespace
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_results import admission_result
    from hermes_cli.turn_exit import turn_exit_code

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = storage_handle(tmp_path)
    seed_session(db, 's', 'cli')

    async def handle(event):
        return 'Gateway is draining for maintenance; your message was not processed.'
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None, _handle_message=handle)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner',
                                 epoch=db.runtime.begin_epoch(instance_id='owner'), persistence=db)
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c', user_id='human'), 'route')
    try:
        receipt = await authority.submit(SessionActor('human', 'p', frozenset({'session:submit'}), 't'),
                                         Submission('r1', SessionRef('p', 's'), {'text': text, 'finite': True}, 'queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 10)
        assert db.admissions.get(admission_id=receipt.admission_id).outcome == outcome
        result = admission_result(authority.persistence, receipt.admission_id)['result']
        assert turn_exit_code(result, kanban_worker=False) == (1 if outcome == 'failed' else 0)
    finally:
        db.close()
