"""Producer retries retain the first admission's route context, not a later speaker.

Native automation still inherits that speaker's memory attribution when it has no
explicit author. The borrowed speaker is route context, not the machine producer.
"""
import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session_ingress_context import native_callback
from hermes_state_runtime import RuntimeStoreError, list_session_admissions
from tests.gateway.test_automation_wake_shared_thread_speaker import _native_owner


def _wake(entry, kind):
    return dict(type=kind, session_id='proc_fixture', delegation_id='deleg_fixture',
                event_id='stable-notice', started_at=1.0, session_key=entry.session_key,
                parent_session_id=entry.session_id, platform='discord', chat_id='42',
                chat_type='thread', thread_id='43', exit_code=0, output='done')


async def _third_speaker(runner, authority, home, monkeypatch):
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '100,200,300')
    adapter = runner.adapters[Platform.DISCORD]
    later = MessageEvent(text='third participant', message_id='human-2',
        source=adapter.build_source(chat_id='42', chat_type='thread', thread_id='43',
                                    user_id='300', user_name='third', message_id='human-2'))
    with native_callback(runner, later, home):
        await authority.admit_native(later)


def _wakes(authority, session_id):
    return [row for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)
            if row['payload'].get('native_text_v1', {}).get('automation')]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['completion', 'watch_match', 'watch_disabled', 'async_delegation'])
async def test_retry_after_third_speaker_keeps_first_admission(tmp_path, monkeypatch, kind):
    runner, authority, entry = await _native_owner(tmp_path, monkeypatch)
    evt = _wake(entry, kind)
    try:
        assert await runner._inject_watch_notification('[notice]', evt, raise_not_accepted=True) is True
        before = _wakes(authority, entry.session_id)
        await _third_speaker(runner, authority, tmp_path, monkeypatch)
        assert await runner._inject_watch_notification('[notice]', evt, raise_not_accepted=True) is True
        assert _wakes(authority, entry.session_id) == before
        assert before[0]['payload']['native_text_v1']['source']['user_id'] == '200'
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('conflict', ['text', 'reset', 'adapter'])
async def test_retry_keeps_content_owner_and_adapter_guards(tmp_path, monkeypatch, conflict):
    from gateway.session_automation import admit_process_notice
    runner, authority, entry = await _native_owner(tmp_path, monkeypatch)
    evt = _wake(entry, 'watch_match')
    adapter = runner.adapters[Platform.DISCORD]
    pin = entry.session_id
    try:
        assert await runner._inject_watch_notification('[notice]', evt, raise_not_accepted=True) is True
        before = _wakes(authority, pin)
        await _third_speaker(runner, authority, tmp_path, monkeypatch)
        if conflict == 'reset':
            runner.session_store.reset_session(entry.session_key)
        event = MessageEvent(text='changed' if conflict == 'text' else before[0]['payload']['text'], internal=True,
            source=runner._restored_source(entry), metadata={'gateway_session_key': entry.session_key,
                                                          'gateway_session_id': pin})
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            await admit_process_notice(runner, authority, object() if conflict == 'adapter' else adapter, event, evt)
        assert _wakes(authority, pin) == before
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_plugin_keyed_injection_after_speaker_change(tmp_path, monkeypatch):
    runner, authority, entry = await _native_owner(tmp_path, monkeypatch)
    # Keep authorization unchanged: both the route's origin and the last speaker are allowed.
    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '100,200')
    runner._running = True
    try:
        assert await runner._dispatch_plugin_message_injection(
            session_key=entry.session_key, content='plugin notice', plugin_id='fixture-plugin') is True
        wakes = _wakes(authority, entry.session_id)
        assert len(wakes) == 1
        assert wakes[0]['payload']['native_text_v1']['source']['user_id'] == '200'
        assert 'message_id' not in wakes[0]['payload']['native_text_v1']['source']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_native_wake_runner_receives_borrowed_speaker_attribution(tmp_path, monkeypatch):
    """The borrowed human is route context; native automation currently has no machine author stamp."""
    from types import SimpleNamespace
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import admission_author, execute_admission
    from gateway.turn_context import TurnContext
    runner, authority, entry = await _native_owner(tmp_path, monkeypatch)
    captured = []

    def infer(message, *, turn_author, **kwargs):
        captured.append((message, turn_author))
        return {'final_response': 'ok', 'messages': []}

    async def turn(event):
        assert event.internal
        assert admission_author.get() is None
        ctx = TurnContext(source=event.source, session_key=entry.session_key,
                          session_id=entry.session_id, message=event.text)
        worker = TurnRunner(runner, ctx)
        worker._run_conversation_with_approval(SimpleNamespace(run_conversation=infer), [], None, None, None)
        return 'ok'

    monkeypatch.setattr(runner, '_handle_message', turn)
    try:
        assert await runner._inject_watch_notification('[notice]', _wake(entry, 'watch_match'),
                                                      raise_not_accepted=True) is True
        row = _wakes(authority, entry.session_id)[0]
        await execute_admission(authority, SessionRef(authority.profile_id, entry.session_id), row)
        assert captured == [(row['payload']['text'], {'id': '200', 'name': 'human', 'is_bot': False})]
    finally:
        authority.db.close()
