"""An edit of a busy ``/queue`` or ``/steer`` command is its own input: the queued turn it becomes keeps
the edit's revision, so native admission keys it ``<message_id>:edit:<revision>`` instead of
refusing it as a changed replay of the original command (``admission_conflict``)."""
import pytest

from tests.gateway.test_messaging_slash_command_fifo import _rows, _start_a
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/queue', '/steer'])  # _Agent cannot steer: the queue fallback
async def test_edited_busy_command_is_queued_as_its_own_turn(monkeypatch, command):
    from hermes_agent.gateway.platforms.event import MessageEvent, MessageType

    rig = await _start_a(monkeypatch, 'queue')
    for text, revision in ((f'{command} C', None), (f'{command} C edited', 'u101')):
        await rig.adapter.handle_message(MessageEvent(text=text, message_type=MessageType.COMMAND,
                                                      source=rig.source, message_id='C', edit_revision=revision))
    await async_wait_until(lambda: len(_rows(rig)) >= 3 or any('admission_conflict' in s for s in rig.sent),
                           what='the edited command to be admitted')
    rig.finish.set()
    await async_wait_until(lambda: [m for m, _ in rig.order].count('C') >= 2, timeout=15, what='both turns to run')
    assert not [s for s in rig.sent if 'admission_conflict' in s or 'RuntimeStoreError' in s], rig.sent
    assert _rows(rig) == ['A', 'C', 'C:edit:u101'], _rows(rig)


@pytest.mark.asyncio
async def test_edited_goal_kickoff_is_admitted_as_its_own_turn(monkeypatch):
    """The same rebuild on the idle path: ``/goal`` kicks off with the command's message id, so an
    edited ``/goal`` (a replaced goal) needs the edit's revision on its kickoff turn too."""
    from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
    from hermes_agent.gateway.session_authority import initialize_session_authority
    from hermes_agent.gateway.session_results import execution_result
    from tests.gateway.test_messaging_slash_command_fifo import _owner

    runner, adapter, sent = _owner(monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key, ran = runner.session_store._generate_session_key(source), []

    async def turn(event, source, quick_key, run_generation):
        ran.append(event.text)
        if (captured := execution_result.get()) is not None:
            captured['result'] = {'final_response': 'ok', 'messages': []}
        return 'ok'

    async def no_judge(**_):  # no goal continuation turns between the two kickoffs
        return None
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    monkeypatch.setattr(runner, '_post_turn_goal_continuation', no_judge)
    rig = type('Rig', (), {'runner': runner, 'authority': authority, 'key': key})

    def idle():
        return not adapter._active_sessions and not runner._is_session_running(key)
    await adapter.handle_message(MessageEvent(text='hello', source=source, message_id='H'))
    await async_wait_until(lambda: ran and idle(), what='a session for the goal')
    for turns, (text, revision) in enumerate((('/goal write a haiku', None), ('/goal write a sonnet', 'u7')), 2):
        await adapter.handle_message(MessageEvent(text=text, message_type=MessageType.COMMAND,
                                                  source=source, message_id='G', edit_revision=revision))
        await async_wait_until(lambda: len(ran) >= turns and idle() or any('admission_conflict' in s for s in sent),
                               what=f'{text} kicked off')
    assert not [s for s in sent if 'admission_conflict' in s or 'RuntimeStoreError' in s], sent
    assert _rows(rig) == ['H', 'G', 'G:edit:u7'], _rows(rig)
    assert 'write a sonnet' in ran[-1], ran
