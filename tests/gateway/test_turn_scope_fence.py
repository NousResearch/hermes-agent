"""A turn the authority executes runs under a ``TurnScope`` carrying its claim's fence.

The fence is the owner epoch and runtime generation the claim stamped on the admission row, so a
store can reject a superseded owner's late write. Real GatewayRunner, DiscordAdapter,
SessionAuthority and drain over a real SessionDB; only the agent's conversation is a stand-in.
"""
import pytest

from tests.gateway.test_messaging_stop_owner_turn import _owner, _until


@pytest.mark.asyncio
async def test_each_claimed_turn_binds_its_admission_fence(monkeypatch):
    from gateway.platforms.event import MessageEvent
    from gateway.session_authority import initialize_session_authority
    from gateway.session_results import execution_result
    from hermes_scope import LOCAL_OWNER, Fence, TurnScope, current_scope
    from hermes_state_runtime import list_session_admissions

    runner, adapter, _sent = _owner(monkeypatch)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    key = runner.session_store._generate_session_key(source)
    seen = {}

    async def turn(event, source, quick_key, run_generation):
        seen[event.text] = current_scope()
        execution_result.get()['result'] = {'final_response': 'done', 'messages': []}
        return 'done'
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)

    for text in ('A', 'B'):
        await adapter.handle_message(MessageEvent(text=text, source=source, message_id=text))
        await _until(lambda: text in seen)
    owner = runner.session_store.peek_session_id(key)
    await _until(lambda: [r['status'] for r in list_session_admissions(
        authority.db, session_id=owner, pending_only=False)] == ['terminal', 'terminal'])
    rows = {r['request_id']: r for r in list_session_admissions(authority.db, session_id=owner, pending_only=False)}
    for text in ('A', 'B'):
        scope = seen[text]
        assert isinstance(scope, TurnScope), f'turn {text} ran without a TurnScope: {scope!r}'
        assert (scope.session, scope.turn) == (owner, rows[text]['admission_id'])
        assert scope.fence == Fence(authority.epoch, rows[text]['generation'])
        assert (scope.owner, scope.role) == (LOCAL_OWNER, 'owner')
    assert seen['B'].fence.generation == seen['A'].fence.generation + 1
