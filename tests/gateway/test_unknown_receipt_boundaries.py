"""Receipt retries preserve transcript boundaries, successor ownership and media identity."""


from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession
from gateway.session_contract import Principal, SessionRef, Submission
from gateway.session_results import admission_result, finish_result
from hermes_state_runtime import (
    RuntimeStoreError,
    admit_session_input,
    begin_runtime_epoch,
    claim_session_input,
    get_session_admission,
    recover_session_inputs,
)


@pytest.mark.asyncio
@pytest.mark.parametrize('rotated', [False, True])
async def test_discard_closure_failure_keeps_unknown_fence_and_retry_is_exact(owner, monkeypatch, rotated):
    from gateway import session_results
    from gateway.session_ingress_media import capture_native_media
    owner.db.create_session('s', source='test')
    owner.sessions['s'] = LiveSession(None, 'route')
    ref = SessionRef(owner.profile_id, 's')
    actor = Principal('human', owner.profile_id, frozenset({'session:submit', 'session:control'}), 't')
    source = Path(owner.db.db_path).parent / 'input.png'
    source.write_bytes(b'image')
    references = capture_native_media([source])
    first = admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
        request_id='lost', payload={'text': 'LOST', 'attachments_v1': {'media': references, 'media_types': ['image/png']}})
    claim = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    target = 's'
    if rotated:
        owner.db.publish_compression_child(parent_session_id='s', child_session_id='child', source='test',
            messages=[{'role': 'assistant', 'content': 'summary'}], require_compression_lease=False)
        target = 'child'
    owner.db.append_message(target, 'user', 'LOST')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
                        request_id='next', payload={'text': 'FOLLOWER'})
    owner.epoch = begin_runtime_epoch(owner.db, instance_id='restart')
    recover_session_inputs(owner.db, epoch=owner.epoch)
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    close = session_results.close_discarded_turn
    def fail_after_close(*args, **kwargs):
        close(*args, **kwargs)
        raise OSError('closure failed')
    monkeypatch.setattr(session_results, 'close_discarded_turn', fail_after_close)
    with pytest.raises(OSError, match='closure failed'):
        await owner.resolve_unknown(actor, ref, first['admission_id'], claim['generation'])
    assert get_session_admission(owner.db, admission_id=first['admission_id'])['status'] == 'unknown'
    assert owner.db.latest_conversation_role(target) == 'user'
    assert Path(references[0]['path']).exists()
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    monkeypatch.setattr(session_results, 'close_discarded_turn', close)
    settled = await owner.resolve_unknown(actor, ref, first['admission_id'], claim['generation'])
    assert not Path(references[0]['path']).exists()
    assert owner.db.latest_conversation_role(target) == 'assistant'
    successor = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    assert successor['request_id'] == 'next'
    owner.db.append_message(target, 'user', 'FOLLOWER')
    assert await owner.resolve_unknown(actor, ref, first['admission_id'], claim['generation']) == settled
    assert owner.db.latest_conversation_role(target) == 'user', 'retry closed a successor turn'


@pytest.mark.asyncio
async def test_discard_closes_turn_lost_while_parked_on_a_tool(owner, monkeypatch):
    """A turn lost waiting on approval/clarify ends on an unanswered assistant ``tool_calls`` row.
    Discard must close it too, or repair prunes the call, drops the empty row and merges the
    discarded input into the follower's request."""
    from agent.agent_runtime_helpers import repair_message_sequence
    owner.db.create_session('s', source='test')
    owner.sessions['s'] = LiveSession(None, 'route')
    ref = SessionRef(owner.profile_id, 's')
    actor = Principal('human', owner.profile_id, frozenset({'session:submit', 'session:control'}), 't')
    first = admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s',
                                request_id='lost', payload={'text': 'LOST'})
    claim = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    owner.db.append_message('s', 'user', 'LOST')
    owner.db.append_message('s', 'assistant', '', tool_calls=[{'id': 'call_gate', 'type': 'function',
        'function': {'name': 'clarify', 'arguments': '{}'}}])
    owner.epoch = begin_runtime_epoch(owner.db, instance_id='restart')
    recover_session_inputs(owner.db, epoch=owner.epoch)
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    await owner.resolve_unknown(actor, ref, first['admission_id'], claim['generation'])
    history = owner.db.get_messages_as_conversation('s') + [{'role': 'user', 'content': 'FOLLOWER'}]
    repair_message_sequence(None, history)
    assert [m['content'] for m in history if m['role'] == 'user'] == ['LOST', 'FOLLOWER'], history
