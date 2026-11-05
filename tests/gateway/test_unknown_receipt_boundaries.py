"""Receipt retries preserve transcript boundaries, successor ownership and media identity."""


from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession
from gateway.session_contract import SessionActor, SessionRef, Submission
from gateway.session_results import admission_result, finish_result
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.core.errors import AuthorityRefusal
from persistence_assembly import store_location
from tests.fixtures.storage_read import conversation
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.mark.asyncio
@pytest.mark.parametrize('rotated', [False, True])
async def test_discard_closure_failure_keeps_unknown_fence_and_retry_is_exact(owner, monkeypatch, rotated):
    from gateway.session_ingress_media import capture_native_media
    from tests.fixtures.storage_holder import admission_store_operation
    seed_session(owner.persistence, 's', 'test')
    owner.sessions['s'] = LiveSession(None, 'route')
    ref = SessionRef(owner.profile_id, 's')
    actor = SessionActor('human', owner.profile_id, frozenset({'session:submit', 'session:control'}), 't')
    source = Path(store_location(owner.persistence.scope)).parent / 'input.png'
    source.write_bytes(b'image')
    references = capture_native_media([source])
    first, _ = owner.persistence.admissions.admit(epoch=owner.epoch, principal_id='human', session_id='s',
        request_id='lost', payload={'text': 'LOST', 'attachments_v1': {'media': references, 'media_types': ['image/png']}})
    claim = owner.persistence.admissions.claim(epoch=owner.epoch, session_id='s')
    target = 's'
    if rotated:
        owner.persistence.lineage.publish_compression_child('s', CompressionChild(session_id='child', source='test'),
            [NewMessage.from_live({'role': 'assistant', 'content': 'summary'})], require_compression_lease=False)
        target = 'child'
    seed_messages(owner.persistence, target, {'role': 'user', 'content': 'LOST'})
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id='human', session_id='s',
                                       request_id='next', payload={'text': 'FOLLOWER'})
    owner.epoch = owner.persistence.runtime.begin_epoch(instance_id='restart')
    owner.persistence.admissions.recover(epoch=owner.epoch)
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    close = admission_store_operation('close_discarded_turn')
    def fail_after_close(*args, **kwargs):
        close(*args, **kwargs)
        raise OSError('closure failed')
    monkeypatch.setattr('persistence.sqlite.authority._admission_store.close_discarded_turn', fail_after_close)
    with pytest.raises(OSError, match='closure failed'):
        await owner.resolve_unknown(actor, ref, first.admission_id, claim.generation)
    assert owner.persistence.admissions.get(admission_id=first.admission_id).status == 'unknown'
    assert owner.persistence.messages.latest_conversation_role(target) == 'user'
    assert Path(references[0]['path']).exists()
    with pytest.raises(AuthorityRefusal, match='unknown_execution'):
        owner.persistence.admissions.claim(epoch=owner.epoch, session_id='s')
    monkeypatch.setattr('persistence.sqlite.authority._admission_store.close_discarded_turn', close)
    settled = await owner.resolve_unknown(actor, ref, first.admission_id, claim.generation)
    from gateway.session_runtime_workers import mutation_tasks
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: not mutation_tasks(owner), timeout=5, what='off-loop media collection')
    assert not Path(references[0]['path']).exists()
    assert owner.persistence.messages.latest_conversation_role(target) == 'assistant'
    successor = owner.persistence.admissions.claim(epoch=owner.epoch, session_id='s')
    assert successor.request_id == 'next'
    seed_messages(owner.persistence, target, {'role': 'user', 'content': 'FOLLOWER'})
    assert await owner.resolve_unknown(actor, ref, first.admission_id, claim.generation) == settled
    assert owner.persistence.messages.latest_conversation_role(target) == 'user', 'retry closed a successor turn'


@pytest.mark.asyncio
async def test_discard_closes_turn_lost_while_parked_on_a_tool(owner, monkeypatch):
    """A turn lost waiting on approval/clarify ends on an unanswered assistant ``tool_calls`` row.
    Discard must close it too, or repair prunes the call, drops the empty row and merges the
    discarded input into the follower's request."""
    from agent.agent_runtime_helpers import repair_message_sequence
    seed_session(owner.persistence, 's', 'test')
    owner.sessions['s'] = LiveSession(None, 'route')
    ref = SessionRef(owner.profile_id, 's')
    actor = SessionActor('human', owner.profile_id, frozenset({'session:submit', 'session:control'}), 't')
    first, _ = owner.persistence.admissions.admit(epoch=owner.epoch, principal_id='human', session_id='s',
                                                  request_id='lost', payload={'text': 'LOST'})
    claim = owner.persistence.admissions.claim(epoch=owner.epoch, session_id='s')
    seed_messages(owner.persistence, 's', {'role': 'user', 'content': 'LOST'},
                  {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_gate', 'type': 'function',
                   'function': {'name': 'clarify', 'arguments': '{}'}}]})
    owner.epoch = owner.persistence.runtime.begin_epoch(instance_id='restart')
    owner.persistence.admissions.recover(epoch=owner.epoch)
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    await owner.resolve_unknown(actor, ref, first.admission_id, claim.generation)
    history = conversation(owner.persistence, 's') + [{'role': 'user', 'content': 'FOLLOWER'}]
    repair_message_sequence(None, history)
    assert [m['content'] for m in history if m['role'] == 'user'] == ['LOST', 'FOLLOWER'], history
