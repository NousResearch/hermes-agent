"""A hosted-room member turn lost across a restart settles instead of wedging the room (#155688).

A ``bot_room`` admission is a producer admission — ``gateway/session_operator.is_viewer_input``
already classifies its ``hosted:`` request id as outside a viewer's own input. Nobody is attached
to the hidden room session to answer "resend it?", so a restart-lost one used to stay ``unknown``:
the head pauses the room's FIFO, ``HostedRoomAuthorityRPC._submit`` refuses every later member turn
with ``unknown_execution``, and the room stays dead until an operator runs
``hermes -p <profile> sessions discard <session> --admission <id>`` by hand.

Both settle ports (boot recovery, the drain's ``unknown_execution``) go through
``_unattended_unknown``, so the classification is what the room needs. A person's turn keeps
pausing: that is GW-10's rule, and its own test covers it.
"""
import pytest


def _hosted_turn_fenced_unknown(hosted_owner):
    """The reported state: a claimed hosted member turn, fenced ``unknown`` by an owner restart."""
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from persistence.gateway.hosted_rooms import HostedTaskIdentity
    authority, loop, principal, _ = hosted_owner
    store = authority.persistence

    def rpc():
        return HostedRoomAuthorityRPC(
            authority, loop, room_id='room', member_id='member', profile='default',
            principal=principal, authorize=lambda *args: True)

    coords = {'profile': 'default', 'source': 'bot_room'}
    coords['session_id'] = rpc().create(**coords, title='Group: room')['session_id']
    rpc().submit(**coords, prompt='lost', task=HostedTaskIdentity('room', 'lost', 'thread', 'turn'),
                 execution_generation=10, on_terminal=lambda receipt: None)
    row = store.admissions.claim(epoch=authority.epoch, session_id=coords['session_id'])
    authority.epoch = store.runtime.begin_epoch(instance_id='cold')
    store.admissions.recover(epoch=authority.epoch)
    return authority, rpc, coords, row


def test_a_restart_lost_room_turn_settles_at_boot_recovery(hosted_owner):
    from gateway.session_unattended_unknown import settle_recovered_unattended
    authority, _rpc, coords, row = _hosted_turn_fenced_unknown(hosted_owner)
    store = authority.persistence
    assert row.status == 'started' and row.request_id.startswith('hosted:')
    assert settle_recovered_unattended(store, authority.epoch) == 1
    settled = next(r for r in store.admissions.list(
        session_id=coords['session_id'], pending_only=False)
        if r.admission_id == row.admission_id)
    # Terminal and interrupted: the lost turn is never requeued or replayed.
    assert (settled.status, settled.outcome) == ('terminal', 'interrupted')


def test_the_room_admits_the_next_member_turn_after_that_settlement(hosted_owner):
    """The reported symptom: the pre-submit ``unknown`` refusal held every later generation out."""
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from persistence.core.errors import AuthorityRefusal
    from persistence.gateway.hosted_rooms import HostedTaskIdentity
    authority, rpc, coords, _row = _hosted_turn_fenced_unknown(hosted_owner)

    def next_turn():
        return rpc().submit(**coords, prompt='next', task=HostedTaskIdentity('room', 'next', 'thread', 'next'),
                            execution_generation=11, on_terminal=lambda receipt: None)

    with pytest.raises(AuthorityRefusal, match='unknown_execution'):
        next_turn()
    settle_recovered_unattended(authority.persistence, authority.epoch)  # the boot pass the restart runs
    receipt = next_turn()
    assert receipt['status'] in {'queued', 'started'}


def test_a_persons_own_lost_turn_still_waits_for_them(hosted_owner):
    """The room carve-out must not widen GW-10: a viewer's turn keeps pausing for ``/discard``."""
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from tests.fixtures.storage_seed import seed_session
    authority = hosted_owner[0]
    store = authority.persistence
    seed_session(store, 'person', 'telegram')
    store.admissions.admit(epoch=authority.epoch, principal_id='human', session_id='person',
                           request_id='local-1', payload={'text': 'lost'})
    person = store.admissions.claim(epoch=authority.epoch, session_id='person')
    authority.epoch = store.runtime.begin_epoch(instance_id='cold-person')
    store.admissions.recover(epoch=authority.epoch)
    settle_recovered_unattended(store, authority.epoch)
    row = next(r for r in store.admissions.list(session_id='person', pending_only=False)
               if r.admission_id == person.admission_id)
    assert row.status == 'unknown'
