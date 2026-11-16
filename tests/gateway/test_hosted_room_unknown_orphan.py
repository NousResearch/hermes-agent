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
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, recover_session_inputs
    authority, loop, principal, _ = hosted_owner

    def rpc():
        return HostedRoomAuthorityRPC(
            authority, loop, room_id='room', member_id='member', profile='default',
            principal=principal, authorize=lambda *args: True)

    coords = {'profile': 'default', 'source': 'bot_room'}
    coords['session_id'] = rpc().create(**coords, title='Group: room')['session_id']
    rpc().submit(**coords, prompt='lost', task=TaskIdentity('room', 'lost', 'thread', 'turn'),
                 execution_generation=10, on_terminal=lambda receipt: None)
    row = claim_session_input(authority.db, epoch=authority.epoch, session_id=coords['session_id'])
    authority.epoch = begin_runtime_epoch(authority.db, instance_id='cold')
    recover_session_inputs(authority.db, epoch=authority.epoch)
    return authority, rpc, coords, row


def test_a_restart_lost_room_turn_settles_at_boot_recovery(hosted_owner):
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from hermes_state_runtime import list_session_admissions
    authority, _rpc, coords, row = _hosted_turn_fenced_unknown(hosted_owner)
    assert row['status'] == 'started' and row['request_id'].startswith('hosted:')
    assert settle_recovered_unattended(authority.db, authority.epoch) == 1
    settled = next(r for r in list_session_admissions(
        authority.db, session_id=coords['session_id'], pending_only=False)
        if r['admission_id'] == row['admission_id'])
    # Terminal and interrupted: the lost turn is never requeued or replayed.
    assert (settled['status'], settled['outcome']) == ('terminal', 'interrupted')


def test_the_room_admits_the_next_member_turn_after_that_settlement(hosted_owner):
    """The reported symptom: the pre-submit ``unknown`` refusal held every later generation out."""
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_unattended_unknown import settle_recovered_unattended
    authority, rpc, coords, _row = _hosted_turn_fenced_unknown(hosted_owner)

    def next_turn():
        return rpc().submit(**coords, prompt='next', task=TaskIdentity('room', 'next', 'thread', 'next'),
                            execution_generation=11, on_terminal=lambda receipt: None)

    from hermes_state_runtime import RuntimeStoreError
    with pytest.raises(RuntimeStoreError, match='unknown_execution'):
        next_turn()
    settle_recovered_unattended(authority.db, authority.epoch)  # the boot pass the restart runs
    receipt = next_turn()
    assert receipt['status'] in {'queued', 'started'}


def test_a_persons_own_lost_turn_still_waits_for_them(hosted_owner):
    """The room carve-out must not widen GW-10: a viewer's turn keeps pausing for ``/discard``."""
    from gateway.session_unattended_unknown import settle_recovered_unattended
    from hermes_state_runtime import (admit_session_input, begin_runtime_epoch, claim_session_input,
                                      list_session_admissions, recover_session_inputs)
    authority = hosted_owner[0]
    db = authority.db
    db.create_session('person', source='telegram')
    admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='person',
                        request_id='local-1', payload={'text': 'lost'})
    person = claim_session_input(db, epoch=authority.epoch, session_id='person')
    authority.epoch = begin_runtime_epoch(db, instance_id='cold-person')
    recover_session_inputs(db, epoch=authority.epoch)
    settle_recovered_unattended(db, authority.epoch)
    row = next(r for r in list_session_admissions(db, session_id='person', pending_only=False)
               if r['admission_id'] == person['admission_id'])
    assert row['status'] == 'unknown'
