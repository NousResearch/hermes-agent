"""Stop of a queued admission the drain claims between the read and the cancel commit stops the
claim instead of refusing ``stale_generation`` while it runs (N57 class: hosted RPC, cron)."""
import asyncio
from types import SimpleNamespace

import pytest

from hermes_state_runtime import claim_session_input, list_session_admissions


def _claim_after_read(authority, session_id):
    """The drain claims the head after the caller read it ``queued``, before the cancel commits."""
    receipt = authority.receipt

    async def claimed(actor, ref, admission_id):
        seen = await receipt(actor, ref, admission_id)
        claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
        return seen
    authority.receipt = claimed


def _hosted(hosted_owner, tmp_path, monkeypatch):
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    authority, loop, principal, agent = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default',
                                 principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    rpc.submit(**coords, session_id=sid, prompt='input', task=TaskIdentity('room', 'task', 'thread', 'turn'),
               execution_generation=1, on_terminal=lambda receipt: None)
    _claim_after_read(authority, sid)
    reply = rpc.interrupt(**coords, session_id=sid, expected_task_id='task')
    [row] = list_session_admissions(authority.db, session_id=sid)
    return reply == {'interrupted': True, 'status': 'interrupted'} and row['status'] == 'started' and agent.interrupted


def _cron(hosted_owner, tmp_path, monkeypatch):
    from gateway import session_cron
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'cron.db')
    db.create_session('s', source='cron')
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='p', instance_id='t', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='t'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='cron-owner'), 'route')
    authority._schedule = lambda ref: None
    actor = Principal('cron-owner', 'p', frozenset({'session:create', 'session:submit', 'session:control'}), 'ticker')
    with db:
        row = admit_session_input(db, epoch=authority.epoch, principal_id='cron-owner', session_id='s',
                                  request_id='cron:job:fire', payload={'text': ''})
        _claim_after_read(authority, 's')
        reply = asyncio.run(session_cron.operation(
            authority, 'cancel', {'session_id': 's', 'admission_id': row['admission_id']}, actor))
        event = getattr(authority, '_cron_cancellations', {}).get(row['admission_id'])
        return reply == {'ok': True} and event is not None and event.is_set()


@pytest.mark.parametrize('surface', [_hosted, _cron], ids=['hosted_rpc_interrupt', 'cron_cancel'])
def test_stop_of_a_queued_row_claimed_mid_stop_stops_the_claim(surface, hosted_owner, tmp_path, monkeypatch):
    assert surface(hosted_owner, tmp_path, monkeypatch), 'the claimed execution was not told to stop'
