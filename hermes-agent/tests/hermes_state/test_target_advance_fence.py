"""Target-advancing mutations (reset, compress/model prepare) fence executing work only.

A queued follower is what /reset and compression exist to serve: it waits on the logical
owner and runs against whatever physical target the mutation publishes.
"""
from contextlib import closing
from dataclasses import asdict

import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def _local_session(db, epoch, cwd):
    from hermes_state_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import build_policy
    sid = local_identity('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(cwd), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return sid


def _mutate(db, epoch, sid, operation, **extra):
    session = db.get_session(sid)
    return rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid,
        request_id=operation + '-' + str(session['runtime_revision']), operation=operation,
        expected_revision=session['runtime_revision'], expected_generation=session['runtime_generation'],
        payload={}, **extra)


@pytest.mark.parametrize('operation', ['reset', 'compress'])
def test_target_advance_allows_queued_follower_and_refuses_started_head(tmp_path, operation):
    prepare = {'_prepare_only': True} if operation == 'compress' else {}
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid = _local_session(db, epoch, tmp_path)
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                               request_id='head', payload={'text': 'head'})
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=sid,
                               request_id='follower', payload={'text': 'follower'})
        # Head and follower both queued (a paused FIFO): the mutation proceeds.
        receipt = _mutate(db, epoch, sid, operation, **prepare)
        assert ('snapshot' in receipt) if operation == 'compress' else receipt['target_session_id'] != sid
        statuses = [r['status'] for r in rt.list_session_admissions(db, session_id=sid)]
        assert statuses == ['queued', 'queued']
        # Once the head is claimed the same mutation is refused and nothing moves.
        head = rt.claim_session_input(db, epoch=epoch, session_id=sid)
        before = db.get_session(sid)
        with pytest.raises(rt.RuntimeStoreError, match='session_busy'):
            _mutate(db, epoch, sid, operation, **prepare)
        after = db.get_session(sid)
        assert (after['runtime_generation'], after['runtime_revision']) == (
            before['runtime_generation'], before['runtime_revision'])
        # The head still settles against its own generation, and the follower stays queued.
        rt.settle_session_input(db, epoch=epoch, admission_id=head['admission_id'],
                                generation=head['generation'], outcome='completed')
        assert [r['request_id'] for r in rt.list_session_admissions(db, session_id=sid)] == ['follower']
