"""Target-advancing mutations (reset, compress/model prepare) fence executing work only.

A queued follower is what /reset and compression exist to serve: it waits on the logical
owner and runs against whatever physical target the mutation publishes.
"""
from dataclasses import asdict

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import (
    admit_session_input, claim_session_input, list_session_admissions, settle_session_input,
)
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import get_session


def _local_session(store, epoch, cwd):
    from gateway.session_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from persistence.authority.bindings import local_session_id
    from gateway.session_policy import build_policy
    sid = local_session_id('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(cwd), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(store, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return sid


def _mutate(holder, epoch, sid, operation, **extra):
    session = get_session(holder, sid)
    return mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id=sid,
        request_id=operation + '-' + str(session['runtime_revision']), operation=operation,
        expected_revision=session['runtime_revision'], expected_generation=session['runtime_generation'],
        payload={}, **extra)[0]


@pytest.mark.parametrize('operation', ['reset', 'compress'])
def test_target_advance_allows_queued_follower_and_refuses_started_head(store, tmp_path, operation):
    prepare = {'prepare_only': True} if operation == 'compress' else {}
    holder = store.holder()
    epoch = begin_runtime_epoch(holder, instance_id='owner')
    sid = _local_session(store, epoch, tmp_path)
    admit_session_input(holder, epoch=epoch, principal_id='human', session_id=sid,
                        request_id='head', payload={'text': 'head'})
    admit_session_input(holder, epoch=epoch, principal_id='human', session_id=sid,
                        request_id='follower', payload={'text': 'follower'})
    # Head and follower both queued (a paused FIFO): the mutation proceeds.
    receipt = _mutate(holder, epoch, sid, operation, **prepare)
    assert ('snapshot' in receipt) if operation == 'compress' else receipt['target_session_id'] != sid
    statuses = [r['status'] for r in list_session_admissions(holder, session_id=sid)]
    assert statuses == ['queued', 'queued']
    # Once the head is claimed the same mutation is refused and nothing moves.
    head = claim_session_input(holder, epoch=epoch, session_id=sid)
    before = get_session(holder, sid)
    with pytest.raises(AuthorityRefusal, match='session_busy'):
        _mutate(holder, epoch, sid, operation, **prepare)
    after = get_session(holder, sid)
    assert (after['runtime_generation'], after['runtime_revision']) == (
        before['runtime_generation'], before['runtime_revision'])
    # The head still settles against its own generation, and the follower stays queued.
    settle_session_input(holder, epoch=epoch, admission_id=head['admission_id'],
                         generation=head['generation'], outcome='completed')
    assert [r['request_id'] for r in list_session_admissions(holder, session_id=sid)] == ['follower']
