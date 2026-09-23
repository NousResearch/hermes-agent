"""Resolving an unknown admission must also retire its physical worker assignment."""
from contextlib import closing
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
import hermes_state_runtime as rt


def hello(process):
    return {'type': 'hello', 'pid': process.pid, 'birth': psutil.Process(process.pid).create_time(),
            'ancestors': [os.getpid()]}


def _local_reset(db, epoch, logical, child):
    """Publish a local policy receipt and a reset child so the physical transcript
    differs from the logical owner (the managed-worker layout after ``/new``)."""
    from hermes_state_local import commit_local_session
    from hermes_state_local_lineage import reset_local_target
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import build_policy
    from dataclasses import asdict
    sid = local_identity('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': '/', 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    reset = SessionEntry(entry.session_key, child, now, now, origin=source, platform=Platform.LOCAL, is_fresh_reset=True)
    reset_local_target(db, epoch=epoch, parent_session_id=sid, entry=reset.to_dict())
    return sid


def test_resolve_unknown_retires_the_physical_worker_row(tmp_path):
    from gateway.session_worker_reservation import reserve_admission_worker
    with closing(SessionDB(tmp_path / 'state.db')) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        logical = _local_reset(db, epoch, 'logical', 'physical')
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=logical,
                               request_id='first', payload={'text': 'first'})
        rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=logical,
                               request_id='second', payload={'text': 'second'})
        row = rt.claim_session_input(db, epoch=epoch, session_id=logical)
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id='profile', _require_admission_open=lambda: None)
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                                             principal_id='human', hello=hello(child))
            assert scope['session_id'] == 'physical' and scope['session_id'] != logical
        finally:
            child.kill()
            child.wait(timeout=5)
        # Owner restart: the started admission and its worker become unknown.
        epoch = rt.begin_runtime_epoch(db, instance_id='restarted')
        rt.recover_session_inputs(db, epoch=epoch)
        assert db._read_one('SELECT status FROM worker_executions')[0] == 'unknown'
        settled = rt.resolve_unknown_session_input(db, epoch=epoch, admission_id=row['admission_id'],
                                                   generation=row['generation'])
        assert settled['status'] == 'terminal'
        assert db._read_one('SELECT status FROM worker_executions')[0] == 'terminal', \
            'Discard left the physical worker assignment unknown'
        # The follower must now be claimable and a fresh managed child reservable on the physical row.
        follower = rt.claim_session_input(db, epoch=epoch, session_id=logical)
        assert follower is not None and follower['request_id'] == 'second'
        authority = SimpleNamespace(db=db, epoch=epoch, profile_id='profile', _require_admission_open=lambda: None)
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=follower['admission_id'], process=child,
                                             principal_id='human', hello=hello(child))
            assert scope['session_id'] == 'physical'
        finally:
            child.kill()
            child.wait(timeout=5)
