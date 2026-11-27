"""Resolving an unknown admission must also retire its physical worker assignment."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from persistence_assembly import open_persistence, profile_scope, store_location
from tests.fixtures.storage_holder import raw_holder
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def hello(process):
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)),
            'ancestors': [os.getpid()]}


def _local_reset(store, epoch, logical, child):
    """Publish a local policy receipt and a reset child so the physical transcript
    differs from the logical owner (the managed-worker layout after ``/new``)."""
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local import commit_local_session
    from persistence.authority.bindings import local_session_id
    from gateway.session_policy import build_policy
    from dataclasses import asdict
    sid = local_session_id('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(Path(store_location(store.scope)).parent), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(store, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    reset = SessionEntry(entry.session_key, child, now, now, origin=source, platform=Platform.LOCAL, is_fresh_reset=True)
    store.bindings.reset_target(epoch=epoch, parent_session_id=sid, entry=reset.to_dict())
    return sid


def test_resolve_unknown_retires_the_physical_worker_row(tmp_path):
    from gateway.session_worker_reservation import reserve_admission_worker
    with open_persistence(profile_scope(home=tmp_path)) as store:
        epoch = store.runtime.begin_epoch(instance_id='owner')
        logical = _local_reset(store, epoch, 'logical', 'physical')
        store.admissions.admit(epoch=epoch, principal_id='human', session_id=logical,
                               request_id='first', payload={'text': 'first'})
        store.admissions.admit(epoch=epoch, principal_id='human', session_id=logical,
                               request_id='second', payload={'text': 'second'})
        row = store.admissions.claim(epoch=epoch, session_id=logical)
        authority = SimpleNamespace(epoch=epoch, profile_id='profile', _require_admission_open=lambda: None,
                                    persistence=storage_handle(tmp_path))
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=row.admission_id, process=child,
                                             principal_id='human', hello=hello(child))
            assert scope['session_id'] == 'physical' and scope['session_id'] != logical
        finally:
            child.kill()
            child.wait(timeout=5)
        # Owner restart: the started admission and its worker become unknown.
        epoch = store.runtime.begin_epoch(instance_id='restarted')
        store.admissions.recover(epoch=epoch)
        assert raw_holder(store)._read_one('SELECT status FROM worker_executions')[0] == 'unknown'
        settled = store.admissions.resolve_unknown(epoch=epoch, admission_id=row.admission_id,
                                                   generation=row.generation)
        assert settled.status == 'terminal'
        assert raw_holder(store)._read_one('SELECT status FROM worker_executions')[0] == 'terminal', \
            'Discard left the physical worker assignment unknown'
        # The follower must now be claimable and a fresh managed child reservable on the physical row.
        follower = store.admissions.claim(epoch=epoch, session_id=logical)
        assert follower is not None and follower.request_id == 'second'
        authority = SimpleNamespace(epoch=epoch, profile_id='profile', _require_admission_open=lambda: None,
                                    persistence=storage_handle(tmp_path))
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=follower.admission_id, process=child,
                                             principal_id='human', hello=hello(child))
            assert scope['session_id'] == 'physical'
        finally:
            child.kill()
            child.wait(timeout=5)
