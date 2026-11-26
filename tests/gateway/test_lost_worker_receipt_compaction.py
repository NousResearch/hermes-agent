"""A managed worker closed by loss or by a reset's orphan discard keeps no full-history read
receipt (dokterdok N41, the N19 residual): every terminal path compacts like execution.finish."""
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import psutil

from gateway.session_authority import admission_row
from hermes_cli.process_identity import worker_birth
from persistence.authority.runtime import Reset
from persistence.conversation.messages import NewMessage
from persistence_assembly import open_persistence, profile_scope, store_location
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

READ = dict(target='s', include_ancestors=False, include_inactive=False, repair_alternation=False,
            include_row_ids=False, include_compacted=False)


def hello(process):
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)), 'ancestors': [os.getpid()]}


def _receipt_sizes(db):
    return [n for (n,) in raw_holder(db)._read_all('SELECT LENGTH(result_json) FROM worker_receipts ORDER BY sequence')]


def _local_session(store, epoch):
    """Commit a local session (policy receipt and route), the conversation a reset publishes a fresh target for."""
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local import commit_local_session
    from gateway.session_policy import build_policy
    from persistence.authority.bindings import local_session_id
    sid = local_session_id('profile', 'human', 'r')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    now = _now()
    entry = SessionEntry('local:' + sid, sid, now, now, origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'cli', 'cwd': str(Path(store_location(store.scope)).parent), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'cli': []}}, private_secrets={})
    commit_local_session(store, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'r', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return sid


def test_lost_admission_worker_compacts_its_history_receipt(tmp_path):
    from gateway.session_worker_reservation import lose_admission_worker, reserve_admission_worker
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'cli')
        db.messages.append_batch('s', [NewMessage.from_live({'role': 'user', 'content': 'x' * 4000})])
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SimpleNamespace(epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None,
                                    persistence=storage_handle(tmp_path))
        db.admissions.admit(epoch=epoch, principal_id='human', session_id='s', request_id='input', payload={})
        row = admission_row(db.admissions.claim(epoch=epoch, session_id='s'))
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            scope = reserve_admission_worker(authority, admission_id=row['admission_id'], process=child,
                                             principal_id='human', hello=hello(child))
            history = db.worker_executions.persist_operation(epoch=epoch, execution_id=scope['execution_id'], session_id='s',
                generation=scope['generation'], sequence=1, operation='compression.history', payload=READ)
            assert history['messages'][0]['content'] == 'x' * 4000
            lose_admission_worker(authority, row, scope)
            assert raw_holder(db)._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
            assert max(_receipt_sizes(db)) < 200, 'a lost worker kept a full-history receipt'
        finally:
            child.stdin.close()
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)


def test_reset_discarded_orphan_worker_compacts_its_history_receipt(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        epoch = db.runtime.begin_epoch(instance_id='old-owner')
        sid = _local_session(db, epoch)
        db.messages.append_batch(sid, [NewMessage.from_live({'role': 'user', 'content': 'x' * 4000})])
        scope = dict(epoch=epoch, execution_id='orphan', session_id=sid, generation=0)
        db.worker_executions.register(**scope, kind='compute', adoption_secret='secret')
        db.worker_executions.persist_operation(**scope, sequence=1, operation='compression.history',
                                               payload=dict(READ, target=sid))
        epoch = db.runtime.begin_epoch(instance_id='new-owner')
        db.admissions.recover(epoch=epoch)  # prior owner's compute -> unknown
        assert raw_holder(db)._read_one('SELECT status FROM worker_executions')[0] == 'unknown'
        db.runtime.mutate(epoch, principal_id='human', session_id=sid, request_id='reset', expected_revision=0,
                          command=Reset(), expected_generation=0, owner=None)
        assert raw_holder(db)._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
        assert max(_receipt_sizes(db)) < 200, 'a reset-discarded worker kept a full-history receipt'
