"""A physical worker excludes its whole FIFO lineage; an explicit reset revokes orphan unknowns."""
from pathlib import Path

import pytest

from persistence.core.errors import AuthorityRefusal
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input, recover_session_inputs
from persistence.sqlite.authority._mutation import mutate_runtime_session
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import get_session
from persistence.sqlite.workers._executions import adopt_worker_execution, register_worker_execution
from persistence_assembly import open_persistence, profile_scope, store_location
from tests.fixtures.storage_holder import raw_holder


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


def test_physical_worker_excludes_logical_fifo_and_second_registration(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        holder = raw_holder(store)
        epoch = begin_runtime_epoch(holder, instance_id='owner')
        logical = _local_reset(store, epoch, 'logical', 'physical')
        register_worker_execution(holder, epoch=epoch, execution_id='standalone-worker', session_id='physical',
                                  generation=0, kind='compute', adoption_secret='secret')
        admit_session_input(holder, epoch=epoch, principal_id='human', session_id=logical,
                            request_id='later', payload={'text': 'later'})
        assert claim_session_input(holder, epoch=epoch, session_id=logical) is None
        with pytest.raises(AuthorityRefusal, match='stale_generation'):
            register_worker_execution(holder, epoch=epoch, execution_id='second-worker', session_id=logical,
                generation=get_session(holder, logical)['runtime_generation'], kind='compute', adoption_secret='secret')


def test_reset_revokes_only_orphan_unknown_worker_and_unblocks_fifo(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        holder = raw_holder(store)
        epoch = begin_runtime_epoch(holder, instance_id='owner')
        logical = _local_reset(store, epoch, 'logical', 'physical')
        generation = get_session(holder, logical)['runtime_generation']
        register_worker_execution(holder, epoch=epoch, execution_id='standalone-worker', session_id=logical,
                                  generation=generation, kind='compute', adoption_secret='secret')
        epoch = begin_runtime_epoch(holder, instance_id='restarted')
        recover_session_inputs(holder, epoch=epoch)
        admit_session_input(holder, epoch=epoch, principal_id='human', session_id=logical,
                            request_id='later', payload={'text': 'later'})
        with pytest.raises(AuthorityRefusal, match='unknown_execution'):
            claim_session_input(holder, epoch=epoch, session_id=logical)
        receipt, _, _ = mutate_runtime_session(holder, epoch=epoch, principal_id='human', session_id=logical,
            request_id='explicit-reset', expected_revision=get_session(holder, logical)['runtime_revision'],
            expected_generation=get_session(holder, logical)['runtime_generation'],
            operation='reset', payload={})
        assert receipt['target_session_id'] != 'physical'
        assert holder._read_one('SELECT status FROM worker_executions')[0] == 'terminal'
        assert claim_session_input(holder, epoch=epoch, session_id=logical)['request_id'] == 'later'
        with pytest.raises(AuthorityRefusal, match='stale_generation'):
            adopt_worker_execution(holder, epoch=epoch, execution_id='standalone-worker',
                session_id=logical, generation=generation, adoption_secret='secret')
