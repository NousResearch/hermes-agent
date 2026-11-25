"""Deletion covers live descendants before their durable claim."""
from types import SimpleNamespace
import pytest
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_delete_cannot_remove_a_live_delegate_without_a_claim(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as db:
        seed_session(db, 's', 'test')
        seed_session(db, 'child', 'test', parent_session_id='s', model_config={'_delegate_from': 's'})
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned', instance_id='owner', epoch=epoch, persistence=storage_handle(tmp_path))
        authority.sessions['s'] = LiveSession(None, 'route')
        authority.sessions['child'] = LiveSession(None, 'child-route')
        # The descendant's execution is a ledger row, not an in-memory task: it must
        # fence the parent's deletion in the same transaction that would delete it.
        raw_holder(db)._execute_write(lambda conn: conn.execute(
            "INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
            "VALUES('child-exec','child','child',?,0,'running','digest')", (epoch,)))
        owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
        try:
            response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {'session_id': 's',
                'request_id': 'delete', 'expected_revision': 0, 'expected_generation': 0,
                'operation': 'delete', 'payload': {}}})
            assert response['error']['message'] == 'session_busy'
            assert db.sessions.get('child') is not None
        finally:
            await owner.close()
