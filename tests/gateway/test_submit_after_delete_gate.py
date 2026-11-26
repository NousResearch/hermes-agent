"""A submit waiting on deletion must revalidate the session before staging its input."""
import asyncio
import threading

import pytest

from gateway.session_contract import Submission
from tests.gateway.test_local_authority_transitions import local_session
from persistence_assembly import close_stores_under
from tests.fixtures.storage_read import session_row


@pytest.mark.asyncio
async def test_submit_waiting_on_delete_reports_not_found(tmp_path, monkeypatch):
    from gateway import session_mutations
    from persistence.core.errors import AuthorityRefusal

    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    attempted = asyncio.Event()
    runtime = session_mutations.authority_persistence(owner).runtime
    original = runtime.mutate

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, 'mutate', blocked)
    handle = owner._handle(ref)
    deletion = asyncio.create_task(session_mutations.mutate_session(owner, connection.actor, ref,
        dict(session_id=ref.session_id, request_id='delete', operation='delete', payload={},
             expected_revision=handle.revision, expected_generation=handle.execution_generation)))
    submission = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)

        async def send():
            attempted.set()
            return await owner.submit(connection.actor, Submission('next', ref, {'text': 'next'}, 'queue'))

        submission = asyncio.create_task(send())
        await asyncio.wait_for(attempted.wait(), 5)
        release.set()
        await asyncio.wait_for(deletion, 5)
        with pytest.raises(AuthorityRefusal, match='not_found'):
            await asyncio.wait_for(submission, 5)
        assert session_row(owner.persistence, ref.session_id) is None
    finally:
        release.set()
        await asyncio.gather(deletion, *([submission] if submission is not None else []), return_exceptions=True)
        await connection.close()
        close_stores_under(owner.persistence.scope.profile_dir)
