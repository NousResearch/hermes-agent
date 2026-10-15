"""Independent review probes: mutation publication survives caller cancellation and retirement."""
import asyncio
from dataclasses import asdict, replace
import json
import threading

import pytest

from gateway.session_contract import Submission
from tests.gateway.test_local_authority_transitions import local_session


async def pending_model(tmp_path, monkeypatch):
    from gateway import session_mutations
    from gateway.session_policy import restore_policy
    owner, connection, ref = await local_session(tmp_path, monkeypatch)
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = session_mutations.mutate_runtime_session

    def blocked(*args, **kwargs):
        if kwargs.get('_prepare_only'):
            return original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        try:
            return original(*args, **kwargs)
        finally:
            completed.set()

    async def prepare(authority, live, payload, prepared):
        old = restore_policy(prepared['snapshot']['receipt']['policy'])
        config = old.config()
        config['model'] = {'default': 'new', 'provider': 'fixture'}
        return {**prepared, 'policy': asdict(replace(old, model='new', config_json=json.dumps(config)))}

    monkeypatch.setattr(session_mutations, 'mutate_runtime_session', blocked)
    monkeypatch.setattr('gateway.session_mutation_model.prepare_model', prepare)
    handle = owner._handle(ref)
    task = asyncio.create_task(session_mutations.mutate_session(owner, connection.actor, ref,
        dict(session_id=ref.session_id, request_id='model', operation='model', payload={'model': 'new'},
             expected_revision=handle.revision, expected_generation=handle.execution_generation)))
    assert await asyncio.to_thread(entered.wait, 5)
    return owner, connection, ref, task, release, completed


@pytest.mark.asyncio
async def test_cancelling_model_request_cannot_strand_next_admission(tmp_path, monkeypatch):
    owner, connection, ref, task, release, completed = await pending_model(tmp_path, monkeypatch)
    seen = []
    async def execute(authority, ref, row):
        seen.append(row['request_id'])
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    try:
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await asyncio.to_thread(completed.wait, 5)
        await owner.submit(connection.actor, Submission('next', ref, {'text': 'next'}, 'queue'))
        await owner.sessions[ref.session_id].task
        assert seen == ['next']
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.to_thread(completed.wait, 5)
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_profile_retirement_waits_for_mutation_thread(tmp_path, monkeypatch):
    from gateway.run_runtime import _retire_profile_authority
    owner, connection, _ref, task, release, completed = await pending_model(tmp_path, monkeypatch)
    retirement = asyncio.create_task(_retire_profile_authority(owner))
    try:
        # Retirement must not finish while the writer can still mutate this profile.
        done, _ = await asyncio.wait({retirement}, timeout=2)
        assert not done, 'profile authority retired while its mutation writer remained alive'
        release.set()
        await asyncio.gather(task, retirement)
    finally:
        release.set()
        await asyncio.gather(task, retirement, return_exceptions=True)
        await asyncio.to_thread(completed.wait, 5)
        await connection.close()
        owner.db.close()
