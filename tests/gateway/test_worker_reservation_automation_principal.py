"""A managed worker adopts an automation admission's reservation as the owner's own OS user (#144139)."""
import os
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle

AUTOMATION = 'automation:agent:main:local:dm:owned'


def _hello(process):
    from hermes_cli.process_identity import worker_birth
    return {'type': 'hello', 'pid': process.pid, 'birth': worker_birth(psutil.Process(process.pid)),
            'ancestors': [os.getpid()]}


@pytest.fixture
def reserved(tmp_path):
    """A started automation admission (a bot DM or completion wake) and the owner's live direct child."""
    db = storage_handle(tmp_path)
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        seed_session(db, 'owned', 'cli')
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SimpleNamespace(persistence=db, epoch=epoch, profile_id=str(tmp_path),
                                    _require_admission_open=lambda: None, wake_after_worker=lambda session_id: None)
        admitted, _ = db.admissions.admit(epoch=epoch, principal_id=AUTOMATION, session_id='owned',
                                          request_id='bot:dm', payload={})
        db.admissions.claim(epoch=epoch, session_id='owned')
        yield authority, admitted, process
    finally:
        process.stdin.close()
        process.wait(timeout=5)


def _connection(authority, subject):
    actor = SimpleNamespace(subject=subject, profile_id=authority.profile_id, capabilities={'worker:adopt'})
    return SimpleNamespace(authority=authority, actor=actor)


@pytest.mark.asyncio
async def test_worker_adopts_an_automation_admission_as_the_owner_user(reserved):
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import owner_worker_subject, reserve_admission_worker
    authority, admitted, process = reserved
    scope = reserve_admission_worker(authority, admission_id=admitted.admission_id, process=process,
                                     principal_id=AUTOMATION, hello=_hello(process),
                                     worker_subject=owner_worker_subject())
    identity = {k: v for k, v in scope.items() if k != 'epoch'}
    receipt = await worker_request(_connection(authority, owner_worker_subject()), None, identity, operation='adopt')
    assert receipt['execution_id'] == scope['execution_id']


@pytest.mark.asyncio
async def test_the_claim_still_refuses_any_other_subject(reserved):
    from gateway.session_worker import worker_request
    from gateway.session_worker_reservation import owner_worker_subject, reserve_admission_worker
    authority, admitted, process = reserved
    scope = reserve_admission_worker(authority, admission_id=admitted.admission_id, process=process,
                                     principal_id=AUTOMATION, hello=_hello(process),
                                     worker_subject=owner_worker_subject())
    identity = {k: v for k, v in scope.items() if k != 'epoch'}
    for subject in (AUTOMATION, 'uid:999999'):
        with pytest.raises(Exception, match='permission_denied'):
            await worker_request(_connection(authority, subject), None, identity, operation='adopt')
