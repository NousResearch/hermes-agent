"""Reference worker addressing matches the runtime assignment lookup."""

import pytest

from tests.contracts._reference import InMemoryBackend

from persistence import PermanentStoreError, ProofRejected
from persistence.conversation.sessions import NewSession
from persistence.workers.executions import WorkerKind
from tests.contracts.workers._support import refused


@pytest.fixture
def backend():
    backend = InMemoryBackend()
    try:
        yield backend
    finally:
        backend.close()


@pytest.mark.parametrize("value", [None, False, {"n": 1}])
def test_adoption_does_not_stringify_session_identity(backend, scope_a, value):
    store_a = backend.open(scope_a)
    sid = str(value)
    store_a.sessions.create(NewSession(session_id=sid, source="cli"))
    epoch = store_a.runtime.begin_epoch(instance_id="owner")
    args = dict(epoch=epoch, execution_id="worker", generation=0, adoption_secret="proof")
    store_a.worker_executions.register(**args, session_id=sid, kind=WorkerKind.COMPUTE)
    with pytest.raises(ProofRejected):
        store_a.worker_executions.adopt(**args, session_id=value)
    assert (store_a.worker_executions.adopt(**args, session_id=sid)).session_id == sid


def test_adoption_binds_the_execution_identifier_before_lookup(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id="s", source="cli"))
    epoch = store_a.runtime.begin_epoch(instance_id="owner")
    args = dict(epoch=epoch, session_id="s", generation=0, adoption_secret="proof")
    store_a.worker_executions.register(**args, execution_id="1", kind=WorkerKind.COMPUTE)
    assert (store_a.worker_executions.adopt(**args, execution_id=True)).execution_id == "1"
    with refused("not_found"):
        store_a.worker_executions.adopt(**args, execution_id=None)
    with pytest.raises(PermanentStoreError, match="Error binding parameter 1"):
        store_a.worker_executions.adopt(**args, execution_id=["1"])
