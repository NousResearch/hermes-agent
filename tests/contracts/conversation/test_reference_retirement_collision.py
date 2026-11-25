"""A reused worker ID cannot replace retired execution evidence.

Every check reads through public verbs, so the case runs on any backend: the session row, the admission's stored payload, the execution record ``verify_claim`` answers, and the stored receipts, which still replay their results (the usage total does not move).
"""

import pytest

from persistence import ExecutionGeneration, PermanentStoreError, SessionId
from persistence.conversation.sessions import NewSession
from persistence.workers.executions import WorkerKind
from tests.contracts.conversation._session_probes import is_retired
from tests.contracts.workers._support import begin_epoch
from tests.contracts._scopes import MEDIA_ROOT


def _evidence(store, sid, worker, admission_id):
    """The retired-execution evidence a failed delete must leave as it was, through public reads."""
    record = store.worker_executions.verify_claim(execution_id=worker["execution_id"], session_id=sid, generation=ExecutionGeneration(0),
                                                  adoption_secret="adoption-" + sid)
    return (store.sessions.get(sid), store.admissions.get(admission_id=admission_id).payload, record)


@pytest.mark.parametrize("reuse", [False, True])
def test_worker_tombstone_collision_rolls_back_retirement(backend, scope_a, reuse):
    store_a = backend.open(scope_a)
    epoch = begin_epoch(backend, scope_a)
    for sid, execution_id in [("old-session", "worker"), ("new-session", "worker" if reuse else "other-worker")]:
        sid = SessionId(sid)
        store_a.sessions.create(NewSession(session_id=sid, source="cli"))
        admission = store_a.admissions.admit(epoch=epoch, principal_id="principal", session_id=sid, request_id="request", payload={"text": sid})[0]
        store_a.admissions.cancel(epoch=epoch, admission_id=admission.admission_id)
        worker = dict(epoch=epoch, execution_id=execution_id, session_id=sid, generation=0)
        store_a.worker_executions.register(**worker, kind=WorkerKind.COMPUTE, adoption_secret="adoption-" + sid)
        store_a.worker_executions.record_usage(**worker, sequence=1, input_tokens=1)
        store_a.worker_executions.finish(**worker, sequence=2)
        if sid == "new-session" and reuse:
            before = _evidence(store_a, sid, worker, admission.admission_id)
            for _ in range(2):
                with pytest.raises(PermanentStoreError):
                    store_a.sessions.delete(sid, media_root=MEDIA_ROOT)
                assert not is_retired(backend, scope_a, sid)
                assert _evidence(store_a, sid, worker, admission.admission_id) == before
                assert before[1] == {"text": sid}
                # Both receipts are still stored: their exact retries replay without re-applying.
                store_a.worker_executions.record_usage(**worker, sequence=1, input_tokens=1)
                store_a.worker_executions.finish(**worker, sequence=2)
                assert store_a.sessions.get(sid).input_tokens == 1
        else:
            assert store_a.sessions.delete(sid, media_root=MEDIA_ROOT)
            assert store_a.sessions.get(sid) is None
            assert is_retired(backend, scope_a, sid)
    assert store_a.sessions.get(SessionId("old-session")) is None
    assert is_retired(backend, scope_a, SessionId("old-session"))
