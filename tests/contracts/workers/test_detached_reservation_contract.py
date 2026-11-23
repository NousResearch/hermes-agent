"""Detached worker declarations are fenced, secret-free reservation metadata."""

import pytest

from tests.contracts.workers._support import SECRET, SESSION, refused, seed


@pytest.mark.parametrize('detach_socket', [None, '/fixture/worker.sock'])
def test_reservation_publishes_only_active_listener_metadata(backend, scope_a, detach_socket):
    epoch = seed(backend, scope_a)
    store = backend.open(scope_a)
    admitted, _ = store.admissions.admit(epoch=epoch, principal_id='owner', session_id=SESSION,
                                         request_id='detached-fixture', payload={'text': 'fixture'})
    store.admissions.claim(epoch=epoch, session_id=SESSION)
    executions = store.worker_executions
    scope = executions.reserve_for_admission(epoch=epoch, admission_id=admitted.admission_id,
        principal_id='owner', profile_id='fixture', pid=42, birth=123.0, secret=SECRET, owner=None,
        detach_socket=detach_socket)
    expected = () if detach_socket is None else (dict(admission_id=admitted.admission_id,
        execution_id=scope.execution_id, session_id=scope.session_id, generation=scope.generation,
        pid=42, birth=123.0, socket=detach_socket),)
    assert executions.detached_reservations(epoch=epoch) == expected
    with refused('stale_epoch'):
        executions.detached_reservations(epoch=epoch + 1)
    executions.lose_reservation(epoch=epoch, admission_id=admitted.admission_id,
                                execution_id=scope.execution_id, generation=scope.generation)
    assert not executions.detached_reservations(epoch=epoch)
