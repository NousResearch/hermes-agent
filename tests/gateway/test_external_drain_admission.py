"""External maintenance blocks new work, not accepted work or its settlement."""
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from hermes_state_runtime import RuntimeStoreError


@pytest.mark.parametrize('transactional', [False, True])
@pytest.mark.parametrize('external,restart,wait,retiring,options,refused', [
    (False, False, False, False, {}, False),
    (True, False, False, False, {}, True),
    (True, False, False, False, {'queue_only': True}, True),
    (True, False, False, False, {'existing_work': True}, False),
    (False, True, False, False, {'existing_work': True}, True),
    (False, True, True, False, {'existing_work': True}, True),
    (False, True, True, False, {'queue_only': True, 'existing_work': True}, False),
    (False, True, True, False, {'queue_only': True}, False),
    (True, True, True, False, {'queue_only': True}, True),
    (True, False, False, True, {'existing_work': True}, True),
])
def test_admission_policy(transactional, external, restart, wait, retiring, options, refused):
    authority = SessionAuthority.__new__(SessionAuthority)
    authority.runner = SimpleNamespace(_draining=restart, _running=True,
        session_runtime_descriptor={'drain_reason': 'restart_wait' if wait else ''})
    if external:
        authority.runner._external_drain_active = True
    authority.retiring = retiring
    authorized = []
    conn = object()

    def admit():
        if transactional:
            authority._admission_gate(authorized.append, **options)(conn)
        else:
            authority._require_admission_open(**options)

    if refused:
        with pytest.raises(RuntimeStoreError) as exc:
            admit()
        assert exc.value.reason == 'runtime_draining'
        assert authorized == []
    else:
        admit()
        assert authorized == ([conn] if transactional else [])


def test_transaction_rechecks_external_drain():
    authority = SessionAuthority.__new__(SessionAuthority)
    authority.runner = SimpleNamespace(_draining=False)
    authority.retiring = False
    authorized = []
    gate = authority._admission_gate(authorized.append, queue_only=True)
    authority.runner._external_drain_active = True
    with pytest.raises(RuntimeStoreError, match='runtime_draining'):
        gate(object())
    assert authorized == []
