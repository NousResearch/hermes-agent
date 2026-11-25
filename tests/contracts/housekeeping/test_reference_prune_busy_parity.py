"""Pruning cannot retire a session with a started admission."""

import pytest

from persistence.conversation.sessions import NewSession

from tests.contracts._scopes import MEDIA_ROOT


@pytest.mark.parametrize("busy", [False, True])
def test_prune_checks_admission_ledger(backend, scope_a, busy):
    store_a = backend.open(scope_a)
    sid = "prune-ledger"
    store_a.sessions.create(NewSession(sid, "cli"))
    if busy:
        epoch = store_a.runtime.begin_epoch(instance_id="prune-test")
        store_a.admissions.admit(epoch=epoch, principal_id="human", session_id=sid, request_id="req", payload={"input": "work"})
        assert store_a.admissions.claim(epoch=epoch, session_id=sid) is not None
    store_a.sessions.end(sid, "done")
    # A row whose ledger holds work is skipped, never refused.
    assert store_a.maintenance.prune_sessions(None, media_root=MEDIA_ROOT).deleted_ids == (() if busy else (sid,))
    assert (store_a.sessions.get(sid) is not None) is busy
