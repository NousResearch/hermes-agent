"""Worker refusals keep their place and leave no trace: a check main makes before the store answers whatever the
epoch, a check main's handler makes answers after the epoch fence, and neither touches the execution or the transcript."""

import sqlite3

import pytest

from persistence.core.errors import AuthorityRefusal
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.workers._support import EXECUTION, SESSION, ident, legacy_persist, parity, seed_worker, store_path

__all__ = ["legacy_store"]


def _untouched(path):
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT status,last_sequence FROM worker_executions WHERE execution_id=?",
                            (EXECUTION,)).fetchall() == [("registered", 0)]
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (SESSION,)).fetchone()[0] == 0


@pytest.mark.parametrize("stale", [False, True])
@pytest.mark.parametrize("operation", ["adopt", "role", "sequence"])
def test_preflight_refusal_preserves_cause_without_mutation(backend, scope_a, legacy_store, stale, operation):
    from hermes_state_runtime import adopt_worker_execution

    path = store_path(scope_a)
    epoch = seed_worker(path)
    store = backend.open(scope_a)
    used = 0 if stale else epoch
    if operation == "adopt":
        args = dict(epoch=used, execution_id=EXECUTION, session_id=SESSION, generation=0, adoption_secret="")
        legacy, verb = (lambda db: adopt_worker_execution(db, **args)), (lambda: store.worker_executions.adopt(**args))
        reason = "invalid_params"
    else:
        idn = ident(used, 0 if operation == "sequence" else 1)
        message = {"role": "bad" if operation == "role" else "user", "content": "text"}
        legacy = legacy_persist("transcript.append", {"messages": [message], "turn_lease_holder": None}, idn)
        verb = lambda: store.worker_executions.append_transcript(**idn, messages=[message], turn_lease_holder=None)
        reason = "stale_epoch" if stale and operation == "role" else "invalid_params"
    _before, after = parity(path, legacy, verb)
    assert isinstance(after[1], AuthorityRefusal) and isinstance(after[1], ValueError)
    assert after[1].reason == reason
    _untouched(path)


def test_list_content_is_refused_only_by_the_epoch_fence(backend, scope_a):
    path = store_path(scope_a)
    seed_worker(path)
    store = backend.open(scope_a)
    idn = ident(0)
    message = {"role": "user", "content": []}
    _before, after = parity(path, legacy_persist("transcript.append", {"messages": [message], "turn_lease_holder": None}, idn),
                           lambda: store.worker_executions.append_transcript(**idn, messages=[message], turn_lease_holder=None))
    assert after[1].reason == "stale_epoch"
    _untouched(path)
