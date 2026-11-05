"""Worker title allocation checks the authority fence before its payload."""

from persistence import AuthorityEpoch
from tests.contracts.workers._support import SESSION, ident, refused, worker


def test_title_epoch_fence_precedes_invalid_title(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    with refused("stale_epoch"):
        store.worker_executions.next_session_title(**ident(AuthorityEpoch(epoch + 1)), base_title=None)
    with refused("invalid_params"):
        store.worker_executions.next_session_title(**ident(epoch), base_title=None)
    # The refused payload left the sequence free.
    store.sessions.set_title(SESSION, "Plan")
    assert store.worker_executions.next_session_title(**ident(epoch), base_title="Plan") == "Plan #2"
