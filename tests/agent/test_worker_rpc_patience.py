"""Contract (NousResearch/hermes-agent#145022 field data): the worker's RPC deadline must
outlast the owner's own transcript-write patience.

The owner answers worker.persist only after committing a transcript write that gets
StoreHolder._TRANSCRIPT_WRITE_PATIENCE_S of lock patience. A worker that gives up first
poisons its own store while the owner still lands the write, and the admission books
'execution unknown' despite a successful save, burning the card's retry budget (field
data: worker waited 20 s while the owner allowed itself 60 s). The storage layer keeps its
patience constant behind the backend wall (scripts/ci/check_backend_imports.py), so the
worker-side deadline is a named constant and this test is the drift guard pinning it to
the holder's patience; it must also stay finite."""
import types

from agent.runtime_session_store import WorkerRPC
from persistence.sqlite._holder import StoreHolder


class _Link:
    def __init__(self):
        self.reader = types.SimpleNamespace(is_alive=lambda: True)
        self.ws = types.SimpleNamespace(close=lambda: None)
        self.timeouts = []

    def send(self, request_id, message):
        pass

    def receive(self, timeout):
        self.timeouts.append(timeout)
        return {'result': 'ok'}


def test_worker_rpc_deadline_outlasts_owner_transcript_patience(tmp_path):
    rpc = WorkerRPC(tmp_path)
    link = _Link()
    rpc.link = link

    assert rpc('worker.persist', session_id='s1', messages=[]) == 'ok'

    assert len(link.timeouts) == 1
    deadline = link.timeouts[0]
    # The invariant: never settle an RPC while the owner is still inside its own write budget.
    assert deadline >= StoreHolder._TRANSCRIPT_WRITE_PATIENCE_S
    # ...and it is still a bounded wait, not an infinite hang.
    assert deadline < 300
