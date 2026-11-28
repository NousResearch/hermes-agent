"""Contract (NousResearch/hermes-agent#145022 field data): the worker's RPC deadline must
outlast the owner's own transcript-write patience.

The owner answers worker.persist only after committing a transcript write that gets
StoreHolder._TRANSCRIPT_WRITE_PATIENCE_S of lock patience. A worker that gives up first
poisons its own store while the owner still lands the write, and the admission books
'execution unknown' despite a successful save, burning the card's retry budget (field
data: worker waited 20 s while the owner allowed itself 60 s). The deadline is DERIVED --
the store's patience comes through the sanctioned persistence door
(persistence_assembly.transcript_write_patience_s()) plus a bounded reply margin -- so
retuning the store retunes the worker with it and the two budgets can never drift apart
again. This test pins the full derivation (a silently dropped margin must fail, not just
a shortened deadline), the drift link, and that the wait stays finite."""
import types

from agent import runtime_session_store
from agent.runtime_session_store import WorkerRPC
from persistence.sqlite._holder import StoreHolder
from persistence_assembly import transcript_write_patience_s


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
    # The full derivation, not a floor: removing the margin must fail this test.
    assert deadline == transcript_write_patience_s() + runtime_session_store._RPC_REPLY_MARGIN_S
    # The invariant, restated against the storage layer's own constant: never settle an
    # RPC while the owner is still inside its own write budget.
    assert deadline > StoreHolder._TRANSCRIPT_WRITE_PATIENCE_S
    # Drift link: the door reports the holder's patience itself, nothing is hardcoded.
    assert transcript_write_patience_s() == StoreHolder._TRANSCRIPT_WRITE_PATIENCE_S
    # ...and it is still a bounded wait, not an infinite hang.
    assert deadline < 300
