"""The admission gate as a callable, read inside the write.

A drain that begins while a write waits on the writer still refuses it, so ``gate`` may be a zero-argument callable returning ``GateRefusal | None``: it reads owner memory only (never the store) and the backend calls it at the gate's fixed point inside the write, on every attempt. These cases flip the callable's answer after the call starts but before the write's gate point, by flipping it from inside the write's own first read.
"""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal
from persistence.authority.admissions import GateRefusal
from persistence.conversation.sessions import NewSession
from tests.contracts.authority.test_bindings_api_contract import api_binding
from tests.contracts.authority.test_bindings_contract import begin

SESSION = "sess-gate-live"
DRAINING = GateRefusal("runtime_draining", field="owner", detail="drain started")


class DrainAfter:
    """A gate that is open on its first ``n`` reads and draining after: the owner's drain flag as the backend sees it."""

    def __init__(self, open_reads: int = 0) -> None:
        self.reads, self.open_reads = 0, open_reads

    def __call__(self) -> GateRefusal | None:
        self.reads += 1
        return None if self.reads <= self.open_reads else DRAINING


def _seed(store) -> int:
    store.sessions.create(NewSession(session_id=SESSION, source="contract"))
    return store.runtime.begin_epoch(instance_id="instance-gate")


def test_a_callable_gate_is_read_inside_the_admission_write(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    open_gate = DrainAfter(open_reads=1)
    # CONTRACT: an open callable gate admits, and is read once per attempt at the gate's point.
    first, created = store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r0", payload={"text": "r0"}, gate=open_gate)
    assert created and open_gate.reads >= 1
    # CONTRACT: a gate that reads draining at the write's gate point refuses with its reason and writes nothing.
    with pytest.raises(AuthorityRefusal) as refused:
        store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r1", payload={"text": "r1"}, gate=DrainAfter())
    assert (refused.value.reason, refused.value.field, refused.value.detail) == ("runtime_draining", "owner", "drain started")
    assert store.admissions.list(session_id=SESSION, pending_only=False) == [first]
    # CONTRACT: an exact replay bypasses the gate, callable or not, without reading it.
    replay_gate = DrainAfter()
    assert store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r0", payload={"text": "r0"}, gate=replay_gate) == (first, False)
    assert replay_gate.reads == 0


def test_a_callable_gate_is_read_inside_the_claim_write(backend, scope_a):
    store = backend.open(scope_a)
    epoch = _seed(store)
    store.admissions.admit(epoch=epoch, principal_id="p", session_id=SESSION, request_id="r0", payload={"text": "r0"})
    with pytest.raises(AuthorityRefusal, match="^runtime_draining$"):
        store.admissions.claim(epoch=epoch, session_id=SESSION, gate=DrainAfter())
    assert [row.status for row in store.admissions.list(session_id=SESSION, pending_only=False)] == ["queued"]
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION, gate=DrainAfter(open_reads=1))
    assert claimed and claimed.status == "started"


def test_a_callable_gate_is_read_inside_the_api_binding_write(backend, scope_a):
    store = backend.open(scope_a)
    epoch = begin(store)
    binding = api_binding("api-gate")
    with pytest.raises(AuthorityRefusal, match="^runtime_draining$"):
        store.bindings.bind_api(epoch=epoch, binding=binding, gate=DrainAfter())
    assert store.sessions.get(binding.session_id) is None
    assert store.bindings.bind_api(epoch=epoch, binding=binding, gate=DrainAfter(open_reads=1)) is None
    assert store.sessions.get(binding.session_id) is not None
