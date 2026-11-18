"""A delegation payload as the worker sent it meets the receipt fences before its own checks, on every ledger verb.

The owner settles a worker's ``delegation.*`` operation by passing the payload it received as a :class:`WorkerDelegationPayload`; a payload no record could hold (a missing or extra key) is judged by its handler, after the epoch, the assignment, the replay and the sequence. Only a payload that is not a mapping refuses first, as every receipt payload does.
"""

from __future__ import annotations

import pytest

from persistence.workers.delegations import WorkerDelegationClaim, WorkerDelegationPayload
from persistence.workers.executions import WorkerExecutionId
from tests.contracts.workers._support import SESSION, ident, refused
from tests.contracts.workers.test_worker_delegations_contract import DELEGATION, PID, BIRTH, LedgerWorker, dispatch_record, event, seed_delegations

_DELIVERY = ("claim", "ack", "drop", "defer", "release")

# verb -> (payload keyword, a payload no record of the verb can hold)
_MALFORMED = {
    "dispatch_delegation": ("dispatch", {"delegation_id": DELEGATION}),
    "complete_delegation": ("completion", {"event": event()}),
    "record_delegation_child": ("child", {"delegation_id": DELEGATION, "entry": {"task_index": 0}, "extra": 1}),
    "read_delegation": ("delegation_id", {}),
    "withdraw_delegation": ("delegation_id", {"delegation_id": DELEGATION, "extra": 1}),
    **{f"{action}_delegation_delivery": ("claim", {"delegation_id": DELEGATION}) for action in _DELIVERY},
}
_IDENTITY = {"worker_pid": PID, "worker_birth": BIRTH}
_DISPATCH_EXTRA = {"owner_started_at": BIRTH}


def _call(worker, verb, payload, **identity):
    keyword, _ = _MALFORMED[verb]
    extra = _IDENTITY | (_DISPATCH_EXTRA if verb == "dispatch_delegation" else {})
    return getattr(worker.store.worker_executions, verb)(**identity, **{keyword: WorkerDelegationPayload(payload)}, **extra)


@pytest.mark.parametrize("verb", sorted(_MALFORMED))
def test_a_malformed_payload_meets_every_fence_first(backend, scope_a, verb):
    epoch = seed_delegations(backend, scope_a)
    worker = LedgerWorker(backend, scope_a, epoch)
    worker.dispatch(sequence=1)
    malformed = _MALFORMED[verb][1]
    # CONTRACT: each fence answers before the payload is judged, with the fence's own refusal.
    with refused("stale_epoch"):
        _call(worker, verb, malformed, **ident(epoch + 1, sequence=2))
    with refused("stale_generation"):
        _call(worker, verb, malformed, **ident(epoch, sequence=2, generation=1))
    with refused("not_found"):
        _call(worker, verb, malformed, **ident(epoch, sequence=1, execution_id=WorkerExecutionId("exec-unknown")))
    with refused("admission_conflict"):
        _call(worker, verb, malformed, **ident(epoch, sequence=1))  # a different payload at a used sequence
    with refused("invalid_params"):
        _call(worker, verb, malformed, **ident(epoch, sequence=3))  # not the next sequence
    # CONTRACT: past the fences the payload refuses with a bare invalid_params and leaves the sequence free.
    with refused("invalid_params") as caught:
        _call(worker, verb, malformed, **ident(epoch, sequence=2))
    assert (caught.value.field, caught.value.detail) == (None, None)
    assert worker.read(sequence=2).state == "running"


@pytest.mark.parametrize("verb", sorted(_MALFORMED))
def test_a_payload_that_is_not_a_mapping_is_refused_before_the_fences(backend, scope_a, verb):
    epoch = seed_delegations(backend, scope_a)
    worker = LedgerWorker(backend, scope_a, epoch)
    for payload in (DELEGATION, None, 7, [DELEGATION]):
        # CONTRACT: the receipt encodes only a mapping, so anything else is invalid_params before the epoch is read.
        with refused("invalid_params"):
            _call(worker, verb, payload, **ident(epoch + 1, sequence=1))
    worker.dispatch(sequence=1)


def test_a_well_formed_wire_payload_is_the_record(backend, scope_a):
    epoch = seed_delegations(backend, scope_a)
    worker = LedgerWorker(backend, scope_a, epoch)
    record = dispatch_record()
    wire = {"delegation_id": record.delegation_id, "task": record.task, "dispatched_at": record.dispatched_at, "session_key": record.session_key,
            "parent_session_id": record.parent_session_id, "origin_ui_session_id": record.origin_ui_session_id, "origin_session_id": record.origin_session_id}
    worker.dispatch(sequence=1)
    # CONTRACT: the wire payload of a record digests as the record, so it replays the record's receipt.
    assert _call(worker, "dispatch_delegation", wire, **ident(epoch, sequence=1)) is None
    via_wire = _call(worker, "read_delegation", {"delegation_id": DELEGATION}, **ident(epoch, sequence=2))
    assert via_wire == worker.read(sequence=2)  # the same receipt replays for the bare id
    worker.complete(sequence=3)
    assert _call(worker, "claim_delegation_delivery", {"delegation_id": DELEGATION, "claim_id": "t"}, **ident(epoch, sequence=4)) is True
    assert worker.call("ack_delegation_delivery", 5, claim=WorkerDelegationClaim(DELEGATION, "t")) is True
    assert worker.pending_event(session_id=SESSION) is None
