"""Every delegation ledger verb takes the worker's verified process identity and checks it before the receipt.

The owner passes the claim's ``pid`` and ``birth`` as ``worker_pid`` and ``worker_birth`` on every ``delegation.*`` operation. The ledger checks them, after the execution id's text, before the receipt transaction runs: a bad identity is ``invalid_params`` where a good one meets the epoch, the assignment, the receipt replay or the sequence. Neither value enters the receipt digest, so a replay under another well-formed identity answers the stored result.
"""

from __future__ import annotations

import math

import pytest

from persistence.workers.delegations import WorkerDelegationChild, WorkerDelegationClaim, WorkerDelegationCompletion, WorkerDelegationPayload
from persistence.workers.executions import WorkerExecutionId
from tests.contracts.workers._support import ident, last_sequence, refused
from tests.contracts.workers.test_worker_delegations_contract import BIRTH, DELEGATION, PID, LedgerWorker, dispatch_record, event, seed_delegations

_DELIVERY = ("claim", "ack", "drop", "defer", "release")

# verb -> (record keyword, a well-formed record)
_RECORDS = {
    "dispatch_delegation": ("dispatch", dispatch_record()),
    "complete_delegation": ("completion", WorkerDelegationCompletion(event(), {"summary": "durable"})),
    "record_delegation_child": ("child", WorkerDelegationChild(DELEGATION, {"task_index": 0})),
    "read_delegation": ("delegation_id", DELEGATION),
    "withdraw_delegation": ("delegation_id", DELEGATION),
    **{f"{action}_delegation_delivery": ("claim", WorkerDelegationClaim(DELEGATION, "t")) for action in _DELIVERY},
}

# a pid that is not a positive int; a birth that is not a finite non-negative int or float
_BAD_IDENTITIES = [
    *({"worker_pid": pid} for pid in (0, -1, True, 4242.0, "4242", None)),
    *({"worker_birth": birth} for birth in (-1, True, math.nan, math.inf, "1700000000", None)),
]


def _call(worker, verb, sequence, *, record=None, epoch=None, generation=0, execution_id=None, **identity):
    keyword, default = _RECORDS[verb]
    extra = {"owner_started_at": BIRTH} if verb == "dispatch_delegation" else {}
    return getattr(worker.store.worker_executions, verb)(
        **ident(worker.epoch if epoch is None else epoch, sequence=sequence, generation=generation, execution_id=worker.execution_id if execution_id is None else execution_id),
        **{keyword: default if record is None else record}, **({"worker_pid": PID, "worker_birth": BIRTH} | identity), **extra)


def _stored(backend, scope, verb):
    """A worker whose delegation is dispatched at sequence 1 and whose ``verb`` holds a receipt; the receipt's sequence and answer."""
    worker = LedgerWorker(backend, scope, seed_delegations(backend, scope))
    worker.dispatch(sequence=1)
    if verb == "dispatch_delegation":
        return worker, 1, None
    return worker, 2, _call(worker, verb, 2)


@pytest.mark.parametrize("verb", sorted(_RECORDS))
def test_a_bad_identity_is_refused_before_the_receipt(backend, scope_a, verb):
    worker, stored, answer = _stored(backend, scope_a, verb)
    following = stored + 1
    for bad in _BAD_IDENTITIES:
        # CONTRACT: the identity is checked before the receipt transaction runs, so each fence a good identity
        # meets answers invalid_params for a bad one.
        with refused("invalid_params"):
            _call(worker, verb, stored, **bad)  # the exact replay of a stored receipt
        with refused("invalid_params"):
            _call(worker, verb, stored, record=WorkerDelegationPayload({"other": 1}), **bad)  # a used sequence
        with refused("invalid_params"):
            _call(worker, verb, following, epoch=worker.epoch + 1, **bad)  # the epoch
        with refused("invalid_params"):
            _call(worker, verb, following, generation=1, **bad)  # the assignment
        with refused("invalid_params"):
            _call(worker, verb, following, execution_id=WorkerExecutionId("exec-unknown"), **bad)
    # CONTRACT: with a good identity the same calls meet the fences and the replay as themselves.
    assert _call(worker, verb, stored) == answer
    with refused("admission_conflict"):
        _call(worker, verb, stored, record=WorkerDelegationPayload({"other": 1}))
    with refused("stale_epoch"):
        _call(worker, verb, following, epoch=worker.epoch + 1)
    with refused("stale_generation"):
        _call(worker, verb, following, generation=1)
    with refused("not_found"):
        _call(worker, verb, following, execution_id=WorkerExecutionId("exec-unknown"))
    # CONTRACT: a refused identity leaves the sequence where it was.
    assert last_sequence(backend, scope_a, worker.epoch) == stored


@pytest.mark.parametrize("verb", sorted(_RECORDS))
def test_the_execution_id_text_is_checked_with_the_identity(backend, scope_a, verb):
    worker, stored, _answer = _stored(backend, scope_a, verb)
    for execution_id in ("", "x" * 513):
        # CONTRACT: the text check refuses the execution id before its lookup would answer not_found.
        with refused("invalid_params"):
            _call(worker, verb, stored + 1, execution_id=WorkerExecutionId(execution_id))
    with refused("not_found"):
        _call(worker, verb, stored + 1, execution_id=WorkerExecutionId("x" * 512))


@pytest.mark.parametrize("verb", sorted(_RECORDS))
def test_a_replay_answers_under_another_well_formed_identity(backend, scope_a, verb):
    worker, stored, answer = _stored(backend, scope_a, verb)
    # CONTRACT: the identity is not part of the operation payload, so the digest and the stored answer hold.
    for identity in ({"worker_pid": PID + 1}, {"worker_birth": BIRTH + 0.5}, {"worker_pid": 1, "worker_birth": 0}):
        assert _call(worker, verb, stored, **identity) == answer
    assert last_sequence(backend, scope_a, worker.epoch) == stored


def test_dispatch_stores_the_probed_start_time_not_the_birth(backend, scope_a):
    worker = LedgerWorker(backend, scope_a, seed_delegations(backend, scope_a))
    # CONTRACT: the row takes the pid and the start time the probe read; the birth only meets the probe.
    worker.call("dispatch_delegation", 1, dispatch=dispatch_record(), worker_pid=PID, worker_birth=BIRTH + 0.5, owner_started_at=BIRTH)
    assert worker.owner() == (PID, BIRTH)
