"""Contract suite for the worker receipt reads (``persistence.workers.receipts.WorkerReceiptReads``).

``claim_origin`` and ``publication_target`` are the owner's reads of the worker receipt stream before it serves a request. Receipts are seeded only through receipted verbs (a usage report and a compression publication) and read back through the verbs under test.
"""

from __future__ import annotations

import pytest

from persistence import ExecutionGeneration, SessionId
from persistence.workers.executions import WorkerExecutionId
from tests.contracts.conftest import require_capability
from tests.contracts.workers._support import EXECUTION, SESSION, ident, refused, worker
from tests.contracts.workers.test_worker_compaction_contract import CHILD, _lease, _publication

OTHER = WorkerExecutionId("exec-other")
GRANDCHILD = SessionId("sess-worker-cont-2")
PUBLISH = "compression.publish"


def _turn_worker(backend, scope):
    require_capability(backend, "transactional_claims")
    return worker(backend, scope)


def _published(backend, scope):
    """A worker with a usage receipt at sequence 1 and a compression publication (SESSION to CHILD) at sequence 2."""
    store, epoch = _turn_worker(backend, scope)
    store.worker_executions.record_usage(**ident(epoch, sequence=1), input_tokens=3, api_call_count=1)
    _lease(store)
    publication = _publication()
    store.worker_executions.publish_compression_child(**ident(epoch, sequence=2), publication=publication)
    return store, epoch, publication


def _target(store, publication, operation=PUBLISH, **overrides):
    args = {"execution_id": EXECUTION, "session_id": SESSION, "generation": ExecutionGeneration(0), "sequence": 2} | overrides
    return store.worker_executions.publication_target(operation=operation, payload=publication.payload(), **args)


def test_claim_origin_follows_the_earliest_publication(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    wx = store.worker_executions
    assert wx.claim_origin(execution_id=EXECUTION, session_id=SESSION) == SESSION
    wx.record_usage(**ident(epoch, sequence=1), input_tokens=1, api_call_count=1)
    # A receipt that records no move leaves the claim where it was addressed.
    assert wx.claim_origin(execution_id=EXECUTION, session_id=SESSION) == SESSION
    _lease(store)
    wx.publish_compression_child(**ident(epoch, sequence=2), publication=_publication())
    # The worker now addresses the child; its claim stays bound to the session it started on.
    assert wx.claim_origin(execution_id=EXECUTION, session_id=CHILD) == SESSION
    assert wx.claim_origin(execution_id=EXECUTION, session_id=SessionId("anything")) == SESSION
    # Another execution, or an id no receipt carries, echoes the addressed session unchanged.
    assert wx.claim_origin(execution_id=OTHER, session_id=CHILD) == CHILD
    for execution_id in (7, None, ""):
        assert wx.claim_origin(execution_id=execution_id, session_id=CHILD) == CHILD


def test_publication_target_follows_only_the_exact_committed_publication(backend, scope_a):
    store, _epoch, publication = _published(backend, scope_a)
    # The exact replay addressed to the parent follows the worker onto the child.
    assert _target(store, publication) == CHILD
    # Any other request addressed to the parent is the ordinary wrong-session refusal.
    for overrides, changed in (({}, _publication(source="other")), ({"sequence": 3}, publication), ({"sequence": 1}, publication)):
        with refused("permission_denied"):
            _target(store, changed, **overrides)
    # Addressing the worker's current session resolves to it.
    assert _target(store, publication, session_id=CHILD, sequence=3) == CHILD


def test_other_operations_answer_the_addressed_session_without_reading(backend, scope_a):
    store, _epoch, publication = _published(backend, scope_a)
    for operation in ("usage.record", "compression.lock", ""):
        assert _target(store, publication, operation=operation, execution_id=OTHER, generation=ExecutionGeneration(9)) == SESSION


def test_a_second_publication_keeps_the_origin_and_the_first_replay(backend, scope_a):
    store, epoch, first = _published(backend, scope_a)
    _lease(store, session_id=CHILD)
    second = _publication(child_session_id=GRANDCHILD)
    store.worker_executions.publish_compression_child(**ident(epoch, sequence=3, session_id=CHILD), publication=second)
    wx = store.worker_executions
    # The claim stays on the session the worker started on, not the parent of the latest move.
    assert wx.claim_origin(execution_id=EXECUTION, session_id=GRANDCHILD) == SESSION
    # Both committed publications replay onto the worker's current session.
    assert _target(store, first) == GRANDCHILD
    assert _target(store, second, session_id=CHILD, sequence=3) == GRANDCHILD
    # The first publication resent to the session it moved the worker to is no replay.
    with refused("permission_denied"):
        _target(store, first, session_id=CHILD)


def test_publication_target_before_any_move_is_the_addressed_assignment(backend, scope_a):
    store, _epoch = _turn_worker(backend, scope_a)
    publication = _publication()
    assert _target(store, publication, sequence=1) == SESSION
    with refused("permission_denied"):
        _target(store, publication, session_id=CHILD, sequence=1)


@pytest.mark.parametrize("execution_id", [OTHER, WorkerExecutionId(""), 7, None])
def test_an_unknown_execution_is_not_found_before_the_generation(backend, scope_a, execution_id):
    store, _epoch, publication = _published(backend, scope_a)
    with refused("not_found"):
        _target(store, publication, execution_id=execution_id, generation=ExecutionGeneration(9))


def test_publication_target_resolution_order(backend, scope_a):
    store, _epoch, publication = _published(backend, scope_a)
    # A wrong session wins over a stale generation.
    with refused("permission_denied"):
        _target(store, _publication(source="other"), generation=ExecutionGeneration(9))
    # The replay resolves onto the child first, then the generation fence applies to it.
    for generation in (ExecutionGeneration(9), "0", True):
        with refused("stale_generation"):
            _target(store, publication, generation=generation)
