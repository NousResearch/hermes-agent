"""Contract suite for the worker receipt transaction every ``sequence``-taking worker verb shares.

Each case drives one receipted verb and pins a step of its order: the pre-store checks, the epoch fence, the assignment, the owner-epoch fence, the replay, conflict and retired receipt, the live status, the strict sequence, the operation's own refusal, and the commit (receipt, sequence, status, revision) as one atomic transition.
"""

from __future__ import annotations

import pytest

from persistence import AuthorityEpoch, ExecutionGeneration, SessionId
from persistence.conversation.sessions import NewSession
from persistence.workers.executions import WorkerExecutionId
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts.workers._support import SESSION, assistant, begin_epoch, ident, last_sequence, refused, revision, user, worker


def _append(store, epoch, content, **overrides):
    return store.worker_executions.append_transcript(**ident(epoch, **overrides), messages=[assistant(content)])


def test_an_exact_duplicate_returns_the_stored_result_without_re_executing(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    first = _append(store, epoch, "hello")
    assert [(r.role, r.content) for r in store.messages.read_transcript(SESSION)] == [("assistant", "hello")]
    assert _append(store, epoch, "hello") == first
    assert len(store.messages.read_transcript(SESSION)) == 1
    assert last_sequence(backend, scope_a, epoch) == 1


def test_a_reused_sequence_with_another_payload_is_admission_conflict(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    _append(store, epoch, "hello")
    with refused("admission_conflict"):
        _append(store, epoch, "tampered")
    assert len(store.messages.read_transcript(SESSION)) == 1


def test_the_digest_distinguishes_surrogate_pairs_and_absent_keys(backend, scope_a):
    # admission_fingerprint encodes utf-8 with surrogatepass: an astral scalar and its surrogate pair are different payloads,
    # and an absent key differs from a key holding its default.
    store, epoch = worker(backend, scope_a)
    _append(store, epoch, "\U0001f600")
    with refused("admission_conflict"):
        _append(store, epoch, "\ud83d\ude00")
    store.worker_executions.record_usage(**ident(epoch, sequence=2), input_tokens=1)
    with refused("admission_conflict"):
        store.worker_executions.record_usage(**ident(epoch, sequence=2), input_tokens=1, output_tokens=0)


def test_pre_store_checks_and_the_strict_sequence_are_invalid_params(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    for sequence in (0, -1, True, 1.0, "1"):
        with refused("invalid_params"):
            _append(store, AuthorityEpoch(99), "x", sequence=sequence)
    with refused("invalid_params"):
        store.worker_executions.set_session_system_prompt(**ident(epoch, sequence=1), system_prompt="x" * (4 * 1024 * 1024 + 1))
    with refused("invalid_params"):  # not the strict successor of 0
        _append(store, epoch, "x", sequence=2)
    assert store.messages.read_transcript(SESSION) == ()
    _append(store, epoch, "first")
    with pytest.raises(ValueError):
        _append(store, epoch, "skipped", sequence=3)


def test_receipts_fence_on_the_epoch_in_both_directions(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    rotated = begin_epoch(backend, scope_a)
    with refused("stale_epoch"):  # the caller is on an old epoch
        _append(store, epoch, "old-epoch")
    with refused("stale_epoch"):  # the execution is still owned by the old epoch: adoption is the only way on
        _append(store, rotated, "unadopted")
    assert store.messages.read_transcript(SESSION) == ()


def test_the_assignment_answers_before_the_epoch_owner_and_the_receipt(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    with refused("not_found"):
        _append(store, epoch, "x", execution_id=WorkerExecutionId("ghost"))
    store.sessions.create(NewSession(session_id=SessionId("elsewhere"), source="contract"))
    with refused("permission_denied"):
        _append(store, epoch, "x", session_id=SessionId("elsewhere"))
    with refused("stale_generation"):
        _append(store, epoch, "x", generation=ExecutionGeneration(1))


def test_replay_precedes_the_terminal_refusal_and_conflict_precedes_replay(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    first = _append(store, epoch, "kept")
    store.worker_executions.finish(**ident(epoch, sequence=2))
    assert _append(store, epoch, "kept") == first
    with refused("admission_conflict"):
        _append(store, epoch, "changed")
    with refused("stale_generation"):
        _append(store, epoch, "late", sequence=3)
    store.worker_executions.finish(**ident(epoch, sequence=2))


def test_a_terminal_execution_refuses_the_replay_of_its_compacted_read_receipts(backend, scope_a):
    # Read receipts (context, history) shrink to a retired marker when the execution ends; mutation receipts stay replayable.
    store, epoch = worker(backend, scope_a)
    read = store.worker_executions.compression_session_context(**ident(epoch, sequence=1), target_session_id=SESSION)
    assert read["id"] == SESSION
    _append(store, epoch, "kept", sequence=2)
    store.worker_executions.finish(**ident(epoch, sequence=3))
    with refused("stale_generation"):
        store.worker_executions.compression_session_context(**ident(epoch, sequence=1), target_session_id=SESSION)
    assert _append(store, epoch, "kept", sequence=2).count == 1


def test_an_operation_refusal_commits_nothing_and_leaves_the_sequence_free(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    before = revision(store)
    with refused("invalid_params"):
        store.worker_executions.append_transcript(**ident(epoch), messages=[{"role": "narrator", "content": "x"}])
    assert (last_sequence(backend, scope_a, epoch), revision(store)) == (0, before)
    _append(store, epoch, "lands")


def test_the_revision_moves_only_when_the_operation_wrote(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    start = revision(store)
    assert store.worker_executions.compression_watermark(**ident(epoch, sequence=1)) == 0
    assert revision(store) == start
    _append(store, epoch, "row", sequence=2)
    assert revision(store) == start + 1
    # A write whose statement matches the row moves it even when the stored values stay as they were.
    store.worker_executions.set_session_system_prompt(**ident(epoch, sequence=3), system_prompt=None)
    assert revision(store) == start + 2
    store.worker_executions.finish(**ident(epoch, sequence=4))
    assert revision(store) == start + 3


def test_crash_between_the_write_and_its_receipt_publishes_nothing_then_retries_exactly_once(backend, scope_a):
    require_failpoints(backend)
    store, epoch = worker(backend, scope_a)
    state = authority_state(backend, scope_a)
    before = revision(store)
    state.failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _append(store, epoch, "crash")
    assert store.messages.read_transcript(SESSION) == ()
    assert (store.sessions.get(SESSION).message_count, last_sequence(backend, scope_a, epoch), revision(store)) == (0, 0, before)
    _append(store, epoch, "crash")
    assert (len(store.messages.read_transcript(SESSION)), last_sequence(backend, scope_a, epoch), revision(store)) == (1, 1, before + 1)
    _append(store, epoch, "crash")
    assert len(store.messages.read_transcript(SESSION)) == 1


def test_receipted_writes_survive_reopen_with_replay_intact(backend, scope_a):
    require_capability(backend, "reopen_durability")
    store, epoch = worker(backend, scope_a)
    first = _append(store, epoch, "durable")
    backend.reopen()
    assert [r.content for r in store.messages.read_transcript(SESSION)] == ["durable"]
    assert _append(store, epoch, "durable") == first
    with refused("admission_conflict"):
        _append(store, epoch, "tampered")
    store.worker_executions.append_transcript(**ident(epoch, sequence=2), messages=[user("next")])
