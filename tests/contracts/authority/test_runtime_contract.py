"""Contract suite for ``SessionRuntimeRepository``: the epoch, ``mutate``'s fence order and receipts, the metadata commands, delete, and the ledger-gated ends."""

from __future__ import annotations

import pytest

from persistence import AuthorityRefusal, FenceStale, StaleFenceKind
from persistence.authority.admissions import GateRefusal
from persistence.authority.runtime import Archive, Delete, Rename, SessionGone, Sidebar, mutation_receipt_key
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts.conftest import require_failpoints

PRINCIPAL = "principal-a"


def _seed(store, session_id: str, parent: str | None = None) -> str:
    return str(store.sessions.create(NewSession(session_id=session_id, source="cli", parent_session_id=parent)))


def _mutate(store, epoch, session_id, command, *, request_id="req-1", expected_revision=None, expected_generation=None, owner=None, **kwargs):
    if expected_revision is None:
        expected_revision = store.sessions.get(session_id).runtime_revision
    return store.runtime.mutate(epoch, principal_id=PRINCIPAL, session_id=session_id, request_id=request_id, expected_revision=expected_revision,
                                command=command, expected_generation=expected_generation, owner=owner, **kwargs)


def test_begin_epoch_is_monotonic_and_current_epoch_reads_it(backend, scope_a):
    store = backend.open(scope_a)
    assert store.runtime.current_epoch() is None
    first = store.runtime.begin_epoch(instance_id="instance-a")
    second = store.runtime.begin_epoch(instance_id="instance-b")
    assert (first, second) == (1, 2) and store.runtime.current_epoch() == 2
    for bad in ("", "x" * 1025, 7):
        with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
            store.runtime.begin_epoch(instance_id=bad)  # type: ignore[arg-type]


def test_epochs_are_scope_local(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    store_a.runtime.begin_epoch(instance_id="a")
    assert store_b.runtime.current_epoch() is None


def test_sidebar_flags_cover_the_compression_lineage_and_bump_each_revision(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-root")
    store.sessions.end("s-root", END_REASON_COMPRESSION)
    _seed(store, "s-tip", parent="s-root")
    outcome = _mutate(store, epoch, "s-tip", Sidebar(archived=True, pinned=True))
    assert outcome.result == {"session_id": "s-tip", "revision": 1, "execution_generation": 0, "operation": "sidebar", "archived": True, "pinned": True}
    assert not outcome.replayed
    # CONTRACT: a first archive owes its plugins the on_session_delete event, naming the affected rows.
    assert outcome.gone is not None and outcome.gone.reason == "archive" and set(outcome.gone.session_ids) == {"s-root", "s-tip"}
    for sid in ("s-root", "s-tip"):
        record = store.sessions.get(sid)
        assert (record.archived, record.pinned, record.runtime_revision) == (True, True, 1), sid
    # CONTRACT: pinning through mutate is a raw lineage column write; it does not clear hidden.
    _mutate(store, epoch, "s-tip", Sidebar(hidden=True), request_id="req-2")
    _mutate(store, epoch, "s-tip", Sidebar(pinned=True), request_id="req-3")
    assert store.sessions.get("s-tip").hidden is True


def test_rename_and_archive_commands(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-name")
    renamed = _mutate(store, epoch, "s-name", Rename("Planning"))
    assert renamed.result["title"] == store.sessions.get("s-name").title == "Planning"
    archived = _mutate(store, epoch, "s-name", Archive(True), request_id="req-2")
    assert archived.result["archived"] is True and archived.gone == SessionGone(("s-name",), "archive")
    # CONTRACT: archiving an archived session owes no event.
    assert _mutate(store, epoch, "s-name", Archive(True), request_id="req-3").gone is None


def test_fence_order_epoch_then_replay_then_existence_revision_generation_guard(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-order")
    first = _mutate(store, epoch, "s-order", Sidebar(pinned=True))
    # CONTRACT: an exact retry replays its receipt even though its revision is now stale, and owes no event.
    replay = _mutate(store, epoch, "s-order", Sidebar(pinned=True), expected_revision=0)
    assert replay.replayed and replay.result == first.result and replay.gone is None
    # CONTRACT: the epoch fence comes before the replay.
    with pytest.raises(FenceStale) as stale:
        _mutate(store, epoch + 5, "s-order", Sidebar(pinned=True), expected_revision=0)
    assert stale.value.kind is StaleFenceKind.STALE_EPOCH
    # CONTRACT: the same identity with another command conflicts.
    with pytest.raises(AuthorityRefusal, match="^admission_conflict$"):
        _mutate(store, epoch, "s-order", Sidebar(pinned=False), expected_revision=0)
    with pytest.raises(AuthorityRefusal, match="^not_found$"):
        _mutate(store, epoch, "s-absent", Sidebar(pinned=True), request_id="r", expected_revision=0)
    with pytest.raises(FenceStale) as revision:
        _mutate(store, epoch, "s-order", Sidebar(hidden=True), request_id="r", expected_revision=0)
    assert revision.value.kind is StaleFenceKind.REVISION_CONFLICT
    with pytest.raises(FenceStale) as generation:
        _mutate(store, epoch, "s-order", Sidebar(hidden=True), request_id="r", expected_generation=9)
    assert generation.value.kind is StaleFenceKind.STALE_GENERATION
    guard = GateRefusal("runtime_coordination_required", field="live")
    with pytest.raises(AuthorityRefusal) as refused:
        _mutate(store, epoch, "s-order", Sidebar(hidden=True), request_id="r", live_guard=guard)
    assert (refused.value.reason, refused.value.field) == ("runtime_coordination_required", "live")
    # CONTRACT: the guard does not stop an exact replay.
    assert _mutate(store, epoch, "s-order", Sidebar(pinned=True), expected_revision=0, live_guard=guard).replayed
    assert store.sessions.get("s-order").hidden is False and store.sessions.get("s-order").runtime_revision == 1


def test_input_shape_refusals_come_before_the_store(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-shape")
    for kwargs in ({"principal_id": ""}, {"request_id": "x" * 1025}, {"expected_revision": -1}, {"expected_revision": True}, {"expected_generation": -1}):
        call = {"principal_id": PRINCIPAL, "session_id": "s-shape", "request_id": "r", "expected_revision": 0, "command": Sidebar(pinned=True), "owner": None, **kwargs}
        with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
            store.runtime.mutate(epoch, **call)
    # CONTRACT: delete, rewind, reset, branch, model and compress require expected_generation.
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        _mutate(store, epoch, "s-shape", Delete())
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        _mutate(store, epoch, "s-shape", Sidebar())
    assert not store.runtime.has_mutation_receipt(PRINCIPAL, "s-shape", "r")


def test_mutation_receipts_carry_a_versioned_key_and_are_found_by_key_alone(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-key")
    _mutate(store, epoch, "s-key", Sidebar(pinned=True), request_id="req-key")
    key, digest = mutation_receipt_key(PRINCIPAL, "s-key", "req-key", operation="sidebar", payload={"pinned": True}, expected_revision=0)
    assert key.startswith("gateway.mutation.v1.") and len(digest) == 64
    # CONTRACT: has_mutation_receipt reads the key alone, whatever the digest.
    assert store.runtime.has_mutation_receipt(PRINCIPAL, "s-key", "req-key")
    assert not store.runtime.has_mutation_receipt(PRINCIPAL, "s-key", "req-other")
    with pytest.raises(AuthorityRefusal, match="^invalid_params$"):
        store.runtime.has_mutation_receipt("", "s-key", "req-key")


def test_crash_before_the_receipt_restores_every_table(backend, scope_a):
    from tests.contracts._reference.authority.mutation import RECEIPT_FAILPOINT

    require_failpoints(backend)
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-crash")
    authority_state(backend, scope_a).failpoints.arm(RECEIPT_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _mutate(store, epoch, "s-crash", Sidebar(archived=True))
    record = store.sessions.get("s-crash")
    assert (record.archived, record.runtime_revision) == (False, 0)
    assert not store.runtime.has_mutation_receipt(PRINCIPAL, "s-crash", "req-1")
    assert _mutate(store, epoch, "s-crash", Sidebar(archived=True)).result["revision"] == 1


def test_delete_retires_the_conversation_and_names_the_ids(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-del")
    store.admissions.admit(epoch=epoch, principal_id="p", session_id="s-del", request_id="q", payload={"text": "q"})
    row = store.sessions.get("s-del")
    with pytest.raises(AuthorityRefusal, match="^session_busy$"):
        _mutate(store, epoch, "s-del", Delete(), expected_generation=row.runtime_generation)
    (queued,) = store.admissions.list(session_id="s-del")
    store.admissions.cancel(epoch=epoch, admission_id=queued.admission_id)
    row = store.sessions.get("s-del")
    outcome = _mutate(store, epoch, "s-del", Delete(), expected_generation=row.runtime_generation)
    assert outcome.result == {"session_id": "s-del", "revision": row.runtime_revision + 1, "execution_generation": row.runtime_generation,
                              "operation": "delete", "deleted_ids": ["s-del"]}
    assert outcome.gone == SessionGone(("s-del",), "delete")
    assert store.sessions.get("s-del") is None and store.runtime.retired_session("s-del")
    # CONTRACT: the receipt outlives the row, so the exact retry replays.
    assert _mutate(store, epoch, "s-del", Delete(), expected_revision=row.runtime_revision, expected_generation=row.runtime_generation).replayed


def test_end_unless_ledger_busy_refuses_while_work_is_pending(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-view")
    record, _ = store.admissions.admit(epoch=epoch, principal_id="p", session_id="s-view", request_id="q", payload={"text": "q"})
    assert store.runtime.end_unless_ledger_busy("s-view", session_key=None, end_reason="tui_close") is False
    assert store.sessions.get("s-view").ended_at is None
    store.admissions.cancel(epoch=epoch, admission_id=record.admission_id)
    assert store.runtime.end_unless_ledger_busy("s-view", session_key=None, end_reason="tui_close") is True
    ended = store.sessions.get("s-view")
    assert ended.end_reason == "tui_close" and ended.ended_at is not None
    # CONTRACT: already ended is True and keeps the first stamp; an unknown id is True (nothing to end).
    assert store.runtime.end_unless_ledger_busy("s-view", session_key=None, end_reason="other") is True
    assert store.sessions.get("s-view").end_reason == "tui_close"
    assert store.runtime.end_unless_ledger_busy("s-none", session_key=None, end_reason="tui_close") is True


def _settled_turn(store, session_id, epoch, request_id="req-1"):
    record, _ = store.admissions.admit(epoch=epoch, principal_id="messaging:x", session_id=session_id, request_id=request_id, payload={"input": 1})
    claimed = store.admissions.claim(epoch=epoch, session_id=session_id)
    return store.admissions.settle(epoch=epoch, admission_id=record.admission_id, generation=claimed.generation, outcome="completed")


def _end(store, session_id, settled, epoch, **overrides):
    kwargs = dict(session_id=session_id, admission_id=settled.admission_id, generation=settled.generation, admission_epoch=settled.owner_epoch, reason="webhook_complete")
    kwargs.update(overrides)
    return store.runtime.end_after_settlement(epoch, **kwargs)


def test_end_after_settlement_ends_only_a_settled_quiet_session(backend, scope_a):
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-hook")
    settled = _settled_turn(store, "s-hook", epoch)
    assert _end(store, "s-none", settled, epoch) is False
    for miss in ({"admission_id": "adm-other"}, {"admission_epoch": epoch + 5}, {"generation": settled.generation + 1}):
        assert _end(store, "s-hook", settled, epoch, **miss) is False
    pending, _ = store.admissions.admit(epoch=epoch, principal_id="messaging:x", session_id="s-hook", request_id="req-2", payload={"input": 2})
    assert _end(store, "s-hook", settled, epoch) is False
    store.admissions.cancel(epoch=epoch, admission_id=pending.admission_id)
    assert _end(store, "s-hook", settled, epoch) is True
    ended = store.sessions.get("s-hook")
    assert ended.end_reason == "webhook_complete" and _end(store, "s-hook", settled, epoch) is False


def test_end_after_settlement_answers_false_on_a_stale_epoch(backend, scope_a):
    # The epoch fence is part of the conditional write: a stale epoch is a miss, not a refusal.
    store = backend.open(scope_a)
    epoch = store.runtime.begin_epoch(instance_id="instance-a")
    _seed(store, "s-hook-fence")
    settled = _settled_turn(store, "s-hook-fence", epoch)
    newer = store.runtime.begin_epoch(instance_id="instance-b")
    assert _end(store, "s-hook-fence", settled, epoch) is False
    # CONTRACT: the admission's own epoch is admission_epoch, so a settlement under the old epoch ends under the new one.
    assert _end(store, "s-hook-fence", settled, newer) is True
