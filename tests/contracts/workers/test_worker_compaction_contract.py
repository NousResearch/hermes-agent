"""Contract suite for the worker's row-carrying compression writes (``persistence.workers.compaction.WorkerCompactionVerbs``).

Each case writes through the worker's receipts and observes the result through the non-worker verbs that own the state (the transcript and model config, the session rows, ``state_meta``, the admission ledger's settlement, the binding receipt and the routing entry). Those reads detect private writes and missing side effects; the replay cases reject changed row ids.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from persistence import ExecutionGeneration, SessionId
from persistence.authority.admissions import AdmissionIntent, AdmissionOutcome
from persistence.authority.bindings import NewSessionBinding, local_session_id
from persistence.conversation.messages import NewMessage, RowVisibility, TranscriptRead, TranscriptView
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind
from persistence.core.errors import CompressionBusy, CompressionInProgress, CompressionPublishRefused, TurnLeaseLost
from persistence.workers.compaction import WorkerCompaction, WorkerCompressionPublication
from session_rules.automation import GoalState
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts.workers._support import EXECUTION, SESSION, begin_epoch, ident, last_sequence, refused, register, worker

HOLDER = "compression-holder-a"
CHILD = SessionId("sess-worker-cont")
ROOT = SessionId("sess-worker-root")
STAMP = datetime(2026, 9, 25, 12, 0, tzinfo=UTC).timestamp()
GOAL = json.dumps({"goal": "ship it", "status": "active"})


def _row(content: str, role: str = "user", **extra) -> dict:
    return {"role": role, "content": content, "timestamp": STAMP, **extra}


def _seed_rows(store, *contents: str, session_id=SESSION) -> None:
    store.messages.append_batch(session_id, tuple(NewMessage(role="user", content=c, timestamp=STAMP) for c in contents))


def _active(store, session_id=SESSION):
    return [(r.content, r.role) for r in store.messages.read_transcript(session_id)]


def _lease(store, holder=HOLDER, session_id=SESSION):
    assert store.coordination.try_acquire_compression_lock(session_id=SessionId(str(session_id)), holder=holder, ttl_seconds=300.0) is True


def _publication(**overrides) -> WorkerCompressionPublication:
    fields: dict = dict(child_session_id=CHILD, source="contract", messages=[_row("summary", "assistant")], compression_lock_holder=HOLDER, require_lease_refresh=True)
    fields.update(overrides)
    return WorkerCompressionPublication(**fields)


def _setup(backend, scope):
    require_capability(backend, "transactional_claims")
    return worker(backend, scope)


def _compact(store, epoch, sequence, compaction, **overrides):
    return store.worker_executions.compact_transcript(**ident(epoch, sequence=sequence, **overrides), compaction=compaction)


# ── payloads ──────────────────────────────────────────────────────────────────


def test_records_send_the_producer_payload_key_for_key():
    assert set(WorkerCompaction(messages=[]).payload()) == {"messages", "model_config_patch", "watermark", "lock_holder", "tail_count",
                                                           "carried_messages", "covered_ids", "unresolved_held", "held_for_cas"}
    payload = _publication(watermark=0).payload()
    assert len(payload) == 14
    assert (payload["watermark"], payload["profile_name"], payload["require_compression_lease"], payload["require_lease_refresh"]) == (0, None, True, True)


def test_compaction_payload_checks_refuse_before_the_sequence_advances(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    store.sessions.create(NewSession(session_id=SessionId("sess-other"), source="contract"))
    foreign = store.messages.append(SessionId("sess-other"), NewMessage(role="user", content="x", timestamp=STAMP))
    for compaction in (WorkerCompaction(messages=[], model_config_patch={"model": "m"}), WorkerCompaction(messages=[], tail_count=-1),
                       WorkerCompaction(messages=[], lock_holder=""), WorkerCompaction(messages=[], watermark=-1),
                       WorkerCompaction(messages=[], covered_ids=[0]), WorkerCompaction(messages=[_row("x", extra=1)])):
        with refused("invalid_params"):
            _compact(store, epoch, 1, compaction)
    with refused("permission_denied"):
        _compact(store, epoch, 1, WorkerCompaction(messages=[_row("x", _row_id=foreign)]))
    assert last_sequence(backend, scope_a, epoch) == 0


# ── compact_transcript ────────────────────────────────────────────────────────


def test_compaction_is_the_messages_compact_inside_the_receipt(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    _seed_rows(store, "a", "b", "keep")
    mark = store.messages.active_watermark(SESSION).last_active
    _seed_rows(store, "late")  # arrives while the summary runs
    _lease(store)
    compaction = WorkerCompaction(messages=[_row("summary", "assistant"), _row("keep")], watermark=mark, lock_holder=HOLDER, tail_count=1,
                                  model_config_patch={"_usage_anchor": 5})
    result = _compact(store, epoch, 1, compaction)
    active = store.messages.read_transcript(SESSION)
    assert [r.content for r in active] == ["summary", "keep", "late"]
    assert result.value == 3
    assert tuple(a.row_id for a in result.annotations) == tuple(r.id for r in active[:2])
    assert tuple(a.message_uid for a in result.annotations) == tuple(r.message_uid for r in active[:2])
    visibility = {r.content: r.visibility for r in store.messages.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL)) if r.visibility is not RowVisibility.ACTIVE}
    assert visibility == {"a": RowVisibility.COMPACTION_ARCHIVED, "b": RowVisibility.COMPACTION_ARCHIVED, "keep": RowVisibility.REWOUND, "late": RowVisibility.REWOUND}
    assert store.sessions.get(SESSION).model_config.get("_usage_anchor") == 5
    # The exact replay returns the recorded outcome and writes nothing; another payload at the sequence conflicts.
    assert _compact(store, epoch, 1, compaction) == result
    assert [r.id for r in store.messages.read_transcript(SESSION)] == [r.id for r in active]
    with refused("admission_conflict"):
        _compact(store, epoch, 1, WorkerCompaction(messages=[_row("other")]))
    assert last_sequence(backend, scope_a, epoch) == 1


def test_compaction_without_the_compression_lease_commits_nothing(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    _seed_rows(store, "a", "b")
    _lease(store, holder="other")
    with pytest.raises(CompressionInProgress):
        _compact(store, epoch, 1, WorkerCompaction(messages=[_row("summary", "assistant")], lock_holder=HOLDER))
    assert (_active(store), last_sequence(backend, scope_a, epoch)) == ([("a", "user"), ("b", "user")], 0)
    # The sequence is still free: an unfenced compaction (micro-compaction) takes it.
    result = _compact(store, epoch, 1, WorkerCompaction(messages=[_row("summary", "assistant")]))
    assert (result.value, _active(store)) == (1, [("summary", "assistant")])


def test_a_carried_tail_longer_than_the_transcript_rewinds_every_row(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    _seed_rows(store, "a", "b", "c")
    _compact(store, epoch, 1, WorkerCompaction(messages=[_row("a"), _row("b"), _row("c")], tail_count=5))
    rows = store.messages.read_transcript(SESSION, TranscriptRead(view=TranscriptView.ALL))
    assert [(r.content, r.visibility) for r in rows if r.visibility is not RowVisibility.ACTIVE] == [
        ("a", RowVisibility.REWOUND), ("b", RowVisibility.REWOUND), ("c", RowVisibility.REWOUND)]


# ── publish_compression_child ─────────────────────────────────────────────────


def _publication_worker(backend, scope, kind=SessionLineageKind.DELEGATE):
    """A fork or reset worker session with a claimed admission and a worker registered at the claimed generation."""
    require_capability(backend, "transactional_claims")
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=ROOT, source="contract"))
    store.sessions.create(NewSession(session_id=SESSION, source="contract", cwd="/work", parent_session_id=ROOT, lineage_kind=kind, display_name="peer"))
    epoch = begin_epoch(backend, scope)
    admission = store.admissions.admit(epoch=epoch, principal_id="principal", session_id=SESSION, request_id="req-1", payload={"input": "hi"}, intent=AdmissionIntent.QUEUE)[0]
    claimed = store.admissions.claim(epoch=epoch, session_id=SESSION)
    assert claimed
    register(backend, scope, epoch, generation=claimed.generation)
    return store, epoch, admission, int(claimed.generation)


def _publish(store, epoch, generation, sequence, publication, session_id=SESSION):
    return store.worker_executions.publish_compression_child(**ident(epoch, generation=generation, sequence=sequence, session_id=session_id), publication=publication)


@pytest.mark.parametrize("kind", [SessionLineageKind.BRANCH, SessionLineageKind.RESET, SessionLineageKind.DELEGATE])
def test_publication_inherits_intent_without_creating_a_new_fork(backend, scope_a, kind):
    store, epoch, _, generation = _publication_worker(backend, scope_a, kind)
    _lease(store)
    _publish(store, epoch, generation, 1, _publication())
    child = store.sessions.get(CHILD)
    assert (child.lineage_kind, child.lineage_origin_id, child.parent_session_id) == (kind, ROOT, SESSION)
    assert store.lineage.lineage(CHILD) == (SESSION, CHILD)
    assert store.lineage.tip(SESSION) == SESSION
    assert not store.lineage.is_fork_child(CHILD)
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="parent", ttl_seconds=300) is True
    assert store.coordination.try_acquire_turn_lease(session_id=CHILD, holder="child", ttl_seconds=300) is False
    # The continuation carries the intent it inherited, so the descendants walk stops at the parent.
    assert [r.session_id for r in store.sessions.descendants(SESSION)] == [SESSION]


def test_publication_child_continues_the_parents_whole_tool_pin(backend, scope_a):
    store, epoch, _, generation = _publication_worker(backend, scope_a)
    pin = {"version": "v1", "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]}
    _lease(store)
    store.sessions.update_tool_pin(SESSION, pin)
    _publish(store, epoch, generation, 1, _publication())
    assert store.sessions.get(CHILD).tool_pin == pin


def test_publication_moves_the_conversation_and_the_worker_in_one_receipt(backend, scope_a):
    store, epoch, admission, generation = _publication_worker(backend, scope_a)
    wx = store.worker_executions
    _seed_rows(store, "a")
    mark = store.messages.active_watermark(SESSION).last_active
    _seed_rows(store, "late")
    store.state_meta.set(f"goal:{SESSION}", GOAL)
    _lease(store)
    publication = _publication(model="m", system_prompt="PREFIX", watermark=mark)
    result = _publish(store, epoch, generation, 1, publication)
    child_rows = store.messages.read_transcript(CHILD)
    assert [r.content for r in child_rows] == ["summary", "late"]
    assert (result.parent, result.session_id, tuple(a.row_id for a in result.annotations)) == (SESSION, CHILD, (child_rows[0].id,))
    parent, child = store.sessions.get(SESSION), store.sessions.get(CHILD)
    assert (parent.end_reason, child.ended_at) == (END_REASON_COMPRESSION, None)
    assert (child.parent_session_id, child.cwd, child.source, child.runtime_generation) == (SESSION, "/work", "contract", generation)
    # A continuation, not a fork: it contends for its parent's turn-lease conversation.
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="turn-a", ttl_seconds=300.0) is True
    assert store.coordination.try_acquire_turn_lease(session_id=CHILD, holder="turn-b", ttl_seconds=300.0) is False
    # Both copies are written through the goal state class: every field of its JSON, the defaults filled in.
    moved = json.loads(GoalState.from_json(GOAL).to_json())
    assert len(moved) == 20 and moved["goal"] == "ship it"
    assert json.loads(store.state_meta.get(f"goal:{CHILD}")) == moved
    assert json.loads(store.state_meta.get(f"goal:{SESSION}")) == dict(moved, status="cleared")
    # The worker now addresses the child; its old assignment is refused.
    assert wx.compression_lock_holder(**ident(epoch, generation=generation, sequence=2, session_id=CHILD)) is None
    with refused("permission_denied"):
        wx.compression_lock_holder(**ident(epoch, generation=generation, sequence=3))
    assert wx.claim_origin(execution_id=EXECUTION, session_id=CHILD) == SESSION
    # The lost-acknowledgement retry still addresses the parent and gets the recorded answer, writing nothing.
    assert _publish(store, epoch, generation, 1, publication) == result
    assert [r.id for r in store.messages.read_transcript(CHILD)] == [r.id for r in child_rows]
    # The same sequence addressed to the child is another request: the digest binds the addressed session.
    with refused("admission_conflict"):
        _publish(store, epoch, generation, 1, publication, session_id=CHILD)
    # The admission's lineage followed the worker: settling it retires the worker on the child.
    store.admissions.settle(epoch=epoch, admission_id=admission.admission_id, generation=ExecutionGeneration(generation), outcome=AdmissionOutcome.COMPLETED)
    with refused("stale_generation"):
        wx.compression_lock_holder(**ident(epoch, generation=generation, sequence=3, session_id=CHILD))


@pytest.mark.parametrize("refusal", ["source", "cwd", "profile", "marker", "child_exists", "lease", "ended", "payload"])
def test_publication_refusals_commit_nothing(backend, scope_a, refusal):
    store, epoch, _, generation = _publication_worker(backend, scope_a)
    _seed_rows(store, "a")
    store.state_meta.set(f"goal:{SESSION}", GOAL)
    _lease(store, holder="other" if refusal == "lease" else HOLDER)
    if refusal == "child_exists":
        store.sessions.create(NewSession(session_id=CHILD, source="contract"))
    elif refusal == "ended":
        store.sessions.end(SESSION, "user_exit")
    publication = {"source": _publication(source="elsewhere"), "cwd": _publication(cwd="/elsewhere"), "profile": _publication(profile_name="another"),
                   "marker": _publication(model_config={"_delegate_from": "x"}), "payload": _publication(require_compression_lease=False)}.get(refusal, _publication())
    expected = {"lease": CompressionBusy, "ended": CompressionPublishRefused}.get(refusal)
    with pytest.raises(expected) if expected else refused("invalid_params" if refusal == "payload" else "permission_denied"):
        _publish(store, epoch, generation, 1, publication)
    assert (store.sessions.get(SESSION).end_reason == "user_exit") is (refusal == "ended")
    if refusal != "child_exists":
        assert store.sessions.get(CHILD) is None
    assert json.loads(store.state_meta.get(f"goal:{SESSION}"))["status"] == "active"
    assert store.state_meta.get(f"goal:{CHILD}") is None
    # The worker never moved and the sequence is still free.
    assert store.worker_executions.compression_lock_holder(**ident(epoch, generation=generation, sequence=1)) in (HOLDER, "other")


def test_publication_refusal_from_the_payload_as_sent_is_the_typed_verbs(backend, scope_a):
    store, epoch, _, generation = _publication_worker(backend, scope_a)
    _seed_rows(store, "a")
    _lease(store)
    store.sessions.end(SESSION, "user_exit")
    with pytest.raises(CompressionPublishRefused):
        store.worker_executions.persist_operation(**ident(epoch, generation=generation), operation="compression.publish", payload=_publication().payload())
    assert store.sessions.get(CHILD) is None
    assert store.worker_executions.compression_lock_holder(**ident(epoch, generation=generation, sequence=1)) == HOLDER


def test_publication_crash_window_publishes_nothing_then_retries_exactly_once(backend, scope_a):
    require_failpoints(backend)
    store, epoch, _, generation = _publication_worker(backend, scope_a)
    _seed_rows(store, "a")
    store.state_meta.set(f"goal:{SESSION}", GOAL)
    _lease(store)
    authority_state(backend, scope_a).failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _publish(store, epoch, generation, 1, _publication())
    assert (store.sessions.get(SESSION).ended_at, store.sessions.get(CHILD), store.state_meta.get(f"goal:{CHILD}")) == (None, None, None)
    assert json.loads(store.state_meta.get(f"goal:{SESSION}"))["status"] == "active"
    assert not store.messages.read_transcript(CHILD)
    assert _publish(store, epoch, generation, 1, _publication()).session_id == CHILD
    assert [r.content for r in store.messages.read_transcript(CHILD)] == ["summary"]


ROUTE = "local-route"


def _bound_worker(backend, scope):
    """A locally bound conversation (binding receipt and routing entry, reserved through the bindings capability) with a worker on it."""
    require_capability(backend, "transactional_claims")
    store = backend.open(scope)
    local = local_session_id(scope.name, "principal", "req-bind")
    epoch = begin_epoch(backend, scope)
    created = "2026-09-25T10:00:00+00:00"
    store.bindings.reserve(epoch=epoch, binding=NewSessionBinding(
        profile_id=scope.name, created_source="cli", session_id=local, route=ROUTE, principal_id="principal", request_id="req-bind",
        policy={"source": "cli", "model": "m", "cwd": "/work", "request_json": "{}"},
        entry={"session_id": str(local), "created_at": created, "updated_at": created,
               "origin": {"chat_id": str(local), "chat_type": "dm", "user_id": "principal", "platform": "local"}}), owner=None)
    register(backend, scope, epoch, session_id=local)
    _lease(store, session_id=local)
    return store, epoch, local


def test_publication_moves_a_local_binding_onto_the_child(backend, scope_a):
    store, epoch, local = _bound_worker(backend, scope_a)
    _publish(store, epoch, 0, 1, _publication(source="cli"), session_id=local)
    binding = store.bindings.receipt(local)
    assert (binding.current_target, binding.lineage) == (CHILD, (local, CHILD))
    assert json.loads(store.routing.load_routing_entries()[ROUTE])["session_id"] == CHILD


def test_publication_refuses_a_route_that_left_the_parent_and_commits_nothing(backend, scope_a):
    store, epoch, local = _bound_worker(backend, scope_a)
    routed = store.routing.load_routing_entries()
    store.routing.upsert_routing_entry(ROUTE, json.dumps({**json.loads(routed[ROUTE]), "session_id": "elsewhere"}))
    with refused("admission_conflict"):
        _publish(store, epoch, 0, 1, _publication(source="cli"), session_id=local)
    assert (store.sessions.get(CHILD), store.sessions.get(local).ended_at, store.bindings.receipt(local).current_target) == (None, None, local)
    # Nothing committed: the same sequence is still free once the route points back at the parent.
    store.routing.upsert_routing_entry(ROUTE, routed[ROUTE])
    _publish(store, epoch, 0, 1, _publication(source="cli"), session_id=local)
    assert store.bindings.receipt(local).current_target == CHILD


# ── append_compression_transcript ─────────────────────────────────────────────


def _append(store, epoch, sequence, messages, *, lock=None, turn=None, ttl=300.0):
    return store.worker_executions.append_compression_transcript(**ident(epoch, sequence=sequence), messages=messages, compression_lock_holder=lock,
                                                                 turn_lease_holder=turn, turn_lease_ttl_seconds=ttl)


def test_compression_append_writes_under_the_turn_lease_with_its_ttl(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="turn", ttl_seconds=300.0) is True
    result = _append(store, epoch, 1, [_row("handoff", "assistant", _compressed_summary_has_user_turn=True)], lock=HOLDER, turn="turn", ttl=30.0)
    stored = store.messages.read_transcript(SESSION)
    assert (result.count, [r.content for r in stored], result.annotations[0].row_id) == (1, ["handoff"], stored[0].id)
    # Another holder's append refuses and commits nothing.
    with pytest.raises(TurnLeaseLost):
        _append(store, epoch, 2, [_row("late")], turn="someone-else")
    assert _append(store, epoch, 1, [_row("handoff", "assistant", _compressed_summary_has_user_turn=True)], lock=HOLDER, turn="turn", ttl=30.0) == result
    assert last_sequence(backend, scope_a, epoch) == 1


def test_compression_append_payload_checks(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    for kwargs in ({"lock": ""}, {"turn": "x" * 1025}, {"ttl": 0.05}, {"ttl": 3601}, {"ttl": float("nan")}):
        with refused("invalid_params"):
            _append(store, epoch, 1, [_row("x")], **kwargs)
    with refused("invalid_params"):
        _append(store, epoch, 1, [_row("x", unknown=1)])
    assert (store.messages.read_transcript(SESSION), last_sequence(backend, scope_a, epoch)) == ((), 0)
    # The compression lock holder rides along but fences nothing: the append guard does not read the lock.
    _lease(store, holder="other")
    assert _append(store, epoch, 1, [_row("x")], lock=HOLDER).count == 1
