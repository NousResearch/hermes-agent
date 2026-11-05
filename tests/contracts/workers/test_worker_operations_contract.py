"""Contract suite for the receipted execution operations of ``WorkerExecutionRepository``: ``usage.main``, ``usage.auxiliary``, ``execution.finish``, ``child.reserve``, the turn lease and an operation applied from its payload as sent."""

from __future__ import annotations

import time

import pytest

from persistence import SessionId
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from persistence.conversation.usage import UsageDelta
from persistence.core.fences import AuthorityEpoch
from persistence.workers.delegations import WorkerDelegationPayload
from persistence.workers.executions import WorkerExecutionId, WorkerStatus, WorkerTurnLease
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts._waiting import wait_past
from tests.contracts.authority.test_coordination import dead_holder, race_on_threads
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts.conversation.test_usage import accounting
from tests.contracts.workers._support import EXECUTION, SESSION, ident, last_sequence, refused, revision, worker

HOLDER = "turn-holder-a"
CHILD_SECRET = "c" * 32


def test_record_usage_lands_aggregate_and_attribution_exactly_once(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    usage = {"input_tokens": 100, "output_tokens": 40, "api_call_count": 2, "estimated_cost_usd": 0.25, "model": "model-x", "billing_provider": "prov"}
    store.worker_executions.record_usage(**ident(epoch), **usage)
    stored = accounting(store, SESSION)
    assert (stored.tokens.input_tokens, stored.tokens.output_tokens, stored.api_call_count) == (100, 40, 2)
    assert stored.estimated_cost_usd == pytest.approx(0.25)
    route = store.usage.recent_route(SESSION)
    assert (route.route.model, route.api_call_count) == ("model-x", 2)
    store.worker_executions.record_usage(**ident(epoch), **usage)
    assert (accounting(store, SESSION).tokens.input_tokens, accounting(store, SESSION).api_call_count) == (100, 2)
    with refused("admission_conflict"):
        store.worker_executions.record_usage(**ident(epoch), input_tokens=999)


def test_record_usage_refuses_out_of_bounds_values_inside_the_receipt(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    for bad in ({"input_tokens": -1}, {"api_call_count": True}, {"output_tokens": 2**53 + 1}, {"estimated_cost_usd": -0.1},
                {"actual_cost_usd": 2e12}, {"absolute": 1}, {"model": "m" * 4097}, {"unknown_key": 1}):
        with refused("invalid_params"):
            store.worker_executions.record_usage(**ident(epoch), **bad)
    assert last_sequence(backend, scope_a, epoch) == 0
    store.worker_executions.record_usage(**ident(epoch), input_tokens=1, source={"not": "a label"})


def test_record_usage_stamps_the_worker_surface_on_an_unset_created_source(backend, scope_a):
    # ``source`` is set aside before the key check and maps through created_surface (``gui`` is ``desktop``); ``local`` counts as unset.
    store, epoch = worker(backend, scope_a, source="local")
    store.worker_executions.record_usage(**ident(epoch), input_tokens=1, source="gui")
    assert store.sessions.get(SESSION).created_source == "desktop"
    store.worker_executions.record_usage(**ident(epoch, sequence=2), input_tokens=1, source="tui")
    assert store.sessions.get(SESSION).created_source == "desktop"


def test_record_usage_absolute_sets_the_counters_without_attribution(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.worker_executions.record_usage(**ident(epoch), input_tokens=10, api_call_count=2, model="m", billing_provider="p")
    store.worker_executions.record_usage(**ident(epoch, sequence=2), input_tokens=4, api_call_count=1, absolute=True)
    stored = accounting(store, SESSION)
    assert (stored.tokens.input_tokens, stored.api_call_count) == (4, 1)
    assert store.usage.recent_route(SESSION).api_call_count == 2


def test_record_auxiliary_usage_never_touches_the_main_aggregate(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    baseline = accounting(store, SESSION)
    store.worker_executions.record_auxiliary_usage(**ident(epoch), task="vision", input_tokens=50, api_call_count=1, model="model-aux")
    assert accounting(store, SESSION) == baseline
    assert store.usage.recent_route(SESSION) is None


def test_record_auxiliary_usage_refuses_bad_reports_and_leaves_the_sequence_free(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    aux = store.worker_executions.record_auxiliary_usage
    with refused("invalid_params"):
        aux(**ident(epoch), task="", input_tokens=1)
    for main_only in ("actual_cost_usd", "cost_status", "cost_source", "billing_mode", "pricing_version", "absolute"):
        with refused("invalid_params"):
            aux(**ident(epoch), task="vision", input_tokens=1, **{main_only: None})
    store.worker_executions.append_transcript(**ident(epoch), messages=[{"role": "assistant", "content": "still-first"}])


def test_usage_repository_and_worker_usage_share_one_aggregate(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.usage.queue_usage(SESSION, UsageDelta(api_call_count=1))
    store.worker_executions.record_usage(**ident(epoch), api_call_count=2, input_tokens=3, model="model-x")
    stored = accounting(store, SESSION)
    assert (stored.api_call_count, stored.tokens.input_tokens) == (3, 3)


def test_usage_crash_window_publishes_nothing_then_retries_exactly_once(backend, scope_a):
    require_failpoints(backend)
    store, epoch = worker(backend, scope_a)
    baseline, before = accounting(store, SESSION), revision(store)
    authority_state(backend, scope_a).failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        store.worker_executions.record_usage(**ident(epoch), input_tokens=10, api_call_count=1, model="model-x")
    assert (accounting(store, SESSION), store.usage.recent_route(SESSION), revision(store)) == (baseline, None, before)
    store.worker_executions.record_usage(**ident(epoch), input_tokens=10, api_call_count=1, model="model-x")
    assert (accounting(store, SESSION).tokens.input_tokens, revision(store)) == (10, before + 1)


def test_finish_is_terminal_receipted_and_replayable(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    store.worker_executions.append_transcript(**ident(epoch), messages=[{"role": "assistant", "content": "work"}])
    store.worker_executions.finish(**ident(epoch, sequence=2))
    store.worker_executions.finish(**ident(epoch, sequence=2))
    with refused("stale_generation"):
        store.worker_executions.append_transcript(**ident(epoch, sequence=3), messages=[{"role": "assistant", "content": "late"}])
    assert len(store.messages.read_transcript(SESSION)) == 1
    assert store.sessions.get(SESSION).ended_at is None


def test_finish_retires_live_children_and_ends_their_rows(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    reserved = _reserve_child(store, epoch).child
    assert store.sessions.get(reserved.session_id).ended_at is None
    store.worker_executions.finish(**ident(epoch, sequence=2))
    child = store.worker_executions.verify_claim(execution_id=reserved.execution_id, session_id=reserved.session_id, generation=reserved.generation,
                                                 adoption_secret=_child_claim(reserved))
    assert child.status is WorkerStatus.TERMINAL
    assert store.sessions.get(reserved.session_id).end_reason == "agent_close"
    with refused("stale_generation"):
        store.worker_executions.append_transcript(epoch=epoch, execution_id=reserved.execution_id, session_id=reserved.session_id,
                                                  generation=reserved.generation, sequence=1, messages=[])


# ── child.reserve ─────────────────────────────────────────────────────────────

_IDENTITY = {"profile_id": "profile-a", "pid": 4242, "birth": 1700000000.5}


def _reserve_child(store, epoch, *, sequence=1, secret=CHILD_SECRET, limits=(10, 1), refused_code=None, execution_id=EXECUTION, session_id=SESSION, generation=0):
    return store.worker_executions.reserve_child(**ident(epoch, sequence=sequence, execution_id=execution_id, session_id=session_id, generation=generation),
                                                 secret=secret, principal="subject-a", max_concurrent_children=limits[0], max_spawn_depth=limits[1],
                                                 refused=refused_code, **_IDENTITY)


def _child_claim(child) -> str:
    """The claim a reserved child presents: the scope its owner minted, bound to the principal."""
    from persistence.authority.vocabulary import admission_fingerprint

    scope = dict(_IDENTITY, session_id=child.session_id, execution_id=child.execution_id, generation=0, secret=CHILD_SECRET)
    return admission_fingerprint(canonical_target=child.session_id, payload=scope | {"principal": "subject-a"})


def test_reserve_child_mints_a_running_child_and_its_subagent_row(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    reservation = _reserve_child(store, epoch)
    child = reservation.child
    assert reservation.refused is None
    assert child.execution_id.startswith(EXECUTION + "/child:") and child.generation == 0
    row = store.sessions.get(child.session_id)
    assert (row.source, row.parent_session_id, row.model_config.get("_delegate_from"), row.profile_name) == ("subagent", SESSION, SESSION, store.sessions.get(SESSION).profile_name)
    record = store.worker_executions.verify_claim(execution_id=child.execution_id, session_id=child.session_id, generation=child.generation, adoption_secret=_child_claim(child))
    assert (record.kind, record.status, record.owner_epoch) == ("child", WorkerStatus.RUNNING, epoch)
    # The exact retry returns the recorded child, never a second one.
    assert _reserve_child(store, epoch) == reservation
    with refused("admission_conflict"):
        _reserve_child(store, epoch, secret="d" * 32)


def test_reserve_child_refusals_are_committed_receipts(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    assert _reserve_child(store, epoch, refused_code="runtime_draining").refused == "runtime_draining"
    assert _reserve_child(store, epoch, sequence=2, limits=(1, 1)).child is not None
    assert _reserve_child(store, epoch, sequence=3, limits=(1, 1)).refused == "delegation_capacity"
    child = _reserve_child(store, epoch, sequence=4, limits=(2, 1)).child
    # A child's own reservation at depth 1 exceeds a spawn depth of 1.
    nested = store.worker_executions.reserve_child(epoch=epoch, execution_id=child.execution_id, session_id=child.session_id, generation=child.generation,
                                                   sequence=1, secret=CHILD_SECRET, principal="subject-a", max_concurrent_children=10, max_spawn_depth=1, **_IDENTITY)
    assert nested.refused == "delegation_depth_exceeded"
    assert last_sequence(backend, scope_a, epoch) == 4


def test_reserve_child_digests_only_the_secret(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    first = _reserve_child(store, epoch)
    # The trusted claim fields, limits and refusal are not digested: a retry with others replays the receipt.
    assert _reserve_child(store, epoch, limits=(0, 0), refused_code="runtime_draining") == first
    for secret in ("short", "x" * 257, 7):
        with refused("invalid_params"):
            _reserve_child(store, epoch, sequence=2, secret=secret)


def test_reserve_child_takes_the_payload_as_sent_and_fences_before_its_shape(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    malformed = WorkerDelegationPayload({"secret": CHILD_SECRET, "extra": 1})
    # A stale epoch answers the fence even when the payload has no reservation to name.
    with refused("stale_epoch"):
        _reserve_child(store, AuthorityEpoch(int(epoch) + 7), secret=malformed)
    for payload in ({"secret": CHILD_SECRET, "extra": 1}, {}, {"other": CHILD_SECRET}):
        with refused("invalid_params"):
            _reserve_child(store, epoch, secret=WorkerDelegationPayload(payload))
    assert last_sequence(backend, scope_a, epoch) == 0
    # The well-formed payload as sent is the typed request: same digest, so the typed retry replays it.
    reservation = _reserve_child(store, epoch, secret=WorkerDelegationPayload({"secret": CHILD_SECRET}))
    assert reservation.child is not None and _reserve_child(store, epoch) == reservation


# ── an operation from its payload as sent ─────────────────────────────────────


def _persist(store, epoch, operation, payload, **identity):
    return store.worker_executions.persist_operation(**ident(epoch, **identity), operation=operation, payload=payload)


def test_persist_operation_shares_the_typed_verbs_receipt(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    result = _persist(store, epoch, "transcript.append", {"messages": [{"role": "assistant", "content": "work"}]})
    assert result["count"] == 1 and len(result["annotations"]) == 1
    # The typed verb's payload is the producer's: its retry replays the receipt the raw request committed, and the reverse.
    assert store.worker_executions.append_transcript(**ident(epoch), messages=[{"role": "assistant", "content": "work"}]).count == 1
    assert store.worker_executions.latest_conversation_role(**ident(epoch, sequence=2)) == "assistant"
    assert _persist(store, epoch, "session.tail_role", {}, sequence=2) == {"value": "assistant"}
    with refused("admission_conflict"):
        _persist(store, epoch, "session.tail_role", {"other": 1}, sequence=2)
    assert [r.content for r in store.messages.read_transcript(SESSION)] == ["work"]
    assert last_sequence(backend, scope_a, epoch) == 2


def test_persist_operation_fences_before_the_payload_shape(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    stale = AuthorityEpoch(int(epoch) + 7)
    # Keys no typed verb could carry still meet the fences first, then the operation's own check.
    for operation, payload in (("usage.main", {"unknown_key": 1}), ("session.create", {}), ("transcript.append", {"messages": [], "extra": 1})):
        with refused("stale_epoch"):
            _persist(store, stale, operation, payload)
        with refused("invalid_params"):
            _persist(store, epoch, operation, payload)
    assert last_sequence(backend, scope_a, epoch) == 0


def test_persist_operation_refuses_operations_outside_its_table_before_the_store(backend, scope_a):
    store, epoch = worker(backend, scope_a)
    stale = AuthorityEpoch(int(epoch) + 7)
    for operation in ("child.reserve", "delegation.read", "session.context", "turn.release", "no.such", 7):
        with refused("invalid_params"):
            _persist(store, stale, operation, {})
    with refused("invalid_params"):
        _persist(store, stale, "execution.finish", [])
    assert last_sequence(backend, scope_a, epoch) == 0


# ── turn lease ────────────────────────────────────────────────────────────────


def _turn(store, epoch, sequence, verb, **kwargs):
    return getattr(store.worker_executions, verb)(**ident(epoch, sequence=sequence), **kwargs)


def _turn_worker(backend, scope):
    require_capability(backend, "transactional_claims")
    return worker(backend, scope)


def test_turn_lease_operations_share_the_coordination_lease(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    coordination = store.coordination
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease(HOLDER, 300.0)) is True
    assert coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is False
    assert _turn(store, epoch, 2, "acquire_turn_lease", lease=WorkerTurnLease(HOLDER, 300.0)) is True
    assert _turn(store, epoch, 3, "renew_turn_lease", lease=WorkerTurnLease(HOLDER, 60)) is True
    assert _turn(store, epoch, 4, "renew_turn_lease", lease=WorkerTurnLease("other", 60)) is False
    _turn(store, epoch, 5, "cleanup_turn_lease", target_session_id=SESSION, holder="other")
    assert coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is False
    _turn(store, epoch, 6, "cleanup_turn_lease", target_session_id=SESSION, holder=HOLDER)
    assert coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    assert _turn(store, epoch, 7, "acquire_turn_lease", lease=WorkerTurnLease(HOLDER, 300.0)) is False
    assert coordination.refresh_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    assert last_sequence(backend, scope_a, epoch) == 7


def test_turn_lease_replay_returns_the_recorded_outcome(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    lease = WorkerTurnLease(HOLDER, 300.0)
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=lease) is True
    store.coordination.release_turn_lease(session_id=SESSION, holder=HOLDER)
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    # The exact duplicate replays True without re-deciding against the new holder, and takes nothing.
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=lease) is True
    assert store.coordination.refresh_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    with refused("admission_conflict"):
        _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease(HOLDER, 60.0))
    with refused("admission_conflict"):
        _turn(store, epoch, 1, "cleanup_turn_lease", target_session_id=SESSION, holder=HOLDER)
    with refused("invalid_params"):
        _turn(store, epoch, 3, "renew_turn_lease", lease=lease)
    store.worker_executions.finish(**ident(epoch, sequence=2))
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=lease) is True
    with refused("stale_generation"):
        _turn(store, epoch, 3, "renew_turn_lease", lease=lease)


def test_turn_lease_fences_and_payload_checks_refuse_without_work(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    with refused("stale_epoch"):
        _turn(store, epoch + 1, 1, "acquire_turn_lease", lease=WorkerTurnLease(HOLDER))
    with refused("not_found"):
        store.worker_executions.acquire_turn_lease(**ident(epoch, execution_id=WorkerExecutionId("ghost")), lease=WorkerTurnLease(HOLDER))
    for bad in (WorkerTurnLease(""), WorkerTurnLease("h" * 1025), WorkerTurnLease(HOLDER, 0.05), WorkerTurnLease(HOLDER, 3601),
                WorkerTurnLease(HOLDER, True), WorkerTurnLease(HOLDER, "300")):
        with refused("invalid_params"):
            _turn(store, epoch, 1, "acquire_turn_lease", lease=bad)
    for target, holder in ((SessionId(""), HOLDER), (SESSION, "")):
        with refused("invalid_params"):
            _turn(store, epoch, 1, "cleanup_turn_lease", target_session_id=target, holder=holder)
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    assert last_sequence(backend, scope_a, epoch) == 0


def test_turn_cleanup_releases_only_along_the_compression_lineage(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    child, foreign = SessionId("sess-worker-cont"), SessionId("sess-foreign")
    store.sessions.end(SESSION, END_REASON_COMPRESSION)
    store.sessions.create(NewSession(session_id=child, source="contract", parent_session_id=SESSION))
    store.sessions.create(NewSession(session_id=foreign, source="contract"))
    assert store.coordination.try_acquire_turn_lease(session_id=child, holder=HOLDER, ttl_seconds=300.0) is True
    for target in (foreign, SessionId("sess-missing")):
        with refused("permission_denied"):
            _turn(store, epoch, 1, "cleanup_turn_lease", target_session_id=target, holder=HOLDER)
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is False
    _turn(store, epoch, 1, "cleanup_turn_lease", target_session_id=child, holder=HOLDER)
    assert store.coordination.try_acquire_turn_lease(session_id=child, holder="other", ttl_seconds=300.0) is True
    # The replay returns without re-authorizing or releasing again.
    _turn(store, epoch, 1, "cleanup_turn_lease", target_session_id=child, holder=HOLDER)
    assert store.coordination.refresh_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True


def test_turn_lease_acquire_reclaims_an_expired_or_dead_holder(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    coordination = store.coordination
    assert coordination.try_acquire_turn_lease(session_id=SESSION, holder="short", ttl_seconds=0.2) is True
    wait_past(time.time() + 0.2)
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn")) is True
    _turn(store, epoch, 2, "cleanup_turn_lease", target_session_id=SESSION, holder="worker-turn")
    assert coordination.try_acquire_turn_lease(session_id=SESSION, holder=dead_holder("turn=t1:platform=cli"), ttl_seconds=300.0) is True
    assert _turn(store, epoch, 3, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn")) is True
    assert coordination.try_acquire_turn_lease(session_id=SESSION, holder="late", ttl_seconds=300.0) is False


def test_turn_lease_crash_window_publishes_nothing_then_retries_exactly_once(backend, scope_a):
    require_failpoints(backend)
    store, epoch = _turn_worker(backend, scope_a)
    authority_state(backend, scope_a).failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn"))
    with refused("invalid_params"):  # the sequence did not advance
        _turn(store, epoch, 2, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn"))
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    store.coordination.release_turn_lease(session_id=SESSION, holder="other")
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn")) is True
    assert store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0) is False


def test_turn_lease_acquire_races_a_coordination_contender_to_one_winner(backend, scope_a):
    store, epoch = _turn_worker(backend, scope_a)
    worker_won, other = race_on_threads([
        lambda: _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn")),
        lambda: store.coordination.try_acquire_turn_lease(session_id=SESSION, holder="other", ttl_seconds=300.0),
    ])
    assert worker_won is not other
    assert _turn(store, epoch, 1, "acquire_turn_lease", lease=WorkerTurnLease("worker-turn")) is worker_won
    assert last_sequence(backend, scope_a, epoch) == 1
